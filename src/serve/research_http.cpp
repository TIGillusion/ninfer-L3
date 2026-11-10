#include "serve/http_server.h"

#include "serve/http_transport.h"
#include "serve/openai_common.h"
#include "serve/research_request.h"

#include <nlohmann/json.hpp>

#include <exception>
#include <string>
#include <utility>
#include <vector>

namespace ninfer::serve {
namespace {

// Shared ingestion for the research endpoints: the exports are prefill-side side channels and
// the single generated token each request produces is discarded. Greedy keeps that token
// deterministic; neither export reads it back.
ninfer::RequestOptions research_scoring_options(bool prompt_logprobs,
                                                std::optional<std::uint32_t> top_k) {
    ninfer::RequestOptions options;
    options.execution.sampling.temperature    = 0.0F;
    options.execution.requested_output_tokens = 1;
    options.execution.allow_prefix_reuse      = true;
    options.top_logprobs                      = top_k;
    options.prompt_logprobs                   = prompt_logprobs;
    return options;
}

void validate_optional_model(const RequestJson& body, const std::string& public_model_id) {
    if (body.contains("model")) {
        validate_openai_model(body.value("model", std::string{}), public_model_id);
    }
}

// Engine-level size guards beyond the parser: tokenize first, then enforce the context
// contract (one generated token of headroom) with a clear message.
ApiException research_parse_error(const std::string& message, const std::string& param) {
    ApiError error;
    error.status  = 400;
    error.type    = "invalid_request_error";
    error.code    = "invalid_request";
    error.message = message;
    error.param   = param;
    return ApiException(error);
}

void log_internal_failure(OperationalLog& log, const char* endpoint,
                          const std::exception& exception) {
    log.http_failure(endpoint,
                     make_internal_request_failure(RequestFailurePhase::Http, exception.what()),
                     {});
}

} // namespace

void HttpServer::handle_score(const httplib::Request& req, httplib::Response& res) {
    try {
        const RequestJson body = parse_json_body(req);
        ResearchScoreRequest request = parse_research_score_request(body);
        validate_optional_model(body, public_model_id_);

        std::vector<ninfer::TokenId> tokens = service_->tokenize_text(request.text);
        const std::uint32_t max_context = service_->engine_options().max_context;
        if (tokens.size() < 2) {
            throw research_parse_error(
                "text must tokenize to at least 2 tokens to score one conditional probability",
                "text");
        }
        if (tokens.size() > static_cast<std::size_t>(max_context) - 1U) {
            throw research_parse_error(
                "text tokenizes to " + std::to_string(tokens.size()) +
                    " tokens, exceeding the scorable maximum of " +
                    std::to_string(max_context - 1U) + " (engine max_context " +
                    std::to_string(max_context) + " minus one generated token); chunk the "
                    "text and score windows separately",
                "text");
        }

        PreparedRequest prepared =
            service_->prepare_raw_tokens(tokens,
                                         research_scoring_options(/*prompt_logprobs=*/true,
                                                                  std::nullopt),
                                         [&req] { return client_disconnected(req); });
        GenerationOutcome outcome = service_->run(prepared, nullptr);
        if (!outcome.prompt_token_logprobs ||
            outcome.prompt_token_logprobs->size() != tokens.size()) {
            ApiError error;
            error.status  = 500;
            error.type    = "internal_error";
            error.message = "engine did not return prompt logprobs for the scored text";
            throw ApiException(error);
        }

        const ScoreAggregate aggregate = aggregate_score(*outcome.prompt_token_logprobs);
        RequestJson logprob_array      = RequestJson::array();
        RequestJson token_array        = RequestJson::array();
        logprob_array.push_back(nullptr);   // position 0 has no conditional probability
        for (std::size_t i = 0; i < tokens.size(); ++i) {
            token_array.push_back(service_->decode_token(static_cast<std::uint32_t>(tokens[i])));
            if (i != 0) { logprob_array.push_back((*outcome.prompt_token_logprobs)[i]); }
        }
        RequestJson response{
            {"object", "score"},
            {"model", public_model_id_},
            {"token_count", tokens.size()},
            {"tokens", std::move(token_array)},
            {"token_logprobs", std::move(logprob_array)},
            {"aggregate",
             {{"scored_tokens", aggregate.scored_tokens},
              {"total_nll", aggregate.total_nll},
              {"mean_nll", aggregate.mean_nll},
              {"perplexity", aggregate.perplexity}}},
        };
        res.set_content(response.dump(), "application/json");
    } catch (const ApiException& exception) {
        write_openai_error(res, exception.error());
    } catch (const std::exception& exception) {
        log_internal_failure(operational_log_, "research_score", exception);
        ApiError error;
        error.status  = 500;
        error.type    = "internal_error";
        error.message = exception.what();
        write_openai_error(res, error);
    }
}

void HttpServer::handle_next_token(const httplib::Request& req, httplib::Response& res) {
    try {
        const RequestJson body = parse_json_body(req);
        validate_optional_model(body, public_model_id_);
        const RequestLimits limits{.default_max_tokens = options_.default_max_tokens};
        // The chat wire adapter requires an explicit model alias; default it to the
        // configured one for the messages carrier.
        RequestJson request_body = body;
        if (!request_body.contains("model")) { request_body["model"] = public_model_id_; }
        ResearchNextTokenRequest request =
            parse_research_next_token_request(request_body, limits);

        PreparedRequest prepared;
        if (request.generation) {
            prepared = service_->prepare(*request.generation, GenerationConsumerMode::Aggregate,
                                         {}, [&req] { return client_disconnected(req); });
        } else {
            std::vector<ninfer::TokenId> tokens =
                service_->tokenize_text(*request.text);
            const std::uint32_t max_context = service_->engine_options().max_context;
            if (tokens.size() > static_cast<std::size_t>(max_context) - 1U) {
                throw research_parse_error(
                    "text tokenizes to " + std::to_string(tokens.size()) +
                        " tokens, exceeding the maximum of " + std::to_string(max_context - 1U),
                    "text");
            }
            prepared = service_->prepare_raw_tokens(
                std::move(tokens),
                research_scoring_options(/*prompt_logprobs=*/false,
                                         static_cast<std::uint32_t>(request.top_k)),
                [&req] { return client_disconnected(req); });
        }
        GenerationOutcome outcome = service_->run(prepared, nullptr);
        if (outcome.next_token_top_logprobs.empty()) {
            ApiError error;
            error.status  = 500;
            error.type    = "internal_error";
            error.message = "engine did not return the next-token distribution";
            throw ApiException(error);
        }

        RequestJson top = RequestJson::array();
        for (const auto& entry : outcome.next_token_top_logprobs) {
            top.push_back({{"token_id", entry.token_id},
                           {"token_text", entry.token_text},
                           {"probability", entry.probability},
                           {"logprob", entry.logprob}});
        }
        RequestJson response{
            {"object", "next_token"},
            {"model", public_model_id_},
            {"prompt_token_count", outcome.prompt_tokens},
            {"top_k", request.top_k},
            {"top", std::move(top)},
        };
        if (outcome.first_token_logprob) {
            response["chosen"] = {
                {"token_text", !outcome.first_token_text.empty() ? outcome.first_token_text
                                                                 : outcome.text},
                {"logprob", *outcome.first_token_logprob}};
        }
        res.set_content(response.dump(), "application/json");
    } catch (const ApiException& exception) {
        write_openai_error(res, exception.error());
    } catch (const std::exception& exception) {
        log_internal_failure(operational_log_, "research_next_token", exception);
        ApiError error;
        error.status  = 500;
        error.type    = "internal_error";
        error.message = exception.what();
        write_openai_error(res, error);
    }
}

} // namespace ninfer::serve
