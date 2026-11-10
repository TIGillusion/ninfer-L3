#include "serve/research_request.h"

#include "serve/openai_chat.h"

#include <cmath>
#include <stdexcept>
#include <utility>

namespace ninfer::serve {
namespace {

constexpr int kResearchTopKMaximum = 100;
constexpr int kResearchDefaultTopK = 10;

ApiException research_bad_request(const std::string& message, const std::string& param,
                                  const std::string& code = "invalid_request") {
    ApiError error;
    error.status  = 400;
    error.type    = "invalid_request_error";
    error.code    = code;
    error.message = message;
    error.param   = param;
    return ApiException(error);
}

} // namespace

ResearchScoreRequest parse_research_score_request(const RequestJson& body) {
    if (!body.is_object()) { throw research_bad_request("body must be a JSON object", ""); }
    ResearchScoreRequest request;
    request.text = body.value("text", std::string{});
    if (request.text.empty()) {
        throw research_bad_request("text must be a non-empty string", "text");
    }
    return request;
}

ResearchNextTokenRequest parse_research_next_token_request(const RequestJson& body,
                                                           const RequestLimits& limits) {
    if (!body.is_object()) { throw research_bad_request("body must be a JSON object", ""); }
    ResearchNextTokenRequest request;
    const bool has_messages = body.contains("messages");
    const bool has_text     = body.contains("text");
    if (has_messages == has_text) {
        throw research_bad_request("exactly one of \"messages\" or \"text\" must be provided",
                                   "");
    }
    request.top_k = body.value("top_k", kResearchDefaultTopK);
    if (request.top_k < 1 || request.top_k > kResearchTopKMaximum) {
        throw research_bad_request("top_k must be in [1," + std::to_string(kResearchTopKMaximum) +
                                       "]",
                                   "top_k");
    }
    if (has_text) {
        request.text = body.value("text", std::string{});
        if (request.text->empty()) {
            throw research_bad_request("text must be a non-empty string", "text");
        }
        return request;
    }
    // The chat wire adapter owns message parsing, validation, and template application. One
    // output token is consumed and discarded; the exported distribution comes from the
    // prefill logits (top_logprobs carries top_k through the neutral request).
    RequestJson chat_body    = body;
    chat_body["max_tokens"]   = 1;
    chat_body["stream"]       = false;
    chat_body["top_logprobs"] = request.top_k;
    request.generation        = parse_chat_completion_request(chat_body, limits).generation;
    return request;
}

ScoreAggregate aggregate_score(const std::vector<float>& logprobs) {
    if (logprobs.size() < 2) {
        throw std::runtime_error("scoring requires at least two prompt tokens");
    }
    ScoreAggregate out;
    out.scored_tokens = logprobs.size() - 1;
    double total_nll  = 0.0;
    for (std::size_t i = 1; i < logprobs.size(); ++i) {
        if (!std::isfinite(logprobs[i])) {
            throw std::runtime_error("engine returned a non-finite prompt logprob");
        }
        total_nll -= static_cast<double>(logprobs[i]);
    }
    out.total_nll  = total_nll;
    out.mean_nll   = total_nll / static_cast<double>(out.scored_tokens);
    out.perplexity = std::exp(out.mean_nll);
    return out;
}

} // namespace ninfer::serve
