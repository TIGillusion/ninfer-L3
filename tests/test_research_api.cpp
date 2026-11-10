// Research API wire contract: request parsing for /v1/score and /v1/next-token plus the
// deterministic score aggregate. CPU-only (no Engine); execution paths are covered by the
// production smoke checks in docs/serving.md.

#include "serve/openai_common.h"
#include "serve/research_request.h"

#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using Json    = ninfer::serve::RequestJson;
namespace srv = ninfer::serve;

int check(bool condition, const std::string& label) {
    if (condition) { return 0; }
    std::cerr << "FAIL: " << label << '\n';
    return 1;
}

template <typename Function>
srv::ApiError api_error(Function&& function) {
    try {
        function();
    } catch (const srv::ApiException& exception) { return exception.error(); }
    return srv::ApiError{.status = 0, .message = "no exception"};
}

Json score_body(const std::string& text) { return Json{{"model", "m"}, {"text", text}}; }

int test_score_parsing() {
    int failures = 0;
    const srv::ResearchScoreRequest parsed =
        srv::parse_research_score_request(score_body("hello world"));
    failures += check(parsed.text == "hello world", "score text parsed");

    const auto empty_text  = api_error([&] { (void)srv::parse_research_score_request(score_body("")); });
    failures += check(empty_text.status == 400 && empty_text.param == "text",
                      "empty score text rejected");
    const auto missing_text = api_error([&] {
        (void)srv::parse_research_score_request(Json{{"model", "m"}});
    });
    failures += check(missing_text.status == 400, "missing score text rejected");
    const auto array_body = api_error([&] {
        (void)srv::parse_research_score_request(Json::array({1, 2}));
    });
    failures += check(array_body.status == 400, "non-object score body rejected");
    return failures;
}

Json next_token_text_body() {
    return Json{{"model", "m"}, {"text", "hello"}, {"top_k", 7}};
}

Json next_token_messages_body() {
    return Json{{"model", "m"},
                {"messages", Json::array({Json{{"role", "user"}, {"content", "hi"}}})},
                {"top_k", 3}};
}

int test_next_token_parsing() {
    int failures = 0;
    const srv::RequestLimits limits{};

    const srv::ResearchNextTokenRequest text =
        srv::parse_research_next_token_request(next_token_text_body(), limits);
    failures += check(text.text && *text.text == "hello", "text carrier set");
    failures += check(!text.generation.has_value(), "text carrier has no generation");
    failures += check(text.top_k == 7, "explicit top_k parsed");

    const srv::ResearchNextTokenRequest messages =
        srv::parse_research_next_token_request(next_token_messages_body(), limits);
    failures += check(messages.generation.has_value(), "messages carrier set");
    failures += check(!messages.text.has_value(), "messages carrier has no text");
    if (messages.generation) {
        failures += check(messages.generation->messages.size() == 1,
                          "messages parsed through the chat adapter");
        failures += check(messages.generation->max_tokens == 1,
                          "one-token output budget injected");
        failures += check(messages.generation->top_logprobs.has_value() &&
                              *messages.generation->top_logprobs == 3,
                          "top_k carried as top_logprobs");
    }

    const srv::ResearchNextTokenRequest defaulted =
        srv::parse_research_next_token_request(
            Json{{"text", "hello"}}, limits);
    failures += check(defaulted.top_k == 10, "top_k default is 10");

    Json both            = next_token_text_body();
    both["messages"]     = Json::array();
    const auto both_body = api_error([&] { (void)srv::parse_research_next_token_request(both, limits); });
    failures += check(both_body.status == 400, "text and messages together rejected");

    const auto neither = api_error([&] {
        (void)srv::parse_research_next_token_request(Json{{"model", "m"}}, limits);
    });
    failures += check(neither.status == 400, "neither carrier rejected");

    for (const int bad_top_k : {0, 101, -1}) {
        Json body             = next_token_text_body();
        body["top_k"]         = bad_top_k;
        const auto out_bounds = api_error([&] {
            (void)srv::parse_research_next_token_request(body, limits);
        });
        failures += check(out_bounds.status == 400 && out_bounds.param == "top_k",
                          "top_k out of range rejected");
    }

    Json empty_text_body  = next_token_text_body();
    empty_text_body["text"] = "";
    const auto empty_text = api_error([&] {
        (void)srv::parse_research_next_token_request(empty_text_body, limits);
    });
    failures += check(empty_text.status == 400 && empty_text.param == "text",
                      "empty next-token text rejected");
    return failures;
}

int test_score_aggregate() {
    int failures = 0;
    // log p2 = ln 0.5, log p3 = ln 0.25: total NLL = ln 2 + ln 4 = ln 8, ppl = sqrt(8).
    const srv::ScoreAggregate aggregate =
        srv::aggregate_score(std::vector<float>{0.0F, std::log(0.5F), std::log(0.25F)});
    failures += check(aggregate.scored_tokens == 2, "aggregate scored token count");
    failures += check(std::abs(aggregate.total_nll - std::log(8.0)) < 1e-6,
                      "aggregate total NLL excludes position 0");
    failures += check(std::abs(aggregate.mean_nll - 0.5 * std::log(8.0)) < 1e-6,
                      "aggregate mean NLL");
    failures += check(std::abs(aggregate.perplexity - std::sqrt(8.0)) < 1e-6,
                      "aggregate perplexity is exp(mean NLL)");

    bool threw_non_finite = false;
    try {
        (void)srv::aggregate_score(std::vector<float>{
            0.0F, std::numeric_limits<float>::infinity()});
    } catch (const std::runtime_error&) { threw_non_finite = true; }
    failures += check(threw_non_finite, "non-finite logprob rejected");

    bool threw_short = false;
    try {
        (void)srv::aggregate_score(std::vector<float>{0.0F});
    } catch (const std::runtime_error&) { threw_short = true; }
    failures += check(threw_short, "single-token scoring rejected");
    return failures;
}

} // namespace

int main() {
    int failures = 0;
    failures += test_score_parsing();
    failures += test_next_token_parsing();
    failures += test_score_aggregate();
    if (failures == 0) { std::cout << "research api test PASS\n"; }
    return failures == 0 ? 0 : 1;
}
