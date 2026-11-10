#pragma once

// Research API wire adapter. Parsing and aggregate math for the dedicated research endpoints
// (POST /v1/score, POST /v1/next-token); the HTTP handlers own execution and rendering.

#include "serve/request.h"
#include "serve/request_json.h"

#include <optional>
#include <string>
#include <vector>

namespace ninfer::serve {

struct ResearchScoreRequest {
    std::string text;
};

// Throws ApiException on any malformed body. "model" is validated by the caller against the
// public model id (the endpoints treat it as optional).
[[nodiscard]] ResearchScoreRequest parse_research_score_request(const RequestJson& body);

struct ResearchNextTokenRequest {
    // Exactly one of the two carriers is set: "messages" goes through the chat wire adapter
    // (template application, validation), "text" through the raw tokenizer.
    std::optional<GenerationRequest> generation;
    std::optional<std::string> text;
    int top_k = 0;
};

[[nodiscard]] ResearchNextTokenRequest
parse_research_next_token_request(const RequestJson& body, const RequestLimits& limits);

// Deterministic aggregate over teacher-forced per-position logprobs (natural log). Entry 0
// (the unconditioned first token) is excluded; the values are log probabilities, so NLL
// accumulates their negation.
struct ScoreAggregate {
    std::size_t scored_tokens = 0;
    double total_nll          = 0.0;
    double mean_nll           = 0.0;
    double perplexity         = 0.0;
};

// Throws std::runtime_error when any logprob is non-finite; values.size() must be >= 2.
[[nodiscard]] ScoreAggregate aggregate_score(const std::vector<float>& logprobs);

} // namespace ninfer::serve
