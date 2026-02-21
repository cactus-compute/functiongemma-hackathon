# Hybrid Routing Strategy Notes (F1 + Speed)

## What the Harness Actually Scores

- `benchmark.py` calls `generate_hybrid(...)` once per case, sequentially.
- F1 is computed from `result["function_calls"]` vs expected calls.
- Time is taken from `result["total_time_ms"]`.
- On-device ratio uses `result["source"] == "on-device"`.

Score components (per difficulty bucket):

- 50%: average F1
- 25%: time score (`1 - avg_time/500ms`, floored at 0)
- 25%: on-device ratio

## F1-Specific Harness Behavior to Exploit

- Match requires exact tool name equality.
- For args, expected keys must exist and values must match after basic normalization:
  - strings: `strip().lower()`
  - non-strings: exact equality
- Extra predicted calls reduce precision.
- Missing calls reduce recall.
- Call order does not matter.
- Extra arguments are tolerated if expected args are correct.

## High-F1 Strategy Set (Beyond `confidence_threshold`)

## 1) Local Output Schema Gate

Before accepting on-device output:

- Ensure every predicted call name exists in `tools`.
- Ensure required args exist for each call.
- Ensure arg types match schema (`integer`, `string`, etc.).
- Fallback to cloud on any schema violation.

Why: removes many low-confidence malformed outputs that would become F1=0.

## 2) Argument Canonicalization Layer

Normalize local args before returning:

- Coerce integer-like strings for integer fields (e.g. `"15"` -> `15`).
- Normalize times to a consistent format (e.g. `5 AM` -> `5:00 AM`) where safe.
- Trim whitespace consistently.

Why: evaluator is strict for non-string type and lightly normalized for string type.

## 3) Intent-Aware Routing

Estimate intent count from prompt shape and lexical cues:

- Single-intent prompts: expect 1 call.
- Multi-intent prompts (`and`, commas, multiple verbs): expect 2+ calls.

Use stricter acceptance for multi-intent local outputs (hard bucket is heavily multi-call).

## 4) Over-Call / Under-Call Guard

Add acceptance rules:

- If prompt appears single-intent but local returns many calls, fallback.
- If prompt appears multi-intent but local returns too few calls, fallback.

Why: F1 heavily penalizes precision/recall mistakes from wrong call counts.

## 5) Difficulty-Conditioned Thresholding

Use dynamic threshold instead of one global value:

- Lower threshold for easy single-tool tasks.
- Higher threshold for medium/hard and multi-intent tasks.

Why: improves on-device ratio where local is usually good while protecting hard-task F1.

## 6) Tool-Set Complexity Heuristic

Route more conservatively when:

- Tool count is large.
- Tool names/descriptions are semantically similar.

Route more aggressively on-device when only 1 tool or clearly disjoint tools.

## 7) Two-Pass Local Consistency Check

Run local inference twice (possibly with different prompt wrapper/temperature settings if available):

- If calls are consistent, accept local even at moderately lower confidence.
- If inconsistent, fallback to cloud.

Why: consistency is often a stronger quality signal than a raw confidence scalar.

## 8) Optional Repair Pass Before Cloud

If local output is close but slightly invalid:

- Attempt deterministic repair (type coercion, required-arg fill if trivially inferable).
- Only fallback if repair fails.

Why: can recover F1 while keeping on-device ratio.

## 9) Tool-RAG / Tool Filtering Knob

Tune `tool_rag_top_k` (if used in local call options):

- Keep small for single-intent to reduce confusion.
- Increase for likely multi-intent prompts.

Why: reduces wrong-tool selection and can improve both speed and accuracy.

## Speed + Concurrency Reality Check

## Are tool calls parallelizable in this harness?

Short answer: tool execution is not happening here.

- The harness does **not** execute tool backends.
- It only evaluates the list of predicted `function_calls`.
- So there is no runtime benefit from parallelizing "tool calls" themselves in `benchmark.py`.

## What is single-threaded vs parallel here?

- Benchmark loop is single-threaded: cases are run one after another.
- Within one case:
  - Baseline strategy is sequential: local first, cloud only on fallback.
  - You may implement speculative parallel local+cloud in `generate_hybrid`, but tradeoffs are:
    - lower tail latency in some cases
    - more cloud usage/cost
    - potential impact on time depending on implementation overhead

## Practical speed advice

- Avoid unnecessary cloud round-trips: fallback only when local output fails strict acceptance.
- Keep local prompt short and deterministic.
- Prefer lightweight deterministic validation/normalization over extra model calls.
- Use timing logs per case to locate slow paths (especially cloud fallback frequency).
