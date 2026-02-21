# Iteration 4 — Research + Bug Fixes + Perfect F1

**Date:** Saturday, February 21, 2026 at 12:13 AM PST

## Score: ~74.9% → 77.8% (consistent 77.8-77.9% across 3 runs)
## F1: ~0.92 → **1.00 (PERFECT across ALL 30 cases)**

## Research Phase

### Web searches performed
1. FunctionGemma prompt format from HuggingFace model card
2. Reddit r/cactuscompute (nothing hackathon-specific found)
3. Cactus SDK parameters (cactus.py deep dive)
4. FunctionGemma fine-tuning articles (Distil Labs 90-97% accuracy)

### Key research findings
1. **Official system prompt**: "You are a model that can do function calling with the following functions" (marked "ESSENTIAL" on HuggingFace). Tested: switching to this prompt actually HURT performance (~71% vs 74.9%). The original prompt works better with Cactus SDK.
2. **Developer role**: HuggingFace uses `"role": "developer"` not `"role": "system"`. Tested: also hurt performance. Reverted.
3. **tool_rag_top_k**: Default is 2, selects top-2 relevant tools. Setting to 0 disables. UNEXPLORED yet.
4. **cloud_handoff field**: Response includes this when confidence below threshold. Currently unused.
5. **Cactus SDK params**: top_p, top_k available but unexplored.

### Conclusion on system prompt
The Cactus SDK likely handles the prompt template internally. Our "You are a helpful assistant that can use tools." works better than the official one, possibly because Cactus prepends the official template and our prompt just adds context.

## Implementation Phase — 4 Changes

### 1. Strip trailing periods from send_message (FIX: message_among_four)
**Impact: Medium F1 0.90 → 1.00**

Root cause: Gemini sometimes returns `"I'll be late."` (with period) vs expected `"I'll be late"`. The `_normalize()` function only does `strip().lower()`, so the period causes a mismatch.

Fix: In `_postprocess_call`, strip trailing periods from send_message's `message` argument.

This fixed `message_among_four` which was consistently F1=0.00 (the only medium case that was broken).

### 2. Add send_message to UNRELIABLE_TOOLS for multi-call
**Impact: Eliminates garbled send_message results in hard cases**

Previously send_message was only in UNRELIABLE_TOOLS_SINGLE (skipped for single-action) but attempted locally in multi-call. Problem: FG often returned "valid" but wrong results like `recipient: "Alice'"` or `recipient: "Lisa's phone number"` that passed structural validation but gave F1=0.

Fix: Added send_message to UNRELIABLE_TOOLS (used in multi-call). Now all send_message sub-actions go to cloud-assist, which is reliable.

### 3. Individual cloud calls per failed sub-action (FIX: search_and_message pattern)
**Impact: Hard F1 ~0.85 → 1.00**

Root cause: When combining multiple failed sub-actions into one cloud call (e.g., "Find Tom in contacts and send him a message"), Gemini interprets this as a SEQUENTIAL workflow and only returns the first step (search_contacts). The second action (send_message) is omitted.

Fix: Instead of `cloud_text = " and ".join(failed_sub_actions)`, make individual cloud calls per failed sub-action. Each gets exactly one function call back.

This fixed:
- search_and_message: F1=0.67 → 1.00 (consistently)
- search_message_weather: F1=0.80 → 1.00 (consistently)
- reminder_and_message: more reliable
- message_weather_alarm: more reliable

### 4. Text-based alarm validation (FIX: alarm_among_five)
**Impact: Catches valid-but-wrong alarm results**

Root cause: FG sometimes returns alarm with wrong hour (e.g., hour=0, minute=9 for "9 AM") that passes structural validation (0≤hour≤23, 0≤minute≤59) but is semantically wrong.

Fix: Parse expected hour/minute from user text (e.g., "9 AM" → hour=9, minute=0) and validate FG's output against it. Reject if hour doesn't match.

This catches cases like alarm_among_five where FG returns wrong-but-valid hours.

## Results — 3 runs

| Run | Easy F1 | Med F1 | Hard F1 | Easy OD | Med OD | Hard OD | Score |
|-----|---------|--------|---------|---------|--------|---------|-------|
| 1   | 1.00    | 1.00   | 1.00    | 6/10    | 3/10   | 10/10   | 77.8% |
| 2   | 1.00    | 1.00   | 1.00    | 6/10    | 3/10   | 10/10   | 77.9% |
| 3   | 1.00    | 1.00   | 1.00    | 6/10    | 3/10   | 10/10   | 77.8% |

**Extremely consistent!** Variance: ±0.1% (was ±3% before).

## Scoring Math
```
Easy:   0.60*1.0 + 0.25*0.6 + 0.15*0 = 0.75
Medium: 0.60*1.0 + 0.25*0.3 + 0.15*0 = 0.675
Hard:   0.60*1.0 + 0.25*1.0 + 0.15*0 = 0.85
Total:  0.20*0.75 + 0.30*0.675 + 0.50*0.85 = 77.75%
```

Speed score is 0 for all levels (avg times > 500ms). Time is NOT the bottleneck.

## What's NOT working locally (contributing to low on-device ratio)
- **Easy cloud cases (4/10)**: message_alice (send_message), alarm_6am (alarm fails 3x), reminder_meeting (create_reminder), search_bob (search_contacts)
- **Medium cloud cases (7/10)**: All unreliable tools + alarm_among_three (alarm fails 3x), music_among_three (music fails 3x), alarm_among_five (alarm wrong values)

## Next Steps (ordered by expected impact)
1. **Investigate why alarm_among_three fails locally** — "Set an alarm for 8:15 AM" with single tool should work
2. **Investigate why music_among_three fails locally** — "Play some jazz music" with single tool should work
3. **Try more retries (4-5)** for medium single-tool path to reduce cloud fallback rate
4. **Try alarm_6am with 4+ retries** — might work some of the time
5. **On-device ratio is now the #1 lever** — each medium case going from cloud→local adds ~0.75% to score
6. **Speed optimization** — if we can get some cases under 500ms, adds up to 2.25% (easy 0.15*0.20=3%, medium 0.15*0.30=4.5%)

## Version info
- Starting from: main-3.py (iteration 3 final)
- Saved as: main-4.py (current best, 77.8%)
