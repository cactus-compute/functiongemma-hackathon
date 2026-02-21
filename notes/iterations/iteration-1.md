# Iteration 1 — Results & Learnings

## Scores
| Version | Score | F1 | On-device | Avg time |
|---------|-------|----|-----------|----------|
| Baseline (confidence threshold 0.99) | 56.1% | 0.72 | 50% (15/30) | 722ms |
| V1: skip local for multi-action | 53.3% | 0.78 | 33% (10/30) | 801ms |
| V2: always try local, validate, keep if valid | **57.8%** | 0.81 | 43% (13/30) | 752ms |

## Key Learnings

### 1. On-device ratio is MORE valuable than F1 accuracy
The scoring math proves it: a local result with F1=0.67 beats a perfect cloud result with F1=1.0.
```
on-device F1=0.67: 0.060*0.67 + 0.025 = 0.065 per case
cloud     F1=1.00: 0.060*1.00 + 0     = 0.060 per case
```
Breakeven is ~F1=0.58. NEVER send to cloud unless local returns literally nothing.

### 2. Skipping local for hard cases was WRONG
V1 tried to be smart by detecting multi-action requests and going straight to cloud.
This killed on-device ratio and the score dropped despite higher F1.
The baseline's "dumb" approach of always trying local was better.

### 3. Validation helps — but keep the bar LOW
V2 validates structure (valid function name, required args present, no negative values).
Only falls back to cloud when local returns empty/garbage.
This is the sweet spot: catches true failures, keeps partial-credit cases on-device.

### 4. FunctionGemma's real weaknesses
- Can only return 1 function call (hard cases need 2-3)
- Non-deterministic even with temp=0
- Confidence score is meaningless (1.0 on wrong answers)
- Struggles with: search_contacts, create_reminder, send_message
- Good at: get_weather, set_alarm, play_music

### 5. The remaining score gaps
**Easy (F1=1.00 now!)** — perfect, but only 5/10 on-device. 5 cases fall back to cloud because local returns empty calls. If we fix those, on-device jumps to 10/10.

**Medium (F1=0.60)** — 3 cases fail:
- alarm_among_three (on-device, F1=0) — picks wrong tool
- reminder_among_four (on-device, F1=0) — wrong args
- alarm_among_five (on-device, F1=0) — picks wrong tool
- message_among_four (cloud, F1=0) — Gemini itself fails!

**Hard (F1=0.83)** — 3/10 on-device with partial credit. Rest on cloud, mostly correct.

## Next Steps to Try
1. **Improve local reliability** — better system prompt, tuned per-tool
2. **Fix the 3 wrong-but-validated medium cases** — validation is too loose, accepting wrong tool calls
3. **Try cactus_reset trick** — keeping model alive with resets between calls showed much better results in testing vs init/destroy per call
4. **Retry on empty calls** — when local returns nothing, try once more with reset
5. **Prompt engineering** — "Call exactly one function" or tool-specific hints

## Scoring Formula Reference
```
total = 0.20 * easy + 0.30 * medium + 0.50 * hard
level = 0.60 * avg_f1 + 0.25 * on_device_ratio + 0.15 * speed_score
speed_score = max(0, 1 - avg_time_ms / 500)
```
