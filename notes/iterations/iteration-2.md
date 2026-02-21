# Iteration 2 — Multi-call Splitting + Retry + Tool-Intent Filtering

## Score: 57.8% → ~59.7% (range 59-60% due to non-determinism)

## What was implemented

### 1. Multi-call splitting for hard cases
Split multi-action messages (detected via " and ", ", ") into individual sub-actions. Call FG once per sub-action instead of once for the whole message.

Example: "Set an alarm for 7:30 AM and check the weather in New York"
→ "Set an alarm for 7:30 AM" + "check the weather in New York"
→ Call FG twice, each time with full tool set

### 2. Model reuse with cactus_reset
Initialize model once, use `cactus_reset()` between sub-calls. Avoids init/destroy overhead.

### 3. Pronoun resolution
Before splitting, resolve pronouns: "send him a message" → "send Tom a message" (where Tom is found in the full message text).

### 4. Retry on empty for single-action cases
If first FG call returns empty/invalid, `cactus_reset()` and try once more before cloud fallback.

### 5. Tool-intent filtering for multi-call
- Detect expected tool per sub-action using keyword matching
- Skip sub-actions for known-unreliable tools (create_reminder, search_contacts)
- Only keep sub-call results where returned tool matches expected tool

## Results table (representative run)

| Level | Avg F1 | On-device | Avg time |
|-------|--------|-----------|----------|
| Easy | 0.90 | 6/10 | 558ms |
| Medium | 0.70-0.80 | 5-6/10 | 753ms |
| Hard | 0.51-0.67 | 7-9/10 | 685ms |
| Overall | 0.74-0.78 | 19-21/30 (63-70%) | ~665ms |

## Key findings

### Multi-call is a net positive but modest
- Hard on-device: 3/10 → 7-9/10 (massive improvement!)
- But hard F1 dropped: 0.83 → 0.51-0.67 (FG gets partial credit but makes errors)
- Net effect: ~+2% total score (on-device bonus > F1 loss)

### Tool-intent filtering helps for specific cases
- Skipping create_reminder/search_contacts sub-calls in hard multi-call prevents F1=0.00
- Cases like `reminder_and_message` now correctly fall back to cloud (F1=1.0)

### Single-action unreliable tool skipping HURTS
- Tried skipping local for unreliable tools in single-action path
- Backfired: even F1=0.00 on-device (0.25 score) beats F1=0.00 cloud (0.00 score)
- Cloud is also non-deterministic (reminder_meeting F1=0 on cloud too!)
- Lesson: NEVER skip local for single-action cases

### Non-determinism is significant
- FG produces different results across runs for same input
- Score variance: ±1-2% between runs
- Makes it hard to evaluate small optimizations
- Cloud (Gemini) also shows occasional F1=0.00 (music_among_three, message_among_four)

### Hard cases detail
| Case | Expected | Multi-call result | Typical F1 |
|------|----------|-------------------|------------|
| message_and_weather | send_message + get_weather | ⚠️ weather right, message wrong | 0.50 |
| alarm_and_weather | set_alarm + get_weather | ✅ both often right | 0.67 |
| timer_and_music | set_timer + play_music | ⚠️ one right, one wrong | 0.67 |
| reminder_and_message | create_reminder + send_message | → cloud (both unreliable) | 1.00 |
| search_and_message | search_contacts + send_message | ⚠️ skipped search, message varies | 0.67 |
| alarm_and_reminder | set_alarm + create_reminder | ⚠️ alarm right, reminder skipped | 0.67 |
| weather_and_music | get_weather + play_music | ✅ both often right | 0.67-1.00 |
| message_weather_alarm | send + weather + alarm | ⚠️ 2/3 often right | 0.50-0.80 |
| timer_music_reminder | timer + music + reminder | ⚠️ reminder skipped, others flaky | 0.00-0.67 |
| search_message_weather | search + message + weather | → cloud (search unreliable) | 0.50-1.00 |

## What NOT to try again
- Don't skip local for single-action cases based on tool unreliability
- Don't add tool-intent validation to single-action path (too aggressive)
- Cloud is NOT a guaranteed F1=1.0 — it fails sometimes too

## Next steps to explore
1. **System prompt tuning** — customize per tool type or per difficulty
2. **Temperature experiments** — try temp>0 for FG (might help with stuck outputs)
3. **Reduce tool set for multi-call** — pass only the expected tool per sub-action (like easy cases)
4. **Fix the persistent F1=0.00 cases** — timer_music_reminder always fails
5. **Address easy cases going to cloud** — alarm_6am, message_alice should work locally
6. **Multiple retries** — try 3x instead of 2x for flaky cases
