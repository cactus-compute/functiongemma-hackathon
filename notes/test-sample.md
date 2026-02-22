# FG Sampling & Pre-Processing Experiments

**Date:** Saturday, February 21, 2026 at 12:58 PM PST

## Key Insight

FunctionGemma 270M is good at **tool selection** and **arg extraction from explicit prompts**, but terrible at **parsing natural language into arg values** (e.g., "7am" → hour=7, minute=0). 

## LFM2 as Rewriter — Dead End

Tested LFM2-VL-450M (690MB on disk) as a prompt rewriter/paraphraser. Results:
- Echoes input back, hallucinates wrong values ("midnight" for "7am"), or rambles
- Tried multiple system prompts and user templates — no reliable paraphrasing
- **Verdict: not viable.** 450M is too small for controlled rewriting.

## Statistical Sampling — Promising (with caveats)

Ran FG 10x on same prompt with temperature > 0, majority vote on results.

### String args: voting works great
```
"What is the weather in Paris?" (temp=0.3, force_tools)
  4/10 → get_weather({"location": "Paris"})  ✅
  1/10 → get_weather({"location": "Europe"}) 
  5/10 → (no calls)
```
Majority vote (excluding no-calls) → correct answer.

### Numeric args: fundamentally broken
```
"Set an alarm for 7am" (temp=0.3, force_tools)
  hour=0 minute=9  |  hour=0 minute=7  |  hour=0 minute=12
  hour=0 minute=8  |  hour=12          |  minutes=-480
```
Not a single sample got hour=7, minute=0. The model cannot parse "7am". No amount of voting fixes this.

### 50% "no calls" issue
With `cactus_reset` between samples, ~50% of calls return no function calls. May be a reset issue. Parse errors also occur at higher temperatures.

## Spoon-Fed Prompts — The Breakthrough

When we rewrite the prompt to include explicit param names, FG nails it:

| Prompt | Result | Correct? |
|--------|--------|----------|
| `"Set an alarm for hour 7 minute 0"` | `set_alarm(hour=7, minute=0)` — 4/5 samples | ✅ |
| `"Send a message to recipient Alice with message hello how are you"` | `send_message(Alice, "hello how are you")` — 2/3 | ✅ |
| `"Set a timer for minutes 10"` | `set_timer(minutes=10)` — 2/3 | ✅ |
| `"Play music song jazz playlist"` | `play_music(song="jazz")` — 2/3 | ✅ |
| `"Create a reminder with title call dentist at time 2:00 PM"` | `set_timer(60)` — wrong tool | ❌ |

4/5 perfect. The `create_reminder` failure is a tool selection issue (confused with `set_timer`), not an arg issue.

## Winning Strategy: Pre-Process + Sample

```
User prompt
    ↓
[Pre-process] Parse NL into explicit form
  "Set alarm for 7am" → "Set an alarm for hour 7 minute 0"
    ↓
[FG × N] Sample 3-5 times with temp > 0
    ↓
[Vote] Majority vote (filter out no-calls and parse errors)
    ↓
Result
```

### Pre-processing scope
Not arbitrary NL rewriting — just **domain-specific parsing** of known value types:
- **Time**: "7am" → "hour 7 minute 0", "3:30pm" → "hour 15 minute 30"
- **Duration**: "5 minutes" → "minutes 5", "an hour" → "minutes 60"
- **The rest** (names, locations, song titles, messages) — FG handles these fine as-is

This is a solved problem — libraries like `dateutil`, `parsedatetime` handle it. Or simple targeted parsing (not brittle regex on full prompts, just on time/duration tokens).

### Why this is good
- **Both models stay on-device** — FG only, no cloud needed for easy/medium
- **Fast** — 3-5 FG calls × ~330ms = ~1-1.6s sequential, parallelizable on device
- **Generalizes** — pre-processing is based on value types, not hardcoded prompts
- **Statistically robust** — voting smooths out FG's inconsistency

### Open questions
- What's causing 50% "no calls" with `cactus_reset`? Init/destroy per sample instead?
- Best temperature for sampling? 0.3 seemed decent.
- Can we fix `create_reminder` tool selection? Maybe fewer tools in the set helps.
- How does this integrate with the hybrid routing in main-4.py?
