# Goal 1: Score Above Baseline on Leaderboard

## Target
Score above **baselinecheck** (currently rank #242, score **53.8%**, F1=0.911, 1000ms, 0% on-device).

## Key Facts
- **Leaderboard**: https://cactusevals.ngrok.app/
- **Submit**: `python submit.py --team "YourTeamName" --location "SF"` — uploads `main.py` directly
- **Rate limit**: 1 submission per hour — be strategic, don't waste submissions
- **Eval set**: Hidden Cactus eval, different from local `benchmark.py` — harder, designed to challenge FunctionGemma
- **Top 10**: Make it to qualitative judging (but we don't need to be top 10 — just above baseline)

## Baseline Analysis
The baselinecheck scored 53.8% with:
- F1 = 0.911 (very high accuracy)
- 1000ms avg latency
- **0% on-device** (everything goes to cloud)

This tells us: the baseline is just sending everything to Gemini cloud. It gets good accuracy but zero on-device bonus and slow speed. We can beat it by:
1. Getting even moderate accuracy with more on-device usage
2. The scoring formula heavily rewards on-device ratio (25% weight)

## Scoring Formula (from benchmark.py)
```
total = 0.20 * easy + 0.30 * medium + 0.50 * hard
level = 0.60 * avg_f1 + 0.25 * on_device_ratio + 0.15 * max(0, 1 - avg_time_ms / 500)
```

## What We Know Works (generalizable techniques from iterations 1-4)
These are proven techniques from `main-4.py` (77.8% on local benchmark):
1. **Tool-set reduction** — detect expected tool via keywords, pass only that tool to FG. Biggest single optimization.
2. **Multi-call splitting** — split "X and Y" into sub-actions, call FG per sub-action with model reuse (`cactus_reset`)
3. **Pronoun resolution** — "send him a message" → "send Tom a message"
4. **Validation + retry** — structural + semantic validation, retry with `cactus_reset` on failure
5. **Cloud-assist for partial failures** — keep local successes, cloud only the failed sub-actions
6. **Unreliable tool routing** — send_message, create_reminder, search_contacts → cloud for single-action
7. **Post-processing** — strip "the " from titles, fix alarm minute quirks, strip trailing periods
8. **Arg override** — parse expected values from text to correct FG's wrong-but-close outputs

## What We DON'T Know About the Hidden Eval
- What tools are available? (could be different from our 7)
- What phrasings are used? (likely more varied than our local benchmark)
- How many easy/medium/hard cases?
- Are there edge cases we haven't seen? (ambiguous requests, no-tool-needed, etc.)

## Strategy for Submission
1. **Start from main-4.py** — our best generalizing version (77.8% local)
2. **Make it more robust** — handle edge cases, unknown tools, varied phrasings
3. **Use `cloud_handoff` field** — FG tells us when confidence is low, use that signal
4. **Test locally first** — run `python benchmark.py` to make sure we haven't regressed
5. **Submit once we're confident** — don't waste the 1/hour limit
6. **Iterate based on leaderboard score** — if we score low, diagnose and fix

## Success Criteria
Score > 53.8% on the leaderboard. Once achieved, shift focus to Goal 2.

## Progress
- [x] Prepare a robust main.py for first submission
- [x] First submission to leaderboard
- [x] **Score above baseline (53.8%) → Achieved 89.15%**

## ✅ GOAL COMPLETE
Submitted `main-4.py` as `main-submitted-1.py`. Score: **89.15%**, F1=0.8333, 220ms avg, 100% on-device.
See `notes/submissions.md` for full details. Focus shifts to Goal 2.
