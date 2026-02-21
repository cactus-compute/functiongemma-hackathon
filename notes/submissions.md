# Leaderboard Submissions

## Submission Process — How It Works
- `submit.py` uploads `main.py` as a file to `https://cactusevals.ngrok.app/eval/submit` via POST
- Server queues it, returns a submission ID. You poll `/eval/status?id=...` for results.
- **Queue can be deep** (we were #9). Each eval takes a few minutes (runs 30 cases with actual FG + cloud calls server-side).
- **Total wait**: ~8 min in queue + ~3 min running = ~11 min total for our first submission.
- The polling script in `submit.py` has no timeout handling — long queues cause connection drops. Polling manually with curl works fine.
- **Rate limit**: 1 submission per hour per team.

## What We Learned About the Hidden Eval
- **30 cases** (same count as local benchmark), also split into easy/medium/hard.
- **Different prompts** — case names visible from progress feed:
  - `music_among_four_v2`, `timer_among_two_v2`, `reminder_music_timer_v2`
  - `alarm_weather_message_v2`, `alarm_among_four_v2`, `play_hotel_california`
  - `alarm_10_15pm`, `weather_alarm_reminder_v2`, `alarm_9am`
  - `weather_and_timer_v2`, `weather_lisbon`, `search_message_alarm_v2`
  - `timer_among_four_v2`, `music_among_two_v2`
- Many are **`_v2` variants** of local benchmark cases — same tools, different phrasings.
- **Same 7 tools** as local (weather, alarm, timer, music, message, reminder, contacts).
- Some new combos: `timer_among_two_v2`, `music_among_four_v2`, `alarm_among_four_v2`.
- Our F1=0.8333 (vs 1.00 local) means ~5 cases had imperfect results — likely from varied phrasings our validation/routing doesn't handle perfectly.

## Key Insight
`main-4.py` (real FG + cloud hybrid) scored **89.15%** on hidden eval — much higher than the local 77.8%. Why? Because the hidden eval also rewards on-device ratio heavily, and our routing kept 100% on-device. The local benchmark was harder to score high on because cloud fallback cases were counted as cloud. On the hidden eval, our validation + retry logic apparently kept everything local.

---

## Submission 1 ✅
- **Date**: Saturday, February 21, 2026 at ~10:50 AM PST
- **File**: `main-submitted-1.py` (copy of `main-4.py` — iteration 4 hybrid)
- **Team**: maxmux
- **Location**: SF
- **Submission ID**: `fdf62af959e34442a38fd04e4ed3797d`
- **Result**:
  - **Score: 89.15%** ← well above baseline (53.8%)
  - **F1: 0.8333**
  - **Avg Time: 220ms**
  - **On-Device: 100%**
- **Notes**: First submission. 30 hidden eval cases (different from local benchmark). Interesting case names visible from progress: `music_among_four_v2`, `timer_among_two_v2`, `reminder_music_timer_v2`, `alarm_weather_message_v2`, `alarm_among_four_v2`, `play_hotel_california`, `alarm_10_15pm`, `weather_alarm_reminder_v2`, `alarm_9am`, `weather_and_timer_v2`, `weather_lisbon`, `search_message_alarm_v2`, `timer_among_four_v2`, `music_among_two_v2`. Many are "_v2" variants suggesting harder/different phrasings than local benchmark. On-device was 100% which means our routing logic kept everything local — big scoring boost.
