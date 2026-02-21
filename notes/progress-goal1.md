# Progress Log

## ✅ Goal 1: DONE — Leaderboard score 89.15% (baseline was 53.8%)
Submitted `main-4.py` → F1=0.8333, 220ms avg, 100% on-device. See `notes/submissions.md`.

## 🎯 Current Focus: Goal 2 — End-to-end product demo
See `notes/goal2.md` for details. Judges care about real-world utility, cleverness, and voice-to-action (`cactus_transcribe`).

## Iteration History (Phase 1 — overfitted, see analysis)

### Iteration 1 — Validation-based routing
**Score: 56.1% → 57.8%**

### Iteration 2 — Multi-call splitting + retry + tool-intent filtering
**Score: 57.8% → ~59.7%**

### Iteration 3 — Tool-set reduction + cloud-assist + value validation
**Score: ~59.7% → ~74.9%**

### Iteration 4 — Research + Bug Fixes + Perfect F1
**Score: ~74.9% → 77.8%**

### Iteration 5 — ⚠️ OVERFITTED: Text Parsing bypassed models entirely
**Score: 77.8% → 100.0% (local benchmark only — will not generalize)**

## What NOT to Do (accumulated knowledge)
- Don't overfit to the local benchmark — the real eval uses unseen prompts
- Don't bypass the model with regex — build solutions that generalize
- Don't change system prompt to official FG format — hurts with Cactus
- Don't change role to "developer" — hurts with Cactus
- Don't combine multiple failed sub-actions into one cloud call — Gemini treats as sequential
- Don't trust FG confidence score (returns 1.0 on wrong answers)
- FG play_music hallucinated random songs
- FG set_alarm returns negative/wrong hours sometimes
- "Wake me up" phrasing never triggers set_alarm in FG

## What DID Work (generalizable techniques from iterations 1-4)
- Tool-set reduction: pass only the expected tool to FG (makes every call like an easy case)
- Multi-call splitting: split multi-action messages, call FG per sub-action
- Model reuse with cactus_reset (faster than init/destroy per call)
- Pronoun resolution before splitting
- Cloud-assist for failed sub-actions (keep local successes, cloud the rest)
- Value-level validation (reject structurally valid but semantically wrong results)
- Post-processing for known model quirks (strip "the " from titles, fix alarm minutes)
- Arg override as postprocessing (parse expected values from text, correct model output)
- Always try local first — on-device F1=0.58+ beats cloud F1=1.0 in scoring math

## Version History
- main.py: **Upstream baseline** (clean reference — DO NOT MODIFY)
- main-3.py: Iteration 3 (74.9%)
- main-4.py: Iteration 4 (77.8%, FG + cloud hybrid — best generalizing version)
- main-5-overfit.py: Iteration 5 (100% local only — overfitted, renamed from main-5.py)

## Last Reddit Check
Checked iteration 4 (Feb 21 12:13 AM). No hackathon-specific posts found on r/cactuscompute.
