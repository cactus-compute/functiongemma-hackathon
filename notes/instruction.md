# Hackathon Agent Instructions

## Overview
Cactus × DeepMind FunctionGemma Hackathon. Two goals — both must be addressed.

**Full rules**: https://github.com/cactus-compute/functiongemma-hackathon?tab=readme-ov-file
**Leaderboard**: https://cactusevals.ngrok.app/
**Hackathon page**: https://sf.aitinkerers.org/p/google-deepmind-x-cactus-compute-global-hackathon-san-francisco

## Goals

### Goal 1: ✅ DONE — Score Above Baseline on Leaderboard
Scored **89.15%** (baseline was 53.8%) with `main-4.py`. F1=0.8333, 220ms avg, 100% on-device.
**Details**: `notes/goal1.md` | **Submission history**: `notes/submissions.md`

### Goal 2: End-to-End Product Demo
Build a hacky but functional Python demo showing real-world use of hybrid routing. Judges care about cleverness, real-world utility, and voice-to-action (Rubric 3: `cactus_transcribe`).
**Details**: `notes/goal2.md`

### Qualitative Judging (what actually wins)
- **Rubric 1**: Quality of hybrid routing algorithm — depth and cleverness
- **Rubric 2**: End-to-end products that execute function calls to solve real-world problems
- **Rubric 3**: Low-latency voice-to-action products, leveraging `cactus_transcribe`

**Scoring above baseline gets us to judging. The demo is what wins.**

## Setup
See `notes/install.md` for full install steps. Quick start:
```bash
cd /Users/maxime/dev/functiongemma-hackathon
source cactus/venv/bin/activate
export GEMINI_API_KEY='...'  # see notes/install.md (DO NOT commit keys)
python benchmark.py          # Run full benchmark (30 cases, ~30s)
python test_harness.py -v    # Diagnostic: see what local produces per case
```

### Submitting to Leaderboard
```bash
python submit.py --team "YourTeamName" --location "SF"
```
- Uploads `main.py` directly — make sure it's the version you want
- **1 submission per hour** — don't waste it
- Hidden eval set, different from local benchmark

## Scoring Formula (for leaderboard)
```
total = 0.20 * easy + 0.30 * medium + 0.50 * hard
level = 0.60 * avg_f1 + 0.25 * on_device_ratio + 0.15 * max(0, 1 - avg_time_ms / 500)
```
- F1 accuracy: 60% weight
- On-device ratio: 25% weight (THIS IS HUGE — partial local beats perfect cloud)
- Speed under 500ms: 15% weight

## Knowledge Base — READ BEFORE EACH ITERATION
1. **`notes/goal1.md`** — Goal 1 details, strategy, progress checklist.
2. **`notes/goal2.md`** — Goal 2 details, demo ideas, available APIs.
3. **`notes/progress.md`** — cumulative brain for Goal 2. What worked, what failed, what to try next.
   - `notes/progress-goal1.md` — archived Goal 1 progress (for reference).
4. **`notes/submissions.md`** — leaderboard submission history (which file, result, notes).
5. **`notes/research.md`** — web search findings. Read this for ideas, add to it when you research.
6. **`notes/iterations/iteration-0-findings.md`** — Initial FunctionGemma exploration (capabilities, knobs, tool reliability).
7. **`notes/strategy.md`** — scoring math and strategic implications.
8. **`notes/analysis-iterations-1-5.md`** — ⚠️ POST-MORTEM: how we overfitted. Read this to avoid repeating the mistake.
9. **`notes/iterations/`** — detailed per-iteration logs.

## How This Runs
An agent runs in a loop via pi (`~/.pi/agent/`) with loop bash (`~/bin/loop`).

### Each iteration MUST:
1. **Start the iteration file with a datetime timestamp** (e.g., `**Date:** Friday, February 20, 2026 at 11:38 PM PST`). This goes at the top of every `notes/iterations/iteration-N.md`.
2. **Read** `notes/progress.md` and the last 2-3 iteration files in `notes/iterations/`.
3. **Read** `notes/goal1.md` and `notes/goal2.md` for current state and priorities.
4. **Pick ONE thing to try** — keep it contained and manageable.
5. **Implement** the change.
6. **Run** `python benchmark.py` to measure the score (for Goal 1 work).
7. **Create** `notes/iterations/iteration-N.md` with results and analysis (do NOT modify past iterations).
8. **Append** findings to `notes/progress.md` — this is the brain, keep it clean and useful.
9. **Call `loop_next`** to hand off to the next iteration.

### Approach: Research → Plan → Implement
Follow the RPI framework (see `/Users/maxime/.pi/agent/skills/rpi/SKILL.md`). Don't jump straight to code. Understand the problem first, form a hypothesis, then implement. This prevents wasted iterations.

### Rules
- **ONE idea per iteration** — don't try 5 things at once.
- **Never modify past iteration files** — append-only history.
- **Always run benchmark.py** for Goal 1 work — no guessing, measure everything.
- **Be bold and creative** — try new prompts, retry strategies, parallel calls, tool-specific routing, temperature tricks, whatever. If it fails, document and move on.
- **Document failures too** — knowing what doesn't work is just as valuable.
- **Use test_harness.py** for quick local-only diagnostics before running full benchmark.

### ⚠️ CRITICAL: Don't Overfit to the Local Benchmark
**Read `notes/analysis-iterations-1-5.md` before continuing.** We already made this mistake once.

The 30 test cases in `benchmark.py` are a **development tool**, not the competition target. The real evaluation uses **unseen prompts** with different phrasings, edge cases, and possibly different tools. A solution that scores 100% locally by regex-matching hardcoded patterns will **fail on unseen data**.

Before every change, ask: **"Would this work on a prompt I've never seen?"**
- ✅ Making FG more reliable (prompt tuning, retries, parameter exploration) — generalizes
- ✅ Smart routing logic (validation, tool-set reduction, confidence calibration) — generalizes
- ✅ Arg postprocessing for known model quirks (strip "the ", fix minute=1→0) — generalizes
- ❌ Regex patterns tuned to specific benchmark phrasings — overfits
- ❌ Bypassing the model entirely with text parsing — overfits
- ❌ Celebrating a local benchmark score without questioning generalization — dangerous

**Build a harder test set.** Add 20-30 cases with varied phrasings to `benchmark.py` (or a separate `benchmark_hard.py`) and test against those too. If your score drops dramatically on unseen phrasings, you're overfitting.

### Versioning main.py
**There should be NO `main.py` in the repo** unless we are actively benchmarking or submitting. This prevents accidental edits to the wrong file.

- **`main-base.py`** — upstream baseline from hackathon repo. Clean reference. DO NOT MODIFY.
- **Experiment files**: `main-N.py` (e.g., `main-6.py`, `main-7.py`). Start from `main-base.py` or `main-4.py` and modify.
- **To benchmark**:
  ```bash
  cp main-7.py main.py        # activate
  python benchmark.py          # run
  rm main.py                   # clean up immediately
  ```
- **To submit**:
  ```bash
  cp main-7.py main.py        # activate
  python submit.py --team "maxmux" --location "SF"
  mv main.py main-submitted-N.py   # rename immediately after, never leave main.py around
  ```
- **Track all submissions** in `notes/submissions.md` — which file, result, notes.
- **Note in the iteration file** which version you're working from.
- **Start fresh from `main-base.py` often** — don't endlessly layer on the same file.

### Current version inventory
- `main-base.py` — **upstream baseline (DO NOT MODIFY)** — clean reference
- `main-3.py` — Iteration 3 (74.9%)
- `main-4.py` — Iteration 4 (77.8%, best generalizing version)
- `main-5-overfit.py` — Iteration 5 (100% local only, overfitted — cautionary tale)
- `main-submitted-1.py` — First leaderboard submission (copy of main-4.py)

### Web Research
You have web search tools (`exa_websearch`, `tavily_websearch`). Use them! Some iterations can be **purely research** — search for FunctionGemma best practices, tool-calling prompt strategies, confidence calibration techniques, etc. Write findings to `notes/research.md` so future iterations benefit.

### Community Intel — r/cactuscompute
Check https://www.reddit.com/r/cactuscompute/ **roughly once per hour** (not every iteration). Other participants or the organizers may post tips, insights, or clarifications that could help. Note the last time you checked in `notes/progress.md` so future iterations know when to check again. Add anything useful to `notes/research.md`.

### Cactus Engine Reference
The on-device inference engine we're using: https://github.com/cactus-compute/cactus — "Energy-efficient inference engine". Check the repo for documentation on parameters, API options, configuration knobs, or anything that could help squeeze more performance out of FunctionGemma locally.

### FunctionGemma Model Reference
The on-device model: https://huggingface.co/google/functiongemma-270m-it — downloaded locally via `cactus download google/functiongemma-270m-it --reconvert`. Check the HuggingFace page for model card details, prompt format, capabilities, limitations, and example usage that could inform our routing and prompting strategy.

### Hackathon Frontpage
https://sf.aitinkerers.org/p/google-deepmind-x-cactus-compute-global-hackathon-san-francisco — the official hackathon page. May contain rules, scoring details, tips, or updates.

### Ideas to Explore (non-exhaustive)
**Goal 1 — Routing:**
- Keep model alive with `cactus_reset` instead of init/destroy per call
- Retry on empty calls (reset + try again)
- System prompt variations per tool type
- Temperature tuning
- `tool_rag_top_k` parameter (0=all tools, N=top N)
- `force_tools` on/off effects
- Validate + retry before cloud fallback
- Pre-parse user message to count actions
- Better arg validation (type checking, range checking)
- Prompt: "Call exactly the function that matches" for medium cases
- For hard multi-call: call FunctionGemma multiple times, once per detected action
- Use `cloud_handoff` field from FG response as a routing signal
- Use `confidence` more intelligently (not as threshold but as one signal among many)

**Goal 2 — Demo:**
- `cactus_transcribe` for voice-to-action (Rubric 3 — big bonus)
- Actually execute function calls (weather API, system commands, etc.)
- Show routing decisions visually ("on-device 45ms" vs "cloud 800ms")
- RAG integration with `cactus_rag_query` for context-aware routing
- Multi-model pipeline: whisper → functiongemma → execution
