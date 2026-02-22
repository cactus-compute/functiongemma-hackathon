# RAG + Whisper + FunctionGemma Pipeline — Test Findings

**Date:** Saturday, February 21, 2026 at 1:50 PM PST  
**Scripts:** `test-rag.py`, `test-transcribe.py`  
**Corpus:** `test_corpus/` (5 markdown files: medications, allergies, symptoms, doctors, health goals)

---

## Architecture

```
🎤 Voice → Whisper (transcribe) → RAG (retrieve context) → FunctionGemma (tool call)
                                                            ↓
                                               Tool name + arguments
```

Three Cactus APIs chained, **all on-device, zero cloud**:
- `cactus_transcribe` — Whisper small, speech-to-text
- `cactus_rag_query` — retrieves relevant chunks from local doc corpus
- `cactus_complete` — FunctionGemma 270M, tool calling

---

## What Works ✅

### Whisper Transcription
- 5/6 test cases transcribed correctly (~1.3s per file, 16kHz mono WAV)
- Handles natural speech well: "Set a reminder to take Warfarin at 8 PM" → perfect
- One miss: "Play Stairway to Heaven" → "Place their way to heaven" (macOS TTS pronunciation issue, not Whisper's fault)

### RAG Retrieval
- **Blazing fast**: 10-25ms per query — negligible in the pipeline
- Finds relevant documents reliably (4/8 top-1 correct, but relevant doc almost always in top-3)
- Scores are tightly clustered (0.0158-0.0164), so ranking isn't great, but retrieval works
- Index auto-builds from `corpus_dir` on `cactus_init`

### RAG + FunctionGemma Together
- **RAG context genuinely helps FG produce tool calls it otherwise refuses**
- Without RAG: 2/7 correct tool calls. With RAG: 4/7 correct
- Key example: "Search my records for recent blood sugar readings" — FG refuses without context, but with RAG injecting health goals it produces the call
- "Search my records for my cardiologist phone number" — same pattern, refuses without RAG, works with it

### Full Voice Pipeline (5/5 ✅)
- End-to-end latency: **2.6–3.9s** (whisper 1.3-2.3s + rag 12-21ms + fg 1.1-1.6s)
- All 5 pipeline test cases produced correct tool calls
- 100% on-device, zero network traffic

---

## What Doesn't Work ❌

### FunctionGemma 270M Limitations (Critical)
- **Extremely literal** — only responds to direct verb phrases it was trained on
- ✅ "Search my records for X" → works
- ✅ "Set a reminder for X at Y" → works  
- ❌ "Can I take ibuprofen?" → refuses or picks wrong tool
- ❌ "Call my cardiologist" → refuses
- ❌ "I'm having chest pain" → refuses or wrong tool
- ❌ "Send a message to Jane saying X" → refuses (inconsistent)
- Custom tool names outside training distribution (e.g., `check_medication_interaction`, `call_emergency_contact`) almost never get called — FG doesn't know them

### RAG Context Can Overwhelm FG
- Injecting full RAG chunks (200+ chars) into FG prompt causes it to refuse or hallucinate
- **Fix:** `summarize_rag_context()` — trim to ~200 chars max, first chunk only
- Even with trimming, FG sometimes parrots RAG content back as search query args instead of extracting user intent
- Example: query="ibuprofen interactions" → FG outputs `search_records({"query": "Ibuprofen Acetaminophen 500mg Max 3000mg..."})` — it's copying the RAG context verbatim

### RAG Ranking Quality
- The embedding model (`lfm2-vl-450m`) doesn't differentiate well between documents
- "Am I allergic to penicillin?" → top result is `health_goals.md` (mood section), not `allergies.md`
- "Who is my emergency contact?" → finds `health_goals.md`, not `doctors.md`
- All scores in 0.015-0.017 range — not enough spread for confident top-1 selection

### FG Non-Determinism
- Same prompt produces different results across runs
- "Set a reminder to take Warfarin at 8 PM" → ✅ in one run, ❌ in next (asks for clarification)
- Greedy decoding (temp=0.0) helps but doesn't eliminate it

---

## Bugs & Workarounds

| Issue | Fix |
|-------|-----|
| Whisper repeats first transcript for all files | Call `cactus_reset(whisper_model)` before each transcription |
| `cactus_rag_query()` Python wrapper returns `[]` always | **Bug in wrapper**: C function returns bytes-written (non-zero = success), but Python treats non-zero as error. Workaround: call `_lib.cactus_rag_query()` directly via ctypes |
| RAG index "Document ID already exists" error on reload | Delete `test_corpus/index.bin` and `test_corpus/data.bin` before `cactus_init`. The `clean_corpus_cache()` helper does this |
| PGRST102 JSON errors in stderr | Telemetry/analytics noise from Cactus SDK — harmless, ignore |

---

## Tool Design Rules for FG 270M

If you want FG to actually call your tools:

1. **Use standard verb-based names**: `search_X`, `set_X`, `get_X`, `create_X`, `send_X` — these are in its training data
2. **Keep parameter schemas minimal** — 1-2 required params max
3. **User prompts must be literal**: "Search my records for X" not "Can you look up X for me?"
4. **Don't use domain-specific tool names**: `check_medication_interaction` will never get called; `search_records` works
5. **RAG context must be SHORT** — 200 chars max, or FG gets confused and copies it verbatim

---

## Recommended Demo Strategy

For the hackathon demo, the architecture should be:

1. **Voice input** (Whisper) — the "wow" factor, covers Rubric 3
2. **RAG from private corpus** — the privacy story, shows local docs stay local
3. **FG tool calling** — use only patterns that reliably work (`search_records`, `set_reminder`)
4. **Post-hoc enrichment** — show RAG context alongside the tool call as "here's what your private records say"

Don't fight FG's limitations. Design the demo flow around what it can do. The impressive part is **three Cactus APIs chained, all on-device, with real private data**, not perfect accuracy.

### Timing Budget
| Step | Time | Notes |
|------|------|-------|
| Whisper | 1.3-2.3s | Dominates latency |
| RAG | 10-25ms | Negligible |
| FunctionGemma | 1.1-1.6s | Second biggest |
| **Total** | **2.6-3.9s** | All on-device |
