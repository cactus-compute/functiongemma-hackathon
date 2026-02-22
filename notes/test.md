# FunctionGemma Systematic Sweep — Feb 21, 2026

## Test Setup

**Script:** `test_sweep.py` — automated sweep across 13 test cases × 7 experiments (~100 individual cactus calls).  
**Model:** `functiongemma-270m-it` running on-device via Cactus engine.  
**System prompt:** `"You are a helpful assistant that can use tools."`  
**Total runtime:** ~113s on M-series Mac.

### Test Cases (13 total)

**Single action, direct phrasing (6):**
- "What's the weather in Tokyo?" → `get_weather(location="Tokyo")`
- "Set an alarm for 7:30 AM." → `set_alarm(hour=7, minute=30)`
- "Send a message to Bob saying hi there." → `send_message(recipient="Bob", message="hi there")`
- "Play Stairway to Heaven." → `play_music(song="Stairway to Heaven")`
- "Set a timer for 10 minutes." → `set_timer(minutes=10)`
- "Remind me to call the dentist at 2 PM." → `create_reminder(time="2 PM")`

**Single action, indirect phrasing (3):**
- "I wonder if I need an umbrella in Seattle today." → `get_weather(location="Seattle")`
- "Make sure I'm up by 6." → `set_alarm(hour=6)`
- "I'm in the mood for some Beatles." → `play_music(...)`

**Multi-action, 2 calls expected (3):**
- "Check the weather in NYC and set an alarm for 9 AM." → `get_weather` + `set_alarm`
- "Send a message to Alice saying I'm on my way, and play some jazz." → `send_message` + `play_music`
- "Set a 15 minute timer and remind me to check the oven at 5 PM." → `set_timer` + `create_reminder`

**Multi-action, 3 calls expected (1):**
- "What's the weather in London, set an alarm for 8 AM, and play Yesterday by the Beatles." → `get_weather` + `set_alarm` + `play_music`

### Tool Definitions (7 total)

`get_weather`, `send_message`, `set_alarm`, `play_music`, `set_timer`, `create_reminder`, `search_contacts` — standard function-calling schemas with typed parameters.

### Tool Set Modes

- **exact** — only the tool(s) the case actually needs
- **three** — target tool(s) + 1-2 unrelated distractors
- **all** — all 7 tools passed in

### Baseline Knobs (unless varied in the experiment)

| Knob | Default |
|------|---------|
| `force_tools` | `True` |
| `tool_rag_top_k` | `0` (disabled) |
| `max_tokens` | `256` |
| `temperature` | not set (model default) |

---

## Experiment 1: Tool Set Size (single actions, 9 cases)

**Question:** Does passing more tools confuse the model?

| Tool mode | fn correct | args correct | avg confidence | avg time |
|-----------|-----------|-------------|---------------|---------|
| exact (1 tool) | 4/9 | 3/9 | 0.871 | 173ms |
| three (3 tools) | 5/9 | 3/9 | 0.953 | 240ms |
| all (7 tools) | 5/9 | 3/9 | 0.985 | 375ms |

**Finding:** More tools does NOT hurt accuracy — it marginally helps. Having extra tool definitions seems to give the model more grounding context. Time increases with more tools (longer prefill), but accuracy holds or improves. The "exact 1 tool" mode was actually worst.

---

## Experiment 2: `tool_rag_top_k` (all 7 tools, single actions, 9 cases)

**Question:** The Cactus SDK has a built-in "Tool RAG" that pre-selects the top-k most relevant tools before the model sees them. Does this help?

| `tool_rag_top_k` | fn correct | args correct | avg confidence | avg time |
|-------------------|-----------|-------------|---------------|---------|
| **0 (disabled)** | **5/9** | **3/9** | 0.982 | 421ms |
| 1 | 1/9 | 1/9 | 0.865 | 300ms |
| 2 (default) | 3/9 | 2/9 | 0.868 | 382ms |
| 3 | 3/9 | 1/9 | 0.976 | 503ms |
| 5 | 4/9 | 3/9 | 0.985 | 445ms |

**Finding:** Tool RAG is actively harmful. At k=1, the RAG engine picks the wrong single tool most of the time (1/9 correct). Even the default k=2 degrades results. As k approaches 7 (all tools), accuracy recovers. **Disable it (`k=0`).**

---

## Experiment 3: `force_tools` (all tools, single actions, 9 cases)

**Question:** `force_tools=True` constrains the model to output in tool-call format. Does it matter?

| `force_tools` | fn correct | args correct | avg confidence | avg time |
|---------------|-----------|-------------|---------------|---------|
| True | 5/9 | 3/9 | 0.983 | 429ms |
| False | 4/9 | 2/9 | 0.982 | 405ms |

**Finding:** Marginal difference. `True` is slightly better. With `False`, the model hallucinated a function name (`timer` instead of `set_timer`), suggesting the constraint prevents name-mangling. Keep it on.

---

## Experiment 4: Temperature (all tools, first 6 single-action cases)

**Question:** Does sampling temperature affect tool-calling quality?

| Temperature | fn correct | args correct | avg confidence | avg time |
|-------------|-----------|-------------|---------------|---------|
| **0.0** | **5/6** | **3/6** | 0.994 | 326ms |
| 0.3 | 2/6 | 1/6 | 0.937 | 296ms |
| 0.7 | 2/6 | 2/6 | 0.494 | 193ms |
| 1.0 | 2/6 | 0/6 | 0.660 | 237ms |
| 1.5 | 1/6 | 0/6 | 0.820 | 286ms |

**Finding:** Greedy decoding (temp=0.0) is overwhelmingly best. Any temperature > 0 causes rapid degradation. At temp=1.0 the model produced nested garbage args (`{"minutes": {"minutes": 1440}}`). At temp=1.5 it hallucinated `"Sleigh All The Way By Train"` for Stairway to Heaven.

---

## Experiment 5: Multi-Action Calls (4 multi-action cases)

**Question:** Can the model produce multiple function calls in one generation?

| Tool mode | fn correct | args correct | calls produced |
|-----------|-----------|-------------|---------------|
| exact | 0/4 | 0/4 | 1 call max |
| all | 0/4 | 0/4 | 0-1 calls |

**Finding:** **Multi-action is fundamentally broken.** The 270M model never produces more than 1 function call per generation, regardless of configuration. It picks the first mentioned action and ignores the rest. This is a model capacity limitation, not a knob problem.

---

## Experiment 6: `max_tokens` on Multi-Action (all tools, 4 multi-action cases)

**Question:** Maybe the model tries to emit multiple calls but runs out of tokens?

| `max_tokens` | fn correct | calls produced |
|--------------|-----------|---------------|
| 64 | 0/4 | 0-1 |
| 128 | 0/4 | 0-1 |
| 256 | 0/4 | 0-1 |
| 512 | 0/4 | 0-1 |

**Finding:** Nope. Even with 512 tokens of headroom, the model stops after 1 call. It's not a truncation issue.

---

## Experiment 7: `tool_rag_top_k` on Multi-Action (all tools, 4 multi-action cases)

**Question:** Does Tool RAG help the model find the right tools for multi-action?

| `tool_rag_top_k` | fn correct | notes |
|-------------------|-----------|-------|
| 0 | 0/4 | 1 correct fn picked, rest ignored |
| 2 | 0/4 | RAG filtered to wrong tools |
| 3 | 0/4 | Got partial fns but wrong args (hour=-12, minute=60) |
| 5 | 0/4 | Same pattern |

**Finding:** No RAG setting helps with multi-action. The model can't do it.

---

## Cross-Cutting Observations

### Confidence is unreliable
The model reports confidence 0.95-0.99 even when:
- Producing **zero** function calls (just a text response)
- Calling the **wrong** function
- Returning **garbage** arguments

Confidence cannot be used as a reliable routing signal.

### Indirect phrasing always fails (0% across all experiments)
- "I wonder if I need an umbrella in Seattle" → never triggers `get_weather`
- "Make sure I'm up by 6" → never triggers `set_alarm`
- "I'm in the mood for some Beatles" → never triggers `play_music`

The model is extremely literal. It needs phrases like "What's the weather in X" or "Set an alarm for X". Any indirect intent should route to cloud.

### Recurring argument bugs
| Bug | Example |
|-----|---------|
| Negative numbers | `set_timer(minutes=-10)` for "10 minutes" |
| Time parsing broken | "7:30 AM" → `hour=0, minute=30` or `hour=10, minute=3` |
| Empty arguments | `play_music({})` — song not extracted |
| Hallucinated content | User said "hi there", model generated "Hi there! I am calling from the office. Thank you." |
| Non-determinism | Same input produces different (often wrong) results across runs |

### Zero-call responses
In many failures, the model returns a conversational text response instead of a function call, even with `force_tools=True`. Example: "I am sorry, but I cannot assist with finding weather information." — despite `get_weather` being in the tool list. This happens most with Tool RAG enabled (which hides the correct tool).

---

## Optimal Configuration

| Knob | Best Value | Reason |
|------|-----------|--------|
| `tool_rag_top_k` | `0` | Disable Tool RAG — it filters out correct tools |
| `force_tools` | `True` | Prevents function name hallucination |
| `temperature` | `0.0` | Greedy decoding is vastly more reliable |
| `max_tokens` | `256` | Sufficient; more doesn't enable multi-call |

## Routing Recommendations

| Scenario | Route |
|----------|-------|
| Single action, direct phrasing | ✅ On-device (~55% accuracy) |
| Single action, indirect phrasing | ❌ Cloud — model is too literal |
| Multi-action (2+ calls) | ❌ Cloud, OR split prompt into N single-action calls to FunctionGemma |
| Confidence as routing signal | ⚠️ Unreliable — do not trust |
| Argument post-processing | Fix known bugs (negative numbers, time parsing) after generation |
