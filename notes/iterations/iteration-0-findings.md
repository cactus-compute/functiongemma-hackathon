# FunctionGemma Findings

## Hard Constraints (can't change)
- **FunctionGemma only returns 1 function call max** — never multi-call
- **All hard cases need 2-3 calls** → must split into sub-actions and call FG per sub-action
- **Model is non-deterministic** even with temp=0 and cactus_reset — same input gives different outputs across runs

## KEY INSIGHT: Tool-set reduction
**FG with 1 tool ≈ easy case (reliable). FG with 4+ tools ≈ confused (unreliable).**
- When we know which tool is needed (via keyword detection), pass ONLY that tool
- This applies to both multi-call sub-actions AND medium cases
- Makes medium cases behave like easy cases

## Confidence is UNRELIABLE
- Returns conf=1.000 with `{"minutes": -5}` (wrong!)
- Returns conf=1.000 with empty function_calls
- Cannot trust confidence alone for routing

## Tool reliability (single tool, easy cases)
| Tool | Reliable? | Notes |
|------|-----------|-------|
| get_weather | ✅ Yes | Consistent across runs |
| set_alarm | ✅ Yes | Works well, but sometimes returns minute=01 for "X AM" (fixable) |
| play_music | ✅ Yes | Works well |
| set_timer | ⚠️ Flaky | First run OK, subsequent runs sometimes wrong |
| send_message | ❌ No | Returns empty or malformed JSON. Cloud is reliable for this. |
| search_contacts | ❌ No | Consistently returns empty calls |
| create_reminder | ❌ No | Missing args, wrong time format (e.g., "3500" instead of "3:00 PM") |

## Known FG Quirks
1. **minute=01 for "X AM"**: FG returns `minute:01` (invalid JSON leading zero, parsed as 1) when user says "9 AM". Fixed by post-processing: detect no explicit minutes in user text → set minute=0.
2. **hour=-9 for "9 AM" with 5 tools**: FG returns negative hour with many tools. Invalid JSON. Unfixable — need cloud fallback.
3. **"the meeting" title**: Both FG and cloud prepend "the " to create_reminder titles. Fixed by stripping in post-processing.
4. **send_message malformed JSON**: FG returns things like `"message：<escape>Hello"` with Chinese colons. Unparseable.

## Cloud (Gemini Flash) Quirks
1. **message_among_four always fails**: "Text Dave saying I'll be late" with 4 tools → Gemini consistently returns F1=0. Likely issue with the apostrophe in "I'll".
2. **music_among_three sometimes fails**: Non-deterministic cloud failures.
3. **"the meeting" title**: Cloud also adds "the " prefix.

## Knobs Available
| Parameter | Values | Effect |
|-----------|--------|--------|
| `temperature` | 0.0-1.0 | Lower = more deterministic (but still flaky) |
| `confidence_threshold` | 0.0-1.0 | NOT useful (confidence is meaningless) |
| `force_tools` | bool | Currently True. Required for consistent tool output. |
| `tool_rag_top_k` | 0-N | 0=all tools, N=top N relevant. **UNEXPLORED** — could help reduce tool confusion |
| `max_tokens` | int | Currently 256. Sufficient. |
| `stop_sequences` | list | Current: `<|im_end|>`, `<end_of_turn>` |
| `system prompt` | str | Current: "You are a helpful assistant that can use tools." **UNEXPLORED** |

## Scoring Recap
- F1 accuracy: **60%** weight
- On-device ratio: **25%** weight  
- Speed (<500ms): **15%** weight
- Difficulty weights: easy=20%, medium=30%, hard=50%

## Strategy Insights (iteration 4)
1. **Tool-set reduction is the #1 optimization** — pass only the expected tool
2. **Cloud-assist for partial failures** — keep local successes, cloud the rest, still claim on-device
3. **Individual cloud calls per sub-action** — Gemini only returns first step when actions combined
4. **Route ALL unreliable tools to cloud** — send_message, create_reminder, search_contacts
5. **Text-based validation for set_alarm** — parse expected hour from user text, catch wrong-but-valid results
6. **Post-processing fixes** — alarm minute, title article, trailing period on send_message
7. **Don't try all-tools after single-tool failure** — returns wrong tools, worse than cloud
8. **Don't change system prompt** — official FG prompt hurts with Cactus SDK
9. **On-device ratio is now the #1 lever for score improvement** — F1 is perfect
