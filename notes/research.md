# Research Notes

Web search findings relevant to FunctionGemma, tool-calling strategies, and hybrid routing.

## FunctionGemma Model Card (HuggingFace) — Searched Feb 21, 2026

### Official System Prompt
```python
message = [
    # ESSENTIAL SYSTEM PROMPT:
    # This line activates the model's function calling logic.
    {"role": "developer", "content": "You are a model that can do function calling with the following functions"},
    {"role": "user", "content": "What's the temperature in London?"}
]
```
**TESTED**: Switching to this prompt (with either "system" or "developer" role) HURTS performance when using Cactus SDK. The Cactus SDK likely handles the prompt template internally. Our original prompt "You are a helpful assistant that can use tools." works better.

### Model Limitations
- FunctionGemma is "not intended for use as a direct dialogue model"
- "Designed to be highly performant after further fine-tuning, as is typical of models this size"
- Built on Gemma 3 270M, "uses a different chat format"
- Total input context: 32K tokens
- Chat format uses `<start_function_call>call:func_name{params}<end_function_call>`

### Unsloth Documentation Notes
- System/developer message MUST be: "You are a model that can do function calling with the following functions"
- "Unsloth versions have this pre-built in if you forget to pass one"
- This confirms Cactus likely pre-builds it too → our custom prompt is additive

## Distil Labs Fine-Tuning — Feb 17, 2026 (r/LocalLLaMA)
- Fine-tuned FunctionGemma 270M using knowledge distillation from 120B teacher
- Achieved 90-97% multi-turn accuracy (base was 10-39%)
- 445× smaller than teacher model while matching performance
- Not applicable in hackathon (can't fine-tune), but shows model potential

## Cactus SDK Parameters (from cactus.py deep dive)

### cactus_complete() full parameter list
| Parameter | Default | Notes |
|-----------|---------|-------|
| temperature | None (model default) | Sampling temperature |
| top_p | None | Top-p sampling |
| top_k | None | Top-k sampling |
| max_tokens | None | Max tokens to generate |
| stop_sequences | None | Stop sequences |
| force_tools | False | Constrain output to tool call format |
| tool_rag_top_k | 2 | Select top-k relevant tools via Tool RAG. 0 = disabled |
| confidence_threshold | 0.7 | Min confidence for local gen (triggers cloud_handoff when below) |

### cloud_handoff field
Response includes `cloud_handoff: bool` — True when confidence below threshold. Could use as routing signal but confidence is known to be unreliable.

### tool_rag_top_k
Default is 2 (auto-selects top-2 relevant tools). Setting to 0 disables Tool RAG. **UNEXPLORED** — could help when we're already passing single tool (disable overhead) or when passing all tools (let RAG pick).

## Reddit r/cactuscompute — Checked Feb 21, 2026 12:13 AM
- No hackathon-specific posts found
- Found general post about Cactus v1.7 with hybrid inference features
- Check again roughly once per hour

## Reddit r/cactuscompute — Checked Feb 21, 2026 1:31 PM PST
Recent posts during hackathon day:
1. **"LOL Classic Hackathon Leakage"** (u/That-Cry3210, ~1 hr ago) — Body text not retrievable. Could be about test set patterns or someone sharing info.
2. **"Error: gemini-2.0-flash no longer available"** (u/Bubbly-Wonder-2599, ~4 hrs ago) — Asking if 2.5 flash is ok. **We already use 2.5 — confirmed correct.**
3. **"Need some help"** (u/Embarrassed_Pack4842, ~9 hrs ago) — HF gated access issue. Solved by accepting HF terms.
4. Several logistics posts: remote access issues, online link questions, team pending status.
5. **"Auto RAG & Local + hybrid Inference"** (u/Henrie_the_dreamer, 4 days ago) — Cactus v1.7 features.

**Takeaway**: Most teams still setting up. No strategy tips or intel shared. We're well ahead with 89.15% score.

## Cactus HN Launch — Key Quote
"We found Qwen3 600m to be great at tool calls for instance." — Could be worth exploring as alternative model, but requires download and different setup.

## Gemini Cloud Behavior Insights
1. **Sequential interpretation**: When given "Find X in contacts and send message to X", Gemini treats as sequential workflow → only returns first step (search_contacts). Fix: make individual cloud calls per sub-action.
2. **Trailing punctuation**: Gemini sometimes adds trailing period to string arguments (e.g., "I'll be late." instead of "I'll be late"). Fix: strip in post-processing.
3. **Gemini-2.5-flash** is the model we're using for cloud calls.
