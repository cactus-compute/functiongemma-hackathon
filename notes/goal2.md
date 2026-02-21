# Goal 2: End-to-End Product Demo

## What the Judges Want
From the hackathon README qualitative judging rubrics:
1. **Rubric 1**: Quality of hybrid routing algorithm — depth and cleverness
2. **Rubric 2**: End-to-end products that execute function calls to solve real-world problems
3. **Rubric 3**: Building low-latency voice-to-action products, leveraging `cactus_transcribe`

## The Idea
Build a **hacky but functional Python demo** that shows a real-world use case of the hybrid routing. Not an app — just a script that takes input and actually executes actions.

This should be **simple, impressive, and demonstrate the edge/cloud hybrid** in a way humans would actually use.

## Available Cactus APIs (from README)
| API | What It Does | Ideas |
|-----|-------------|-------|
| `cactus_init` | Load a model | FunctionGemma, Whisper, LFM2, RAG models |
| `cactus_complete` | Text completion + tool calling | Core function calling |
| `cactus_transcribe` | Speech-to-text (Whisper) | **Voice commands** → function calls (Rubric 3!) |
| `cactus_embed` | Text embeddings | Semantic similarity, tool matching |
| `cactus_rag_query` | RAG over local documents | Context-aware function calling |
| `cactus_reset` | Reset model state | Reuse between calls |

## Available Models (from cactus download)
- `google/functiongemma-270m-it` — tool calling (already have)
- `weights/whisper-small` — speech transcription
- `weights/lfm2-vl-450m` — general completion
- `weights/lfm2-rag` — RAG-enabled completion

## Bonus Points: Voice-to-Action (Rubric 3)
The judges explicitly call out `cactus_transcribe`. Pipeline:
```
Voice (WAV) → cactus_transcribe → text → generate_hybrid → function calls → execute
```

This is a huge differentiator. Most teams will only do text → function calls.

## Demo Ideas (brainstorm)

### Option A: Voice Assistant That Actually Does Things
- Record voice command → transcribe → route → execute
- "Set an alarm for 7 AM" → actually sets system alarm (or prints confirmation)
- "What's the weather in SF?" → actually calls a weather API
- "Text Mom saying I'll be late" → actually sends an iMessage (via osascript)
- Show the routing decision: "Handled on-device in 45ms" vs "Routed to cloud in 800ms"
- **Why it's good**: Covers all 3 rubrics. Voice + routing + real execution.

### Option B: Smart Home Controller
- Voice/text commands → function calls → execute home automation
- Lights, thermostat, music, timers
- Show how on-device handles simple commands instantly, cloud handles complex ones
- **Why it's good**: Practical IoT use case for edge inference.

### Option C: Personal Productivity Agent
- "Remind me to call dentist at 2pm, check weather, and play focus music"
- Splits into sub-actions, routes each intelligently, executes all
- Shows multi-tool orchestration with hybrid routing
- **Why it's good**: Demonstrates the hard multi-action cases well.

## Tech Stack (keep it hacky)
- Python scripts only — no web framework, no frontend
- Real API calls where possible (weather API, system commands)
- Print-based UI showing routing decisions and execution
- Maybe a simple CLI loop: speak/type → see routing → see result

## TODO
- [ ] Decide on demo concept
- [ ] Download whisper model (`cactus download weights/whisper-small`)
- [ ] Test `cactus_transcribe` with a sample WAV
- [ ] Build the execution layer (actually call APIs / run system commands)
- [ ] Wire it all together: voice → transcribe → hybrid route → execute → display
- [ ] Prepare a brief walkthrough/script for demo
