<img src="assets/banner.png" alt="Logo" style="border-radius: 30px; width: 100%;">

## Context
- Cactus runs Google DeepMind's FunctionGemma at up to 3000 toks/sec prefill speed on M4 Macs.
- While decode speed reaches 200 tokens/sec, all without GPU, to remain energy-efficient. 
- FunctionGemma is great at tool calling, but small models are not the smartest for some tasks. 
- There is a need to dynamically combine edge and cloud (Gemini 3.1) to get the best of both worlds. 
- Cactus developed various strategies for choosing when to fall back to Gemini or FunctionGemma.

## Challenge
- FunctionGemma is just a tool-call model, but tool calling is the core of agentic systems. 
- You MUST design new strategies that decide when to stick with on-device or fall to cloud. 
- You will be objectively ranked on tool-call correctness, speed and edge/cloud ratio (priortize local). 
- You can focus on prompting, tool description patterns, confidence score algorithms, anything!

## Submissions
- Do not modify the `generate_hybrid` keep the hybrid interface compatible with benchmark.py.
- Submit to the leaderboard `python submit.py --team "YourTeamName"`, 1x every 10mins.
- The dataset is a hidden Cactus eval, EXTREMELY difficult for FunctionGemma by design.
- For transparency, hackers can see live rankings on the [leaderboard](https://khalilah-unbibulous-carlena.ngrok-free.dev), 
- The top 10 in each location will make it to judging.

## Qualitative Judging 
- **Rubric 1**: The quality of your hybrid routing algorithm, depth and cleverness.
- **Rubric 2**: End-to-end products that execute function calls to solve real-world problems. 
- **Rubric 3**: Building low-latency voice-to-action products, leveraging `cactus_transcribe`.

## Setup (clone this repo and hollistically follow)
- Step 1: Fork this repo, clone to your Mac, open terminal.
- Step 2: `git clone https://github.com/cactus-compute/cactus`
- Step 3: `cd cactus && source ./setup && cd ..`
- Step 4: `cactus build --python`
- Step 5: `cactus download google/functiongemma-270m-it --reconvert`
- Step 6: Get cactus key from the [cactus website](https://cactuscompute.com/dashboard/api-keys)
- Sept 7: Run `cactus auth` and enter your token when prompted.
- Step 8: `pip install google-genai`
- Step 9: Obtain Gemini API key from [Google AI Studio](https://aistudio.google.com/api-keys)
- Step 10: `export GEMINI_API_KEY="your-api-key-here"`
- Step 11: Join the [Reddit channel](https://www.reddit.com/r/cactuscompute/), ask any technical questions there.
- Step 12: read and run `python example.py`
- Step 13: read and run `python benchmark.py` to understand how objective scoring works.
- Note: Final objective score will be done on held-out evals, top 10 are then judged subjectively.

## Submissions
- Do not modify the `generate_hybrid` keep the hybrid interface compatible with benchmark.py.
- Submit to the leaderboard `python submit.py --team "YourTeamName"`
- Record a demo video, and submit along with your demo video. 
- Submission link will be provided!

## Quick Example

```python
import json
from cactus import cactus_init, cactus_complete, cactus_destroy

model = cactus_init("weights/lfm2-vl-450m")
messages = [{"role": "user", "content": "What is 2+2?"}]
response = json.loads(cactus_complete(model, messages))
print(response["response"])

cactus_destroy(model)
```

## API Reference

### `cactus_init(model_path, corpus_dir=None)`

| Parameter | Type | Description |
|-----------|------|-------------|
| `model_path` | `str` | Path to model weights directory |
| `corpus_dir` | `str` | (Optional) dir of txt/md files for auto-RAG |

```python
model = cactus_init("weights/lfm2-vl-450m")
model = cactus_init("weights/lfm2-rag", corpus_dir="./documents")
```

### `cactus_complete(model, messages, **options)`

| Parameter | Type | Description |
|-----------|------|-------------|
| `model` | handle | Model handle from `cactus_init` |
| `messages` | `list\|str` | List of message dicts or JSON string |
| `tools` | `list` | Optional tool definitions for function calling |
| `temperature` | `float` | Sampling temperature |
| `top_p` | `float` | Top-p sampling |
| `top_k` | `int` | Top-k sampling |
| `max_tokens` | `int` | Maximum tokens to generate |
| `stop_sequences` | `list` | Stop sequences |
| `include_stop_sequences` | `bool` | Include matched stop sequences in output (default: `False`) |
| `force_tools` | `bool` | Constrain output to tool call format |
| `tool_rag_top_k` | `int` | Select top-k relevant tools via Tool RAG (default: 2, 0 = use all tools) |
| `confidence_threshold` | `float` | Minimum confidence for local generation (default: 0.7, triggers cloud_handoff when below) |
| `callback` | `fn` | Streaming callback `fn(token, token_id, user_data)` |

```python
# Basic completion
messages = [{"role": "user", "content": "Hello!"}]
response = cactus_complete(model, messages, max_tokens=100)
print(json.loads(response)["response"])
```

```python
# Completion with tools
tools = [{
    "name": "get_weather",
    "description": "Get weather for a location",
    "parameters": {
        "type": "object",
        "properties": {"location": {"type": "string"}},
        "required": ["location"]
    }
}]

response = cactus_complete(model, messages, tools=tools)
cactus_complete(model, messages, callback=on_token)
```

**Response format** (all fields always present):
```json
{
    "success": true,
    "error": null,
    "cloud_handoff": false,
    "response": "Hello! How can I help?",
    "function_calls": [],
    "confidence": 0.85,
    "time_to_first_token_ms": 45.2,
    "total_time_ms": 163.7,
    "prefill_tps": 619.5,
    "decode_tps": 168.4,
    "ram_usage_mb": 245.67,
    "prefill_tokens": 28,
    "decode_tokens": 50,
    "total_tokens": 78
}
```

**Cloud handoff response** (when model detects low confidence):
```json
{
    "success": false,
    "error": null,
    "cloud_handoff": true,
    "response": null,
    "function_calls": [],
    "confidence": 0.18,
    "time_to_first_token_ms": 45.2,
    "total_time_ms": 45.2,
    "prefill_tps": 619.5,
    "decode_tps": 0.0,
    "ram_usage_mb": 245.67,
    "prefill_tokens": 28,
    "decode_tokens": 0,
    "total_tokens": 28
}
```

- When `cloud_handoff` is `True`, the model's confidence dropped below `confidence_threshold` (default: 0.7) and recommends deferring to a cloud-based model for better results. 

- You will NOT rely on this, hackers must design custom strategies to fall-back to cloud, that maximizes on-devices and correctness, while minimizing end-to-end latency!

### `cactus_transcribe(model, audio_path, prompt="")`

| Parameter | Type | Description |
|-----------|------|-------------|
| `model` | handle | Whisper model handle |
| `audio_path` | `str` | Path to audio file (WAV) |
| `prompt` | `str` | Whisper prompt for language/task |

```python
whisper = cactus_init("weights/whisper-small")
prompt = "<|startoftranscript|><|en|><|transcribe|><|notimestamps|>"
response = cactus_transcribe(whisper, "audio.wav", prompt=prompt)
print(json.loads(response)["response"])
cactus_destroy(whisper)
```

### `cactus_embed(model, text, normalize=False)`

| Parameter | Type | Description |
|-----------|------|-------------|
| `model` | handle | Model handle |
| `text` | `str` | Text to embed |
| `normalize` | `bool` | L2-normalize embeddings (default: False) |

```python
embedding = cactus_embed(model, "Hello world")
print(f"Dimension: {len(embedding)}")
```

### `cactus_reset(model)`

Reset model state (clear KV cache). Call between unrelated conversations.

```python
cactus_reset(model)
```

### `cactus_stop(model)`

Stop an ongoing generation (useful with streaming callbacks).

```python
cactus_stop(model)
```

### `cactus_destroy(model)`

Free model memory. Always call when done.

```python
cactus_destroy(model)
```

### `cactus_get_last_error()`

Get the last error message, or `None` if no error.

```python
error = cactus_get_last_error()
if error:
    print(f"Error: {error}")
```

### `cactus_rag_query(model, query, top_k=5)`

Query RAG corpus for relevant text chunks. Requires model initialized with `corpus_dir`.

| Parameter | Type | Description |
|-----------|------|-------------|
| `model` | handle | Model handle (must have corpus_dir set) |
| `query` | `str` | Query text |
| `top_k` | `int` | Number of chunks to retrieve (default: 5) |

```python
model = cactus_init("weights/lfm2-rag", corpus_dir="./documents")
chunks = cactus_rag_query(model, "What is machine learning?", top_k=3)
for chunk in chunks:
    print(f"Score: {chunk['score']:.2f} - {chunk['text'][:100]}...")
```

## Next steps:
- Join the [Reddit channel](https://www.reddit.com/r/cactuscompute/), ask any technical questions there.