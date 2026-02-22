#!/usr/bin/env python3
"""
Quick playground for calling FunctionGemma (on-device via Cactus).

Usage:
    source cactus/venv/bin/activate
    python test.py

Edit the TOOLS and PROMPT below to experiment. Just prints the raw output
so you can build intuition about what the model does.

 ┌───────────────────────────────────────┬────────────────────────────────────────┐
 │ Path                                  │ Tools passed to FG                     │
 ├───────────────────────────────────────┼────────────────────────────────────────┤
 │ Multi-action, tool detected           │ 1 tool (the detected one)              │
 ├───────────────────────────────────────┼────────────────────────────────────────┤
 │ Multi-action, tool NOT detected       │ All tools                              │
 ├───────────────────────────────────────┼────────────────────────────────────────┤
 │ Single-action medium, tool detected   │ 1 tool (up to 3 retries)               │
 ├───────────────────────────────────────┼────────────────────────────────────────┤
 │ Single-action easy (1 tool available) │ 1 tool (trivially — it's the only one) │
 ├───────────────────────────────────────┼────────────────────────────────────────┤
 │ Single-action, tool NOT detected      │ All tools                              │
 └───────────────────────────────────────┴────────────────────────────────────────┘
"""

import sys

sys.path.insert(0, "cactus/python/src")

import json

from cactus import cactus_complete, cactus_destroy, cactus_init

MODEL_PATH = "cactus/weights/functiongemma-270m-it"

# ── Tools ─────────────────────────────────────────────────────────────
# Define as many or as few as you like. Each is a standard function-calling schema.

TOOLS = [
    {
        "name": "get_weather",
        "description": "Get current weather for a location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "City name"},
            },
            "required": ["location"],
        },
    },
    {
        "name": "send_message",
        "description": "Send a message to a contact",
        "parameters": {
            "type": "object",
            "properties": {
                "recipient": {"type": "string", "description": "Name of the person"},
                "message": {"type": "string", "description": "The message content"},
            },
            "required": ["recipient", "message"],
        },
    },
    {
        "name": "set_alarm",
        "description": "Set an alarm for a given time",
        "parameters": {
            "type": "object",
            "properties": {
                "hour": {"type": "integer", "description": "Hour (0-23)"},
                "minute": {"type": "integer", "description": "Minute (0-59)"},
            },
            "required": ["hour", "minute"],
        },
    },
    {
        "name": "play_music",
        "description": "Play a song or playlist",
        "parameters": {
            "type": "object",
            "properties": {
                "song": {"type": "string", "description": "Song or playlist name"},
            },
            "required": ["song"],
        },
    },
    {
        "name": "set_timer",
        "description": "Set a countdown timer",
        "parameters": {
            "type": "object",
            "properties": {
                "minutes": {"type": "integer", "description": "Number of minutes"},
            },
            "required": ["minutes"],
        },
    },
    {
        "name": "create_reminder",
        "description": "Create a reminder with a title and time",
        "parameters": {
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "Reminder title"},
                "time": {
                    "type": "string",
                    "description": "Time for the reminder (e.g. 3:00 PM)",
                },
            },
            "required": ["title", "time"],
        },
    },
    {
        "name": "search_contacts",
        "description": "Search for a contact by name",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Name to search for"},
            },
            "required": ["query"],
        },
    },
]

# ── Prompt ────────────────────────────────────────────────────────────
# Change this to whatever you want to test.

PROMPT = "What's the weather in Tokyo?"

# ── Knobs ─────────────────────────────────────────────────────────────
# Tweak these to see how the model behaves differently.

SYSTEM_MSG = "You are a helpful assistant that can use tools."
FORCE_TOOLS = True
MAX_TOKENS = 256
TOOL_RAG_TOP_K = (
    0  # 0 = disabled (pass all tools). Default 2 = auto-picks top-2, can miss!
)
TEMPERATURE = None  # None = model default
TOP_P = None
TOP_K = None


# ── Run ───────────────────────────────────────────────────────────────


def call_cactus(prompt, tools, **kwargs):
    """Single call to FunctionGemma. Returns parsed dict + raw string."""
    model = cactus_init(MODEL_PATH)

    messages = [
        {"role": "system", "content": kwargs.get("system", SYSTEM_MSG)},
        {"role": "user", "content": prompt},
    ]

    cactus_tools = [{"type": "function", "function": t} for t in tools]

    # Build optional params
    params = dict(
        tools=cactus_tools,
        force_tools=kwargs.get("force_tools", FORCE_TOOLS),
        max_tokens=kwargs.get("max_tokens", MAX_TOKENS),
        stop_sequences=["<|im_end|>", "<end_of_turn>"],
    )
    if kwargs.get("temperature", TEMPERATURE) is not None:
        params["temperature"] = kwargs.get("temperature", TEMPERATURE)
    if kwargs.get("top_p", TOP_P) is not None:
        params["top_p"] = kwargs.get("top_p", TOP_P)
    if kwargs.get("top_k", TOP_K) is not None:
        params["top_k"] = kwargs.get("top_k", TOP_K)
    if kwargs.get("tool_rag_top_k", TOOL_RAG_TOP_K) is not None:
        params["tool_rag_top_k"] = kwargs.get("tool_rag_top_k", TOOL_RAG_TOP_K)

    raw_str = cactus_complete(model, messages, **params)
    cactus_destroy(model)

    try:
        parsed = json.loads(raw_str)
    except json.JSONDecodeError:
        parsed = {"_raw_parse_error": True}

    return raw_str, parsed


def pretty_print(raw_str, parsed):
    """Print results in a readable way."""
    print("─" * 60)
    print(f"Raw output:\n{raw_str}\n")
    print(f"Parsed JSON:\n{json.dumps(parsed, indent=2)}\n")

    calls = parsed.get("function_calls", [])
    conf = parsed.get("confidence", "?")
    ms = parsed.get("total_time_ms", "?")
    handoff = parsed.get("cloud_handoff", "?")

    print(f"Function calls: {len(calls)}")
    for c in calls:
        print(f"  → {c['name']}({json.dumps(c.get('arguments', {}))})")
    print(f"Confidence: {conf}")
    print(f"Time: {ms} ms")
    print(f"Cloud handoff: {handoff}")
    print("─" * 60)


if __name__ == "__main__":
    print(f"\nPrompt: {PROMPT!r}")
    print(f"Tools:  {[t['name'] for t in TOOLS]}")
    print(
        f"Knobs:  force_tools={FORCE_TOOLS}, max_tokens={MAX_TOKENS}, "
        f"tool_rag_top_k={TOOL_RAG_TOP_K}, temp={TEMPERATURE}\n"
    )

    raw_str, parsed = call_cactus(PROMPT, TOOLS)
    pretty_print(raw_str, parsed)
