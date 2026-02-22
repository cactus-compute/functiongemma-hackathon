#!/usr/bin/env python3
"""
Interactive playground for FunctionGemma via Cactus.

Usage:
    source cactus/venv/bin/activate
    python test-max.py "What's the weather in Tokyo?"
    python test-max.py "Set an alarm for 7am and play some jazz"
    python test-max.py "Remind me to call dentist at 2pm"
"""

import json
import sys

sys.path.insert(0, "cactus/python/src")

from cactus import cactus_complete, cactus_destroy, cactus_init

MODEL_PATH = "cactus/weights/functiongemma-270m-it"

# ── Tools ─────────────────────────────────────────────────────────────

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

# ── Knobs ─────────────────────────────────────────────────────────────

SYSTEM_MSG = "You are a helpful assistant that can use tools."
FORCE_TOOLS = False  # Raw output — no forcing
MAX_TOKENS = 256
TOOL_RAG_TOP_K = 0  # 0 = pass all tools (no RAG filtering)
TEMPERATURE = None
TOP_P = None
TOP_K = None

# ── Run ───────────────────────────────────────────────────────────────


def call_cactus(prompt, tools):
    """Single call to FunctionGemma. Returns raw string + parsed dict."""
    model = cactus_init(MODEL_PATH)

    messages = [
        {"role": "system", "content": SYSTEM_MSG},
        {"role": "user", "content": prompt},
    ]

    cactus_tools = [{"type": "function", "function": t} for t in tools]

    params = dict(
        tools=cactus_tools,
        force_tools=FORCE_TOOLS,
        max_tokens=MAX_TOKENS,
        stop_sequences=["<|im_end|>", "<end_of_turn>"],
    )
    if TOOL_RAG_TOP_K is not None:
        params["tool_rag_top_k"] = TOOL_RAG_TOP_K
    if TEMPERATURE is not None:
        params["temperature"] = TEMPERATURE
    if TOP_P is not None:
        params["top_p"] = TOP_P
    if TOP_K is not None:
        params["top_k"] = TOP_K

    raw_str = cactus_complete(model, messages, **params)
    cactus_destroy(model)
    print(raw_str)

    try:
        parsed = json.loads(raw_str)
    except json.JSONDecodeError:
        parsed = {"_raw_parse_error": True}

    return raw_str, parsed


def pretty_print(raw_str, parsed):
    """Print results nicely."""
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
    if len(sys.argv) < 2:
        print('Usage: python test-max.py "your prompt here"')
        print('Example: python test-max.py "What\'s the weather in Tokyo?"')
        sys.exit(1)

    prompt = " ".join(sys.argv[1:])

    print(f"\nPrompt: {prompt!r}")
    print(f"Tools:  {[t['name'] for t in TOOLS]}")
    print(
        f"Knobs:  force_tools={FORCE_TOOLS}, max_tokens={MAX_TOKENS}, "
        f"tool_rag_top_k={TOOL_RAG_TOP_K}, temp={TEMPERATURE}\n"
    )

    raw_str, parsed = call_cactus(prompt, TOOLS)
    pretty_print(raw_str, parsed)
