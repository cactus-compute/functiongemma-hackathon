#!/usr/bin/env python3
"""
Statistical sampling: run FunctionGemma N times with temp > 0
and see how results vary. Can we get better accuracy with majority vote?

Usage:
    source cactus/venv/bin/activate
    python test-fg-sample.py "Set an alarm for 7am"
    python test-fg-sample.py "Play some jazz" --n 5 --temp 0.7
"""

import sys
import json
import argparse
from collections import Counter

sys.path.insert(0, "cactus/python/src")

from cactus import cactus_init, cactus_complete, cactus_destroy, cactus_reset

MODEL_PATH = "cactus/weights/functiongemma-270m-it"

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

SYSTEM_MSG = "You are a helpful assistant that can use tools."


def sample_fg(prompt, n=10, temperature=0.8, force_tools=False):
    """Run FG n times with temperature, collect all results."""
    model = cactus_init(MODEL_PATH)

    messages = [
        {"role": "system", "content": SYSTEM_MSG},
        {"role": "user", "content": prompt},
    ]
    cactus_tools = [{"type": "function", "function": t} for t in TOOLS]

    results = []
    for i in range(n):
        cactus_reset(model)
        raw = cactus_complete(
            model,
            messages,
            tools=cactus_tools,
            force_tools=force_tools,
            max_tokens=256,
            temperature=temperature,
            tool_rag_top_k=0,
            stop_sequences=["<|im_end|>", "<end_of_turn>"],
        )
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            parsed = {"_parse_error": True, "_raw": raw[:200], "function_calls": [], "confidence": 0}
        results.append(parsed)

    cactus_destroy(model)
    return results


def analyze(results):
    """Analyze the distribution of results."""
    # Extract function call signatures for voting
    signatures = []
    for r in results:
        calls = r.get("function_calls", [])
        if calls:
            # Signature = tuple of (name, frozen args)
            sig = tuple(
                (c["name"], json.dumps(c.get("arguments", {}), sort_keys=True))
                for c in calls
            )
        else:
            sig = ("NO_CALLS",)
        signatures.append(sig)

    # Count votes
    counter = Counter(signatures)

    # Tool name distribution (ignoring args)
    tool_names = Counter()
    for r in results:
        for c in r.get("function_calls", []):
            tool_names[c["name"]] += 1

    return counter, tool_names


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("prompt", help="The prompt to test")
    parser.add_argument("--n", type=int, default=10, help="Number of samples")
    parser.add_argument("--temp", type=float, default=0.8, help="Temperature")
    parser.add_argument("--force", action="store_true", help="Use force_tools")
    args = parser.parse_args()

    print(f"\nPrompt:      {args.prompt!r}")
    print(f"Samples:     {args.n}")
    print(f"Temperature: {args.temp}")
    print(f"Force tools: {args.force}")
    print(f"Tools:       {[t['name'] for t in TOOLS]}")
    print()

    results = sample_fg(args.prompt, n=args.n, temperature=args.temp, force_tools=args.force)
    votes, tool_names = analyze(results)

    # Print each sample
    print("── Individual Samples ──")
    for i, r in enumerate(results):
        calls = r.get("function_calls", [])
        conf = r.get("confidence", "?")
        ms = r.get("total_time_ms", "?")
        if calls:
            call_strs = [f"{c['name']}({json.dumps(c.get('arguments', {}))})" for c in calls]
            print(f"  [{i+1:2d}] {' + '.join(call_strs):60s}  conf={conf}  {ms}ms")
        else:
            print(f"  [{i+1:2d}] {'(no calls)':60s}  conf={conf}  {ms}ms")

    # Print vote summary
    print(f"\n── Vote Distribution ({args.n} samples) ──")
    for sig, count in votes.most_common():
        pct = count / args.n * 100
        if sig == ("NO_CALLS",):
            label = "(no calls)"
        else:
            label = " + ".join(f"{name}({args_str})" for name, args_str in sig)
        bar = "█" * count
        print(f"  {count:2d}/{args.n} ({pct:5.1f}%) {bar}  {label}")

    # Print tool name distribution
    print(f"\n── Tool Selection ──")
    for name, count in tool_names.most_common():
        print(f"  {name}: {count}/{args.n}")

    # Winner
    winner_sig, winner_count = votes.most_common(1)[0]
    print(f"\n── Winner ({winner_count}/{args.n} votes) ──")
    if winner_sig == ("NO_CALLS",):
        print("  (no calls)")
    else:
        for name, args_str in winner_sig:
            print(f"  → {name}({args_str})")

    # Timing stats
    times = [r.get("total_time_ms", 0) for r in results]
    print(f"\n── Timing ──")
    print(f"  Total: {sum(times):.0f}ms for {args.n} samples")
    print(f"  Avg:   {sum(times)/len(times):.0f}ms per sample")
    print(f"  Min:   {min(times):.0f}ms  Max: {max(times):.0f}ms")


if __name__ == "__main__":
    main()
