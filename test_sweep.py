#!/usr/bin/env python3
"""
Systematic sweep of FunctionGemma behavior across inputs & knobs.
Runs fast (~300ms per call), so we can do ~100 tests in ~30s.
"""

import sys
sys.path.insert(0, "cactus/python/src")

import json, time, os
os.environ["CACTUS_NO_CLOUD_TELE"] = "1"

from cactus import cactus_init, cactus_complete, cactus_destroy

MODEL_PATH = "cactus/weights/functiongemma-270m-it"

# ── Tool definitions ──────────────────────────────────────────────────

TOOLS = {
    "weather": {
        "name": "get_weather",
        "description": "Get current weather for a location",
        "parameters": {"type": "object", "properties": {
            "location": {"type": "string", "description": "City name"},
        }, "required": ["location"]},
    },
    "message": {
        "name": "send_message",
        "description": "Send a message to a contact",
        "parameters": {"type": "object", "properties": {
            "recipient": {"type": "string", "description": "Name of the person"},
            "message": {"type": "string", "description": "The message content"},
        }, "required": ["recipient", "message"]},
    },
    "alarm": {
        "name": "set_alarm",
        "description": "Set an alarm for a given time",
        "parameters": {"type": "object", "properties": {
            "hour": {"type": "integer", "description": "Hour (0-23)"},
            "minute": {"type": "integer", "description": "Minute (0-59)"},
        }, "required": ["hour", "minute"]},
    },
    "music": {
        "name": "play_music",
        "description": "Play a song or playlist",
        "parameters": {"type": "object", "properties": {
            "song": {"type": "string", "description": "Song or playlist name"},
        }, "required": ["song"]},
    },
    "timer": {
        "name": "set_timer",
        "description": "Set a countdown timer",
        "parameters": {"type": "object", "properties": {
            "minutes": {"type": "integer", "description": "Number of minutes"},
        }, "required": ["minutes"]},
    },
    "reminder": {
        "name": "create_reminder",
        "description": "Create a reminder with a title and time",
        "parameters": {"type": "object", "properties": {
            "title": {"type": "string", "description": "Reminder title"},
            "time": {"type": "string", "description": "Time for the reminder (e.g. 3:00 PM)"},
        }, "required": ["title", "time"]},
    },
    "contacts": {
        "name": "search_contacts",
        "description": "Search for a contact by name",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string", "description": "Name to search for"},
        }, "required": ["query"]},
    },
}

ALL_TOOLS = list(TOOLS.values())

# ── Test cases ────────────────────────────────────────────────────────

CASES = [
    # --- Single action, clear intent ---
    {"id": "single_weather",    "prompt": "What's the weather in Tokyo?",
     "expect_fn": "get_weather", "expect_args": {"location": "Tokyo"}, "type": "single"},
    {"id": "single_alarm",      "prompt": "Set an alarm for 7:30 AM.",
     "expect_fn": "set_alarm",   "expect_args": {"hour": 7, "minute": 30}, "type": "single"},
    {"id": "single_message",    "prompt": "Send a message to Bob saying hi there.",
     "expect_fn": "send_message","expect_args": {"recipient": "Bob"}, "type": "single"},
    {"id": "single_music",      "prompt": "Play Stairway to Heaven.",
     "expect_fn": "play_music",  "expect_args": {"song": "Stairway to Heaven"}, "type": "single"},
    {"id": "single_timer",      "prompt": "Set a timer for 10 minutes.",
     "expect_fn": "set_timer",   "expect_args": {"minutes": 10}, "type": "single"},
    {"id": "single_reminder",   "prompt": "Remind me to call the dentist at 2 PM.",
     "expect_fn": "create_reminder", "expect_args": {"time": "2 PM"}, "type": "single"},

    # --- Single action, indirect phrasing ---
    {"id": "indirect_weather",  "prompt": "I wonder if I need an umbrella in Seattle today.",
     "expect_fn": "get_weather", "expect_args": {"location": "Seattle"}, "type": "single"},
    {"id": "indirect_alarm",    "prompt": "Make sure I'm up by 6.",
     "expect_fn": "set_alarm",   "expect_args": {"hour": 6}, "type": "single"},
    {"id": "indirect_music",    "prompt": "I'm in the mood for some Beatles.",
     "expect_fn": "play_music",  "expect_args": {}, "type": "single"},

    # --- Multi action (2 calls expected) ---
    {"id": "multi_weather_alarm",  "prompt": "Check the weather in NYC and set an alarm for 9 AM.",
     "expect_fns": ["get_weather", "set_alarm"], "type": "multi"},
    {"id": "multi_msg_music",      "prompt": "Send a message to Alice saying I'm on my way, and play some jazz.",
     "expect_fns": ["send_message", "play_music"], "type": "multi"},
    {"id": "multi_timer_reminder", "prompt": "Set a 15 minute timer and remind me to check the oven at 5 PM.",
     "expect_fns": ["set_timer", "create_reminder"], "type": "multi"},

    # --- Multi action (3 calls) ---
    {"id": "multi_three",          "prompt": "What's the weather in London, set an alarm for 8 AM, and play Yesterday by the Beatles.",
     "expect_fns": ["get_weather", "set_alarm", "play_music"], "type": "multi3"},
]

# ── Tool set variations ───────────────────────────────────────────────

def tools_for(case, mode):
    """Return tool list based on mode."""
    if mode == "exact":
        # Only the tool(s) needed
        if case["type"] == "single":
            for k, v in TOOLS.items():
                if v["name"] == case["expect_fn"]:
                    return [v]
        else:
            fns = case["expect_fns"]
            return [v for v in TOOLS.values() if v["name"] in fns]
    elif mode == "three":
        # Target tool(s) + 1-2 distractors
        if case["type"] == "single":
            needed = {case["expect_fn"]}
        else:
            needed = set(case["expect_fns"])
        result = [v for v in TOOLS.values() if v["name"] in needed]
        for v in TOOLS.values():
            if v["name"] not in needed and len(result) < 3:
                result.append(v)
        return result
    elif mode == "all":
        return ALL_TOOLS
    return ALL_TOOLS


def check_result(case, parsed):
    """Check if result matches expectations. Returns (correct_fn, correct_args, n_calls)."""
    calls = parsed.get("function_calls", [])
    n_calls = len(calls)

    if case["type"] == "single":
        if n_calls == 0:
            return False, False, 0
        first = calls[0]
        fn_ok = first["name"] == case["expect_fn"]
        args_ok = all(
            str(first.get("arguments", {}).get(k, "")).lower().find(str(v).lower()) >= 0
            for k, v in case.get("expect_args", {}).items()
            if v  # skip empty expected args
        )
        return fn_ok, fn_ok and args_ok, n_calls
    else:
        expected = set(case["expect_fns"])
        got = set(c["name"] for c in calls)
        fn_ok = expected.issubset(got)
        return fn_ok, fn_ok, n_calls


def run_one(prompt, tools, force_tools=True, max_tokens=256, tool_rag_top_k=0, temperature=None):
    """Run a single cactus call. Returns (raw_str, parsed, elapsed_ms)."""
    model = cactus_init(MODEL_PATH)
    messages = [
        {"role": "system", "content": "You are a helpful assistant that can use tools."},
        {"role": "user", "content": prompt},
    ]
    cactus_tools = [{"type": "function", "function": t} for t in tools]
    params = dict(
        tools=cactus_tools,
        force_tools=force_tools,
        max_tokens=max_tokens,
        stop_sequences=["<|im_end|>", "<end_of_turn>"],
        tool_rag_top_k=tool_rag_top_k,
    )
    if temperature is not None:
        params["temperature"] = temperature

    raw_str = cactus_complete(model, messages, **params)
    cactus_destroy(model)

    try:
        parsed = json.loads(raw_str)
    except json.JSONDecodeError:
        parsed = {"function_calls": [], "confidence": 0, "total_time_ms": 0}

    return raw_str, parsed


# ── Experiments ───────────────────────────────────────────────────────

def run_experiment(name, cases, tool_mode, knobs, verbose=False):
    """Run a set of cases with given tool mode and knobs. Returns results list."""
    results = []
    for case in cases:
        tools = tools_for(case, tool_mode)
        raw_str, parsed = run_one(case["prompt"], tools, **knobs)
        fn_ok, args_ok, n_calls = check_result(case, parsed)
        calls = parsed.get("function_calls", [])

        r = {
            "id": case["id"],
            "type": case["type"],
            "tool_mode": tool_mode,
            "fn_ok": fn_ok,
            "args_ok": args_ok,
            "n_calls": n_calls,
            "confidence": parsed.get("confidence", 0),
            "time_ms": parsed.get("total_time_ms", 0),
            "cloud_handoff": parsed.get("cloud_handoff", None),
            "calls": [(c["name"], c.get("arguments", {})) for c in calls],
            "response": parsed.get("response", "")[:80],
        }
        results.append(r)

        if verbose:
            mark = "✅" if args_ok else ("⚠️" if fn_ok else "❌")
            call_str = ", ".join(f"{c['name']}({json.dumps(c.get('arguments',{}))})" for c in calls) or "(none)"
            print(f"  {mark} {case['id']:25s} → {call_str:60s}  conf={parsed.get('confidence',0):.2f}  {parsed.get('total_time_ms',0):.0f}ms")

    return results


def summarize(name, results):
    """Print summary stats for an experiment."""
    total = len(results)
    fn_correct = sum(1 for r in results if r["fn_ok"])
    args_correct = sum(1 for r in results if r["args_ok"])
    avg_conf = sum(r["confidence"] for r in results) / total if total else 0
    avg_time = sum(r["time_ms"] for r in results) / total if total else 0
    zero_calls = sum(1 for r in results if r["n_calls"] == 0)

    print(f"\n{'='*70}")
    print(f"  {name}")
    print(f"  fn correct: {fn_correct}/{total}  |  args correct: {args_correct}/{total}  |  "
          f"avg conf: {avg_conf:.3f}  |  avg time: {avg_time:.0f}ms  |  zero-call: {zero_calls}")
    print(f"{'='*70}")


if __name__ == "__main__":
    start = time.time()

    single_cases = [c for c in CASES if c["type"] == "single"]
    multi_cases = [c for c in CASES if c["type"].startswith("multi")]
    all_cases = CASES

    # ── Experiment 1: Tool set size (single actions) ──────────────────
    print("\n\n" + "█" * 70)
    print("  EXPERIMENT 1: How does # of tools affect single-action accuracy?")
    print("█" * 70)

    for tool_mode in ["exact", "three", "all"]:
        print(f"\n── tool_mode={tool_mode} ──")
        r = run_experiment(f"single/{tool_mode}", single_cases, tool_mode,
                          {"force_tools": True, "tool_rag_top_k": 0}, verbose=True)
        summarize(f"Single actions, tools={tool_mode}", r)

    # ── Experiment 2: tool_rag_top_k (all tools, single actions) ──────
    print("\n\n" + "█" * 70)
    print("  EXPERIMENT 2: tool_rag_top_k effect (all 7 tools, single actions)")
    print("█" * 70)

    for rag_k in [0, 1, 2, 3, 5]:
        print(f"\n── tool_rag_top_k={rag_k} ──")
        r = run_experiment(f"rag_k={rag_k}", single_cases, "all",
                          {"force_tools": True, "tool_rag_top_k": rag_k}, verbose=True)
        summarize(f"tool_rag_top_k={rag_k} (all tools, single)", r)

    # ── Experiment 3: force_tools on vs off ───────────────────────────
    print("\n\n" + "█" * 70)
    print("  EXPERIMENT 3: force_tools=True vs False (all tools, single actions)")
    print("█" * 70)

    for ft in [True, False]:
        print(f"\n── force_tools={ft} ──")
        r = run_experiment(f"force_tools={ft}", single_cases, "all",
                          {"force_tools": ft, "tool_rag_top_k": 0}, verbose=True)
        summarize(f"force_tools={ft}", r)

    # ── Experiment 4: Temperature (all tools, single actions) ─────────
    print("\n\n" + "█" * 70)
    print("  EXPERIMENT 4: Temperature effect (all tools, single actions)")
    print("█" * 70)

    for temp in [0.0, 0.3, 0.7, 1.0, 1.5]:
        print(f"\n── temperature={temp} ──")
        r = run_experiment(f"temp={temp}", single_cases[:6], "all",  # fewer cases to save time
                          {"force_tools": True, "tool_rag_top_k": 0, "temperature": temp}, verbose=True)
        summarize(f"temperature={temp}", r)

    # ── Experiment 5: Multi-action calls ──────────────────────────────
    print("\n\n" + "█" * 70)
    print("  EXPERIMENT 5: Multi-action calls (exact tools vs all tools)")
    print("█" * 70)

    for tool_mode in ["exact", "all"]:
        print(f"\n── tool_mode={tool_mode} ──")
        r = run_experiment(f"multi/{tool_mode}", multi_cases, tool_mode,
                          {"force_tools": True, "tool_rag_top_k": 0}, verbose=True)
        summarize(f"Multi-action, tools={tool_mode}", r)

    # ── Experiment 6: max_tokens effect on multi-action ───────────────
    print("\n\n" + "█" * 70)
    print("  EXPERIMENT 6: max_tokens effect on multi-action (all tools)")
    print("█" * 70)

    for mt in [64, 128, 256, 512]:
        print(f"\n── max_tokens={mt} ──")
        r = run_experiment(f"max_tokens={mt}", multi_cases, "all",
                          {"force_tools": True, "tool_rag_top_k": 0, "max_tokens": mt}, verbose=True)
        summarize(f"max_tokens={mt} (multi-action)", r)

    # ── Experiment 7: tool_rag_top_k on multi-action ──────────────────
    print("\n\n" + "█" * 70)
    print("  EXPERIMENT 7: tool_rag_top_k on multi-action (all tools)")
    print("█" * 70)

    for rag_k in [0, 2, 3, 5]:
        print(f"\n── tool_rag_top_k={rag_k} ──")
        r = run_experiment(f"multi_rag_k={rag_k}", multi_cases, "all",
                          {"force_tools": True, "tool_rag_top_k": rag_k}, verbose=True)
        summarize(f"multi tool_rag_top_k={rag_k}", r)

    elapsed = time.time() - start
    print(f"\n\nTotal time: {elapsed:.1f}s ({len(CASES)} cases × multiple experiments)")
