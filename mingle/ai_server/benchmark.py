
import sys, os
import pathlib

_REPO_ROOT = str(pathlib.Path(__file__).resolve().parents[2])
sys.path.insert(0, os.path.join(_REPO_ROOT, "cactus/python/src"))
os.environ["CACTUS_NO_CLOUD_TELE"] = "1"

import json
from main import generate_hybrid


############## Tool definitions ##############

TOOL_GET_WEATHER = {
    "name": "get_weather",
    "description": "Get current weather for a location",
    "parameters": {
        "type": "object",
        "properties": {
            "location": {"type": "string", "description": "City name"}
        },
        "required": ["location"],
    },
}

TOOL_SET_ALARM = {
    "name": "set_alarm",
    "description": "Set an alarm for a given time",
    "parameters": {
        "type": "object",
        "properties": {
            "hour": {"type": "integer", "description": "Hour to set the alarm for"},
            "minute": {"type": "integer", "description": "Minute to set the alarm for"},
        },
        "required": ["hour", "minute"],
    },
}

TOOL_SEND_MESSAGE = {
    "name": "send_message",
    "description": "Send a message to a contact",
    "parameters": {
        "type": "object",
        "properties": {
            "recipient": {"type": "string", "description": "Name of the person to send the message to"},
            "message": {"type": "string", "description": "The message content to send"},
        },
        "required": ["recipient", "message"],
    },
}

TOOL_CREATE_REMINDER = {
    "name": "create_reminder",
    "description": "Create a reminder with a title and time",
    "parameters": {
        "type": "object",
        "properties": {
            "title": {"type": "string", "description": "Reminder title"},
            "time": {"type": "string", "description": "Time for the reminder (e.g. 3:00 PM)"},
        },
        "required": ["title", "time"],
    },
}

TOOL_SEARCH_CONTACTS = {
    "name": "search_contacts",
    "description": "Search for a contact by name",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Name to search for"},
        },
        "required": ["query"],
    },
}

TOOL_PLAY_MUSIC = {
    "name": "play_music",
    "description": "Play a song or playlist",
    "parameters": {
        "type": "object",
        "properties": {
            "song": {"type": "string", "description": "Song or playlist name"},
        },
        "required": ["song"],
    },
}

TOOL_SET_TIMER = {
    "name": "set_timer",
    "description": "Set a countdown timer",
    "parameters": {
        "type": "object",
        "properties": {
            "minutes": {"type": "integer", "description": "Number of minutes"},
        },
        "required": ["minutes"],
    },
}


############## Benchmark cases ##############

BENCHMARKS = [
    # ===== Easy: 1 tool, direct request =====
    {
        "name": "weather_sf",
        "difficulty": "easy",
        "messages": [{"role": "user", "content": "What is the weather in San Francisco?"}],
        "tools": [TOOL_GET_WEATHER],
        "expected_calls": [{"name": "get_weather", "arguments": {"location": "San Francisco"}}],
    },
    {
        "name": "alarm_10am",
        "difficulty": "easy",
        "messages": [{"role": "user", "content": "Set an alarm for 10 AM."}],
        "tools": [TOOL_SET_ALARM],
        "expected_calls": [{"name": "set_alarm", "arguments": {"hour": 10, "minute": 0}}],
    },
    {
        "name": "message_alice",
        "difficulty": "easy",
        "messages": [{"role": "user", "content": "Send a message to Alice saying good morning."}],
        "tools": [TOOL_SEND_MESSAGE],
        "expected_calls": [{"name": "send_message", "arguments": {"recipient": "Alice", "message": "good morning"}}],
    },
    {
        "name": "weather_london",
        "difficulty": "easy",
        "messages": [{"role": "user", "content": "What's the weather like in London?"}],
        "tools": [TOOL_GET_WEATHER],
        "expected_calls": [{"name": "get_weather", "arguments": {"location": "London"}}],
    },
    {
        "name": "alarm_6am",
        "difficulty": "easy",
        "messages": [{"role": "user", "content": "Wake me up at 6 AM."}],
        "tools": [TOOL_SET_ALARM],
        "expected_calls": [{"name": "set_alarm", "arguments": {"hour": 6, "minute": 0}}],
    },
    {
        "name": "play_bohemian",
        "difficulty": "easy",
        "messages": [{"role": "user", "content": "Play Bohemian Rhapsody."}],
        "tools": [TOOL_PLAY_MUSIC],
        "expected_calls": [{"name": "play_music", "arguments": {"song": "Bohemian Rhapsody"}}],
    },
    {
        "name": "timer_5min",
        "difficulty": "easy",
        "messages": [{"role": "user", "content": "Set a timer for 5 minutes."}],
        "tools": [TOOL_SET_TIMER],
        "expected_calls": [{"name": "set_timer", "arguments": {"minutes": 5}}],
    },
    {
        "name": "reminder_meeting",
        "difficulty": "easy",
        "messages": [{"role": "user", "content": "Remind me about the meeting at 3:00 PM."}],
        "tools": [TOOL_CREATE_REMINDER],
        "expected_calls": [{"name": "create_reminder", "arguments": {"title": "meeting", "time": "3:00 PM"}}],
    },
    {
        "name": "search_bob",
        "difficulty": "easy",
        "messages": [{"role": "user", "content": "Find Bob in my contacts."}],
        "tools": [TOOL_SEARCH_CONTACTS],
        "expected_calls": [{"name": "search_contacts", "arguments": {"query": "Bob"}}],
    },
    {
        "name": "weather_paris",
        "difficulty": "easy",
        "messages": [{"role": "user", "content": "How's the weather in Paris?"}],
        "tools": [TOOL_GET_WEATHER],
        "expected_calls": [{"name": "get_weather", "arguments": {"location": "Paris"}}],
    },

    # ===== Medium: 2-3 tools, must pick the right one =====
    {
        "name": "message_among_three",
        "difficulty": "medium",
        "messages": [{"role": "user", "content": "Send a message to John saying hello."}],
        "tools": [TOOL_GET_WEATHER, TOOL_SEND_MESSAGE, TOOL_SET_ALARM],
        "expected_calls": [{"name": "send_message", "arguments": {"recipient": "John", "message": "hello"}}],
    },
    {
        "name": "weather_among_two",
        "difficulty": "medium",
        "messages": [{"role": "user", "content": "What's the weather in Tokyo?"}],
        "tools": [TOOL_GET_WEATHER, TOOL_SEND_MESSAGE],
        "expected_calls": [{"name": "get_weather", "arguments": {"location": "Tokyo"}}],
    },
    {
        "name": "alarm_among_three",
        "difficulty": "medium",
        "messages": [{"role": "user", "content": "Set an alarm for 8:15 AM."}],
        "tools": [TOOL_SEND_MESSAGE, TOOL_SET_ALARM, TOOL_GET_WEATHER],
        "expected_calls": [{"name": "set_alarm", "arguments": {"hour": 8, "minute": 15}}],
    },
    {
        "name": "music_among_three",
        "difficulty": "medium",
        "messages": [{"role": "user", "content": "Play some jazz music."}],
        "tools": [TOOL_SET_ALARM, TOOL_PLAY_MUSIC, TOOL_GET_WEATHER],
        "expected_calls": [{"name": "play_music", "arguments": {"song": "jazz"}}],
    },
    {
        "name": "reminder_among_four",
        "difficulty": "medium",
        "messages": [{"role": "user", "content": "Remind me to call the dentist at 2:00 PM."}],
        "tools": [TOOL_GET_WEATHER, TOOL_SEND_MESSAGE, TOOL_CREATE_REMINDER, TOOL_SET_ALARM],
        "expected_calls": [{"name": "create_reminder", "arguments": {"title": "call the dentist", "time": "2:00 PM"}}],
    },
    {
        "name": "timer_among_three",
        "difficulty": "medium",
        "messages": [{"role": "user", "content": "Set a timer for 10 minutes."}],
        "tools": [TOOL_SET_ALARM, TOOL_SET_TIMER, TOOL_PLAY_MUSIC],
        "expected_calls": [{"name": "set_timer", "arguments": {"minutes": 10}}],
    },
    {
        "name": "search_among_four",
        "difficulty": "medium",
        "messages": [{"role": "user", "content": "Look up Sarah in my contacts."}],
        "tools": [TOOL_SEND_MESSAGE, TOOL_GET_WEATHER, TOOL_SEARCH_CONTACTS, TOOL_SET_ALARM],
        "expected_calls": [{"name": "search_contacts", "arguments": {"query": "Sarah"}}],
    },
    {
        "name": "weather_among_four",
        "difficulty": "medium",
        "messages": [{"role": "user", "content": "What's the weather in Berlin?"}],
        "tools": [TOOL_SEND_MESSAGE, TOOL_SET_ALARM, TOOL_PLAY_MUSIC, TOOL_GET_WEATHER],
        "expected_calls": [{"name": "get_weather", "arguments": {"location": "Berlin"}}],
    },
    {
        "name": "message_among_four",
        "difficulty": "medium",
        "messages": [{"role": "user", "content": "Text Dave saying I'll be late."}],
        "tools": [TOOL_GET_WEATHER, TOOL_SET_TIMER, TOOL_SEND_MESSAGE, TOOL_PLAY_MUSIC],
        "expected_calls": [{"name": "send_message", "arguments": {"recipient": "Dave", "message": "I'll be late"}}],
    },
    {
        "name": "alarm_among_five",
        "difficulty": "medium",
        "messages": [{"role": "user", "content": "Set an alarm for 9 AM."}],
        "tools": [TOOL_SEND_MESSAGE, TOOL_GET_WEATHER, TOOL_PLAY_MUSIC, TOOL_SET_TIMER, TOOL_SET_ALARM],
        "expected_calls": [{"name": "set_alarm", "arguments": {"hour": 9, "minute": 0}}],
    },

    # ===== Hard: multiple tools needed, multi-call =====
    {
        "name": "message_and_weather",
        "difficulty": "hard",
        "messages": [{"role": "user", "content": "Send a message to Bob saying hi and get the weather in London."}],
        "tools": [TOOL_GET_WEATHER, TOOL_SEND_MESSAGE, TOOL_SET_ALARM],
        "expected_calls": [
            {"name": "send_message", "arguments": {"recipient": "Bob", "message": "hi"}},
            {"name": "get_weather", "arguments": {"location": "London"}},
        ],
    },
    {
        "name": "alarm_and_weather",
        "difficulty": "hard",
        "messages": [{"role": "user", "content": "Set an alarm for 7:30 AM and check the weather in New York."}],
        "tools": [TOOL_GET_WEATHER, TOOL_SET_ALARM, TOOL_SEND_MESSAGE],
        "expected_calls": [
            {"name": "set_alarm", "arguments": {"hour": 7, "minute": 30}},
            {"name": "get_weather", "arguments": {"location": "New York"}},
        ],
    },
    {
        "name": "timer_and_music",
        "difficulty": "hard",
        "messages": [{"role": "user", "content": "Set a timer for 20 minutes and play lo-fi beats."}],
        "tools": [TOOL_SET_TIMER, TOOL_PLAY_MUSIC, TOOL_GET_WEATHER, TOOL_SET_ALARM],
        "expected_calls": [
            {"name": "set_timer", "arguments": {"minutes": 20}},
            {"name": "play_music", "arguments": {"song": "lo-fi beats"}},
        ],
    },
    {
        "name": "reminder_and_message",
        "difficulty": "hard",
        "messages": [{"role": "user", "content": "Remind me about groceries at 5:00 PM and text Lisa saying see you tonight."}],
        "tools": [TOOL_CREATE_REMINDER, TOOL_SEND_MESSAGE, TOOL_GET_WEATHER, TOOL_SET_ALARM],
        "expected_calls": [
            {"name": "create_reminder", "arguments": {"title": "groceries", "time": "5:00 PM"}},
            {"name": "send_message", "arguments": {"recipient": "Lisa", "message": "see you tonight"}},
        ],
    },
    {
        "name": "search_and_message",
        "difficulty": "hard",
        "messages": [{"role": "user", "content": "Find Tom in my contacts and send him a message saying happy birthday."}],
        "tools": [TOOL_SEARCH_CONTACTS, TOOL_SEND_MESSAGE, TOOL_GET_WEATHER, TOOL_PLAY_MUSIC],
        "expected_calls": [
            {"name": "search_contacts", "arguments": {"query": "Tom"}},
            {"name": "send_message", "arguments": {"recipient": "Tom", "message": "happy birthday"}},
        ],
    },
    {
        "name": "alarm_and_reminder",
        "difficulty": "hard",
        "messages": [{"role": "user", "content": "Set an alarm for 6:45 AM and remind me to take medicine at 7:00 AM."}],
        "tools": [TOOL_SET_ALARM, TOOL_CREATE_REMINDER, TOOL_SEND_MESSAGE, TOOL_PLAY_MUSIC],
        "expected_calls": [
            {"name": "set_alarm", "arguments": {"hour": 6, "minute": 45}},
            {"name": "create_reminder", "arguments": {"title": "take medicine", "time": "7:00 AM"}},
        ],
    },
    {
        "name": "weather_and_music",
        "difficulty": "hard",
        "messages": [{"role": "user", "content": "Check the weather in Miami and play summer hits."}],
        "tools": [TOOL_GET_WEATHER, TOOL_PLAY_MUSIC, TOOL_SET_TIMER, TOOL_SEND_MESSAGE],
        "expected_calls": [
            {"name": "get_weather", "arguments": {"location": "Miami"}},
            {"name": "play_music", "arguments": {"song": "summer hits"}},
        ],
    },
    {
        "name": "message_weather_alarm",
        "difficulty": "hard",
        "messages": [{"role": "user", "content": "Text Emma saying good night, check the weather in Chicago, and set an alarm for 5 AM."}],
        "tools": [TOOL_SEND_MESSAGE, TOOL_GET_WEATHER, TOOL_SET_ALARM, TOOL_PLAY_MUSIC, TOOL_SET_TIMER],
        "expected_calls": [
            {"name": "send_message", "arguments": {"recipient": "Emma", "message": "good night"}},
            {"name": "get_weather", "arguments": {"location": "Chicago"}},
            {"name": "set_alarm", "arguments": {"hour": 5, "minute": 0}},
        ],
    },
    {
        "name": "timer_music_reminder",
        "difficulty": "hard",
        "messages": [{"role": "user", "content": "Set a 15 minute timer, play classical music, and remind me to stretch at 4:00 PM."}],
        "tools": [TOOL_SET_TIMER, TOOL_PLAY_MUSIC, TOOL_CREATE_REMINDER, TOOL_GET_WEATHER, TOOL_SEND_MESSAGE],
        "expected_calls": [
            {"name": "set_timer", "arguments": {"minutes": 15}},
            {"name": "play_music", "arguments": {"song": "classical music"}},
            {"name": "create_reminder", "arguments": {"title": "stretch", "time": "4:00 PM"}},
        ],
    },
    {
        "name": "search_message_weather",
        "difficulty": "hard",
        "messages": [{"role": "user", "content": "Look up Jake in my contacts, send him a message saying let's meet, and check the weather in Seattle."}],
        "tools": [TOOL_SEARCH_CONTACTS, TOOL_SEND_MESSAGE, TOOL_GET_WEATHER, TOOL_SET_ALARM, TOOL_PLAY_MUSIC],
        "expected_calls": [
            {"name": "search_contacts", "arguments": {"query": "Jake"}},
            {"name": "send_message", "arguments": {"recipient": "Jake", "message": "let's meet"}},
            {"name": "get_weather", "arguments": {"location": "Seattle"}},
        ],
    },
]


def _normalize(v):
    """Normalize a value for comparison."""
    if isinstance(v, str):
        return v.strip().lower()
    return v


def _call_matches(predicted, expected):
    """Check if a predicted call matches an expected call (name + argument values)."""
    if predicted["name"] != expected["name"]:
        return False
    pred_args = predicted.get("arguments", {})
    exp_args = expected.get("arguments", {})
    for key, exp_val in exp_args.items():
        if key not in pred_args:
            return False
        if _normalize(pred_args[key]) != _normalize(exp_val):
            return False
    return True


def compute_f1(predicted_calls, expected_calls):
    """Compute F1 score between predicted and expected function calls."""
    if not predicted_calls and not expected_calls:
        return 1.0
    if not predicted_calls or not expected_calls:
        return 0.0

    matched = 0
    used = set()
    for exp in expected_calls:
        for i, pred in enumerate(predicted_calls):
            if i not in used and _call_matches(pred, exp):
                matched += 1
                used.add(i)
                break

    precision = matched / len(predicted_calls)
    recall = matched / len(expected_calls)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def run_benchmark(benchmarks=None):
    """Run all benchmark cases and print results."""
    if benchmarks is None:
        benchmarks = BENCHMARKS

    total = len(benchmarks)
    results = []
    for i, case in enumerate(benchmarks, 1):
        print(f"[{i}/{total}] Running: {case['name']} ({case['difficulty']})...", end=" ", flush=True)
        result = generate_hybrid(case["messages"], case["tools"])
        f1 = compute_f1(result["function_calls"], case["expected_calls"])
        source = result.get("source", "unknown")
        print(f"F1={f1:.2f} | {result['total_time_ms']:.0f}ms | {source}")
        results.append({
            "name": case["name"],
            "difficulty": case["difficulty"],
            "total_time_ms": result["total_time_ms"],
            "f1": f1,
            "source": source,
            "predicted": result["function_calls"],
            "expected": case["expected_calls"],
        })

    print("\n=== Benchmark Results ===\n")
    print(f"  {'#':>2} | {'Difficulty':<10} | {'Name':<28} | {'Time (ms)':>10} | {'F1':>5} | Source")
    print(f"  {'--':>2}-+-{'-'*10}-+-{'-'*28}-+-{'-'*10}-+-{'-'*5}-+-{'-'*20}")
    for i, r in enumerate(results, 1):
        print(f"  {i:>2} | {r['difficulty']:<10} | {r['name']:<28} | {r['total_time_ms']:>10.2f} | {r['f1']:>5.2f} | {r['source']}")

    print(f"\n--- Summary ---")
    for difficulty in ["easy", "medium", "hard"]:
        group = [r for r in results if r["difficulty"] == difficulty]
        if not group:
            continue
        avg_f1 = sum(r["f1"] for r in group) / len(group)
        avg_time = sum(r["total_time_ms"] for r in group) / len(group)
        on_device = sum(1 for r in group if r["source"] == "on-device")
        cloud = len(group) - on_device
        print(f"  {difficulty:<8} avg F1={avg_f1:.2f}  avg time={avg_time:.2f}ms  on-device={on_device}/{len(group)} cloud={cloud}/{len(group)}")

    avg_f1 = sum(r["f1"] for r in results) / len(results)
    avg_time = sum(r["total_time_ms"] for r in results) / len(results)
    total_time = sum(r["total_time_ms"] for r in results)
    on_device_total = sum(1 for r in results if r["source"] == "on-device")
    cloud_total = len(results) - on_device_total
    print(f"  {'overall':<8} avg F1={avg_f1:.2f}  avg time={avg_time:.2f}ms  total time={total_time:.2f}ms")
    print(f"           on-device={on_device_total}/{len(results)} ({100*on_device_total/len(results):.0f}%)  cloud={cloud_total}/{len(results)} ({100*cloud_total/len(results):.0f}%)")

    # Total score
    score = compute_total_score(results)
    print(f"\n{'='*50}")
    print(f"  TOTAL SCORE: {score:.1f}%")
    print(f"{'='*50}")

    return results


def compute_total_score(results):
    """
    Compute a total score from 0-100% as a weighted sum across difficulty levels.

    Components (per difficulty level):
      - F1 score (50%): accuracy of tool calls
      - Time score (25%): faster is better, capped at 500ms baseline
      - On-device ratio (25%): higher on-device usage is better

    Difficulty weights:
      - easy: 20%
      - medium: 30%
      - hard: 50%
    """
    difficulty_weights = {"easy": 0.20, "medium": 0.30, "hard": 0.50}
    time_baseline_ms = 500  # anything under this gets full marks

    total_score = 0
    for difficulty, weight in difficulty_weights.items():
        group = [r for r in results if r["difficulty"] == difficulty]
        if not group:
            continue

        avg_f1 = sum(r["f1"] for r in group) / len(group)
        avg_time = sum(r["total_time_ms"] for r in group) / len(group)
        on_device_ratio = sum(1 for r in group if r["source"] == "on-device") / len(group)

        time_score = max(0, 1 - avg_time / time_baseline_ms)

        level_score = (0.60 * avg_f1) + (0.15 * time_score) + (0.25 * on_device_ratio)
        total_score += weight * level_score

    return total_score * 100


if __name__ == "__main__":
    run_benchmark()
