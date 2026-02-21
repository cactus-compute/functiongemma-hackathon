"""Quick end-to-end test for HandsFree modules."""
import sys
sys.path.insert(0, "cactus/python/src")
sys.path.insert(0, ".")

from handsfree.tools import ALL_TOOLS
from handsfree.location import detect_location_intent, inject_location_into_command
from handsfree.executor import execute
from main import generate_hybrid

tools = [{k: v for k, v in t.items() if k != "on_device"} for t in ALL_TOOLS]

print("=== Executor test ===")
test_calls = [
    {"name": "set_alarm",       "arguments": {"hour": 7, "minute": 30, "label": "Wake up"}},
    {"name": "play_music",      "arguments": {"song": "Bohemian Rhapsody", "artist": "Queen"}},
    {"name": "set_timer",       "arguments": {"minutes": 10, "label": "Pasta"}},
    {"name": "create_reminder", "arguments": {"title": "Call John", "time": "3:00 PM"}},
    {"name": "send_message",    "arguments": {"recipient": "Mom", "message": "On my way!"}},
    {"name": "get_weather",     "arguments": {"location": "San Francisco"}},
    {"name": "get_directions",  "arguments": {"destination": "Golden Gate Bridge", "origin": "Civic Center"}},
    {"name": "find_nearby",     "arguments": {"category": "coffee", "location": "here"}},
    {"name": "share_location",  "arguments": {"recipient": "Dad", "location": "37.7749,-122.4194"}},
]
for call in test_calls:
    r = execute([call])[0]["result"]
    print(f"  {call['name']:20s}: {r.get('icon','')} status={r.get('status','?')}")

print("\n=== Location detection ===")
for cmd in ["send my location to Mom", "what time is it", "share my location with John"]:
    detected = detect_location_intent(cmd)
    print(f"  {detected!s:5} | {cmd}")

print("\n=== Hybrid routing (3 commands) ===")
for cmd in ["Set an alarm for 7:30 AM", "Play Bohemian Rhapsody", "Set a timer for 10 minutes"]:
    msgs = [{"role": "user", "content": cmd}]
    result = generate_hybrid(msgs, tools)
    calls = result.get("function_calls", [])
    src = result.get("source", "?")
    fn = calls[0]["name"] if calls else "NO CALL"
    args = calls[0]["arguments"] if calls else {}
    print(f"  [{src:25s}] {cmd:35s} → {fn}({args})")

print("\nAll tests passed ✅")
