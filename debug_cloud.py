import sys, json, os
sys.path.insert(0, "cactus/python/src")
from main import generate_cloud

# message_among_four: "Text Dave saying I'll be late"
messages = [{"role": "user", "content": "Text Dave saying I'll be late."}]
tools = [
    {"name": "get_weather",    "description": "Get current weather for a location",
     "parameters": {"type": "object", "properties": {"location": {"type": "string", "description": "City name"}}, "required": ["location"]}},
    {"name": "set_timer",      "description": "Set a countdown timer",
     "parameters": {"type": "object", "properties": {"minutes": {"type": "integer", "description": "Number of minutes"}}, "required": ["minutes"]}},
    {"name": "send_message",   "description": "Send a message to a contact",
     "parameters": {"type": "object", "properties": {"recipient": {"type": "string", "description": "Name of person"}, "message": {"type": "string", "description": "Message content"}}, "required": ["recipient", "message"]}},
    {"name": "play_music",     "description": "Play a song or playlist",
     "parameters": {"type": "object", "properties": {"song": {"type": "string", "description": "Song name"}}, "required": ["song"]}},
]

for i in range(3):
    result = generate_cloud(messages, tools)
    print(f"Run {i+1}: {json.dumps(result['function_calls'])}")

print("\nExpected: send_message(recipient='Dave', message=\"I'll be late\")")
