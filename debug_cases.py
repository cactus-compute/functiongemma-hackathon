import sys, json
sys.path.insert(0, "cactus/python/src")
from main import generate_cactus, _preflight, _validate

ALARM_TOOL = {
    "name": "set_alarm",
    "description": "Set an alarm for a given time",
    "parameters": {
        "type": "object",
        "properties": {
            "hour":   {"type": "integer", "description": "Hour to set the alarm for"},
            "minute": {"type": "integer", "description": "Minute to set the alarm for"},
        },
        "required": ["hour", "minute"],
    },
}
REMINDER_TOOL = {
    "name": "create_reminder",
    "description": "Create a reminder with a title and time",
    "parameters": {
        "type": "object",
        "properties": {
            "title": {"type": "string", "description": "Reminder title"},
            "time":  {"type": "string", "description": "Time for the reminder (e.g. 3:00 PM)"},
        },
        "required": ["title", "time"],
    },
}

cases = [
    ("alarm_10am",      [{"role": "user", "content": "Set an alarm for 10 AM."}],          [ALARM_TOOL]),
    ("alarm_9am",       [{"role": "user", "content": "Set an alarm for 9 AM."}],            [ALARM_TOOL]),
    ("alarm_6am",       [{"role": "user", "content": "Wake me up at 6 AM."}],               [ALARM_TOOL]),
    ("reminder_meeting",[{"role": "user", "content": "Remind me about the meeting at 3:00 PM."}], [REMINDER_TOOL]),
    ("timer_7min",      [{"role": "user", "content": "Set a timer for 7 minutes."}],
     [{"name": "set_timer", "description": "Set a countdown timer",
       "parameters": {"type": "object", "properties": {"minutes": {"type": "integer", "description": "Number of minutes"}}, "required": ["minutes"]}}]),
]

for name, messages, tools in cases:
    result = generate_cactus(messages, tools)
    complexity = _preflight(messages, tools)
    valid, reason = _validate(result, tools, complexity, messages)
    print(f"{name}: calls={json.dumps(result['function_calls'])}  conf={result['confidence']:.3f}  valid={valid}  reason={reason}")
