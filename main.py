
import sys
sys.path.insert(0, "cactus/python/src")
functiongemma_path = "cactus/weights/functiongemma-270m-it"

import json, os, time
from cactus import cactus_init, cactus_complete, cactus_destroy
from google import genai
from google.genai import types


def generate_cactus(messages, tools):
    """Run function calling on-device via FunctionGemma + Cactus."""
    model = cactus_init(functiongemma_path)

    cactus_tools = [{
        "type": "function",
        "function": t,
    } for t in tools]

    raw_str = cactus_complete(
        model,
        [{"role": "system", "content": "You are a helpful assistant that can use tools."}] + messages,
        tools=cactus_tools,
        force_tools=True,
        max_tokens=256,
        stop_sequences=["<|im_end|>", "<end_of_turn>"],
    )

    cactus_destroy(model)

    try:
        raw = json.loads(raw_str)
    except json.JSONDecodeError:
        return {
            "function_calls": [],
            "total_time_ms": 0,
            "confidence": 0,
        }

    return {
        "function_calls": raw.get("function_calls", []),
        "total_time_ms": raw.get("total_time_ms", 0),
        "confidence": raw.get("confidence", 0),
    }


def generate_cloud(messages, tools):
    """Run function calling via Gemini Cloud API."""
    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

    gemini_tools = [
        types.Tool(function_declarations=[
            types.FunctionDeclaration(
                name=t["name"],
                description=t["description"],
                parameters=types.Schema(
                    type="OBJECT",
                    properties={
                        k: types.Schema(type=v["type"].upper(), description=v.get("description", ""))
                        for k, v in t["parameters"]["properties"].items()
                    },
                    required=t["parameters"].get("required", []),
                ),
            )
            for t in tools
        ])
    ]

    contents = [m["content"] for m in messages if m["role"] == "user"]

    start_time = time.time()

    gemini_response = client.models.generate_content(
        model="gemini-2.5-flash", #gemini-1.5-flash-8b
        contents=contents,
        config=types.GenerateContentConfig(tools=gemini_tools),
    )

    total_time_ms = (time.time() - start_time) * 1000

    function_calls = []
    for candidate in gemini_response.candidates:
        for part in candidate.content.parts:
            if part.function_call:
                function_calls.append({
                    "name": part.function_call.name,
                    "arguments": dict(part.function_call.args),
                })

    return {
        "function_calls": function_calls,
        "total_time_ms": total_time_ms,
    }


def generate_hybrid(messages, tools, confidence_threshold=0.99):
    """
    Hybrid strategy:
      1) Run on-device first.
      2) Validate local tool calls against tool schema + user intent.
      3) Use dynamic confidence thresholds by complexity.
      4) Fall back to cloud only when local output looks unreliable.
    """
    import re

    user_text = " ".join(m.get("content", "") for m in messages if m.get("role") == "user")
    user_text_l = user_text.lower()
    tool_map = {t["name"]: t for t in tools}
    available = set(tool_map.keys())

    def _normalize_text(s):
        return " ".join(str(s).strip().lower().split())

    def _extract_intents(text_l, available_tools):
        intent_patterns = {
            "get_weather": [r"\bweather\b", r"\bforecast\b", r"\btemperature\b"],
            "set_alarm": [r"\balarm\b", r"\bwake me up\b"],
            "send_message": [r"\bsend\b", r"\btext\b", r"\bmessage\b"],
            "create_reminder": [r"\bremind\b", r"\breminder\b"],
            "search_contacts": [r"\bcontacts\b", r"\blook up\b", r"\bfind\b", r"\bsearch\b"],
            "play_music": [r"\bplay\b", r"\bmusic\b", r"\bsong\b", r"\bplaylist\b"],
            "set_timer": [r"\btimer\b", r"\bcountdown\b"],
        }

        intents = set()
        for tool_name, patterns in intent_patterns.items():
            if tool_name not in available_tools:
                continue
            if any(re.search(p, text_l) for p in patterns):
                intents.add(tool_name)

        # Resolve "find X in my contacts and send message" ambiguity:
        # keep both when explicitly requested; otherwise bias to contact search for pure lookup phrasing.
        if "search_contacts" in intents and "send_message" in intents:
            has_send_action = bool(re.search(r"\b(send|text)\b", text_l))
            if not has_send_action:
                intents.discard("send_message")

        return intents

    def _extract_alarm_time(text):
        m = re.search(r"(?:for|at)\s*(\d{1,2})(?::(\d{2}))?\s*(am|pm)\b", text, re.I)
        if not m:
            return None
        hour = int(m.group(1))
        minute = int(m.group(2) or 0)
        mer = m.group(3).lower()
        if mer == "pm" and hour != 12:
            hour += 12
        if mer == "am" and hour == 12:
            hour = 0
        return {"hour": hour, "minute": minute}

    def _extract_timer_minutes(text):
        m = re.search(r"(\d+)\s*(?:minute|min)\b", text, re.I)
        if not m:
            return None
        return int(m.group(1))

    def _extract_weather_location(text):
        m = re.search(r"weather(?:\s+like)?\s+in\s+([A-Za-z][A-Za-z\s\-']+?)(?:[\.\,\?\!]|\s+and\b|$)", text, re.I)
        if not m:
            return None
        return m.group(1).strip()

    def _extract_search_query(text):
        patterns = [
            r"(?:find|look up|search for)\s+([A-Za-z][A-Za-z\s\-']+?)\s+(?:in|from)\s+my\s+contacts",
            r"(?:find|look up|search for)\s+([A-Za-z][A-Za-z\s\-']+?)\b",
        ]
        for p in patterns:
            m = re.search(p, text, re.I)
            if m:
                return m.group(1).strip()
        return None

    def _extract_message_fields(text):
        patterns = [
            r"(?:send|text)\s+(?:a\s+message\s+to\s+)?([A-Za-z][A-Za-z\s\-']+?)\s+saying\s+(.+?)(?:[\.\!\?]|$)",
            r"text\s+([A-Za-z][A-Za-z\s\-']+?)\s+saying\s+(.+?)(?:[\.\!\?]|$)",
            r"send\s+(?:him|her|them)\s+a\s+message\s+saying\s+(.+?)(?:[\.\!\?]|$)",
        ]
        for p in patterns:
            m = re.search(p, text, re.I)
            if not m:
                continue
            if len(m.groups()) == 2:
                return {"recipient": m.group(1).strip(), "message": m.group(2).strip().strip("'\"")}
            if len(m.groups()) == 1:
                return {"message": m.group(1).strip().strip("'\"")}
        return None

    def _extract_reminder_fields(text):
        m = re.search(
            r"remind me(?:\s+to|\s+about)?\s+(.+?)\s+at\s+(\d{1,2}:\d{2}\s*(?:AM|PM|am|pm))",
            text,
            re.I,
        )
        if not m:
            return None
        return {"title": m.group(1).strip(), "time": m.group(2).upper()}

    def _extract_music_song(text):
        m = re.search(r"\bplay\s+(.+?)(?:[\.\,\?\!]|\s+and\b|$)", text, re.I)
        if not m:
            return None
        song = m.group(1).strip()
        song = re.sub(r"\bmusic\b", "", song, flags=re.I).strip()
        song = re.sub(r"\bsome\b", "", song, flags=re.I).strip()
        return song or None

    def _schema_valid(call):
        name = call.get("name")
        args = call.get("arguments", {})
        if name not in tool_map:
            return False
        required = tool_map[name].get("parameters", {}).get("required", [])
        props = tool_map[name].get("parameters", {}).get("properties", {})
        for key in required:
            if key not in args:
                return False
        for k, v in args.items():
            if k not in props:
                continue
            ptype = props[k].get("type", "").lower()
            if ptype == "integer" and not isinstance(v, int):
                return False
            if ptype == "string" and not isinstance(v, str):
                return False
        return True

    def _semantic_valid(calls, text, intents):
        if not calls:
            return False

        call_names = [c.get("name") for c in calls]
        call_set = set(call_names)

        if any(n not in available for n in call_set):
            return False

        # Expected intents should be covered when detected.
        if intents and not intents.issubset(call_set):
            return False

        # Multi-intent commands should produce at least as many calls as intents.
        if len(intents) >= 2 and len(calls) < len(intents):
            return False

        # Tool-specific slot checks based on user utterance.
        expected_alarm = _extract_alarm_time(text)
        expected_timer = _extract_timer_minutes(text)
        expected_loc = _extract_weather_location(text)
        expected_query = _extract_search_query(text)
        expected_msg = _extract_message_fields(text)
        expected_rem = _extract_reminder_fields(text)
        expected_song = _extract_music_song(text)

        for c in calls:
            name = c.get("name")
            args = c.get("arguments", {})

            if name == "set_alarm":
                if not (0 <= int(args.get("hour", -1)) <= 23 and 0 <= int(args.get("minute", -1)) <= 59):
                    return False
                if expected_alarm:
                    if args.get("hour") != expected_alarm["hour"] or args.get("minute") != expected_alarm["minute"]:
                        return False

            if name == "set_timer":
                mins = args.get("minutes")
                if not isinstance(mins, int) or mins <= 0:
                    return False
                if expected_timer is not None and mins != expected_timer:
                    return False

            if name == "get_weather":
                loc = args.get("location")
                if not isinstance(loc, str) or not loc.strip():
                    return False
                if expected_loc and _normalize_text(loc) != _normalize_text(expected_loc):
                    return False

            if name == "search_contacts":
                q = args.get("query")
                if not isinstance(q, str) or not q.strip():
                    return False
                if expected_query and _normalize_text(q) != _normalize_text(expected_query):
                    return False

            if name == "send_message":
                rec = args.get("recipient")
                msg = args.get("message")
                if not (isinstance(rec, str) and rec.strip() and isinstance(msg, str) and msg.strip()):
                    return False
                if expected_msg:
                    if "recipient" in expected_msg and _normalize_text(rec) != _normalize_text(expected_msg["recipient"]):
                        return False
                    if "message" in expected_msg and _normalize_text(msg) != _normalize_text(expected_msg["message"]):
                        return False

            if name == "create_reminder":
                title = args.get("title")
                tval = args.get("time")
                if not (isinstance(title, str) and title.strip() and isinstance(tval, str) and tval.strip()):
                    return False
                if expected_rem:
                    if _normalize_text(title) != _normalize_text(expected_rem["title"]):
                        return False
                    if _normalize_text(tval) != _normalize_text(expected_rem["time"]):
                        return False

            if name == "play_music":
                song = args.get("song")
                if not isinstance(song, str) or not song.strip():
                    return False
                if expected_song and _normalize_text(song) != _normalize_text(expected_song):
                    return False

        return True

    intents = _extract_intents(user_text_l, available)
    local = generate_cactus(messages, tools)
    local_calls = local.get("function_calls", [])
    local_conf = local.get("confidence", 0.0)

    schema_ok = bool(local_calls) and all(_schema_valid(c) for c in local_calls)
    semantic_ok = schema_ok and _semantic_valid(local_calls, user_text, intents)

    # Dynamic threshold: higher bar for more complex/multi-intent prompts.
    base_thr = confidence_threshold
    if len(intents) <= 1:
        dyn_thr = min(base_thr, 0.80)
    elif len(intents) == 2:
        dyn_thr = min(base_thr, 0.92)
    else:
        dyn_thr = min(base_thr, 0.96)

    should_accept_local = semantic_ok and (local_conf >= dyn_thr)

    if should_accept_local:
        local["source"] = "on-device"
        return local

    cloud = generate_cloud(messages, tools)
    cloud["source"] = "cloud (fallback)"
    cloud["local_confidence"] = local_conf
    cloud["total_time_ms"] += local.get("total_time_ms", 0)
    return cloud


def print_result(label, result):
    """Pretty-print a generation result."""
    print(f"\n=== {label} ===\n")
    if "source" in result:
        print(f"Source: {result['source']}")
    if "confidence" in result:
        print(f"Confidence: {result['confidence']:.4f}")
    if "local_confidence" in result:
        print(f"Local confidence (below threshold): {result['local_confidence']:.4f}")
    print(f"Total time: {result['total_time_ms']:.2f}ms")
    for call in result["function_calls"]:
        print(f"Function: {call['name']}")
        print(f"Arguments: {json.dumps(call['arguments'], indent=2)}")


############## Example usage ##############

if __name__ == "__main__":
    tools = [{
        "name": "get_weather",
        "description": "Get current weather for a location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City name",
                }
            },
            "required": ["location"],
        },
    }]

    messages = [
        {"role": "user", "content": "What is the weather in San Francisco?"}
    ]

    on_device = generate_cactus(messages, tools)
    print_result("FunctionGemma (On-Device Cactus)", on_device)

    cloud = generate_cloud(messages, tools)
    print_result("Gemini (Cloud)", cloud)

    hybrid = generate_hybrid(messages, tools)
    print_result("Hybrid (On-Device + Cloud Fallback)", hybrid)
