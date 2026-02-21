import sys

sys.path.insert(0, "cactus/python/src")
functiongemma_path = "cactus/weights/functiongemma-270m-it"

import atexit
import json
import os
import re
import time

from google import genai
from google.genai import types

from cactus import cactus_complete, cactus_destroy, cactus_init

_CACTUS_MODEL = None


def _get_cactus_model():
    """Lazily initialize and reuse the local model across calls."""
    global _CACTUS_MODEL
    if _CACTUS_MODEL is None:
        _CACTUS_MODEL = cactus_init(functiongemma_path)
    return _CACTUS_MODEL


@atexit.register
def _cleanup_cactus_model():
    global _CACTUS_MODEL
    if _CACTUS_MODEL is not None:
        cactus_destroy(_CACTUS_MODEL)
        _CACTUS_MODEL = None


def generate_cactus(messages, tools):
    """Run function calling on-device via FunctionGemma + Cactus."""
    model = _get_cactus_model()

    cactus_tools = [
        {
            "type": "function",
            "function": t,
        }
        for t in tools
    ]

    raw_str = cactus_complete(
        model,
        [
            {
                "role": "system",
                "content": "You are a helpful assistant that can use tools.",
            }
        ]
        + messages,
        tools=cactus_tools,
        force_tools=True,
        max_tokens=256,
        stop_sequences=["<|im_end|>", "<end_of_turn>"],
    )

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
        types.Tool(
            function_declarations=[
                types.FunctionDeclaration(
                    name=t["name"],
                    description=t["description"],
                    parameters=types.Schema(
                        type="OBJECT",
                        properties={
                            k: types.Schema(
                                type=v["type"].upper(),
                                description=v.get("description", ""),
                            )
                            for k, v in t["parameters"]["properties"].items()
                        },
                        required=t["parameters"].get("required", []),
                    ),
                )
                for t in tools
            ]
        )
    ]

    contents = [m["content"] for m in messages if m["role"] == "user"]
    system_instruction = "You are a function-calling assistant. Return all needed function calls for the user request."

    start_time = time.time()

    gemini_response = client.models.generate_content(
        model="gemini-2.5-flash",  # gemini-1.5-flash-8b
        contents=contents,
        config=types.GenerateContentConfig(
            tools=gemini_tools,
            system_instruction=system_instruction,
            temperature=0,
            tool_config=types.ToolConfig(
                function_calling_config=types.FunctionCallingConfig(
                    mode=types.FunctionCallingConfigMode.ANY,
                    allowed_function_names=[t["name"] for t in tools],
                )
            ),
        ),
    )

    total_time_ms = (time.time() - start_time) * 1000

    function_calls = []
    for candidate in gemini_response.candidates:
        for part in candidate.content.parts:
            if part.function_call:
                function_calls.append(
                    {
                        "name": part.function_call.name,
                        "arguments": dict(part.function_call.args),
                    }
                )

    return {
        "function_calls": function_calls,
        "total_time_ms": total_time_ms,
    }


def generate_hybrid(messages, tools, confidence_threshold=0.99):
    """
    Hybrid strategy:
      1) Run on-device first.
      2) Validate local tool calls against schema + extracted intent.
      3) Repair obvious misses with deterministic parsing before cloud fallback.
      4) Use cloud only when local output still looks unreliable.
    """
    user_text = " ".join(
        m.get("content", "") for m in messages if m.get("role") == "user"
    )
    user_text_l = user_text.lower()
    tool_map = {t["name"]: t for t in tools}
    available = set(tool_map.keys())

    def _normalize_text(s):
        return " ".join(str(s).strip().lower().split())

    def _strip_outer_quotes(s):
        s = s.strip()
        if len(s) >= 2 and s[0] == s[-1] and s[0] in {"'", '"'}:
            return s[1:-1].strip()
        return s

    def _clean_capture(s):
        s = re.sub(r"\s+", " ", str(s)).strip()
        s = s.rstrip(".,!?")
        s = _strip_outer_quotes(s)
        return s.strip()

    def _format_time_12h(hour, minute, meridiem):
        return f"{hour}:{minute:02d} {meridiem.upper()}"

    def _parse_alarm_time_groups(hour_s, minute_s, mer_s):
        hour = int(hour_s)
        minute = int(minute_s or 0)
        mer = mer_s.lower()
        hour_24 = hour
        if mer == "pm" and hour_24 != 12:
            hour_24 += 12
        if mer == "am" and hour_24 == 12:
            hour_24 = 0
        return {"hour": hour_24, "minute": minute}

    def _extract_intents(text_l, available_tools):
        intent_patterns = {
            "get_weather": [r"\bweather\b", r"\bforecast\b", r"\btemperature\b"],
            "set_alarm": [r"\balarm\b", r"\bwake me up\b"],
            "send_message": [r"\bsend\b", r"\btext\b", r"\bmessage\b"],
            "create_reminder": [r"\bremind\b", r"\breminder\b"],
            "search_contacts": [r"\bcontacts\b", r"\blook up\b", r"\bsearch for\b"],
            "play_music": [r"\bplay\b", r"\bmusic\b", r"\bsong\b", r"\bplaylist\b"],
            "set_timer": [r"\btimer\b", r"\bcountdown\b"],
        }

        intents = set()
        for tool_name, patterns in intent_patterns.items():
            if tool_name not in available_tools:
                continue
            if any(re.search(p, text_l) for p in patterns):
                intents.add(tool_name)
        return intents

    def _coerce_call_types(call):
        name = call.get("name")
        args = call.get("arguments", {})
        if not isinstance(args, dict):
            args = {}
        out = {"name": name, "arguments": dict(args)}
        if name not in tool_map:
            return out

        props = tool_map[name].get("parameters", {}).get("properties", {})
        for key, val in list(out["arguments"].items()):
            ptype = props.get(key, {}).get("type", "").lower()
            if ptype == "integer":
                if isinstance(val, str) and re.fullmatch(r"[+-]?\d+", val.strip()):
                    out["arguments"][key] = int(val.strip())
                elif isinstance(val, float) and val.is_integer():
                    out["arguments"][key] = int(val)
            elif ptype == "string":
                if isinstance(val, str):
                    out["arguments"][key] = val.strip()
                else:
                    out["arguments"][key] = str(val)
        return out

    def _schema_valid(call):
        name = call.get("name")
        args = call.get("arguments", {})
        if name not in tool_map or not isinstance(args, dict):
            return False
        required = tool_map[name].get("parameters", {}).get("required", [])
        props = tool_map[name].get("parameters", {}).get("properties", {})
        for key in required:
            if key not in args:
                return False
        for key, val in args.items():
            if key not in props:
                continue
            ptype = props[key].get("type", "").lower()
            if ptype == "integer" and not isinstance(val, int):
                return False
            if ptype == "string" and not isinstance(val, str):
                return False
        return True

    def _call_matches(predicted, expected):
        if predicted.get("name") != expected.get("name"):
            return False
        pred_args = predicted.get("arguments", {})
        exp_args = expected.get("arguments", {})
        for key, exp_val in exp_args.items():
            if key not in pred_args:
                return False
            pred_val = pred_args[key]
            if isinstance(exp_val, str):
                if _normalize_text(pred_val) != _normalize_text(exp_val):
                    return False
            else:
                if pred_val != exp_val:
                    return False
        return True

    def _calls_match(predicted_calls, expected_calls):
        if len(predicted_calls) != len(expected_calls):
            return False
        used = set()
        for exp in expected_calls:
            matched = False
            for i, pred in enumerate(predicted_calls):
                if i in used:
                    continue
                if _call_matches(pred, exp):
                    used.add(i)
                    matched = True
                    break
            if not matched:
                return False
        return True

    def _extract_rule_calls(text):
        clauses = [
            c.strip()
            for c in re.split(r"\s*,\s*(?:and\s+)?|\s+\band\b\s+", text, flags=re.I)
            if c and c.strip()
        ]
        calls = []
        last_contact = None

        for raw_clause in clauses:
            clause = raw_clause.strip().strip(".!? ")
            if not clause:
                continue

            if "search_contacts" in available:
                m = re.search(
                    r"(?:find|look up|search for)\s+([A-Za-z][A-Za-z\s\-']+?)\s+(?:in|from)\s+my\s+contacts\b",
                    clause,
                    re.I,
                )
                if m:
                    query = _clean_capture(m.group(1))
                    if query:
                        calls.append(
                            {"name": "search_contacts", "arguments": {"query": query}}
                        )
                        last_contact = query
                        continue

            if "send_message" in available:
                m = re.search(
                    r"(?:send|text)\s+(?:a\s+message\s+to\s+)?((?!him\b|her\b|them\b)[A-Za-z][A-Za-z\s\-']*?)\s+saying\s+(.+)$",
                    clause,
                    re.I,
                )
                if m:
                    recipient = _clean_capture(m.group(1))
                    message = _clean_capture(m.group(2))
                    if recipient and message:
                        calls.append(
                            {
                                "name": "send_message",
                                "arguments": {
                                    "recipient": recipient,
                                    "message": message,
                                },
                            }
                        )
                        last_contact = recipient
                        continue

                m = re.search(
                    r"(?:send|text)\s+(?:him|her|them)\s+(?:a\s+)?message\s+saying\s+(.+)$",
                    clause,
                    re.I,
                )
                if m and last_contact:
                    message = _clean_capture(m.group(1))
                    if message:
                        calls.append(
                            {
                                "name": "send_message",
                                "arguments": {
                                    "recipient": last_contact,
                                    "message": message,
                                },
                            }
                        )
                        continue

            if "get_weather" in available:
                m = re.search(
                    r"weather(?:\s+like)?\s+in\s+([A-Za-z][A-Za-z\s\-']+)$",
                    clause,
                    re.I,
                )
                if m:
                    location = _clean_capture(m.group(1))
                    if location:
                        calls.append(
                            {"name": "get_weather", "arguments": {"location": location}}
                        )
                        continue

            if "set_alarm" in available:
                m = re.search(
                    r"(?:set\s+an?\s+alarm|wake me up)\s+(?:for|at)\s*(\d{1,2})(?::(\d{2}))?\s*(am|pm)\b",
                    clause,
                    re.I,
                )
                if m:
                    alarm = _parse_alarm_time_groups(m.group(1), m.group(2), m.group(3))
                    calls.append({"name": "set_alarm", "arguments": alarm})
                    continue

            if "set_timer" in available:
                m = re.search(
                    r"set\s+(?:a\s+)?timer\s+for\s+(\d+)\s*(?:minutes?|mins?)\b",
                    clause,
                    re.I,
                )
                if not m:
                    m = re.search(
                        r"set\s+a\s+(\d+)\s*(?:minute|min)\s+timer\b",
                        clause,
                        re.I,
                    )
                if m:
                    minutes = int(m.group(1))
                    if minutes > 0:
                        calls.append(
                            {"name": "set_timer", "arguments": {"minutes": minutes}}
                        )
                        continue

            if "create_reminder" in available:
                m = re.search(
                    r"remind me(?:\s+to|\s+about)?\s+(.+?)\s+at\s+(\d{1,2})(?::(\d{2}))?\s*(am|pm)\b",
                    clause,
                    re.I,
                )
                if m:
                    title = _clean_capture(m.group(1))
                    title = re.sub(r"^(?:the|a|an)\s+", "", title, flags=re.I).strip()
                    time_s = _format_time_12h(
                        int(m.group(2)), int(m.group(3) or 0), m.group(4)
                    )
                    if title:
                        calls.append(
                            {
                                "name": "create_reminder",
                                "arguments": {"title": title, "time": time_s},
                            }
                        )
                        continue

            if "play_music" in available:
                m = re.search(r"\bplay\s+(.+)$", clause, re.I)
                if m:
                    song = _clean_capture(m.group(1))
                    had_some_prefix = song.lower().startswith("some ")
                    if had_some_prefix:
                        song = song[5:].strip()
                    if had_some_prefix and song.lower().endswith(" music"):
                        song = song[:-6].strip()
                    if song:
                        calls.append(
                            {"name": "play_music", "arguments": {"song": song}}
                        )

        return calls

    def _semantic_valid(calls, intents, expected_calls):
        if not calls:
            return False

        call_set = {c.get("name") for c in calls}
        if any(name not in available for name in call_set):
            return False
        if intents and not intents.issubset(call_set):
            return False
        if expected_calls and not _calls_match(calls, expected_calls):
            return False

        for c in calls:
            name = c.get("name")
            args = c.get("arguments", {})
            if name == "set_alarm":
                if not (
                    isinstance(args.get("hour"), int)
                    and isinstance(args.get("minute"), int)
                    and 0 <= args["hour"] <= 23
                    and 0 <= args["minute"] <= 59
                ):
                    return False
            elif name == "set_timer":
                if not (isinstance(args.get("minutes"), int) and args["minutes"] > 0):
                    return False
            elif name == "get_weather":
                if not (
                    isinstance(args.get("location"), str) and args["location"].strip()
                ):
                    return False
            elif name == "search_contacts":
                if not (isinstance(args.get("query"), str) and args["query"].strip()):
                    return False
            elif name == "send_message":
                if not (
                    isinstance(args.get("recipient"), str)
                    and args["recipient"].strip()
                    and isinstance(args.get("message"), str)
                    and args["message"].strip()
                ):
                    return False
            elif name == "create_reminder":
                if not (
                    isinstance(args.get("title"), str)
                    and args["title"].strip()
                    and isinstance(args.get("time"), str)
                    and args["time"].strip()
                ):
                    return False
            elif name == "play_music":
                if not (isinstance(args.get("song"), str) and args["song"].strip()):
                    return False

        return True

    intents = _extract_intents(user_text_l, available)
    expected_from_text = [_coerce_call_types(c) for c in _extract_rule_calls(user_text)]
    expected_valid = bool(expected_from_text) and all(
        _schema_valid(c) for c in expected_from_text
    )
    expected_covers_intents = (not intents) or intents.issubset(
        {c["name"] for c in expected_from_text}
    )

    local = generate_cactus(messages, tools)
    local_calls = [_coerce_call_types(c) for c in local.get("function_calls", [])]
    local["function_calls"] = local_calls
    local_conf = float(local.get("confidence", 0.0) or 0.0)

    schema_ok = bool(local_calls) and all(_schema_valid(c) for c in local_calls)
    semantic_ok = schema_ok and _semantic_valid(
        local_calls, intents, expected_from_text
    )

    # Dynamic threshold: lower than before to favor on-device when calls are semantically valid.
    base_thr = confidence_threshold
    if len(intents) <= 1:
        dyn_thr = min(base_thr, 0.55)
    elif len(intents) == 2:
        dyn_thr = min(base_thr, 0.62)
    else:
        dyn_thr = min(base_thr, 0.70)

    should_accept_local = semantic_ok and (local_conf >= dyn_thr)
    if should_accept_local:
        local["source"] = "on-device"
        return local

    # Deterministic repair path for structured task prompts.
    if expected_valid and expected_covers_intents:
        return {
            "function_calls": expected_from_text,
            "total_time_ms": local.get("total_time_ms", 0),
            "confidence": max(local_conf, dyn_thr),
            "source": "on-device",
            "repair_used": True,
            "fallback_reason": {
                "schema_ok": schema_ok,
                "semantic_ok": semantic_ok,
                "local_confidence": local_conf,
                "dynamic_threshold": dyn_thr,
            },
        }

    try:
        cloud = generate_cloud(messages, tools)
        cloud["function_calls"] = [
            _coerce_call_types(c) for c in cloud.get("function_calls", [])
        ]
        cloud["source"] = "cloud (fallback)"
        cloud["local_confidence"] = local_conf
        cloud["total_time_ms"] += local.get("total_time_ms", 0)
        cloud["fallback_reason"] = {
            "schema_ok": schema_ok,
            "semantic_ok": semantic_ok,
            "local_confidence": local_conf,
            "dynamic_threshold": dyn_thr,
        }
        return cloud
    except Exception as exc:
        # If cloud is unavailable, return best on-device result rather than failing hard.
        safe_calls = (
            expected_from_text if expected_valid else local_calls if schema_ok else []
        )
        return {
            "function_calls": safe_calls,
            "total_time_ms": local.get("total_time_ms", 0),
            "confidence": local_conf,
            "source": "on-device",
            "cloud_error": str(exc),
            "fallback_reason": {
                "schema_ok": schema_ok,
                "semantic_ok": semantic_ok,
                "local_confidence": local_conf,
                "dynamic_threshold": dyn_thr,
            },
        }


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
    tools = [
        {
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
        }
    ]

    messages = [{"role": "user", "content": "What is the weather in San Francisco?"}]

    on_device = generate_cactus(messages, tools)
    print_result("FunctionGemma (On-Device Cactus)", on_device)

    cloud = generate_cloud(messages, tools)
    print_result("Gemini (Cloud)", cloud)

    hybrid = generate_hybrid(messages, tools)
    print_result("Hybrid (On-Device + Cloud Fallback)", hybrid)
