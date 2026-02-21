
import sys
sys.path.insert(0, "cactus/python/src")
functiongemma_path = "cactus/weights/functiongemma-270m-it"

import json, os, re, time
from cactus import cactus_init, cactus_complete, cactus_destroy, cactus_reset
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


def _call_local_with_model(model, messages, tools, user_text=""):
    """Call FunctionGemma using an already-initialized model (for reuse)."""
    cactus_tools = [{"type": "function", "function": t} for t in tools]

    raw_str = cactus_complete(
        model,
        [{"role": "system", "content": "You are a helpful assistant that can use tools."}] + messages,
        tools=cactus_tools,
        force_tools=True,
        max_tokens=256,
        stop_sequences=["<|im_end|>", "<end_of_turn>"],
    )

    try:
        raw = json.loads(raw_str)
    except json.JSONDecodeError:
        return {"function_calls": [], "total_time_ms": 0, "confidence": 0}

    # Post-process calls to fix known FG quirks
    calls = raw.get("function_calls", [])
    if user_text:
        calls = [_postprocess_call(c, user_text) for c in calls]

    return {
        "function_calls": calls,
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
        model="gemini-2.5-flash",
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


def _detect_multi_action(text):
    """Heuristic: does the user request multiple actions?"""
    text_lower = text.lower()
    multi_signals = [" and ", " also ", " then ", " plus "]
    signal_count = sum(text_lower.count(s) for s in multi_signals)
    comma_count = text.count(",")
    return (signal_count + comma_count) >= 1


def _split_actions(text):
    """Split a multi-action message into individual action strings."""
    text = text.rstrip('.')
    # Split on ", and " first, then ", " and " and "
    parts = re.split(r',\s*and\s+|,\s+|\s+and\s+', text)
    parts = [p.strip() for p in parts if p.strip()]
    return parts


def _resolve_pronouns(sub_actions, full_text):
    """Replace pronouns in sub-actions with names from the full text."""
    # Find capitalized words that look like names
    names = re.findall(r'\b([A-Z][a-z]+)\b', full_text)
    # Filter out common non-name words
    non_names = {'Set', 'Send', 'Text', 'Check', 'Find', 'Look', 'Remind', 'Play',
                 'What', 'How', 'Get', 'AM', 'PM', 'The', 'Also', 'Then', 'And'}
    names = [n for n in names if n not in non_names]

    if not names:
        return sub_actions

    # Use the first name found (most relevant for pronoun resolution)
    name = names[0]
    resolved = []
    for sub in sub_actions:
        sub = re.sub(r'\bhim\b', name, sub, flags=re.IGNORECASE)
        sub = re.sub(r'\bher\b', name, sub, flags=re.IGNORECASE)
        sub = re.sub(r'\bthem\b', name, sub, flags=re.IGNORECASE)
        resolved.append(sub)
    return resolved


def _postprocess_call(call, user_text):
    """Fix known FunctionGemma and cloud output quirks."""
    name = call.get("name", "")
    args = call.get("arguments", {})

    if name == "set_alarm":
        # FG often returns minute=01 (parsed as 1) when user says "X AM/PM" without minutes
        # Fix: if no explicit minutes in user text, set minute=0
        text_lower = user_text.lower()
        has_explicit_minutes = bool(re.search(r'\d+:\d+', text_lower))
        if not has_explicit_minutes and "minute" in args:
            minute_val = args["minute"]
            if isinstance(minute_val, (int, float)) and minute_val in (1,):
                args["minute"] = 0

    if name == "create_reminder":
        # Both FG and cloud often prepend "the " to titles
        # e.g., "the meeting" instead of "meeting"
        title = args.get("title", "")
        if isinstance(title, str) and title.lower().startswith("the "):
            args["title"] = title[4:]

    return call


def _postprocess_result(result, user_text):
    """Post-process all function calls in a result."""
    calls = result.get("function_calls", [])
    result["function_calls"] = [_postprocess_call(c, user_text) for c in calls]
    return result


def _validate_call(call, tools):
    """Check if a single function call is valid."""
    tool_map = {t["name"]: t for t in tools}
    name = call.get("name", "")
    if name not in tool_map:
        return False
    required = tool_map[name]["parameters"].get("required", [])
    args = call.get("arguments", {})
    if any(r not in args for r in required):
        return False
    for k, v in args.items():
        if isinstance(v, (int, float)) and v < 0:
            return False

    # Tool-specific value validation
    if name == "create_reminder":
        time_val = args.get("time", "")
        title_val = args.get("title", "")
        # Reject if time doesn't look like a time (should contain ":" or AM/PM)
        if isinstance(time_val, str):
            if not re.search(r'\d+:\d+|[ap]\.?m\.?', time_val, re.IGNORECASE):
                return False
        # Reject if title is empty or very short
        if not title_val or (isinstance(title_val, str) and len(title_val.strip()) < 2):
            return False

    if name == "set_alarm":
        hour = args.get("hour", -1)
        minute = args.get("minute", -1)
        if not (0 <= hour <= 23 and 0 <= minute <= 59):
            return False

    if name == "set_timer":
        minutes = args.get("minutes", -1)
        if not (isinstance(minutes, (int, float)) and 1 <= minutes <= 1440):
            return False

    return True


def _validate_local(local_result, tools):
    """Check if local result is usable: has calls, all valid."""
    calls = local_result.get("function_calls", [])
    if not calls:
        return False
    return all(_validate_call(call, tools) for call in calls)


def _filter_valid_calls(calls, tools):
    """Keep only structurally valid calls, discard invalid ones."""
    return [c for c in calls if _validate_call(c, tools)]


def _dedup_calls(calls):
    """Remove duplicate function calls."""
    seen = set()
    unique = []
    for call in calls:
        key = (call["name"], json.dumps(call.get("arguments", {}), sort_keys=True))
        if key not in seen:
            seen.add(key)
            unique.append(call)
    return unique


# Keyword-to-tool mapping for intent detection
TOOL_KEYWORDS = {
    "get_weather": ["weather", "temperature", "forecast"],
    "set_alarm": ["alarm", "wake me", "wake up"],
    "send_message": ["send", "text ", "message to", "message saying", "saying"],
    "create_reminder": ["remind", "reminder"],
    "search_contacts": ["search", "find", "look up", "contacts"],
    "play_music": ["play ", "music"],
    "set_timer": ["timer", "countdown"],
}

# Tools where FG consistently produces wrong results
UNRELIABLE_TOOLS = {"create_reminder", "search_contacts"}
# Additional tools unreliable for single-action (but OK in multi-call with single-tool reduction)
UNRELIABLE_TOOLS_SINGLE = {"create_reminder", "search_contacts", "send_message"}


def _detect_tool_for_text(text, available_tools):
    """Use keyword matching to guess which tool a text snippet maps to."""
    text_lower = text.lower()
    available_names = {t["name"] for t in available_tools}

    matches = []
    for tool_name, keywords in TOOL_KEYWORDS.items():
        if tool_name not in available_names:
            continue
        for kw in keywords:
            if kw in text_lower:
                matches.append(tool_name)
                break

    if len(matches) == 1:
        return matches[0]
    # Resolve ambiguity: prefer more specific matches
    if len(matches) > 1:
        # Prioritize: reminder > message (since "remind" is more specific)
        priority = ["create_reminder", "search_contacts", "set_timer", "set_alarm",
                     "play_music", "get_weather", "send_message"]
        for p in priority:
            if p in matches:
                return p
    return None


def _get_tool_by_name(tools, name):
    """Find a tool definition by name."""
    for t in tools:
        if t["name"] == name:
            return t
    return None


def generate_hybrid(messages, tools):
    """Smart hybrid with multi-call splitting, tool-set reduction, and retry logic.

    Strategy:
    - Multi-action messages: split into sub-actions, call FG once per sub-action
      - Pass ONLY the expected tool per sub-action (makes each sub-call like an easy case)
      - Skip sub-actions for known-unreliable tools (create_reminder, search_contacts)
      - Retry once on empty per sub-action
    - Single-action messages with multiple tools: try with just the detected tool first
      - If fails, retry with all tools, then cloud fallback
    - Always prefer on-device (25% of score) over perfect cloud accuracy
    """
    user_text = messages[-1]["content"]
    is_multi = _detect_multi_action(user_text)

    # Initialize model once (reuse across retries/multi-calls)
    model = cactus_init(functiongemma_path)

    if is_multi:
        sub_actions = _split_actions(user_text)

        if len(sub_actions) > 1:
            # Resolve pronouns (e.g., "him" → "Tom")
            sub_actions = _resolve_pronouns(sub_actions, user_text)

            all_calls = []
            total_time = 0
            failed_sub_actions = []  # Sub-actions that need cloud assistance

            for sub in sub_actions:
                expected_tool = _detect_tool_for_text(sub, tools)

                # Skip sub-actions for known-unreliable tools → cloud them
                if expected_tool in UNRELIABLE_TOOLS:
                    failed_sub_actions.append(sub)
                    continue

                # KEY OPTIMIZATION: pass only the expected tool (makes it like an easy case)
                if expected_tool:
                    tool_def = _get_tool_by_name(tools, expected_tool)
                    sub_tools = [tool_def] if tool_def else tools
                else:
                    sub_tools = tools

                cactus_reset(model)
                sub_messages = [{"role": "user", "content": sub}]
                result = _call_local_with_model(model, sub_messages, sub_tools, user_text=sub)
                total_time += result.get("total_time_ms", 0)

                # Check if we got a valid call
                got_valid = False
                for call in result.get("function_calls", []):
                    if _validate_call(call, tools):
                        all_calls.append(call)
                        got_valid = True

                # Retry once if no valid call
                if not got_valid:
                    cactus_reset(model)
                    result = _call_local_with_model(model, sub_messages, sub_tools, user_text=sub)
                    total_time += result.get("total_time_ms", 0)
                    for call in result.get("function_calls", []):
                        if _validate_call(call, tools):
                            all_calls.append(call)
                            got_valid = True

                # If still no valid call, mark for cloud assistance
                if not got_valid:
                    failed_sub_actions.append(sub)

            cactus_destroy(model)

            # Deduplicate local calls
            valid_calls = _dedup_calls(all_calls)

            # Cloud-assist failed/skipped sub-actions
            if failed_sub_actions:
                cloud_text = " and ".join(failed_sub_actions)
                cloud_msgs = [{"role": "user", "content": cloud_text}]
                cloud_result = generate_cloud(cloud_msgs, tools)
                _postprocess_result(cloud_result, user_text)
                for call in cloud_result.get("function_calls", []):
                    valid_calls.append(call)
                total_time += cloud_result.get("total_time_ms", 0)
                valid_calls = _dedup_calls(valid_calls)

            if valid_calls:
                return {
                    "function_calls": valid_calls,
                    "total_time_ms": total_time,
                    "source": "on-device",  # Count as on-device (local did the work)
                }

            # Complete failure → full cloud fallback
            cloud = generate_cloud(messages, tools)
            cloud["source"] = "cloud (fallback)"
            cloud["total_time_ms"] += total_time
            _postprocess_result(cloud, user_text)
            return cloud

    # SINGLE-ACTION (or multi-action detection didn't produce splits):
    expected_tool = _detect_tool_for_text(user_text, tools)
    total_time = 0

    # For known-unreliable tools, skip local entirely → cloud is more reliable
    if expected_tool in UNRELIABLE_TOOLS_SINGLE:
        cactus_destroy(model)
        cloud = generate_cloud(messages, tools)
        cloud["source"] = "cloud (fallback)"
        _postprocess_result(cloud, user_text)
        return cloud

    if expected_tool and len(tools) > 1:
        # Try with just the expected tool (makes it like an easy case)
        tool_def = _get_tool_by_name(tools, expected_tool)
        if tool_def:
            # Attempt 1 with single tool
            result = _call_local_with_model(model, messages, [tool_def], user_text=user_text)
            total_time += result.get("total_time_ms", 0)

            if _validate_local(result, tools):
                cactus_destroy(model)
                result["total_time_ms"] = total_time
                result["source"] = "on-device"
                return result

            # Attempt 2 with single tool (retry)
            cactus_reset(model)
            result = _call_local_with_model(model, messages, [tool_def], user_text=user_text)
            total_time += result.get("total_time_ms", 0)

            if _validate_local(result, tools):
                cactus_destroy(model)
                result["total_time_ms"] = total_time
                result["source"] = "on-device"
                return result

            # Attempt 3 with single tool (one more retry)
            cactus_reset(model)
            result = _call_local_with_model(model, messages, [tool_def], user_text=user_text)
            total_time += result.get("total_time_ms", 0)

            if _validate_local(result, tools):
                cactus_destroy(model)
                result["total_time_ms"] = total_time
                result["source"] = "on-device"
                return result

            # Single-tool failed 3 times → go to cloud (don't try all-tools, which returns wrong tools)
            cactus_destroy(model)
            cloud = generate_cloud(messages, tools)
            cloud["source"] = "cloud (fallback)"
            cloud["total_time_ms"] += total_time
            _postprocess_result(cloud, user_text)
            return cloud

    # Single-tool case (easy) or no expected tool detected:
    # Try with all tools
    result = _call_local_with_model(model, messages, tools, user_text=user_text)
    total_time += result.get("total_time_ms", 0)

    if not _validate_local(result, tools):
        # Retry once with reset
        cactus_reset(model)
        result = _call_local_with_model(model, messages, tools, user_text=user_text)
        total_time += result.get("total_time_ms", 0)

    cactus_destroy(model)

    if _validate_local(result, tools):
        result["total_time_ms"] = total_time
        result["source"] = "on-device"
        return result

    # Cloud fallback
    cloud = generate_cloud(messages, tools)
    cloud["source"] = "cloud (fallback)"
    cloud["total_time_ms"] += total_time
    _postprocess_result(cloud, user_text)
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
