
import sys
sys.path.insert(0, "cactus/python/src")
functiongemma_path = "cactus/weights/functiongemma-270m-it"

import json, os, re, time

# Regex to detect round-hour phrasing like "10 AM", "6AM" (no explicit minutes)
_ROUND_HOUR_RE = re.compile(r'\b(\d{1,2})\s*(am|pm)\b', re.I)
_EXPLICIT_MINUTE_RE = re.compile(r'\b\d{1,2}:\d{2}')
# ISO 8601 datetime — model sometimes emits these instead of "3:00 PM"
_ISO_DATETIME_RE = re.compile(r'^\d{4}-\d{2}-\d{2}T')
from cactus import cactus_init, cactus_complete, cactus_destroy
from google import genai
from google.genai import types


# ── Action-intent keyword map for pre-flight analysis ────────────────────────
# Maps intent category → distinctive keywords that appear in user requests.
_ACTION_KEYWORDS = {
    "weather":  ["weather", "forecast", "temperature"],
    "alarm":    ["alarm", "wake"],
    "message":  ["message", "text", "send"],
    "reminder": ["remind"],
    "contacts": ["contacts"],
    "music":    ["play", "music", "song"],
    "timer":    ["timer"],
}


def _preflight(messages, tools):
    """
    Checkpoint 1 — Pre-flight: Analyse request complexity before calling any model.

    Detects how many distinct action categories the user is asking for.
    If 2+ categories appear, the request likely requires multiple tool calls
    (multi-intent), which FunctionGemma reliably fails at.
    """
    user_text = " ".join(m["content"] for m in messages if m["role"] == "user").lower()

    matched_categories = set()
    for category, keywords in _ACTION_KEYWORDS.items():
        if any(kw in user_text for kw in keywords):
            matched_categories.add(category)

    return {
        "matched_categories": matched_categories,
        "num_actions": len(matched_categories),
        "multi_intent": len(matched_categories) >= 2,
        "num_tools": len(tools),
    }


def _get_tool_category(tool):
    """Map a tool to its action category using its name and description."""
    search_text = (tool["name"].replace("_", " ") + " " + tool.get("description", "")).lower()
    for category, keywords in _ACTION_KEYWORDS.items():
        if any(kw in search_text for kw in keywords):
            return category
    return None


def _validate(result, tools, complexity, messages):
    """
    Checkpoint 2 — Post-flight: Inspect on-device output for correctness signals.

    Checks:
      1. At least one function call returned.
      2. Multi-intent requests produce 2+ calls.
      3. Every called function exists in the provided tool list.
      4. All required parameters are present.
      5. Integer-typed parameters carry actual integer values (not strings).
      6. For multi-tool requests, the called function's category matches the user's
         detected intent (catches wrong-tool selection by the small model).

    Returns (is_valid: bool, reason: str).
    """
    calls = result.get("function_calls", [])
    tool_map = {t["name"]: t for t in tools}
    tool_names = set(tool_map)

    # 1. Must produce at least one call
    if not calls:
        return False, "no_calls"

    # 2. Multi-intent: expect 2+ calls
    if complexity["multi_intent"] and len(calls) < 2:
        return False, "multi_intent_needs_more_calls"

    for call in calls:
        fn_name = call.get("name", "")

        # 3. Called function must exist in the tool list
        if fn_name not in tool_names:
            return False, f"unknown_function:{fn_name}"

        tool = tool_map[fn_name]
        props = tool["parameters"].get("properties", {})
        required = tool["parameters"].get("required", [])
        args = call.get("arguments", {})

        # 4. All required parameters present
        for req in required:
            if req not in args:
                return False, f"missing_required_param:{req}"

        # 5. Integer parameters must carry integer values (not strings like "5")
        for param, spec in props.items():
            if param in args and spec.get("type") == "integer":
                if not isinstance(args[param], int):
                    return False, f"type_mismatch:{param}={repr(args[param])}_should_be_int"

        # 6. Tool-specific semantic sanity checks
        if fn_name == "set_alarm":
            hour = args.get("hour")
            minute = args.get("minute")
            if isinstance(hour, int) and not (0 <= hour <= 23):
                return False, f"alarm_hour_out_of_range:{hour}"
            if isinstance(minute, int) and not (0 <= minute <= 59):
                return False, f"alarm_minute_out_of_range:{minute}"
            # If user said a round hour ("10 AM", "6 AM") without explicit minutes,
            # the minute should be 0. Any other value means the model hallucinated.
            user_text = " ".join(m["content"] for m in messages if m["role"] == "user")
            round_hour_match = _ROUND_HOUR_RE.search(user_text)
            has_explicit_minute = bool(_EXPLICIT_MINUTE_RE.search(user_text))
            if round_hour_match and not has_explicit_minute:
                if isinstance(minute, int) and minute != 0:
                    return False, f"alarm_minute_should_be_0_for_round_hour:{minute}"
                # Also verify the hour in the response matches what the user asked
                requested_hour = int(round_hour_match.group(1))
                period = round_hour_match.group(2).lower()
                if period == "pm" and requested_hour != 12:
                    requested_hour += 12
                elif period == "am" and requested_hour == 12:
                    requested_hour = 0
                if isinstance(hour, int) and hour != requested_hour:
                    return False, f"alarm_hour_mismatch:{hour}_expected:{requested_hour}"

        elif fn_name == "set_timer":
            minutes = args.get("minutes")
            if isinstance(minutes, int) and minutes <= 0:
                return False, f"timer_minutes_non_positive:{minutes}"

        elif fn_name == "create_reminder":
            time_val = str(args.get("time", ""))
            title_val = str(args.get("title", ""))
            # Reject ISO 8601 datetime strings — model should produce "3:00 PM" style
            if _ISO_DATETIME_RE.match(time_val):
                return False, f"reminder_time_is_iso_datetime:{time_val}"
            # Reject titles that begin with "Reminder" — model is just echoing the prompt
            if title_val.lower().startswith("reminder"):
                return False, f"reminder_title_has_filler_prefix:{title_val}"

    # 6. Semantic intent check: only when multiple tools could be chosen
    if len(tools) > 1 and complexity["matched_categories"]:
        for call in calls:
            fn_name = call.get("name", "")
            tool = tool_map.get(fn_name)
            if not tool:
                continue
            fn_category = _get_tool_category(tool)
            # If the tool has a recognised category that doesn't match what the
            # user asked for, the small model picked the wrong tool.
            if fn_category is not None and fn_category not in complexity["matched_categories"]:
                return False, f"intent_mismatch:{fn_name}(category:{fn_category})"

    return True, "ok"


def _generate_cactus_with_system(messages, tools, system_message):
    """Run FunctionGemma on-device with a custom system message (for retries)."""
    model = cactus_init(functiongemma_path)
    cactus_tools = [{"type": "function", "function": t} for t in tools]

    raw_str = cactus_complete(
        model,
        [{"role": "system", "content": system_message}] + messages,
        tools=cactus_tools,
        force_tools=True,
        max_tokens=256,
        stop_sequences=["<|im_end|>", "<end_of_turn>"],
    )
    cactus_destroy(model)

    try:
        raw = json.loads(raw_str)
    except json.JSONDecodeError:
        return {"function_calls": [], "total_time_ms": 0, "confidence": 0}

    return {
        "function_calls": raw.get("function_calls", []),
        "total_time_ms": raw.get("total_time_ms", 0),
        "confidence": raw.get("confidence", 0),
    }


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

    for attempt in range(3):
        try:
            gemini_response = client.models.generate_content(
                model="gemini-2.5-flash-lite",
                contents=contents,
                config=types.GenerateContentConfig(
                    tools=gemini_tools,
                    temperature=0.0,
                    system_instruction=(
                        "Use the EXACT words from the user's request as argument values. "
                        "Do not paraphrase, expand contractions, or alter the wording. "
                        "Do not add trailing periods or punctuation to extracted phrases."
                    ),
                ),
            )
            break
        except Exception as e:
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)  # exponential back-off: 1s, 2s

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
    Multi-checkpoint hybrid routing strategy.

    Checkpoint 1 — Pre-flight (before any model call):
      Analyse the user's request for complexity signals.
      - Multi-intent (2+ distinct action categories) → skip local entirely and
        go straight to cloud. Small models consistently fail multi-call tasks.

    Checkpoint 2 — Post-flight (after FunctionGemma returns):
      Validate the local result structurally and semantically:
        • Function exists in tool list (hallucination check)
        • All required parameters present
        • Integer params carry integer values, not strings
        • For multi-tool requests: called function matches user's intent category
      If valid, apply a relaxed confidence threshold:
        • Single-tool requests: threshold = 0 (validation alone is sufficient)
        • Multi-tool requests: threshold = 0.6
      Trust the on-device result if it clears both hurdles.

    Checkpoint 3 — Retry (before escalating to cloud):
      If validation failed, re-run FunctionGemma with an explicit system prompt
      that emphasises correct parameter types and tool selection.
      Accept the retry on-device if it passes validation with confidence ≥ 0.5.
      Only escalate to cloud if retry also fails.
    """

    # ── Checkpoint 1: Pre-flight complexity analysis ──────────────────────
    complexity = _preflight(messages, tools)

    if complexity["multi_intent"]:
        # Multi-call tasks: go straight to cloud, no point running local first
        cloud = generate_cloud(messages, tools)
        # Cloud completeness check: if fewer calls returned than detected intents,
        # retry once with explicit N-call instruction.
        if len(cloud.get("function_calls", [])) < complexity["num_actions"]:
            retry_messages = messages + [{
                "role": "user",
                "content": (
                    f"Important: this request requires EXACTLY {complexity['num_actions']} separate "
                    f"function calls — one per action. Please call all {complexity['num_actions']} "
                    f"relevant tools now."
                )
            }]
            cloud2 = generate_cloud(retry_messages, tools)
            # Only prefer retry if it returned more calls (strictly an improvement)
            if len(cloud2.get("function_calls", [])) > len(cloud.get("function_calls", [])):
                cloud2["source"] = "cloud (pre-flight: multi-intent)"
                cloud2["total_time_ms"] += cloud["total_time_ms"]
                return cloud2
        cloud["source"] = "cloud (pre-flight: multi-intent)"
        return cloud

    # ── On-device inference ───────────────────────────────────────────────
    local = generate_cactus(messages, tools)
    total_local_time = local["total_time_ms"]

    # ── Checkpoint 2: Post-flight validation ─────────────────────────────
    valid, reason = _validate(local, tools, complexity, messages)

    if valid:
        # Single-tool: if it passed validation the function + params are correct;
        # confidence score adds no useful signal, so threshold = 0.
        # Multi-tool: require moderate confidence on top of validation.
        relaxed_threshold = 0.0 if complexity["num_tools"] == 1 else 0.6
        if local["confidence"] >= relaxed_threshold:
            local["source"] = "on-device"
            return local

    # ── Checkpoint 3: Retry with enhanced system prompt ───────────────────
    retry_system = (
        "You are a precise function-calling assistant. "
        "You MUST call one of the provided tools to fulfil the user's request. "
        "IMPORTANT rules:\n"
        "- Use integer values (not strings) for integer-type parameters.\n"
        "- For alarms: if the user says '10 AM' with no minutes, set minute=0.\n"
        "- For timers: minutes must be a positive integer.\n"
        "- For reminders: use a short title (2-4 words) and a simple time like '3:00 PM'.\n"
        "- Include every required parameter. Choose the tool that best matches the request."
    )
    retry = _generate_cactus_with_system(messages, tools, retry_system)
    total_local_time += retry["total_time_ms"]

    valid_retry, _ = _validate(retry, tools, complexity, messages)
    if valid_retry and retry["confidence"] >= 0.5:
        retry["total_time_ms"] = total_local_time
        retry["source"] = "on-device (retry)"
        return retry

    # ── Cloud fallback ────────────────────────────────────────────────────
    cloud = generate_cloud(messages, tools)
    cloud["source"] = "cloud (fallback)"
    cloud["local_confidence"] = local.get("confidence", 0)
    cloud["total_time_ms"] += total_local_time
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
