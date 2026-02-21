
import sys
sys.path.insert(0, "cactus/python/src")
functiongemma_path = "cactus/weights/functiongemma-270m-it"

import json, os, re, time
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
        model="gemini-2.5-flash-lite",
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


FAIL_FAST_THRESHOLD = 0.55
INTENT_WEIGHT_CAP = 0.6
ARG_IMPLICIT_WEIGHT = 0.55
TOOL_AMBIGUITY_WEIGHT = 0.6
THRESHOLD_MODULATION = 0.20
THRESHOLD_FLOOR = 0.70


def _last_user_message(messages):
    for message in reversed(messages):
        if message.get("role") == "user":
            return message.get("content", "")
    return ""


def _estimate_intent_count(last_user_message):
    lowered = f" {last_user_message.lower()} "
    normalized = lowered.replace("after that", "|")
    normalized = re.sub(r"\b(and|also|then)\b", "|", normalized)
    normalized = re.sub(r"[,:;?]", "|", normalized)
    chunks = [chunk.strip() for chunk in normalized.split("|") if chunk.strip()]
    return max(1, len(chunks))


def _required_tool_args(tools):
    required_args = []
    for tool in tools:
        params = tool.get("parameters", {})
        properties = params.get("properties", {})
        for arg_name in params.get("required", []):
            arg_schema = properties.get(arg_name, {})
            arg_type = str(arg_schema.get("type", "string")).lower()
            required_args.append((arg_name, arg_type))
    return required_args


def _arg_explicitness(last_user_message, tools):
    required_args = _required_tool_args(tools)
    if not required_args:
        return 1.0

    text = last_user_message
    has_quoted = bool(re.search(r"(['\"])[^'\"]+\1", text))
    has_proper_noun = bool(re.search(r"\b[A-Z][a-z]+\b", text))
    has_numeric = bool(re.search(r"\b\d+(?:[:.]\d+)?\b", text))
    has_date_like = bool(re.search(r"\b(?:\d{1,2}:\d{2}\s?(?:AM|PM|am|pm)?|\d{4}-\d{2}-\d{2}|jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\b", text))
    has_bool = bool(re.search(r"\b(true|false|yes|no|on|off)\b", text, flags=re.IGNORECASE))

    explicit = 0
    for _, arg_type in required_args:
        if arg_type in {"integer", "number"}:
            explicit += int(has_numeric or has_date_like)
        elif arg_type == "boolean":
            explicit += int(has_bool)
        else:
            explicit += int(has_quoted or has_proper_noun or has_numeric or has_date_like)

    return explicit / len(required_args)


def _tokenize_for_jaccard(text):
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def _tool_ambiguity_flag(tools):
    descriptions = [tool.get("description", "") for tool in tools if tool.get("description")]
    for i in range(len(descriptions)):
        for j in range(i + 1, len(descriptions)):
            left = _tokenize_for_jaccard(descriptions[i])
            right = _tokenize_for_jaccard(descriptions[j])
            if not left and not right:
                continue
            similarity = len(left & right) / len(left | right)
            if similarity > 0.4:
                return 1.0
    return 0.0


def _compute_complexity(messages, tools):
    last_user_message = _last_user_message(messages)
    intent_count = _estimate_intent_count(last_user_message)
    arg_explicitness = _arg_explicitness(last_user_message, tools)
    tool_ambiguity_flag = _tool_ambiguity_flag(tools)

    complexity = (
        min(intent_count / 3.0, INTENT_WEIGHT_CAP)
        + (1 - arg_explicitness) * ARG_IMPLICIT_WEIGHT
        + tool_ambiguity_flag * TOOL_AMBIGUITY_WEIGHT
    )
    return max(0.0, min(1.0, complexity))


def _is_structurally_valid(local_result, tools):
    tool_map = {tool["name"]: tool for tool in tools}
    primitive_types = {"string", "integer", "number", "boolean"}

    function_calls = local_result.get("function_calls", [])
    for call in function_calls:
        call_name = call.get("name")
        if call_name not in tool_map:
            return False

        tool_schema = tool_map[call_name].get("parameters", {})
        required = tool_schema.get("required", [])
        properties = tool_schema.get("properties", {})
        args = call.get("arguments", {}) or {}

        if any(required_arg not in args for required_arg in required):
            return False

        for arg_name, arg_value in args.items():
            expected_type = str(properties.get(arg_name, {}).get("type", "")).lower()
            if expected_type in primitive_types and arg_value is None:
                return False

    return True


def generate_hybrid(messages, tools, confidence_threshold=0.99):
    """Hybrid inference with fail-fast pre-routing.

    Computes a cheap complexity score before any inference. High-complexity
    queries are routed directly to cloud, avoiding the double-latency penalty
    of running local inference that is likely to fail anyway.
    """
    FAIL_FAST_COMPLEXITY = 0.4277
    CONFIDENCE_BASE = 0.6639
    CONFIDENCE_SCALE = 0.3126
    INTENT_WEIGHT = 0.2682
    ARG_DIFFICULTY_WEIGHT = 0.1325
    TOOL_PRESSURE_WEIGHT = 0.2872
    TOOL_RELIABILITY_WEIGHT = 0.4380

    def get_last_user_text(msgs):
        for message in reversed(msgs):
            if message.get("role") == "user":
                return message.get("content", "")
        return ""

    def compute_intent_score(last_user_text):
        segments = re.split(r"\band\b|\bthen\b|\balso\b|\bafter\b|[,;]", last_user_text.lower())
        segments = [s.strip() for s in segments if len(s.strip()) >= 3]
        segment_count = len(segments)
        return max(0.0, min((segment_count - 1) / 2.0, 1.0))

    def arg_difficulty_for_required_args(available_tools):
        difficulties = []
        for tool in available_tools:
            params = tool.get("parameters", {})
            properties = params.get("properties", {})
            for arg_name in params.get("required", []):
                arg_type = str(properties.get(arg_name, {}).get("type", "")).lower()
                arg_key = str(arg_name).lower()
                combined = f"{arg_key} {arg_type}"

                if any(token in combined for token in ("time", "duration", "hour", "minute", "when")):
                    difficulties.append(0.8)
                elif any(token in combined for token in ("location", "city", "place")):
                    difficulties.append(0.2)
                elif any(token in combined for token in ("contact", "person", "name", "recipient", "to")):
                    difficulties.append(0.7)
                elif any(token in combined for token in ("query", "search", "term", "keyword")):
                    difficulties.append(0.6)
                else:
                    difficulties.append(0.4)

        if not difficulties:
            return 0.3
        return sum(difficulties) / len(difficulties)

    def compute_tool_pressure(available_tools):
        return max(0.0, min((len(available_tools) - 1) / 4.0, 1.0))

    def compute_tool_reliability_penalty(available_tools):
        """
        Score how unreliable FunctionGemma tends to be for the given tool set.
        Based on empirical observation: weather/location tools succeed;
        alarm/timer/message/search/reminder/music tools fail at high confidence.
        Returns 0.0 (reliable) to 1.0 (unreliable).
        """
        UNRELIABLE_PATTERNS = ("alarm", "timer", "message", "search", "reminder", "music", "contact", "note")
        RELIABLE_PATTERNS = ("weather", "location", "forecast")

        scores = []
        for tool in available_tools:
            name = tool.get("name", "").lower()
            desc = tool.get("description", "").lower()
            combined = f"{name} {desc}"

            if any(p in combined for p in RELIABLE_PATTERNS):
                scores.append(0.1)
            elif any(p in combined for p in UNRELIABLE_PATTERNS):
                scores.append(0.9)
            else:
                scores.append(0.5)  # unknown tool — be moderately cautious

        if not scores:
            return 0.5
        return sum(scores) / len(scores)

    def is_tool_name_valid(result, available_tools):
        calls = result.get("function_calls", [])
        if not calls:
            return True
        tool_names = {tool["name"] for tool in available_tools}
        return all(call.get("name") in tool_names for call in calls)

    last_user_text = get_last_user_text(messages)
    intent_score = compute_intent_score(last_user_text)
    arg_difficulty = arg_difficulty_for_required_args(tools)
    tool_pressure = compute_tool_pressure(tools)
    reliability_penalty = compute_tool_reliability_penalty(tools)

    complexity = (
        (intent_score * INTENT_WEIGHT)
        + (arg_difficulty * ARG_DIFFICULTY_WEIGHT)
        + (tool_pressure * TOOL_PRESSURE_WEIGHT)
        + (reliability_penalty * TOOL_RELIABILITY_WEIGHT)
    )
    complexity = max(0.0, min(complexity, 1.0))

    if complexity >= FAIL_FAST_COMPLEXITY:
        cloud = generate_cloud(messages, tools)
        cloud["source"] = "cloud (complexity skip)"
        cloud["local_confidence"] = None
        return cloud

    local = generate_cactus(messages, tools)

    if not is_tool_name_valid(local, tools):
        cloud = generate_cloud(messages, tools)
        cloud["source"] = "cloud (invalid local)"
        cloud["local_confidence"] = local["confidence"]
        cloud["total_time_ms"] += local["total_time_ms"]
        return cloud

    effective_threshold = CONFIDENCE_BASE + (complexity * CONFIDENCE_SCALE)
    effective_threshold = min(effective_threshold, 0.95)
    if local["confidence"] >= effective_threshold:
        local["source"] = "on-device"
        return local

    cloud = generate_cloud(messages, tools)
    cloud["source"] = "cloud (fallback)"
    cloud["local_confidence"] = local["confidence"]
    cloud["total_time_ms"] += local["total_time_ms"]
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
