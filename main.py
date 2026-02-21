
import sys
sys.path.insert(0, "cactus/python/src")
functiongemma_path = "cactus/weights/functiongemma-270m-it"

import json, os, time
from cactus import cactus_init, cactus_complete, cactus_destroy
from google import genai
from google.genai import types

DEFAULT_LOCAL_CONFIDENCE_THRESHOLD = 0.99
DEFAULT_LOCAL_TEMPERATURE = 0.0
DEFAULT_CLOUD_TEMPERATURE = 0.0
DEFAULT_MAX_TOKENS = 256
DEFAULT_STOP_SEQUENCES = ["<|im_end|>", "<end_of_turn>"]
DEFAULT_CLOUD_MODEL_FALLBACKS = ["gemini-2.5-flash", "gemini-2.5-pro"]

DEFAULT_SYSTEM_INSTRUCTION = "You are a helpful assistant that can use tools."
REPAIR_SYSTEM_INSTRUCTION = (
    "You must respond using one or more tool calls from the provided tool list. "
    "Do not respond with plain text."
)
CLOUD_MULTI_INTENT_REPAIR_SYSTEM_INSTRUCTION = (
    "You must respond with function calls only. If the user requests multiple actions, "
    "output one function call per action in the same order. Do not omit requested actions "
    "and do not reply with plain text."
)

MULTI_INTENT_SEPARATORS = [" and then ", " then ", " and also ", ", and ", " also "]


def _to_cactus_tools(tools):
    return [{"type": "function", "function": tool} for tool in tools]


def _run_cactus(messages, tools, system_instruction, temperature=DEFAULT_LOCAL_TEMPERATURE):
    model = cactus_init(functiongemma_path)

    raw_str = cactus_complete(
        model,
        [{"role": "system", "content": system_instruction}] + messages,
        tools=_to_cactus_tools(tools),
        force_tools=True,
        temperature=temperature,
        max_tokens=DEFAULT_MAX_TOKENS,
        stop_sequences=DEFAULT_STOP_SEQUENCES,
    )

    cactus_destroy(model)

    try:
        raw = json.loads(raw_str)
    except json.JSONDecodeError:
        return {
            "function_calls": [],
            "total_time_ms": 0,
            "confidence": 0,
            "raw_text": raw_str,
        }

    return {
        "function_calls": raw.get("function_calls", []),
        "total_time_ms": raw.get("total_time_ms", 0),
        "confidence": raw.get("confidence", 0),
        "raw_text": raw.get("response"),
    }


def _extract_latest_user_text(messages):
    for message in reversed(messages):
        if message.get("role") == "user":
            return message.get("content", "")
    return ""


def _estimate_expected_action_count(messages):
    latest_user_text = _extract_latest_user_text(messages).lower()
    if not latest_user_text:
        return 1
    for separator in MULTI_INTENT_SEPARATORS:
        if separator in latest_user_text:
            return 2
    return 1


def _is_argument_type_valid(value, schema_type):
    if not schema_type:
        return True
    normalized_type = str(schema_type).lower()
    if normalized_type == "string":
        return isinstance(value, str)
    if normalized_type == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if normalized_type == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if normalized_type == "boolean":
        return isinstance(value, bool)
    if normalized_type == "object":
        return isinstance(value, dict)
    if normalized_type == "array":
        return isinstance(value, list)
    return True


def _has_obvious_invalid_time_values(arguments):
    minutes = arguments.get("minutes")
    if isinstance(minutes, int) and minutes < 0:
        return True

    hour = arguments.get("hour")
    if isinstance(hour, int) and (hour < 0 or hour > 23):
        return True

    minute = arguments.get("minute")
    if isinstance(minute, int) and (minute < 0 or minute > 59):
        return True

    return False


def _validate_local_function_calls(function_calls, tools, confidence, confidence_threshold, expected_action_count):
    if not function_calls:
        return False, "no_function_calls"

    if confidence < confidence_threshold:
        return False, "low_confidence"

    tools_by_name = {tool.get("name"): tool for tool in tools}

    for function_call in function_calls:
        if not isinstance(function_call, dict):
            return False, "invalid_arguments"

        function_name = function_call.get("name")
        if not isinstance(function_name, str):
            return False, "invalid_arguments"

        tool_definition = tools_by_name.get(function_name)
        if tool_definition is None:
            return False, "unknown_tool"

        arguments = function_call.get("arguments")
        if not isinstance(arguments, dict):
            return False, "invalid_arguments"

        parameters = tool_definition.get("parameters", {}) if isinstance(tool_definition, dict) else {}
        required_arguments = parameters.get("required", []) if isinstance(parameters, dict) else []
        properties = parameters.get("properties", {}) if isinstance(parameters, dict) else {}

        for required_argument in required_arguments:
            if required_argument not in arguments:
                return False, "invalid_arguments"

        if isinstance(properties, dict):
            for argument_name, argument_value in arguments.items():
                property_schema = properties.get(argument_name, {})
                schema_type = property_schema.get("type") if isinstance(property_schema, dict) else None
                if not _is_argument_type_valid(argument_value, schema_type):
                    return False, "invalid_arguments"

        if _has_obvious_invalid_time_values(arguments):
            return False, "invalid_arguments"

    if expected_action_count > 1 and len(function_calls) < expected_action_count:
        return False, "multi_intent_incomplete"

    return True, None


def _run_cloud(messages, tools, system_instruction=None):
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

    if system_instruction:
        contents = [f"{system_instruction}\n\nUser request: {_extract_latest_user_text(messages)}"]
    else:
        contents = [m["content"] for m in messages if m["role"] == "user"]

    configured_model = os.environ.get("GEMINI_MODEL", "").strip()
    model_candidates = []
    if configured_model:
        model_candidates.append(configured_model)
    for fallback_model in DEFAULT_CLOUD_MODEL_FALLBACKS:
        if fallback_model not in model_candidates:
            model_candidates.append(fallback_model)

    start_time = time.time()
    gemini_response = None
    last_error = None
    for model_name in model_candidates:
        try:
            gemini_response = client.models.generate_content(
                model=model_name,
                contents=contents,
                config=types.GenerateContentConfig(
                    tools=gemini_tools,
                    temperature=DEFAULT_CLOUD_TEMPERATURE,
                ),
            )
            break
        except Exception as error:
            last_error = error
            continue

    if gemini_response is None:
        raise last_error if last_error is not None else RuntimeError("Gemini cloud call failed")

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


def generate_cactus(messages, tools):
    """Run function calling on-device via FunctionGemma + Cactus."""
    return _run_cactus(messages, tools, DEFAULT_SYSTEM_INSTRUCTION, temperature=DEFAULT_LOCAL_TEMPERATURE)


def generate_cloud(messages, tools):
    """Run function calling via Gemini Cloud API."""
    return _run_cloud(messages, tools)


def generate_hybrid(messages, tools, confidence_threshold=0.99):
    """Hybrid inference: validate local calls, repair no-call outputs, then fallback to cloud."""
    expected_action_count = _estimate_expected_action_count(messages)
    threshold = confidence_threshold if confidence_threshold is not None else DEFAULT_LOCAL_CONFIDENCE_THRESHOLD

    local = generate_cactus(messages, tools)
    is_local_accepted, fallback_reason = _validate_local_function_calls(
        local.get("function_calls", []),
        tools,
        local.get("confidence", 0),
        threshold,
        expected_action_count,
    )

    if is_local_accepted:
        local["source"] = "on-device"
        return local

    local_time_ms = local.get("total_time_ms", 0)
    local_confidence = local.get("confidence", 0)

    if fallback_reason == "no_function_calls":
        repaired_local = _run_cactus(
            messages,
            tools,
            REPAIR_SYSTEM_INSTRUCTION,
            temperature=DEFAULT_LOCAL_TEMPERATURE,
        )
        local_time_ms += repaired_local.get("total_time_ms", 0)
        is_repair_accepted, repair_reason = _validate_local_function_calls(
            repaired_local.get("function_calls", []),
            tools,
            repaired_local.get("confidence", 0),
            threshold,
            expected_action_count,
        )
        if is_repair_accepted:
            repaired_local["source"] = "on-device"
            repaired_local["total_time_ms"] = local_time_ms
            repaired_local["used_repair_pass"] = True
            return repaired_local
        fallback_reason = repair_reason or fallback_reason

    cloud = generate_cloud(messages, tools)

    if expected_action_count > 1 and len(cloud.get("function_calls", [])) < expected_action_count:
        repaired_cloud = _run_cloud(
            messages,
            tools,
            CLOUD_MULTI_INTENT_REPAIR_SYSTEM_INSTRUCTION,
        )
        repaired_cloud["total_time_ms"] += cloud.get("total_time_ms", 0)
        if len(repaired_cloud.get("function_calls", [])) > len(cloud.get("function_calls", [])):
            cloud = repaired_cloud
            cloud["used_cloud_repair"] = True

    cloud["source"] = "cloud (fallback)"
    cloud["local_confidence"] = local_confidence
    cloud["fallback_reason"] = fallback_reason
    cloud["total_time_ms"] += local_time_ms
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
