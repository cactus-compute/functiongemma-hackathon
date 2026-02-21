
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
    # Offline-trained SVM parameters (balanced class_weight + 4 extra positives).
    MEAN = [0.08695652173913043, 2.3043478260869565, 0.4739130434782608, 2.260869565217391, 0.34782608695652173, 0.9130434782608695]
    SCALE = [0.18951734537133363, 1.158514138649933, 0.22109881974071516, 2.1713027807276126, 0.47628048478710105, 0.2817713347133852]
    SV = [
        [-0.4588314677411235, -1.1258799375612023, 1.4748471154398053, 0.34040873587189124, 1.369306393762915, 0.308606699924184],
        [-0.4588314677411235, -1.1258799375612023, 1.4748471154398053, -0.12014425971949091, 1.369306393762915, 0.308606699924184],
        [-0.4588314677411235, -1.1258799375612023, 0.5702742179700581, 1.7220677226460377, 1.369306393762915, 0.308606699924184],
        [-0.4588314677411235, 0.6004693000326412, -0.33429867949968867, -0.5806972553108731, -0.7302967433402213, 0.308606699924184],
        [-0.4588314677411235, 1.463643918829563, 1.4748471154398053, 0.8009617314632734, -0.7302967433402213, -3.2403703492039297],
        [-0.4588314677411235, 0.6004693000326412, 1.4748471154398053, 0.34040873587189124, -0.7302967433402213, -3.2403703492039297],
        [-0.4588314677411235, 1.463643918829563, 0.5702742179700581, 1.7220677226460377, -0.7302967433402213, 0.308606699924184],
        [-0.4588314677411235, 1.463643918829563, 1.0225606667049314, 1.2615147270546556, -0.7302967433402213, 0.308606699924184],
        [2.179449471770337, -0.26270531876428055, 0.11798776923518471, -0.12014425971949091, -0.7302967433402213, 0.308606699924184],
        [-0.4588314677411235, -1.1258799375612023, -1.2388715769694356, -1.0412502509022552, 1.369306393762915, 0.308606699924184],
        [-0.4588314677411235, -1.1258799375612023, -1.2388715769694356, -1.0412502509022552, 1.369306393762915, 0.308606699924184],
        [-0.4588314677411235, 0.6004693000326412, -0.33429867949968867, -0.5806972553108731, -0.7302967433402213, 0.308606699924184],
        [-0.4588314677411235, 1.463643918829563, -1.2388715769694356, -1.0412502509022552, -0.7302967433402213, 0.308606699924184],
        [-0.4588314677411235, -1.1258799375612023, -0.33429867949968867, -0.5806972553108731, 1.369306393762915, 0.308606699924184],
        [-0.4588314677411235, 0.6004693000326412, -0.33429867949968867, -1.0412502509022552, -0.7302967433402213, 0.308606699924184],
    ]
    DUAL_COEF = [[-0.018830803763000666, -0.8846153846153846, -0.49125934086967427, -0.8846153846153846, -0.18455697756276257, -0.3332584673601857, -0.16909775723866555, -0.5605699994469142, -0.8207074003547666, 0.0007820782857653614, 0.5394615840054955, 1.15, 0.7559134964678993, 1.15, 0.7513543570675789]]
    INTERCEPT = [-0.482540305745601]
    GAMMA = 0.1666666666666667

    SVM_DECISION_THRESHOLD = -0.3

    CATEGORY_CONFIDENCE_THRESHOLD = {
        0: 0.45,
        1: 0.72,
        2: 0.90,
        3: 0.90,
        4: 0.88,
        5: 0.90,
        6: 0.88,
        7: 0.85,
    }

    CATEGORY_MAP = [
        ("weather", 0), ("forecast", 0), ("location", 0),
        ("play", 1),
        ("alarm", 2), ("timer", 3), ("reminder", 4),
        ("message", 5), ("contact", 5),
        ("search", 6), ("note", 6),
    ]

    def get_last_user_text(msgs):
        for message in reversed(msgs):
            if message.get("role") == "user":
                return message.get("content", "")
        return ""

    def extract_features(msgs, available_tools):
        last_user_text = get_last_user_text(msgs)

        segments = re.split(r"\band\b|\bthen\b|\balso\b|\bafter\b|[,;]", last_user_text.lower())
        segments = [s.strip() for s in segments if len(s.strip()) >= 3]
        intent_score = max(0.0, min((len(segments) - 1) / 2.0, 1.0))

        tool_count = len(available_tools)

        difficulties = []
        for tool in available_tools:
            for arg in tool.get("parameters", {}).get("required", []):
                key = arg.lower()
                if any(t in key for t in ("time", "duration", "hour", "minute", "when")):
                    difficulties.append(0.8)
                elif any(t in key for t in ("location", "city", "place")):
                    difficulties.append(0.2)
                elif any(t in key for t in ("contact", "person", "name", "recipient")):
                    difficulties.append(0.7)
                elif any(t in key for t in ("query", "search", "term", "keyword")):
                    difficulties.append(0.6)
                else:
                    difficulties.append(0.4)
        arg_difficulty = sum(difficulties) / len(difficulties) if difficulties else 0.3

        categories = []
        for tool in available_tools:
            combined = f"{tool.get('name', '').lower()} {tool.get('description', '').lower()}"
            matched = None
            for pattern, cat in CATEGORY_MAP:
                if pattern in combined:
                    matched = cat
                    break
            if matched is not None:
                categories.append(matched)
        category = max(categories) if categories else 7

        single_tool = int(len(available_tools) == 1)

        has_proper_noun = bool(re.search(r"\b[A-Z][a-z]+\b", last_user_text))
        has_numeric = bool(re.search(r"\b\d+(?:[:.]\d+)?\b", last_user_text))
        has_quoted = bool(re.search(r"['\"][^'\"]+['\"]", last_user_text))
        explicit_value = int(has_proper_noun or has_numeric or has_quoted)

        return [intent_score, float(tool_count), arg_difficulty, float(category), float(single_tool), float(explicit_value)]

    def svm_predict(features):
        x = []
        for i, value in enumerate(features):
            denom = SCALE[i] if SCALE[i] != 0 else 1.0
            x.append((value - MEAN[i]) / denom)

        decision = INTERCEPT[0]
        for coef, sv in zip(DUAL_COEF[0], SV):
            sq = 0.0
            for xi, svi in zip(x, sv):
                diff = svi - xi
                sq += diff * diff
            kernel = pow(2.718281828459045, -GAMMA * sq)
            decision += coef * kernel
        return decision

    features = extract_features(messages, tools)
    task_category = int(features[3])

    decision_score = svm_predict(features)
    if decision_score <= SVM_DECISION_THRESHOLD:
        cloud = generate_cloud(messages, tools)
        cloud["source"] = "cloud (svm skip)"
        cloud["local_confidence"] = None
        return cloud

    local = generate_cactus(messages, tools)

    tool_names = {t["name"] for t in tools}
    if any(c.get("name") not in tool_names for c in local.get("function_calls", [])):
        cloud = generate_cloud(messages, tools)
        cloud["source"] = "cloud (invalid local)"
        cloud["local_confidence"] = local["confidence"]
        cloud["total_time_ms"] += local["total_time_ms"]
        return cloud

    threshold = CATEGORY_CONFIDENCE_THRESHOLD.get(task_category, 0.85)
    if local["confidence"] >= threshold:
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
