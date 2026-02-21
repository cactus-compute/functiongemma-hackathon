
import sys
sys.path.insert(0, "cactus/python/src")
functiongemma_path = "cactus/weights/functiongemma-270m-it"

import json, math, os, re, time
from concurrent.futures import ThreadPoolExecutor
from cactus import cactus_init, cactus_complete, cactus_destroy
from google import genai
from query_decompose_regex import decompose_query as _decompose_query_regex
from google.genai import types


def generate_cactus(messages, tools):
    """Run function calling on-device via FunctionGemma + Cactus with nucleus sampling."""
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
        temperature=0.2,
        top_p=0.95,
        top_k=50,
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


_CATEGORY_MAP = [
    ("weather", 0), ("forecast", 0), ("location", 0),
    ("play", 1),
    ("alarm", 2), ("timer", 3), ("reminder", 4),
    ("message", 5), ("contact", 5),
    ("search", 6), ("note", 6),
]


def _load_svm_gate(path="svm_gate.json"):
    with open(path) as f:
        return json.load(f)


_SVM_GATE = _load_svm_gate()


def _extract_features(user_text, tools):
    """Return [intent_score, tool_count, arg_difficulty, category, single_tool, explicit_value]."""
    segments = re.split(r"\band\b|\bthen\b|\balso\b|\bafter\b|[,;]", user_text.lower())
    segments = [s.strip() for s in segments if len(s.strip()) >= 3]
    intent_score = max(0.0, min((len(segments) - 1) / 2.0, 1.0))

    difficulties = []
    for tool in tools:
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
    for tool in tools:
        combined = f"{tool.get('name', '').lower()} {tool.get('description', '').lower()}"
        matched = next((cat for pat, cat in _CATEGORY_MAP if pat in combined), None)
        if matched is not None:
            categories.append(matched)
    category = max(categories) if categories else 7

    has_proper_noun = bool(re.search(r"\b[A-Z][a-z]+\b", user_text))
    has_numeric = bool(re.search(r"\b\d+(?:[:.]\d+)?\b", user_text))
    has_quoted = bool(re.search(r"['\"][^'\"]+['\"]", user_text))
    explicit_value = int(has_proper_noun or has_numeric or has_quoted)

    return [
        intent_score,
        float(len(tools)),
        arg_difficulty,
        float(category),
        float(int(len(tools) == 1)),
        float(explicit_value),
    ]


def _svm_predict_local(features, gate=_SVM_GATE):
    """Return True when SVM predicts the query can be handled locally (label=1)."""
    mean = gate["mean"]
    scale = gate["scale"]
    svs = gate["support_vectors"]
    dual = gate["dual_coef"][0]
    intercept = gate["intercept"][0]
    gamma = gate["gamma"]

    x = [(f - m) / (s if s != 0 else 1.0) for f, m, s in zip(features, mean, scale)]

    decision = intercept
    for coef, sv in zip(dual, svs):
        sq = sum((xi - svi) ** 2 for xi, svi in zip(x, sv))
        decision += coef * math.exp(-gamma * sq)

    return decision > 0


def _decompose_query(user_text):
    """Use regex to split a compound query into sub-queries."""
    return _decompose_query_regex(user_text)


def _route_subquery(user_text, tools):
    """SVM gate: predict=1 → local cactus, predict=0 → cloud."""
    features = _extract_features(user_text, tools)
    msgs = [{"role": "user", "content": user_text}]
    if _svm_predict_local(features):
        result = generate_cactus(msgs, tools)
        result["source"] = "on-device"
    else:
        result = generate_cloud(msgs, tools)
        result["source"] = "cloud"
    return result


def generate_hybrid(messages, tools):
    """Decompose via FunctionGemma, then SVM-route each sub-query."""
    user_text = next(
        (m["content"] for m in reversed(messages) if m["role"] == "user"), ""
    )

    start = time.time()
    sub_queries = _decompose_query(user_text)
    decompose_ms = (time.time() - start) * 1000

    if not sub_queries or len(sub_queries) <= 1:
        query = sub_queries[0] if sub_queries else user_text
        result = _route_subquery(query, tools)
        result["total_time_ms"] += decompose_ms
        return result

    fan_start = time.time()
    with ThreadPoolExecutor(max_workers=len(sub_queries)) as pool:
        results = list(pool.map(lambda sq: _route_subquery(sq, tools), sub_queries))
    fan_ms = (time.time() - fan_start) * 1000

    all_calls = []
    seen = set()
    for r in results:
        for fc in r.get("function_calls", []):
            key = (fc.get("name"), json.dumps(fc.get("arguments", {}), sort_keys=True))
            if key not in seen:
                seen.add(key)
                all_calls.append(fc)

    any_cloud = any(r.get("source") == "cloud" for r in results)
    return {
        "function_calls": all_calls,
        "total_time_ms": decompose_ms + fan_ms,
        "confidence": min((r.get("confidence", 0) for r in results), default=0),
        "source": "hybrid" if any_cloud else "on-device",
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
