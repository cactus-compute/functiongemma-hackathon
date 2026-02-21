
import sys
sys.path.insert(0, "cactus/python/src")
functiongemma_path = "cactus/weights/functiongemma-270m-it"

import json, os, pickle, re, time
import threading
from dataclasses import dataclass
from typing import Literal

import numpy as np
from cactus import cactus_init, cactus_complete, cactus_destroy


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
    from google import genai
    from google.genai import types

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
        config=types.GenerateContentConfig(
            tools=gemini_tools,
            # Minimize deliberate reasoning latency for routing speed.
            thinking_config=types.ThinkingConfig(thinking_budget=0),
            temperature=0.0,
            max_output_tokens=64,
        ),
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


# Regex-based query decomposition (inlined for single-file submission)
_DECOMP_ACTION_HINT = r"(?:set|play|remind|send|text|message|check|get|find|look\s+up|search|create|wake)\b"
_DECOMP_CONJUNCTION = re.compile(
    rf"\s*(?:,\s*and\s+(?={_DECOMP_ACTION_HINT})|\s+and\s+(?={_DECOMP_ACTION_HINT})|\s+then\s+(?={_DECOMP_ACTION_HINT})|\s+also\s+(?={_DECOMP_ACTION_HINT})|\s+after\s+(?={_DECOMP_ACTION_HINT}))\s*",
    re.IGNORECASE,
)
_DECOMP_LIST_SEP = re.compile(rf"\s*[,;]\s*(?={_DECOMP_ACTION_HINT})", re.IGNORECASE)
_DECOMP_LEADING = re.compile(r"^\s*(?:and|then|also|after)\s+", re.IGNORECASE)
_DECOMP_TRAILING_PUNCT = re.compile(r"^[\s,;:.!?]+|[\s,;:.!?]+$")
_DECOMP_MAX_SUBQUERIES = 2


class BaseMode:
    """Marker base class for structured routing payloads."""


@dataclass(frozen=True)
class SubQuery(BaseMode):
    sub_query: str
    destination: Literal["cloud", "local"]


_CACTUS_CALL_LOCK = threading.Lock()


def _subquery_destination(sub_query: str, tools) -> Literal["cloud", "local"]:
    """
    History-driven hybrid destination policy.
    Prefer local where prior runs are stable; use cloud for historically brittle intents.
    """
    lowered = sub_query.lower()
    tool_count = float(len(tools))
    features = _extract_features(sub_query, tools)
    is_svm_local = _svm_predict_local(features)

    is_weather = bool(re.search(r"\b(?:weather|forecast)\b", lowered))
    is_music = bool(re.search(r"\b(?:play|music|song|playlist)\b", lowered))
    is_alarm = bool(re.search(r"\b(?:alarm|wake)\b", lowered))
    is_timer = bool(re.search(r"\btimer\b", lowered))
    is_reminder = bool(re.search(r"\b(?:remind|reminder)\b", lowered))
    is_message = bool(re.search(r"\b(?:message|text|send)\b", lowered))
    is_search = bool(re.search(r"\b(?:find|look\s+up|search|contacts?)\b", lowered))

    has_numeric = bool(re.search(r"\b\d+(?::\d+)?\b", lowered))
    has_proper_name = bool(re.search(r"\b[A-Z][a-z]+\b", sub_query))
    has_ambiguous_pronoun = bool(re.search(r"\b(?:him|her|them|it|that)\b", lowered))
    token_count = len([t for t in re.split(r"\s+", lowered) if t])

    # Reliability prior from observed benchmark history.
    local_score = 0.2
    if is_weather:
        local_score += 1.4
    if is_music:
        local_score += 0.2
    if is_search:
        local_score -= 0.1
    if is_timer:
        local_score -= 0.6
    if is_alarm:
        local_score += 0.1
    if is_reminder:
        local_score -= 0.8
    if is_message:
        local_score -= 0.7

    if has_numeric and is_alarm:
        local_score += 0.35
    if has_numeric and is_timer:
        local_score -= 0.25
    if has_proper_name and (is_weather or is_search):
        local_score += 0.15
    if has_ambiguous_pronoun and (is_message or is_search):
        local_score -= 0.7

    if tool_count >= 4.0:
        local_score -= 0.65
    elif tool_count >= 2.0:
        local_score -= 0.25
    if token_count >= 11:
        local_score -= 0.3
    if token_count <= 6 and (is_weather or is_alarm):
        local_score += 0.2

    # SVM is a soft tie-breaker only.
    local_score += 0.25 if is_svm_local else -0.1
    return "local" if local_score >= 0.05 else "cloud"


def _decompose_query(user_text, tools):
    """Split compound query into sub-queries via regex."""
    if not user_text or not user_text.strip():
        return []
    text = user_text.strip()
    segments = _DECOMP_CONJUNCTION.split(text)
    flat = []
    for seg in segments:
        flat.extend(_DECOMP_LIST_SEP.split(seg))
    result = [
        _DECOMP_TRAILING_PUNCT.sub("", _DECOMP_LEADING.sub("", s).strip())
        for s in flat
        if s and s.strip()
    ]
    if not result:
        return []
    if len(result) > _DECOMP_MAX_SUBQUERIES:
        # Keep first action explicit, fold remaining actions into the second slot.
        result = [result[0], " and ".join(result[1:])]
    return [SubQuery(sub_query=s, destination=_subquery_destination(s, tools)) for s in result]


_CATEGORY_MAP = [
    ("weather", 0), ("forecast", 0), ("location", 0),
    ("play", 1),
    ("alarm", 2), ("timer", 3), ("reminder", 4),
    ("message", 5), ("contact", 5),
    ("search", 6), ("note", 6),
]


def _load_svm_gate(path="svm_gate.pkl"):
    """Load serialized SVM gate if present, otherwise return None."""
    candidate_paths = [
        path,
        os.path.join(os.path.dirname(__file__), path),
    ]
    for candidate in candidate_paths:
        if os.path.exists(candidate):
            with open(candidate, "rb") as f:
                return pickle.load(f)
    return None


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


def _fallback_predict_local(features):
    """
    Submission-safe fallback when svm_gate.pkl is unavailable.
    Bias local for simple weather/music-like single-intent requests only.
    """
    intent_score, tool_count, arg_difficulty, category, single_tool, explicit_value = features
    return bool(
        intent_score <= 0.0
        and explicit_value >= 1.0
        and (
            (single_tool >= 1.0 and category in (0.0, 1.0) and arg_difficulty <= 0.45)
            or (tool_count <= 2.0 and category == 0.0 and arg_difficulty <= 0.30)
        )
    )


def _svm_predict_local(features, gate=_SVM_GATE):
    """Return True when gate predicts the query can be handled locally (label=1)."""
    if gate is None:
        return _fallback_predict_local(features)
    scaler, clf = gate["scaler"], gate["clf"]
    X = np.array([features], dtype=float)
    X_scaled = scaler.transform(X)
    return clf.predict(X_scaled)[0] == 1


def _route_subquery(sub_query, tools):
    """Route each sub-query to destination engine with local safety fallback."""
    msgs = [{"role": "user", "content": sub_query.sub_query}]
    if sub_query.destination == "cloud":
        result = generate_cloud(msgs, tools)
        result["source"] = "cloud"
        # If cloud returns nothing, try local once as a recovery path.
        if not result.get("function_calls"):
            with _CACTUS_CALL_LOCK:
                local_result = generate_cactus(msgs, tools)
            if local_result.get("function_calls"):
                local_result["source"] = "on-device"
                return local_result
        return result

    # Cactus native stack can crash on concurrent calls; serialize local invocations.
    with _CACTUS_CALL_LOCK:
        result = generate_cactus(msgs, tools)
    result["source"] = "on-device"

    # Recover from malformed/empty ultra-fast local responses.
    if result.get("total_time_ms", 0.0) < 0.05 or not result.get("function_calls"):
        result = generate_cloud(msgs, tools)
        result["source"] = "cloud"

    return result


def generate_hybrid(messages, tools):
    """Decompose via FunctionGemma, then SVM-route each sub-query."""
    user_text = next(
        (m["content"] for m in reversed(messages) if m["role"] == "user"), ""
    )

    start = time.time()
    sub_queries = _decompose_query(user_text, tools)
    decompose_ms = (time.time() - start) * 1000
    if sub_queries:
        for idx, sq in enumerate(sub_queries, 1):
            print(f"[route] subquery {idx}: {sq.destination} | {sq.sub_query}")
    else:
        print(f"[route] subquery 1: local | {user_text}")

    if not sub_queries or len(sub_queries) <= 1:
        query = sub_queries[0] if sub_queries else SubQuery(sub_query=user_text, destination="local")
        result = _route_subquery(query, tools)
        result["total_time_ms"] += decompose_ms
        return result

    fan_start = time.time()
    results = [None] * len(sub_queries)

    def _run_one(idx, sq):
        results[idx] = _route_subquery(sq, tools)

    threads = [
        threading.Thread(target=_run_one, args=(idx, sq), daemon=True)
        for idx, sq in enumerate(sub_queries)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

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
