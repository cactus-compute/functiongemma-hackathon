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
      2) Score local calls with generic schema + grounding checks (tool-agnostic).
      3) Accept strong local calls; otherwise fallback to cloud.
    """
    user_text = " ".join(
        m.get("content", "") for m in messages if m.get("role") == "user"
    )
    user_text_l = user_text.lower()
    user_tokens = set(re.findall(r"[a-z0-9]+", user_text_l))
    tool_map = {t["name"]: t for t in tools}

    def _coerce_value(value, schema):
        if not isinstance(schema, dict):
            return value
        type_name = str(schema.get("type", "")).lower()

        if type_name == "integer":
            if isinstance(value, bool):
                return value
            if isinstance(value, int):
                return value
            if isinstance(value, float) and value.is_integer():
                return int(value)
            if isinstance(value, str) and re.fullmatch(r"[+-]?\d+", value.strip()):
                return int(value.strip())
            return value

        if type_name == "number":
            if isinstance(value, bool):
                return value
            if isinstance(value, (int, float)):
                return value
            if isinstance(value, str):
                try:
                    return float(value.strip())
                except ValueError:
                    return value
            return value

        if type_name == "boolean":
            if isinstance(value, bool):
                return value
            if isinstance(value, str):
                v = value.strip().lower()
                if v in {"true", "yes", "1", "on"}:
                    return True
                if v in {"false", "no", "0", "off"}:
                    return False
            return value

        if type_name == "string":
            if isinstance(value, str):
                return value.strip()
            return str(value)

        if type_name == "array" and isinstance(value, list):
            item_schema = schema.get("items", {})
            return [_coerce_value(v, item_schema) for v in value]

        return value

    def _value_matches_type(value, schema):
        if not isinstance(schema, dict):
            return True
        type_name = str(schema.get("type", "")).lower()
        if not type_name:
            return True
        if type_name == "integer":
            return isinstance(value, int) and not isinstance(value, bool)
        if type_name == "number":
            return isinstance(value, (int, float)) and not isinstance(value, bool)
        if type_name == "boolean":
            return isinstance(value, bool)
        if type_name == "string":
            return isinstance(value, str)
        if type_name == "array":
            if not isinstance(value, list):
                return False
            item_schema = schema.get("items", {})
            return all(_value_matches_type(v, item_schema) for v in value)
        if type_name == "object":
            return isinstance(value, dict)
        return True

    def _tokenize_tool(tool):
        parts = [
            tool.get("name", ""),
            tool.get("description", ""),
        ]
        params = tool.get("parameters", {}).get("properties", {})
        for p_name, p_schema in params.items():
            parts.append(str(p_name))
            if isinstance(p_schema, dict):
                parts.append(str(p_schema.get("description", "")))
        raw = " ".join(parts).replace("_", " ").lower()
        tokens = {t for t in re.findall(r"[a-z0-9]+", raw) if len(t) > 2}
        return tokens

    tool_tokens = {name: _tokenize_tool(tool) for name, tool in tool_map.items()}

    def _tool_relevance(name):
        tokens = tool_tokens.get(name, set())
        if not tokens:
            return 0.0
        overlap = len(tokens & user_tokens)
        return overlap / max(1, len(tokens))

    def _coerce_call(call):
        name = call.get("name")
        args = call.get("arguments", {})
        if not isinstance(args, dict):
            args = {}
        out = {"name": name, "arguments": dict(args)}
        tool = tool_map.get(name)
        if not tool:
            return out
        props = tool.get("parameters", {}).get("properties", {})
        for key, value in list(out["arguments"].items()):
            if key in props:
                out["arguments"][key] = _coerce_value(value, props[key])
        return out

    def _schema_stats(call):
        name = call.get("name")
        args = call.get("arguments", {})
        if name not in tool_map or not isinstance(args, dict):
            return {
                "valid": False,
                "required_coverage": 0.0,
                "type_pass": 0.0,
                "unknown_arg_ratio": 1.0,
            }

        tool = tool_map[name]
        params = tool.get("parameters", {})
        required = params.get("required", []) or []
        props = params.get("properties", {}) or {}

        required_present = sum(1 for key in required if key in args)
        required_coverage = required_present / max(1, len(required))

        checked = 0
        passed = 0
        unknown = 0
        for key, value in args.items():
            if key not in props:
                unknown += 1
                continue
            checked += 1
            if _value_matches_type(value, props[key]):
                passed += 1
        type_pass = passed / max(1, checked)
        unknown_ratio = unknown / max(1, len(args))

        valid = required_coverage >= 1.0 and type_pass >= 1.0 and unknown_ratio <= 0.4
        return {
            "valid": valid,
            "required_coverage": required_coverage,
            "type_pass": type_pass,
            "unknown_arg_ratio": unknown_ratio,
        }

    def _argument_grounding_score(call):
        args = call.get("arguments", {})
        if not isinstance(args, dict) or not args:
            return 0.0
        hit = 0.0
        total = 0
        for value in args.values():
            if isinstance(value, str):
                total += 1
                val = value.strip().lower()
                if not val:
                    continue
                if val in user_text_l:
                    hit += 1.0
                    continue
                val_tokens = set(re.findall(r"[a-z0-9]+", val))
                if not val_tokens:
                    continue
                overlap = len(val_tokens & user_tokens) / len(val_tokens)
                hit += overlap
            elif isinstance(value, (int, float)) and not isinstance(value, bool):
                total += 1
                if str(value) in user_text_l:
                    hit += 1.0
        return hit / max(1, total)

    def _estimate_action_count(text):
        if not text.strip():
            return 1
        separators = re.findall(r"\b(?:and|then|also)\b|,", text.lower())
        return max(1, min(4, len(separators) + 1))

    def _expected_tool_names():
        """Tool names that are semantically relevant to the user text (intent coverage)."""
        return {
            name for name in tool_map
            if _tool_relevance(name) >= 0.15
        }

    def _intent_coverage_ok(calls):
        """True if the set of calls covers every expected intent (tool)."""
        expected = _expected_tool_names()
        if not expected:
            return True
        called = {c.get("name") for c in calls if c.get("name") in tool_map}
        return expected <= called

    def _dedupe_calls(calls):
        out = []
        seen = set()
        for call in calls:
            key = json.dumps(
                {"name": call.get("name"), "arguments": call.get("arguments", {})},
                sort_keys=True,
            )
            if key in seen:
                continue
            seen.add(key)
            out.append(call)
        return out

    action_count_hint = _estimate_action_count(user_text)

    def _score_candidate(calls, conf):
        call_scores = []
        for call in calls:
            stats = _schema_stats(call)
            relevance = _tool_relevance(call.get("name"))
            grounding = _argument_grounding_score(call)
            score = (
                0.45 * stats["required_coverage"]
                + 0.30 * stats["type_pass"]
                + 0.15 * grounding
                + 0.10 * relevance
                - 0.20 * stats["unknown_arg_ratio"]
            )
            call_scores.append(
                {
                    "call": call,
                    "stats": stats,
                    "score": max(0.0, min(1.0, score)),
                }
            )

        strong_calls = [
            c for c in call_scores if c["stats"]["valid"] and c["score"] >= 0.55
        ]
        strong_calls = sorted(strong_calls, key=lambda x: x["score"], reverse=True)
        strong_calls = strong_calls[: action_count_hint + 1]
        selected_calls = [c["call"] for c in strong_calls]

        all_schema_valid = bool(calls) and all(c["stats"]["valid"] for c in call_scores)
        mean_quality = (
            sum(c["score"] for c in call_scores) / len(call_scores)
            if call_scores
            else 0.0
        )
        action_ratio = min(1.0, len(calls) / max(1, action_count_hint))
        reliability = 0.50 * mean_quality + 0.30 * conf + 0.20 * action_ratio
        return {
            "strong_calls": strong_calls,
            "selected_calls": selected_calls,
            "all_schema_valid": all_schema_valid,
            "mean_quality": mean_quality,
            "reliability": reliability,
        }

    local = generate_cactus(messages, tools)
    local_calls_raw = [_coerce_call(c) for c in local.get("function_calls", [])]
    local_calls = [c for c in local_calls_raw if c.get("name") in tool_map]
    local_calls = _dedupe_calls(local_calls)
    local["function_calls"] = local_calls
    local_conf = float(local.get("confidence", 0.0) or 0.0)
    local_eval = _score_candidate(local_calls, local_conf)

    if (not local_eval["all_schema_valid"]) or local_eval["mean_quality"] < 0.58:
        refine_messages = list(messages) + [
            {
                "role": "user",
                "content": (
                    "Use only the provided tools. Return calls only for explicit intents in this request. "
                    "Every returned call must include all required arguments with correctly typed values."
                ),
            }
        ]
        local_refine = generate_cactus(refine_messages, tools)
        refine_calls_raw = [
            _coerce_call(c) for c in local_refine.get("function_calls", [])
        ]
        refine_calls = [c for c in refine_calls_raw if c.get("name") in tool_map]
        refine_calls = _dedupe_calls(refine_calls)
        refine_conf = float(local_refine.get("confidence", 0.0) or 0.0)
        refine_eval = _score_candidate(refine_calls, refine_conf)

        if refine_eval["reliability"] > local_eval["reliability"]:
            local = local_refine
            local_calls = refine_calls
            local_conf = refine_conf
            local["function_calls"] = local_calls
            local_eval = refine_eval

    selected_local_calls = local_eval["selected_calls"]
    all_schema_valid = local_eval["all_schema_valid"]
    mean_quality = local_eval["mean_quality"]
    reliability = local_eval["reliability"]

    # Stricter for multi-intent: higher confidence bar and require call count + intent coverage.
    dyn_thr = min(confidence_threshold, 0.72 + 0.08 * max(0, action_count_hint - 1))
    multi_intent = action_count_hint >= 2
    call_count_ok = (
        len(selected_local_calls) >= action_count_hint
        if multi_intent
        else True
    )
    # Multi-intent: every expected tool must be called. Single-intent: the call must match an expected tool.
    intent_covered = (
        _intent_coverage_ok(selected_local_calls)
        if multi_intent
        else (not _expected_tool_names() or any(c.get("name") in _expected_tool_names() for c in selected_local_calls))
    )
    should_accept_local = (
        bool(local_calls)
        and all_schema_valid
        and call_count_ok
        and intent_covered
        and (local_conf >= dyn_thr or (reliability >= 0.70 and local_conf >= 0.45))
    )

    if should_accept_local:
        local["source"] = "on-device"
        return local

    # If raw local output is noisy but we still have high-confidence valid calls, keep only strong calls.
    if selected_local_calls:
        selected_stats = [_schema_stats(c) for c in selected_local_calls]
        if all(s["valid"] for s in selected_stats):
            selected_quality = sum(
                c["score"] for c in local_eval["strong_calls"]
            ) / len(local_eval["strong_calls"])
            repair_call_count_ok = (
                len(selected_local_calls) >= action_count_hint
                if multi_intent
                else True
            )
            repair_intent_ok = _intent_coverage_ok(selected_local_calls) if multi_intent else True
            if (
                selected_quality >= 0.72
                and local_conf >= 0.40
                and repair_call_count_ok
                and repair_intent_ok
            ):
                return {
                    "function_calls": selected_local_calls,
                    "total_time_ms": local.get("total_time_ms", 0),
                    "confidence": local_conf,
                    "source": "on-device",
                    "repair_used": True,
                    "fallback_reason": {
                        "all_schema_valid": all_schema_valid,
                        "mean_quality": mean_quality,
                        "local_confidence": local_conf,
                        "dynamic_threshold": dyn_thr,
                    },
                }

    try:
        cloud = generate_cloud(messages, tools)
        cloud_calls = [_coerce_call(c) for c in cloud.get("function_calls", [])]
        cloud["function_calls"] = [c for c in cloud_calls if c.get("name") in tool_map]
        cloud["source"] = "cloud (fallback)"
        cloud["local_confidence"] = local_conf
        cloud["total_time_ms"] += local.get("total_time_ms", 0)
        cloud["fallback_reason"] = {
            "all_schema_valid": all_schema_valid,
            "mean_quality": mean_quality,
            "local_confidence": local_conf,
            "dynamic_threshold": dyn_thr,
        }
        return cloud
    except Exception as exc:
        # If cloud is unavailable, return best on-device result rather than failing hard.
        safe_calls = selected_local_calls or local_calls
        return {
            "function_calls": safe_calls,
            "total_time_ms": local.get("total_time_ms", 0),
            "confidence": local_conf,
            "source": "on-device",
            "cloud_error": str(exc),
            "fallback_reason": {
                "all_schema_valid": all_schema_valid,
                "mean_quality": mean_quality,
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
