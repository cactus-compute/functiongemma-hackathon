
import sys
import os as _os

# Resolve repo root (two levels up from mingle/ai_server/)
_REPO_ROOT = _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../.."))
sys.path.insert(0, _os.path.join(_REPO_ROOT, "cactus/python/src"))
functiongemma_path = _os.path.join(_REPO_ROOT, "cactus/weights/functiongemma-270m-it")

import json, os, time

# Load .env from repo root if present (dev convenience — does not override existing env vars)
try:
    from dotenv import load_dotenv
    load_dotenv(_os.path.join(_REPO_ROOT, ".env"), override=False)
except ImportError:
    pass

from cactus import cactus_init, cactus_complete, cactus_destroy
from google import genai
from google.genai import types


# --- Model persistence singleton ---
_cactus_model = None

def _get_cactus_model():
    global _cactus_model
    if _cactus_model is None:
        _cactus_model = cactus_init(functiongemma_path)
    return _cactus_model


# --- Complexity classifier ---
_MULTI_ACTION_KW = ["and", "also", "then", "plus", "as well", "both", "additionally"]
_ACTION_VERBS = ["set", "send", "check", "play", "find", "remind", "text", "get", "search"]

def _classify_complexity(messages, tools) -> str:
    user_text = " ".join(
        m["content"] for m in messages if m["role"] == "user"
    ).lower()
    tool_count = len(tools)
    conjunction_count = sum(1 for kw in _MULTI_ACTION_KW if f" {kw} " in f" {user_text} ")
    verb_count = sum(1 for v in _ACTION_VERBS if v in user_text.split())
    if conjunction_count >= 1 and verb_count >= 2:
        return "hard"
    if tool_count >= 4 and verb_count >= 2:
        return "hard"
    if tool_count >= 3:
        return "medium"
    return "easy"


# Per-complexity routing table
_COMPLEXITY_CONFIG = {
    "easy":   {"tool_rag_top_k": 1, "confidence_threshold": 0.75, "max_tokens": 128},
    "medium": {"tool_rag_top_k": 2, "confidence_threshold": 0.82, "max_tokens": 192},
    "hard":   {"tool_rag_top_k": 0, "confidence_threshold": 0.97, "max_tokens": 320},
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

    gemini_response = client.models.generate_content(
        model="gemini-2.0-flash",
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


def generate_hybrid(messages, tools, confidence_threshold=None):
    """Hybrid inference: classify complexity, route to on-device or cloud.

    Uses a model persistence singleton to avoid re-initialising Cactus on
    every call (major latency improvement). Complexity-aware routing lowers
    confidence thresholds for simple requests so more work stays on-device.
    """
    complexity = _classify_complexity(messages, tools)
    cfg = _COMPLEXITY_CONFIG[complexity]

    # Use caller-supplied threshold if provided, otherwise use per-complexity default
    threshold = confidence_threshold if confidence_threshold is not None else cfg["confidence_threshold"]

    model = _get_cactus_model()

    # Pass all tools — tool_rag_top_k in cactus_complete handles native RAG filtering
    cactus_tools = [{"type": "function", "function": t} for t in tools]

    raw_str = cactus_complete(
        model,
        [{"role": "system", "content": "You are a helpful assistant that can use tools."}] + messages,
        tools=cactus_tools,
        force_tools=True,
        max_tokens=cfg["max_tokens"],
        tool_rag_top_k=cfg["tool_rag_top_k"],      # native Cactus RAG tool filtering
        confidence_threshold=threshold,             # native Cactus confidence gate
        stop_sequences=["<|im_end|>", "<end_of_turn>"],
    )

    try:
        raw = json.loads(raw_str)
    except json.JSONDecodeError:
        raw = {}

    local_confidence = raw.get("confidence", 0)
    local_function_calls = raw.get("function_calls", [])
    local_time_ms = raw.get("total_time_ms", 0)
    cloud_handoff = raw.get("cloud_handoff", False)

    # Accept on-device result: not a cloud_handoff, confidence met, and non-empty calls
    if not cloud_handoff and local_confidence >= threshold and local_function_calls:
        return {
            "function_calls": local_function_calls,
            "total_time_ms": local_time_ms,
            "confidence": local_confidence,
            "source": "on-device",
            "complexity": complexity,
        }

    # Fall back to cloud
    cloud = generate_cloud(messages, tools)
    cloud["source"] = "cloud (fallback)"
    cloud["local_confidence"] = local_confidence
    cloud["total_time_ms"] += local_time_ms
    cloud["complexity"] = complexity
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
