
import sys
sys.path.insert(0, "cactus/python/src")
functiongemma_path = "cactus/weights/functiongemma-270m-it"

import json, os, time
from concurrent.futures import ThreadPoolExecutor
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


def generate_hybrid(messages, tools, confidence_threshold=0.99):
    """Hybrid strategy: neural decomposition via FunctionGemma, then fan-out."""
    user_text = next(
        (m["content"] for m in reversed(messages) if m["role"] == "user"), ""
    )

    # --- neural classification + decomposition in one FunctionGemma call ---
    start = time.time()
    decompose_tool = [{
        "type": "function",
        "function": {
            "name": "decompose_query",
            "description": "Break a user request into simple, single-action sub-queries. "
                           "If the request is already a single action, return it as-is in a one-element list.",
            "parameters": {
                "type": "object",
                "properties": {
                    "subqueries": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of simple sub-queries",
                    }
                },
                "required": ["subqueries"],
            },
        },
    }]
    model = cactus_init(functiongemma_path)
    raw_str = cactus_complete(
        model,
        [{
            "role": "system", 
            "content": "You are a query decomposer. Use the decompose_query tool to break multi-hop queries into simple single-hop queries. If the query is single-hop native, return the query as is."
        },
         {"role": "user", "content": user_text}],
        tools=decompose_tool,
        force_tools=True,
        max_tokens=256,
        stop_sequences=["<|im_end|>", "<end_of_turn>"],
    )
    cactus_destroy(model)

    sub_queries = None
    try:
        raw = json.loads(raw_str)
        for fc in raw.get("function_calls", []):
            subs = fc.get("arguments", {}).get("subqueries", [])
            if isinstance(subs, list) and subs:
                sub_queries = [s for s in subs if isinstance(s, str) and s.strip()]
                break
    except (json.JSONDecodeError, KeyError, TypeError):
        pass

    decompose_ms = (time.time() - start) * 1000

    # Model returned <=1 sub-query -> simple request, run directly with original messages
    if not sub_queries or len(sub_queries) <= 1:
        local = generate_cactus(messages, tools)
        local["total_time_ms"] += decompose_ms
        local["source"] = "on-device"
        return local

    # --- compound: fan-out sub-queries concurrently ---
    def _run_subquery(sq):
        return generate_cactus([{"role": "user", "content": sq}], tools)

    fan_start = time.time()
    with ThreadPoolExecutor(max_workers=len(sub_queries)) as pool:
        results = list(pool.map(_run_subquery, sub_queries))
    fan_ms = (time.time() - fan_start) * 1000

    all_calls = []
    seen = set()
    for r in results:
        for fc in r.get("function_calls", []):
            key = (fc.get("name"), json.dumps(fc.get("arguments", {}), sort_keys=True))
            if key not in seen:
                seen.add(key)
                all_calls.append(fc)

    return {
        "function_calls": all_calls,
        "total_time_ms": decompose_ms + fan_ms,
        "confidence": min((r.get("confidence", 0) for r in results), default=0),
        "source": "on-device",
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
