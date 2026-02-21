import sys
sys.path.insert(0, "cactus/python/src")
functiongemma_path = "cactus/weights/functiongemma-270m-it"
import json, os, time, re
from cactus import cactus_init, cactus_complete, cactus_destroy
from google import genai
from google.genai import types


def generate_cactus(messages, tools, system_msg="You are a helpful assistant that can use tools."):
    """Run function calling on-device via FunctionGemma + Cactus."""
    model = cactus_init(functiongemma_path)
    cactus_tools = [{"function": t} for t in tools]
    raw_str = cactus_complete(
        model,
        [{"role": "developer", "content": system_msg}] + messages,
        tools=cactus_tools,
        force_tools=True,
        max_tokens=256,
        stop_sequences=["<end_of_turn>"],
        confidence_threshold=0.0,
    )
    cactus_destroy(model)
    try:
        patched_str = re.sub(r'([:\s\[,])0+(\d+)', r'\1\2', raw_str)
        patched_str = re.sub(r'"true"|"false"|"TRUE"|"FALSE"', lambda m: m.group(0).lower().replace('"', ''), patched_str)
        raw = json.loads(patched_str)
    except json.JSONDecodeError:
        return {"function_calls": [], "total_time_ms": 0, "confidence": 0, "cloud_handoff": False}
    return {
        "function_calls": raw.get("function_calls", []),
        "total_time_ms":  raw.get("total_time_ms", 0),
        "confidence":     raw.get("confidence", 0),
        "cloud_handoff":  raw.get("cloud_handoff", False),
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
    return {"function_calls": function_calls, "total_time_ms": total_time_ms}


def generate_hybrid(messages, tools, confidence_threshold=0.99):

    # ══════════════════════════════════════════════════════════
    # CHECKPOINT 1 — PRE-FLIGHT
    # Analyze the request before calling any model.
    # Uses 5 signals to decide if this is too complex for local.
    # Zero model calls — pure text analysis, runs in microseconds.
    # ══════════════════════════════════════════════════════════

    # Get user message
    user_message = ""
    for m in reversed(messages):
        if m.get("role") == "user":
            user_message = m.get("content", "")
            break
    msg = user_message.lower()

    # -- Signal 1: Message length --
    word_count = len(user_message.split())
    if word_count <= 8:
        s_length = 0.0
    elif word_count <= 20:
        s_length = 0.2
    elif word_count <= 40:
        s_length = 0.5
    else:
        s_length = 0.8

    # -- Signal 2: Action verb count --
    action_verbs = [
        "look up", "send", "text", "get", "check",
        "find", "set", "create", "remind", "play",
        "start", "search", "book", "wake", "call"
    ]
    found_verbs = []
    for verb in sorted(action_verbs, key=len, reverse=True):
        if " " in verb:
            if verb in msg: found_verbs.append(verb)
        else:
            if re.search(rf"\b{verb}\b", msg): found_verbs.append(verb)
    verb_count = len(found_verbs)
    if verb_count <= 1: s_verbs = 0.0
    elif verb_count == 2: s_verbs = 0.8
    else: s_verbs = 1.0

    # -- Explicit multi-step signal --
    s_multi = 1.0 if (" and " in msg and verb_count > 1) or verb_count > 1 else 0.0

    # -- Signal 3: Negations and conditionals --
    # Small models ignore these and produce wrong calls.
    neg_patterns  = [r"\bnot\b", r"\bnever\b", r"\bexcept\b", r"\bwithout\b", r"\bno\b"]
    cond_patterns = [r"\bif\b", r"\bunless\b", r"\bonly\s+when\b", r"\bonly\s+if\b", r"\bwhen\b"]
    neg_cond_hits = sum(1 for p in neg_patterns + cond_patterns if re.search(p, msg))
    if neg_cond_hits == 0:
        s_neg = 0.0
    elif neg_cond_hits == 1:
        s_neg = 0.3
    elif neg_cond_hits == 2:
        s_neg = 0.6
    else:
        s_neg = 0.9

    # -- Signal 4: Tool count --
    # More tools = harder selection for a small model.
    tool_count = len(tools)
    if tool_count <= 2:
        s_tools = 0.0
    elif tool_count <= 5:
        s_tools = 0.2
    elif tool_count <= 10:
        s_tools = 0.5
    else:
        s_tools = 0.8

    # -- Signal 5: Tool name/description similarity --
    # Similar tools (set_alarm vs set_timer) cause confusion.
    def jaccard(a, b):
        wa, wb = set(a.lower().split()), set(b.lower().split())
        return len(wa & wb) / len(wa | wb) if wa and wb else 0.0

    descs = [f"{t.get('name','')} {t.get('description','')}" for t in tools]
    max_sim = 0.0
    for i in range(len(descs)):
        for j in range(i + 1, len(descs)):
            max_sim = max(max_sim, jaccard(descs[i], descs[j]))
    if max_sim < 0.2:
        s_sim = 0.0
    elif max_sim < 0.4:
        s_sim = 0.3
    elif max_sim < 0.6:
        s_sim = 0.6
    else:
        s_sim = 0.9

    # -- Weighted composite score --
    score = (
        s_length * 0.10 +
        s_verbs  * 0.20 +
        s_multi  * 0.40 +
        s_neg    * 0.20 +
        s_tools  * 0.10 +
        s_sim    * 0.10
    )

    # -- Route to cloud immediately if too complex --
    if score >= 0.40:
        cloud = generate_cloud(messages, tools)
        cloud["source"] = f"cloud (preflight score={score:.2f})"
        return cloud

    # ══════════════════════════════════════════════════════════
    # CHECKPOINT 2 — RUN LOCAL + POST-FLIGHT VALIDATION
    # Run FunctionGemma locally, then validate the output.
    # Check: valid function name, required params present, types ok.
    # ══════════════════════════════════════════════════════════
    local = generate_cactus(messages, tools)
    available_names = {t["name"] for t in tools}

    def is_valid(result):
        calls = result.get("function_calls", [])
        if not calls:
            return False, "no function calls returned"
        tools_by_name = {t["name"]: t for t in tools}
        for call in calls:
            name = call.get("name", "")
            args = call.get("arguments", {})
            if name not in tools_by_name:
                return False, f"hallucinated tool name: {name}"
            required = tools_by_name[name].get("parameters", {}).get("required", [])
            for param in required:
                if param not in args:
                    return False, f"missing required param '{param}' in {name}"
            props = tools_by_name[name].get("parameters", {}).get("properties", {})
            for param, value in args.items():
                if param not in props:
                    continue
                expected_type = props[param].get("type", "")
                if expected_type == "integer" and not isinstance(value, int):
                    try:
                        int(str(value))
                    except (ValueError, TypeError):
                        return False, f"param '{param}' not coercible to int"
                elif expected_type == "number" and not isinstance(value, (int, float)):
                    try:
                        float(str(value))
                    except (ValueError, TypeError):
                        return False, f"param '{param}' not coercible to number"
                elif expected_type == "string":
                    if str(value).strip() == "" and param in required:
                        return False, f"required string param '{param}' is empty"
                    elif str(value).strip() != "":
                        val_clean = re.sub(r'[^\w\s]', '', str(value).lower()).strip()
                        msg_clean = re.sub(r'[^\w\s]', '', msg).strip()
                        if val_clean and val_clean not in msg_clean:
                            words = val_clean.split()
                            match_count = sum(1 for w in words if w in msg_clean)
                            if match_count == 0:
                                return False, f"hallucinated string not in prompt: {value}"
        return True, "ok"

    valid, reason = is_valid(local)
    if valid:
        local["function_calls"] = [
            c for c in local["function_calls"] if c.get("name") in available_names
        ]
        local["source"] = "on-device"
        return local

    # ══════════════════════════════════════════════════════════
    # CHECKPOINT 3 — RETRY LOCALLY WITH STRONGER PROMPT
    # Before paying for a cloud call, retry once locally with
    # a more explicit system prompt. Costs ~300ms but free.
    # ══════════════════════════════════════════════════════════
    retry_system = (
        "You MUST call one of the provided tools. "
        "Do not write any text. Only call the most relevant tool."
    )
    retry = generate_cactus(messages, tools, system_msg=retry_system)
    valid_retry, retry_reason = is_valid(retry)
    if valid_retry:
        retry["function_calls"] = [
            c for c in retry["function_calls"] if c.get("name") in available_names
        ]
        retry["source"] = "on-device (retry)"
        retry["total_time_ms"] += local["total_time_ms"]
        return retry

    # ══════════════════════════════════════════════════════════
    # FALLBACK — CLOUD
    # Both local attempts failed validation. Escalate to Gemini.
    # ══════════════════════════════════════════════════════════
    cloud = generate_cloud(messages, tools)
    cloud["source"] = "cloud (postflight fallback)"
    cloud["local_confidence"] = local.get("confidence", 0)
    cloud["total_time_ms"] += local["total_time_ms"] + retry["total_time_ms"]
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
                "location": {"type": "string", "description": "City name"}
            },
            "required": ["location"],
        },
    }]
    messages = [{"role": "user", "content": "What is the weather in San Francisco?"}]

    on_device = generate_cactus(messages, tools)
    print_result("FunctionGemma (On-Device Cactus)", on_device)

    cloud = generate_cloud(messages, tools)
    print_result("Gemini (Cloud)", cloud)

    hybrid = generate_hybrid(messages, tools)
    print_result("Hybrid (On-Device + Cloud Fallback)", hybrid)
