
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

    raw = _parse_cactus_output(raw_str)

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


_TIME_12H_RE = re.compile(r"\b(\d{1,2})(?::(\d{2}))?\s*(AM|PM)\b", re.IGNORECASE)
_MINUTES_RE = re.compile(r"\b(\d+)\s*(?:minutes?|mins?)\b", re.IGNORECASE)
_KEYWORD_STOPWORDS = {
    "a", "an", "the", "for", "to", "with", "and", "or", "of",
    "set", "get", "create", "play", "search", "send", "current", "given",
    "name", "time", "message", "contact", "weather", "alarm", "timer", "song",
}
_TOOL_HINTS = {
    "get_weather": {"weather", "forecast", "temperature"},
    "set_alarm": {"alarm", "wake"},
    "send_message": {"message", "text", "sms"},
    "create_reminder": {"remind", "reminder"},
    "search_contacts": {"contacts", "contact", "find", "lookup", "look", "search"},
    "play_music": {"play", "music", "song", "playlist"},
    "set_timer": {"timer", "countdown"},
}


def _find_matching_delimiter(text, start_idx, open_char, close_char):
    """Find the closing delimiter index for a nested region in text.

    The scan skips characters inside quoted strings and honors escape
    sequences so braces/brackets in string literals do not affect depth.

    :param text: Source text to scan.
    :type text: str
    :param start_idx: Index of the opening delimiter.
    :type start_idx: int
    :param open_char: Opening delimiter character (for example ``"{"``).
    :type open_char: str
    :param close_char: Closing delimiter character (for example ``"}"``).
    :type close_char: str
    :return: Index of the matching closing delimiter, or ``None``.
    :rtype: int | None
    """
    depth = 0
    in_string = False
    escaped = False
    for idx in range(start_idx, len(text)):
        ch = text[idx]
        if escaped:
            escaped = False
            continue
        if ch == "\\":
            escaped = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == open_char:
            depth += 1
        elif ch == close_char:
            depth -= 1
            if depth == 0:
                return idx
    return None


def _sanitize_json_numbers(raw):
    """Normalize JSON numeric tokens that use leading zeros.

    :param raw: Raw JSON-like text emitted by the model.
    :type raw: str
    :return: Sanitized JSON-like text.
    :rtype: str
    """
    # Some generations emit leading-zero integers (for example: minute=01).
    return re.sub(r'(:\s*)0+(\d)(?=\s*[,}\]])', r"\1\2", raw)


def _safe_json_loads(raw):
    """Parse JSON with light repairs for common malformed outputs.

    The function first attempts strict parsing, then retries after small
    sanitizations (leading-zero numbers and trailing commas).

    :param raw: Raw JSON-like text.
    :type raw: str | Any
    :return: Parsed object, or ``None`` if parsing fails.
    :rtype: dict | list | None
    """
    if not isinstance(raw, str) or not raw.strip():
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    sanitized = _sanitize_json_numbers(raw).replace(",}", "}").replace(",]", "]")
    try:
        return json.loads(sanitized)
    except json.JSONDecodeError:
        return None


def _extract_numeric_field(raw, field_name, default=0.0):
    """Extract a top-level numeric field from raw JSON-like text.

    :param raw: Raw JSON-like text.
    :type raw: str | Any
    :param field_name: Field key to extract.
    :type field_name: str
    :param default: Fallback value when extraction fails.
    :type default: float
    :return: Extracted numeric value or ``default``.
    :rtype: float
    """
    if not isinstance(raw, str):
        return default
    match = re.search(rf'"{re.escape(field_name)}"\s*:\s*(-?\d+(?:\.\d+)?)', raw)
    if not match:
        return default
    try:
        return float(match.group(1))
    except ValueError:
        return default


def _extract_function_calls_from_raw(raw):
    """Recover ``function_calls`` from malformed model output.

    The function tries a structured parse of the ``function_calls`` array and
    falls back to regex-based recovery when the JSON is partially corrupted.

    :param raw: Raw model output text.
    :type raw: str | Any
    :return: Recovered function-call list.
    :rtype: list[dict]
    """
    if not isinstance(raw, str):
        return []

    marker = '"function_calls"'
    marker_idx = raw.find(marker)
    if marker_idx == -1:
        return []

    array_start = raw.find("[", marker_idx)
    if array_start == -1:
        return []

    array_end = _find_matching_delimiter(raw, array_start, "[", "]")
    if array_end is None:
        return []

    array_text = raw[array_start:array_end + 1]
    parsed_array = _safe_json_loads(array_text)
    if isinstance(parsed_array, list):
        calls = []
        for item in parsed_array:
            if not isinstance(item, dict):
                continue
            name = item.get("name")
            arguments = item.get("arguments", {})
            if isinstance(name, str) and isinstance(arguments, dict):
                calls.append({"name": name, "arguments": arguments})
        if calls:
            return calls

    # Fallback parser: recover each call name and best-effort arguments.
    calls = []
    name_pattern = re.compile(r'"name"\s*:\s*"([^"]+)"')
    for match in name_pattern.finditer(array_text):
        name = match.group(1)
        args_idx = array_text.find('"arguments"', match.end())
        if args_idx == -1:
            calls.append({"name": name, "arguments": {}})
            continue
        brace_start = array_text.find("{", args_idx)
        if brace_start == -1:
            calls.append({"name": name, "arguments": {}})
            continue
        brace_end = _find_matching_delimiter(array_text, brace_start, "{", "}")
        if brace_end is None:
            calls.append({"name": name, "arguments": {}})
            continue
        args_text = array_text[brace_start:brace_end + 1]
        parsed_args = _safe_json_loads(args_text)
        if not isinstance(parsed_args, dict):
            parsed_args = {}
        calls.append({"name": name, "arguments": parsed_args})
    return calls


def _parse_cactus_output(raw):
    """Parse Cactus output into minimal fields required by the harness.

    :param raw: Raw model output string.
    :type raw: str | Any
    :return: Parsed payload with ``function_calls``, ``total_time_ms``,
        and ``confidence`` keys.
    :rtype: dict
    """
    parsed = _safe_json_loads(raw)
    if isinstance(parsed, dict):
        return {
            "function_calls": parsed.get("function_calls", []),
            "total_time_ms": parsed.get("total_time_ms", 0),
            "confidence": parsed.get("confidence", 0),
        }

    return {
        "function_calls": _extract_function_calls_from_raw(raw),
        "total_time_ms": _extract_numeric_field(raw, "total_time_ms", 0.0),
        "confidence": _extract_numeric_field(raw, "confidence", 0.0),
    }


def _latest_user_text(messages):
    """Concatenate user-message contents into a single query string.

    :param messages: Chat message objects.
    :type messages: list[dict]
    :return: Joined user text in original order.
    :rtype: str
    """
    parts = []
    for message in messages:
        if message.get("role") == "user":
            content = message.get("content", "")
            if isinstance(content, str):
                parts.append(content.strip())
    return " ".join(parts).strip()


def _normalize_time_text(value):
    """Normalize first 12-hour time mention to ``H:MM AM/PM`` format.

    :param value: Raw time-like value.
    :type value: str | Any
    :return: Normalized time string, or stripped original value.
    :rtype: str | Any
    """
    if not isinstance(value, str):
        return value
    match = _TIME_12H_RE.search(value.strip())
    if not match:
        return value.strip()
    hour = int(match.group(1))
    minute = int(match.group(2) or 0)
    am_pm = match.group(3).upper()
    return f"{hour}:{minute:02d} {am_pm}"


def _extract_time_parts(text):
    """Extract hour/minute tuple from the first 12-hour time mention.

    :param text: Source text.
    :type text: str | Any
    :return: ``(hour, minute)`` if found, otherwise ``None``.
    :rtype: tuple[int, int] | None
    """
    if not isinstance(text, str):
        return None
    match = _TIME_12H_RE.search(text)
    if not match:
        return None
    hour = int(match.group(1))
    minute = int(match.group(2) or 0)
    return hour, minute


def _extract_minutes(text):
    """Extract timer minutes from phrases like ``"15 minutes"``.

    :param text: Source text.
    :type text: str | Any
    :return: Parsed minute count or ``None``.
    :rtype: int | None
    """
    if not isinstance(text, str):
        return None
    match = _MINUTES_RE.search(text)
    if not match:
        return None
    return int(match.group(1))


def _coerce_value(value, expected_type):
    """Coerce an argument value to a schema-declared primitive type.

    Supports ``integer``, ``number``, ``boolean``, and ``string`` coercion.
    Returns ``None`` when safe coercion is not possible.

    :param value: Raw value to coerce.
    :type value: Any
    :param expected_type: JSON-schema type name.
    :type expected_type: str | None
    :return: Coerced value or ``None``.
    :rtype: Any | None
    """
    schema_type = (expected_type or "").lower()
    if schema_type == "integer":
        if isinstance(value, bool):
            return None
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            if value.is_integer():
                return int(value)
            return None
        if isinstance(value, str):
            match = re.search(r"-?\d+", value.strip())
            if match:
                return int(match.group(0))
        return None

    if schema_type == "number":
        if isinstance(value, bool):
            return None
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value.strip())
            except ValueError:
                return None
        return None

    if schema_type == "boolean":
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"true", "yes", "1"}:
                return True
            if lowered in {"false", "no", "0"}:
                return False
        return None

    if value is None:
        return None
    return str(value).strip() if schema_type == "string" else value


def _clean_text_value(value):
    """Trim and lightly normalize extracted text fragments.

    :param value: Raw text value.
    :type value: str | Any
    :return: Cleaned text value.
    :rtype: str | Any
    """
    if not isinstance(value, str):
        return value
    value = re.sub(r"\s+", " ", value).strip()
    value = value.strip("\"'")
    value = value.rstrip(".,!?")
    return value.strip()


def _infer_argument_from_text(arg_name, user_text):
    """Infer a tool argument from user text using regex heuristics.

    :param arg_name: Target argument key (for example ``"recipient"``).
    :type arg_name: str
    :param user_text: Aggregated user request text.
    :type user_text: str | Any
    :return: Inferred argument value or ``None`` when not found.
    :rtype: str | int | None
    """
    if not isinstance(user_text, str):
        return None

    if arg_name in {"hour", "minute"}:
        time_parts = _extract_time_parts(user_text)
        if not time_parts:
            return None
        return time_parts[0] if arg_name == "hour" else time_parts[1]

    if arg_name == "minutes":
        return _extract_minutes(user_text)

    if arg_name == "time":
        time_match = _TIME_12H_RE.search(user_text)
        if not time_match:
            return None
        return _normalize_time_text(time_match.group(0))

    patterns = {
        "location": [
            r"\bweather(?:\s+like)?\s+in\s+([A-Za-z][A-Za-z\s'-]*?)(?:\s+and\s+|[.?!,]|$)",
        ],
        "query": [
            r"\b(?:find|look up|search for|search)\s+([A-Za-z][A-Za-z'-]*)",
        ],
        "recipient": [
            r"\bsend(?:\s+a)?\s+message\s+to\s+([A-Za-z][A-Za-z'-]*)",
            r"\btext\s+([A-Za-z][A-Za-z'-]*)",
        ],
        "message": [
            r"\bsaying\s+(.+?)(?:\s+and\s+(?:get|check|set|play|remind|look|find|search|text|send)\b|[.?!]|$)",
        ],
        "song": [
            r"\bplay(?:\s+some)?\s+(.+?)(?:\s+and\s+(?:get|check|set|send|text|remind|look|find|search)\b|[.?!]|$)",
        ],
        "title": [
            r"\bremind me(?:\s+(?:about|to))?\s+(.+?)(?:\s+at\s+\d{1,2}(?::\d{2})?\s*(?:AM|PM)|\s+and\s+|[.?!]|$)",
        ],
    }

    for pattern in patterns.get(arg_name, []):
        match = re.search(pattern, user_text, flags=re.IGNORECASE)
        if not match:
            continue
        value = _clean_text_value(match.group(1))
        if arg_name == "song":
            value = re.sub(r"\s+music$", "", value, flags=re.IGNORECASE).strip()
        if arg_name == "title":
            value = re.sub(r"^the\s+", "", value, flags=re.IGNORECASE).strip()
        if value:
            return value

    # Handle "send him/her" by reusing a searched contact name in the same request.
    if arg_name == "recipient" and re.search(r"\bsend\s+(him|her)\b", user_text, flags=re.IGNORECASE):
        match = re.search(r"\b(?:find|look up|search for|search)\s+([A-Za-z][A-Za-z'-]*)", user_text, flags=re.IGNORECASE)
        if match:
            return _clean_text_value(match.group(1))

    return None


def _build_tool_index(tools):
    """Build a lookup map for tool schema metadata by tool name.

    :param tools: Tool declarations passed to the model.
    :type tools: list[dict]
    :return: Mapping of tool name to description/properties/required metadata.
    :rtype: dict[str, dict]
    """
    tool_index = {}
    for tool in tools:
        name = tool.get("name")
        if not name:
            continue
        parameters = tool.get("parameters", {})
        properties = parameters.get("properties", {})
        required = list(parameters.get("required", []))
        tool_index[name] = {
            "description": tool.get("description", ""),
            "properties": properties,
            "required": required,
        }
    return tool_index


def _normalize_calls(raw_calls, tool_index, user_text):
    """Validate and normalize model-produced function calls.

    The routine filters unknown tools, coerces argument types, normalizes
    time fields, fills missing required arguments from user text heuristics,
    and de-duplicates calls.

    :param raw_calls: Raw function-call list from model output.
    :type raw_calls: list[dict] | Any
    :param tool_index: Tool metadata index from :func:`_build_tool_index`.
    :type tool_index: dict[str, dict]
    :param user_text: Aggregated user text for backfilling arguments.
    :type user_text: str
    :return: Normalized function-call list.
    :rtype: list[dict]
    """
    if not isinstance(raw_calls, list):
        return []

    normalized_calls = []
    seen = set()
    for call in raw_calls:
        if not isinstance(call, dict):
            continue

        name = call.get("name")
        if name not in tool_index:
            continue

        raw_args = call.get("arguments", {})
        if not isinstance(raw_args, dict):
            raw_args = {}

        properties = tool_index[name]["properties"]
        required = tool_index[name]["required"]
        normalized_args = {}

        for key, value in raw_args.items():
            if key not in properties:
                continue
            expected_type = properties[key].get("type", "string")
            coerced = _coerce_value(value, expected_type)
            if coerced is None:
                continue
            if key.lower() == "time" and isinstance(coerced, str):
                coerced = _normalize_time_text(coerced)
            if isinstance(coerced, str) and not coerced:
                continue
            normalized_args[key] = coerced

        # Backfill common time fields when required by schema.
        required_set = set(required)
        if "hour" in required_set and "minute" in required_set:
            if "hour" not in normalized_args or "minute" not in normalized_args:
                extracted = _extract_time_parts(raw_args.get("time")) or _extract_time_parts(user_text)
                if extracted:
                    normalized_args.setdefault("hour", extracted[0])
                    normalized_args.setdefault("minute", extracted[1])

        if "minutes" in required_set and "minutes" not in normalized_args:
            extracted_minutes = _extract_minutes(str(raw_args.get("minutes", ""))) or _extract_minutes(user_text)
            if extracted_minutes is not None:
                normalized_args["minutes"] = extracted_minutes

        if "time" in required_set and "time" in normalized_args:
            normalized_args["time"] = _normalize_time_text(normalized_args["time"])

        # Fill missing required fields from user text when possible.
        for req in required:
            if req in normalized_args:
                continue
            inferred = _infer_argument_from_text(req, user_text)
            if inferred is None:
                continue
            expected_type = properties.get(req, {}).get("type", "string")
            coerced = _coerce_value(inferred, expected_type)
            if coerced is None:
                continue
            if req.lower() == "time" and isinstance(coerced, str):
                coerced = _normalize_time_text(coerced)
            if isinstance(coerced, str):
                coerced = _clean_text_value(coerced)
            if coerced == "":
                continue
            normalized_args[req] = coerced

        missing_required = any(req not in normalized_args for req in required)
        if missing_required:
            continue

        key = (name, json.dumps(normalized_args, sort_keys=True))
        if key in seen:
            continue
        seen.add(key)
        normalized_calls.append({"name": name, "arguments": normalized_args})

    # If strict normalization dropped everything, keep best-effort original valid-shape calls.
    if normalized_calls:
        return normalized_calls

    fallback = []
    seen = set()
    for call in raw_calls:
        if not isinstance(call, dict):
            continue
        name = call.get("name")
        arguments = call.get("arguments", {})
        if name in tool_index and isinstance(arguments, dict):
            key = (name, json.dumps(arguments, sort_keys=True))
            if key in seen:
                continue
            seen.add(key)
            fallback.append({"name": name, "arguments": arguments})
    return fallback


def _tool_keywords(tool):
    """Generate lexical keywords for a tool from schema plus hand-tuned hints.

    :param tool: Tool declaration.
    :type tool: dict
    :return: Keyword set for intent/relevance matching.
    :rtype: set[str]
    """
    name = tool.get("name", "")
    description = tool.get("description", "")
    raw_tokens = re.findall(r"[a-zA-Z0-9_']+", f"{name} {description}".lower())
    keywords = set()
    for token in raw_tokens:
        for piece in token.split("_"):
            piece = piece.strip("'")
            if not piece or piece in _KEYWORD_STOPWORDS or len(piece) < 3:
                continue
            keywords.add(piece)
    keywords.update(_TOOL_HINTS.get(name, set()))
    return keywords


def _tool_is_mentioned(tool, user_text):
    """Check whether tool keywords appear in user text.

    :param tool: Tool declaration.
    :type tool: dict
    :param user_text: Aggregated user text.
    :type user_text: str
    :return: ``True`` if any tool keyword is present.
    :rtype: bool
    """
    text = user_text.lower()
    for keyword in _tool_keywords(tool):
        if re.search(rf"\b{re.escape(keyword)}\b", text):
            return True
    return False


def _estimate_intent_count(user_text, tools):
    """Estimate expected number of tool intents from text and tool mentions.

    :param user_text: Aggregated user text.
    :type user_text: str
    :param tools: Available tools for this case.
    :type tools: list[dict]
    :return: Estimated intent count, bounded to ``[1, 3]``.
    :rtype: int
    """
    text = user_text.lower()
    if not text:
        return 1

    matched_tools = 0
    for tool in tools:
        keywords = _tool_keywords(tool)
        if any(re.search(rf"\b{re.escape(keyword)}\b", text) for keyword in keywords):
            matched_tools += 1

    separator_estimate = text.count(" and ") + text.count(", and ")
    if matched_tools == 0:
        return max(1, min(3, 1 + separator_estimate))
    if matched_tools == 1 and separator_estimate > 0:
        return 2
    return max(1, min(3, matched_tools))


def _call_relevance_score(call_name, user_text, tools):
    """Compute lexical relevance between a predicted call and user text.

    :param call_name: Predicted tool name.
    :type call_name: str
    :param user_text: Aggregated user text.
    :type user_text: str
    :param tools: Available tools for this case.
    :type tools: list[dict]
    :return: Keyword-overlap score.
    :rtype: int
    """
    text = user_text.lower()
    tool = next((tool for tool in tools if tool.get("name") == call_name), None)
    if not tool:
        return 0
    keywords = _tool_keywords(tool)
    return sum(1 for keyword in keywords if re.search(rf"\b{re.escape(keyword)}\b", text))


def _rule_based_calls(user_text, tools, tool_index):
    """Build deterministic fallback calls by direct text extraction.

    :param user_text: Aggregated user text.
    :type user_text: str
    :param tools: Available tools for this case.
    :type tools: list[dict]
    :param tool_index: Tool metadata index.
    :type tool_index: dict[str, dict]
    :return: Deterministically inferred function calls.
    :rtype: list[dict]
    """
    calls = []
    seen = set()
    for tool in tools:
        name = tool.get("name")
        if name not in tool_index:
            continue
        if not _tool_is_mentioned(tool, user_text):
            continue

        properties = tool_index[name]["properties"]
        required = tool_index[name]["required"]
        arguments = {}
        valid = True
        for req in required:
            inferred = _infer_argument_from_text(req, user_text)
            if inferred is None:
                valid = False
                break
            expected_type = properties.get(req, {}).get("type", "string")
            coerced = _coerce_value(inferred, expected_type)
            if coerced is None:
                valid = False
                break
            if req.lower() == "time" and isinstance(coerced, str):
                coerced = _normalize_time_text(coerced)
            if isinstance(coerced, str):
                coerced = _clean_text_value(coerced)
            arguments[req] = coerced

        if not valid:
            continue

        key = (name, json.dumps(arguments, sort_keys=True))
        if key in seen:
            continue
        seen.add(key)
        calls.append({"name": name, "arguments": arguments})
    return calls


def _score_candidate(calls, intent_count, user_text, tools):
    """Score a candidate call list for selection in hybrid routing.

    Higher score rewards lexical relevance and argument completeness while
    penalizing mismatch between call count and estimated intent count.

    :param calls: Candidate function calls.
    :type calls: list[dict]
    :param intent_count: Expected number of intents.
    :type intent_count: int
    :param user_text: Aggregated user text.
    :type user_text: str
    :param tools: Available tools for this case.
    :type tools: list[dict]
    :return: Candidate quality score.
    :rtype: int
    """
    if not calls:
        return -10
    relevance = sum(_call_relevance_score(call["name"], user_text, tools) for call in calls)
    count_penalty = abs(len(calls) - intent_count)
    arg_bonus = sum(min(len(call.get("arguments", {})), 3) for call in calls)
    return (5 * relevance) + arg_bonus - (3 * count_penalty)


def _prune_calls(calls, target_count, user_text, tools):
    """Keep the top-N most relevant calls by lexical relevance score.

    :param calls: Candidate function calls.
    :type calls: list[dict]
    :param target_count: Maximum number of calls to retain.
    :type target_count: int
    :param user_text: Aggregated user text.
    :type user_text: str
    :param tools: Available tools for this case.
    :type tools: list[dict]
    :return: Pruned function-call list.
    :rtype: list[dict]
    """
    if target_count <= 0 or len(calls) <= target_count:
        return calls

    scored_calls = []
    for call in calls:
        score = _call_relevance_score(call["name"], user_text, tools)
        score += min(len(call.get("arguments", {})), 3) * 0.01
        scored_calls.append((score, call))

    scored_calls.sort(key=lambda item: item[0], reverse=True)
    return [call for _, call in scored_calls[:target_count]]


def generate_hybrid(messages, tools, confidence_threshold=0.99):
    """Local-only hybrid strategy with validation + normalization to maximize F1 while staying on-device."""
    local = generate_cactus(messages, tools)
    user_text = _latest_user_text(messages)
    tool_index = _build_tool_index(tools)

    normalized_calls = _normalize_calls(local.get("function_calls", []), tool_index, user_text)
    estimated_intents = _estimate_intent_count(user_text, tools)
    rule_calls = _rule_based_calls(user_text, tools, tool_index)

    if _score_candidate(rule_calls, estimated_intents, user_text, tools) > _score_candidate(normalized_calls, estimated_intents, user_text, tools):
        normalized_calls = rule_calls

    if len(normalized_calls) > estimated_intents:
        # Keep likely-intended calls when model over-calls (helps precision on single-intent prompts).
        normalized_calls = _prune_calls(normalized_calls, estimated_intents, user_text, tools)

    # Use confidence threshold as strictness control for low-confidence generations.
    if local.get("confidence", 0) < confidence_threshold and len(normalized_calls) > max(1, estimated_intents):
        normalized_calls = _prune_calls(normalized_calls, max(1, estimated_intents), user_text, tools)

    local["function_calls"] = normalized_calls
    local["source"] = "on-device"

    # Cloud fallback intentionally disabled for 100% on-device routing.
    # Keep this block for quick rollback to hybrid cloud routing:
    #
    # if local.get("confidence", 0) < confidence_threshold:
    #     cloud = generate_cloud(messages, tools)
    #     cloud["source"] = "cloud (fallback)"
    #     cloud["local_confidence"] = local.get("confidence", 0)
    #     cloud["total_time_ms"] += local.get("total_time_ms", 0)
    #     return cloud

    return local


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
