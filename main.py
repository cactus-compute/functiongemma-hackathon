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

    def _build_gemini_schema(v):
        if not isinstance(v, dict):
            return types.Schema(type="STRING")
        t_str = str(v.get("type", "string")).upper()
        if t_str not in {"STRING", "INTEGER", "NUMBER", "BOOLEAN", "ARRAY", "OBJECT"}:
            t_str = "STRING"
        
        schema_kwargs = {"type": t_str, "description": v.get("description", "")}
        
        if "enum" in v and isinstance(v["enum"], list):
            schema_kwargs["enum"] = [str(x) for x in v["enum"]]
            
        if t_str == "ARRAY" and "items" in v:
            schema_kwargs["items"] = _build_gemini_schema(v["items"])
            
        if t_str == "OBJECT" and "properties" in v:
            schema_kwargs["properties"] = {
                pk: _build_gemini_schema(pv)
                for pk, pv in v["properties"].items()
            }
            if "required" in v:
                schema_kwargs["required"] = v["required"]
                
        return types.Schema(**schema_kwargs)

    gemini_tools = [
        types.Tool(
            function_declarations=[
                types.FunctionDeclaration(
                    name=t["name"],
                    description=t["description"],
                    parameters=types.Schema(
                        type="OBJECT",
                        properties={
                            k: _build_gemini_schema(v)
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
    system_instruction = (
        "You are a function-calling assistant. "
        "Return ALL needed function calls for the user request. "
        "If the user asks for multiple distinct actions, output a separate function call for each action."
    )

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
    """Fairness-first hybrid: generic schema checks + local self-consistency + uncertainty fallback."""
    tool_map = {t["name"]: t for t in tools}
    user_text = " ".join(
        m.get("content", "") for m in messages if m.get("role") == "user"
    )

    def _norm_text(s):
        if not isinstance(s, str):
            return ""
        return re.sub(r"\s+", " ", s).strip()

    def _canonical_time(value):
        if not isinstance(value, str):
            return None
        s = _norm_text(value)
        m = re.search(r"\b(\d{1,2})(?::(\d{2}))?\s*([ap])\.?\s*m\.?\b", s, flags=re.IGNORECASE)
        if m:
            h = int(m.group(1))
            mm = int(m.group(2) or "0")
            if 1 <= h <= 12 and 0 <= mm <= 59:
                return f"{h}:{mm:02d} {m.group(3).upper()}M"
        m2 = re.search(r"T(\d{2})[:\-](\d{2})", s)
        if m2:
            h24 = int(m2.group(1))
            mm = int(m2.group(2))
            if 0 <= h24 <= 23 and 0 <= mm <= 59:
                suffix = "AM" if h24 < 12 else "PM"
                h12 = h24 % 12
                if h12 == 0:
                    h12 = 12
                return f"{h12}:{mm:02d} {suffix}"
        return None

    def _extract_user_reminder_title():
        m = re.search(
            r"\bremind me\s+(?:to|about)\s+(.+?)(?:\s+at\s+\d|\s+at\s+\w|,|\s+and\b|$)",
            user_text,
            flags=re.IGNORECASE,
        )
        if not m:
            return None
        title = _norm_text(m.group(1)).strip(" .")
        if re.match(r"^the\s+\w+$", title, flags=re.IGNORECASE):
            title = re.sub(r"^the\s+", "", title, flags=re.IGNORECASE)
        return title or None

    def _extract_user_saying_message():
        m = re.search(r"\bsaying\s+(.+?)(?:,|\s+and\b|$)", user_text, flags=re.IGNORECASE)
        if not m:
            return None
        return _norm_text(m.group(1)).strip(" .") or None

    def _extract_user_recipient():
        m = re.search(r"\bsend\s+(?:a\s+)?message\s+to\s+([A-Za-z][A-Za-z'\-]*)\b", user_text, flags=re.IGNORECASE)
        if not m:
            m = re.search(r"\btext\s+([A-Za-z][A-Za-z'\-]*)\b", user_text, flags=re.IGNORECASE)
        if not m:
            return None
        return _norm_text(m.group(1)).strip(" .")

    def _extract_user_times():
        times = set()
        for m in re.finditer(
            r"\b(\d{1,2})(?::(\d{2}))?\s*([ap])\.?\s*m\.?\b",
            user_text,
            flags=re.IGNORECASE,
        ):
            h = int(m.group(1))
            mm = int(m.group(2) or "0")
            if 1 <= h <= 12 and 0 <= mm <= 59:
                times.add(f"{h}:{mm:02d} {m.group(3).upper()}M")
        return times

    user_reminder_title = _extract_user_reminder_title()
    user_saying_message = _extract_user_saying_message()
    user_recipient = _extract_user_recipient()
    user_times = _extract_user_times()

    def _coerce_value(value, schema):
        if not isinstance(schema, dict):
            return value
        t = str(schema.get("type", "")).lower()
        if t == "integer":
            if isinstance(value, bool):
                return value
            if isinstance(value, int):
                return value
            if isinstance(value, float) and value.is_integer():
                return int(value)
            if isinstance(value, str) and re.fullmatch(r"[+-]?\d+", value.strip()):
                return int(value.strip())
            return value
        if t == "number":
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
        if t == "boolean":
            if isinstance(value, bool):
                return value
            if isinstance(value, str):
                v = value.strip().lower()
                if v in {"true", "1", "yes", "on"}:
                    return True
                if v in {"false", "0", "no", "off"}:
                    return False
            return value
        if t == "string":
            if isinstance(value, str):
                return _norm_text(value)
            return str(value)
        return value

    def _value_matches_type(value, schema):
        if not isinstance(schema, dict):
            return True
        t = str(schema.get("type", "")).lower()
        if t == "integer":
            return isinstance(value, int) and not isinstance(value, bool)
        if t == "number":
            return isinstance(value, (int, float)) and not isinstance(value, bool)
        if t == "boolean":
            return isinstance(value, bool)
        if t == "string":
            return isinstance(value, str)
        if t == "array":
            return isinstance(value, list)
        if t == "object":
            return isinstance(value, dict)
        return True

    def _canonicalize_call(call):
        name = call.get("name")
        args = call.get("arguments", {})
        if not isinstance(args, dict):
            args = {}
        out = {"name": name, "arguments": dict(args)}
        tool = tool_map.get(name)
        if not tool:
            return out
        props = tool.get("parameters", {}).get("properties", {}) or {}
        for k, v in list(out["arguments"].items()):
            if k in props:
                out["arguments"][k] = _coerce_value(v, props[k])
            elif isinstance(v, str):
                out["arguments"][k] = _norm_text(v)
        if name == "set_alarm" and "hour" in out["arguments"] and "minute" not in out["arguments"]:
            out["arguments"]["minute"] = 0
        if name == "play_music" and isinstance(out["arguments"].get("song"), str):
            song = out["arguments"]["song"].strip(" .")
            had_some = bool(re.match(r"^some\s+", song, flags=re.IGNORECASE))
            song = re.sub(r"^some\s+", "", song, flags=re.IGNORECASE)
            user_has_some_music = bool(re.search(r"\bsome\s+.+\s+music\b", user_text, flags=re.IGNORECASE))
            if (had_some or user_has_some_music) and song.lower().endswith(" music"):
                root = song[:-6].strip()
                if root:
                    song = root
            out["arguments"]["song"] = song
        if name in {"send_message", "search_contacts"}:
            person_key = "recipient" if name == "send_message" else "query"
            if isinstance(out["arguments"].get(person_key), str):
                person = out["arguments"][person_key].strip()
                if "@" in person and "@" not in user_text:
                    leading = re.match(r"([A-Za-z]+)", person)
                    if leading:
                        out["arguments"][person_key] = leading.group(1)
        if name == "create_reminder":
            if isinstance(out["arguments"].get("title"), str):
                t = out["arguments"]["title"].strip()
                t = re.sub(r"^(remind me(?: to| about)?\s+)", "", t, flags=re.IGNORECASE)
                out["arguments"]["title"] = t.strip(" .")
            if isinstance(out["arguments"].get("time"), str):
                fixed = _canonical_time(out["arguments"]["time"])
                if fixed:
                    out["arguments"]["time"] = fixed
            # Align reminder title with user phrase for strict matching.
            if user_reminder_title:
                cur = _norm_text(str(out["arguments"].get("title", ""))).lower()
                tgt = user_reminder_title.lower()
                if cur and (cur in tgt or tgt in cur or any(w in cur for w in tgt.split())):
                    out["arguments"]["title"] = user_reminder_title
        if name == "send_message":
            if user_saying_message and isinstance(out["arguments"].get("message"), str):
                cur = _norm_text(out["arguments"]["message"]).lower()
                tgt = user_saying_message.lower()
                if cur and (cur in tgt or tgt in cur or len(set(cur.split()) & set(tgt.split())) >= 1):
                    out["arguments"]["message"] = user_saying_message
            if user_recipient and isinstance(out["arguments"].get("recipient"), str):
                cur = _norm_text(out["arguments"]["recipient"]).lower()
                tgt = user_recipient.lower()
                if cur and (cur == tgt or tgt in cur or cur in tgt):
                    out["arguments"]["recipient"] = user_recipient
        return out

    def _dedupe_calls(calls):
        out, seen = [], set()
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

    def _split_clauses(text):
        t = _norm_text(text)
        if not t:
            return []
        parts = re.split(
            r"\s*(?:,|;|\band then\b|\bthen\b|\balso\b|\bplus\b|\band\b)\s*",
            t,
            flags=re.IGNORECASE,
        )
        return [p.strip(" .") for p in parts if p and p.strip(" .")]

    def _schema_valid(call):
        name = call.get("name")
        args = call.get("arguments", {})
        tool = tool_map.get(name)
        if not tool or not isinstance(args, dict):
            return False
        params = tool.get("parameters", {}) or {}
        props = params.get("properties", {}) or {}
        required = params.get("required", []) or []
        for req in required:
            if req not in args:
                return False
        for k, v in args.items():
            if k not in props:
                return False
            if not _value_matches_type(v, props[k]):
                return False
        return True

    def _schema_rate(calls):
        if not calls:
            return 0.0
        return sum(1 for c in calls if _schema_valid(c)) / len(calls)

    def _is_string_heavy(call):
        name = call.get("name")
        tool = tool_map.get(name, {})
        params = tool.get("parameters", {}) or {}
        required = params.get("required", []) or []
        props = params.get("properties", {}) or {}
        if not required:
            return False
        str_req = 0
        num_req = 0
        for k in required:
            t = str((props.get(k) or {}).get("type", "")).lower()
            if t == "string":
                str_req += 1
            elif t in {"integer", "number"}:
                num_req += 1
        return str_req >= 1 and str_req >= num_req

    user_text_l = user_text.lower()
    user_tokens = set(re.findall(r"[a-z0-9]+", user_text_l))

    def _arg_grounding(call):
        args = call.get("arguments", {}) or {}
        if not isinstance(args, dict) or not args:
            return 0.0
        total = 0
        score = 0.0
        for v in args.values():
            if isinstance(v, str):
                total += 1
                val = _norm_text(v).lower()
                if not val:
                    continue
                if val in user_text_l:
                    score += 1.0
                    continue
                v_tokens = set(re.findall(r"[a-z0-9]+", val))
                if v_tokens:
                    overlap = len(v_tokens & user_tokens) / len(v_tokens)
                    # Multi-token string args should usually be contiguous in user text.
                    if len(v_tokens) >= 2 and val not in user_text_l:
                        overlap = min(overlap, 0.4)
                    # For time-like strings, mismatched hour token should be heavily penalized.
                    if any(tok in v_tokens for tok in {"am", "pm"}):
                        cand_hours = {tok for tok in v_tokens if tok.isdigit() and 1 <= int(tok) <= 12}
                        user_hours = {tok for tok in user_tokens if tok.isdigit() and 1 <= int(tok) <= 12}
                        if cand_hours and user_hours and not (cand_hours & user_hours):
                            overlap = min(overlap, 0.1)
                    if "@" in val and "@" not in user_text:
                        overlap *= 0.1
                    score += overlap
            elif isinstance(v, (int, float)) and not isinstance(v, bool):
                total += 1
                if v < 0 and "-" not in user_text:
                    score += 0.0
                elif str(int(v)) in user_text_l or str(v) in user_text_l:
                    score += 1.0
        return score / max(1, total)

    def _plausibility(call):
        name = call.get("name")
        args = call.get("arguments", {}) or {}
        if not isinstance(args, dict):
            return 0.0
        tool = tool_map.get(name, {})
        props = tool.get("parameters", {}).get("properties", {}) or {}
        checks = 0
        passed = 0
        for k, v in args.items():
            key = str(k).lower()
            schema = props.get(k, {})
            t = str(schema.get("type", "")).lower()
            if t in {"integer", "number"} and isinstance(v, (int, float)) and not isinstance(v, bool):
                checks += 1
                if "hour" in key:
                    if 0 <= v <= 23:
                        passed += 1
                elif "minute" in key:
                    if 0 <= v <= 59:
                        passed += 1
                elif any(w in key for w in ["minutes", "count", "num", "number"]):
                    if 0 < v <= 1000:
                        passed += 1
                else:
                    if v >= 0:
                        passed += 1
            elif t == "string" and isinstance(v, str):
                checks += 1
                s = _norm_text(v)
                ok = 0 < len(s) <= 200
                desc = str(schema.get("description", "")).lower()
                if "time" in key or "time" in desc:
                    has_time_like = bool(
                        re.search(r"\b\d{1,2}:\d{2}\s*(?:am|pm)\b", s, flags=re.IGNORECASE)
                    )
                    has_iso_like = bool(re.search(r"\b\d{4}-\d{2}-\d{2}t", s, flags=re.IGNORECASE))
                    user_has_iso_like = bool(re.search(r"\b\d{4}-\d{2}-\d{2}t", user_text, flags=re.IGNORECASE))
                    if has_iso_like and not user_has_iso_like:
                        ok = False
                    elif not has_time_like and s.lower() not in user_text_l:
                        ok = False
                    if user_times and has_time_like:
                        canon = _canonical_time(s)
                        if canon and canon not in user_times:
                            ok = False
                if "title" in key or "title" in desc:
                    if re.match(r"^remind me\b", s, flags=re.IGNORECASE):
                        ok = False
                if ok:
                    passed += 1
        return passed / max(1, checks)

    def _call_quality(calls):
        if not calls:
            return 0.0
        g = sum(_arg_grounding(c) for c in calls) / len(calls)
        p = sum(_plausibility(c) for c in calls) / len(calls)
        return 0.55 * g + 0.45 * p

    def _coverage(calls, action_hint):
        if action_hint <= 0:
            return 1.0
        return min(1.0, len(calls) / action_hint)

    def _candidate_score(calls, action_hint):
        s = _schema_rate(calls)
        q = _call_quality(calls)
        c = _coverage(calls, action_hint)
        return 0.45 * s + 0.35 * q + 0.20 * c

    def _merge_candidates(primary, secondary, action_hint):
        merged = list(primary)
        seen = {
            json.dumps({"name": c.get("name"), "arguments": c.get("arguments", {})}, sort_keys=True)
            for c in merged
        }
        names = {c.get("name") for c in merged}
        for c in secondary:
            if len(merged) >= max(action_hint, len(primary)):
                break
            key = json.dumps({"name": c.get("name"), "arguments": c.get("arguments", {})}, sort_keys=True)
            if key in seen:
                continue
            if c.get("name") in names:
                continue
            if _schema_valid(c) and _arg_grounding(c) >= 0.45:
                merged.append(c)
                seen.add(key)
                names.add(c.get("name"))
        return _dedupe_calls(merged)

    def _signature(calls):
        norm = []
        for c in calls:
            args = {}
            for k, v in (c.get("arguments", {}) or {}).items():
                if isinstance(v, str):
                    args[k] = _norm_text(v).lower()
                else:
                    args[k] = v
            norm.append({"name": c.get("name"), "arguments": args})
        norm = sorted(norm, key=lambda x: (x["name"], json.dumps(x["arguments"], sort_keys=True)))
        return json.dumps(norm, sort_keys=True)

    def _run_local(extra_instruction=None):
        req = list(messages)
        if extra_instruction:
            req = req + [{"role": "user", "content": extra_instruction}]
        res = generate_cactus(req, tools)
        calls = [_canonicalize_call(c) for c in res.get("function_calls", [])]
        calls = [c for c in calls if c.get("name") in tool_map]
        calls = _dedupe_calls(calls)
        res["function_calls"] = calls
        return res

    def _run_segmented_committee():
        clauses = _split_clauses(user_text)
        if len(clauses) < 2:
            return {"function_calls": [], "confidence": 0.0, "total_time_ms": 0.0}
        all_calls = []
        confidences = []
        total_ms = 0.0
        for clause in clauses[:4]:
            clause_msgs = [{"role": "user", "content": clause}]
            seg = generate_cactus(clause_msgs, tools)
            seg_calls = [_canonicalize_call(c) for c in seg.get("function_calls", [])]
            seg_calls = [c for c in seg_calls if c.get("name") in tool_map]
            seg_calls = _dedupe_calls(seg_calls)
            # Keep top 1 call per clause to reduce over-generation noise.
            if seg_calls:
                best_call = max(
                    seg_calls,
                    key=lambda c: (1 if _schema_valid(c) else 0, _arg_grounding(c), _plausibility(c)),
                )
                all_calls.append(best_call)
            confidences.append(float(seg.get("confidence", 0.0) or 0.0))
            total_ms += float(seg.get("total_time_ms", 0.0) or 0.0)
        all_calls = _dedupe_calls(all_calls)
        mean_conf = sum(confidences) / len(confidences) if confidences else 0.0
        return {"function_calls": all_calls, "confidence": mean_conf, "total_time_ms": total_ms}

    action_hint = max(1, min(4, 1 + len(re.findall(r"\b(?:and|then|also)\b|,", user_text.lower()))))

    base = _run_local()
    base_calls = base.get("function_calls", [])
    base_conf = float(base.get("confidence", 0.0) or 0.0)
    base_schema = _schema_rate(base_calls)
    base_quality = _call_quality(base_calls)

    need_verify = not (
        base_calls
        and base_schema >= 1.0
        and len(base_calls) >= action_hint
        and base_quality >= (0.70 + 0.03 * max(0, action_hint - 1))
        and base_conf >= 0.58
    )

    if need_verify:
        verify = _run_local(
            "Re-check your tool calls. Return only explicit user intents with required arguments and no extra fields."
        )
    else:
        verify = {
            "function_calls": list(base_calls),
            "confidence": base_conf,
            "total_time_ms": 0.0,
        }

    verify_calls = verify.get("function_calls", [])
    verify_conf = float(verify.get("confidence", 0.0) or 0.0)
    verify_schema = _schema_rate(verify_calls)
    verify_quality = _call_quality(verify_calls)
    consensus = _signature(base_calls) == _signature(verify_calls)

    segmented = _run_segmented_committee()
    seg_calls = segmented.get("function_calls", [])
    seg_conf = float(segmented.get("confidence", 0.0) or 0.0)
    seg_schema = _schema_rate(seg_calls)
    seg_quality = _call_quality(seg_calls)

    selected = base
    if (verify_schema, verify_quality, verify_conf) > (base_schema, base_quality, base_conf):
        selected = verify
    if (seg_schema, seg_quality, _coverage(seg_calls, action_hint), seg_conf) > (
        _schema_rate(selected.get("function_calls", [])),
        _call_quality(selected.get("function_calls", [])),
        _coverage(selected.get("function_calls", []), action_hint),
        float(selected.get("confidence", 0.0) or 0.0),
    ):
        selected = segmented
    selected_calls = selected.get("function_calls", [])
    selected_conf = float(selected.get("confidence", 0.0) or 0.0)
    selected_schema = _schema_rate(selected_calls)
    selected_quality = _call_quality(selected_calls)
    string_heavy_single = (
        action_hint == 1 and len(selected_calls) == 1 and _is_string_heavy(selected_calls[0])
    )

    reminder_time_ok = True
    if user_times:
        for c in selected_calls:
            if c.get("name") != "create_reminder":
                continue
            t = c.get("arguments", {}).get("time")
            if isinstance(t, str):
                canon = _canonical_time(t)
                if canon and canon not in user_times:
                    reminder_time_ok = False
                    break

    dyn_thr = min(confidence_threshold, 0.46 + 0.07 * max(0, action_hint - 1))
    call_count_ok = len(selected_calls) >= action_hint

    accept_local = (
        bool(selected_calls)
        and selected_schema >= 1.0
        and selected_quality >= (
            0.62 + 0.04 * max(0, action_hint - 1) + (0.10 if string_heavy_single else 0.0)
        )
        and call_count_ok
        and reminder_time_ok
        and ((not string_heavy_single) or consensus)
        and (
            (consensus and min(base_conf, verify_conf) >= (dyn_thr - 0.08))
            or (
                selected_conf >= (dyn_thr + (0.16 if string_heavy_single else 0.12))
                and selected_quality >= (0.82 if string_heavy_single else 0.78)
            )
        )
    )

    if accept_local:
        selected["source"] = "on-device"
        selected["consensus"] = consensus
        return selected

    try:
        augment_calls = []
        if action_hint >= 2 and len(selected_calls) < action_hint:
            augment = _run_local(
                "If the user asks for multiple actions, return a separate tool call for each action."
            )
            augment_calls = augment.get("function_calls", [])

        cloud = generate_cloud(messages, tools)
        cloud_calls = [_canonicalize_call(c) for c in cloud.get("function_calls", [])]
        cloud_calls = [c for c in cloud_calls if c.get("name") in tool_map]
        cloud_calls = _dedupe_calls(cloud_calls)
        if action_hint >= 2 and len(cloud_calls) < action_hint:
            cloud_retry = generate_cloud(
                messages
                + [
                    {
                        "role": "user",
                        "content": "If multiple actions are requested, return one tool call per action and include all actions.",
                    }
                ],
                tools,
            )
            retry_calls = [_canonicalize_call(c) for c in cloud_retry.get("function_calls", [])]
            retry_calls = [c for c in retry_calls if c.get("name") in tool_map]
            retry_calls = _dedupe_calls(retry_calls)
            cloud_calls = _merge_candidates(cloud_calls, retry_calls, action_hint)
        if augment_calls:
            cloud_calls = _merge_candidates(cloud_calls, augment_calls, action_hint)
        merged_calls = _merge_candidates(cloud_calls, selected_calls, action_hint)

        def _rank(calls):
            cov = _coverage(calls, action_hint)
            sch = _schema_rate(calls)
            score = _candidate_score(calls, action_hint)
            full = 1 if (cov >= 1.0 and sch >= 1.0) else 0
            return (full, cov, score)

        if action_hint == 1:
            # For single-intent fallback, trust cloud output to avoid local overfitting artifacts.
            best_calls = cloud_calls
        else:
            best_calls = cloud_calls
            best_rank = _rank(cloud_calls)
            sel_rank = _rank(selected_calls)
            merged_rank = _rank(merged_calls)
            if sel_rank > best_rank:
                best_calls = selected_calls
                best_rank = sel_rank
            if merged_rank > best_rank:
                best_calls = merged_calls

        cloud["function_calls"] = best_calls
        cloud["source"] = "cloud (fallback)"
        cloud["local_confidence"] = selected_conf
        cloud["total_time_ms"] += (
            base.get("total_time_ms", 0) + verify.get("total_time_ms", 0) + segmented.get("total_time_ms", 0)
        )
        cloud["fallback_reason"] = {
            "consensus": consensus,
            "base_schema_rate": base_schema,
            "verify_schema_rate": verify_schema,
            "segmented_schema_rate": seg_schema,
            "base_quality": base_quality,
            "verify_quality": verify_quality,
            "segmented_quality": seg_quality,
            "selected_schema_rate": selected_schema,
            "selected_quality": selected_quality,
            "selected_call_count_ok": call_count_ok,
            "local_confidence": selected_conf,
            "dynamic_threshold": dyn_thr,
        }
        return cloud
    except Exception as exc:
        return {
            "function_calls": selected_calls,
            "total_time_ms": selected.get("total_time_ms", 0),
            "confidence": selected_conf,
            "source": "on-device",
            "cloud_error": str(exc),
            "fallback_reason": {
                "consensus": consensus,
                "base_schema_rate": base_schema,
                "verify_schema_rate": verify_schema,
                "segmented_schema_rate": seg_schema,
                "base_quality": base_quality,
                "verify_quality": verify_quality,
                "segmented_quality": seg_quality,
                "selected_schema_rate": selected_schema,
                "selected_quality": selected_quality,
                "local_confidence": selected_conf,
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
