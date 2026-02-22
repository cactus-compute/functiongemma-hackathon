#!/usr/bin/env python3
"""
Test script for Cactus RAG + FunctionGemma tool calling.

Demonstrates a private health companion pipeline:
  1. Query local health documents via RAG (cactus_rag_query)
  2. Inject retrieved context into FunctionGemma prompt
  3. FunctionGemma produces a tool call with context-aware arguments

Optionally adds Whisper transcription for the full voice→RAG→tool-call pipeline.

Usage:
    source cactus/venv/bin/activate
    python test-rag.py                    # Run all RAG tests
    python test-rag.py --pipeline         # Full: voice → transcribe → RAG → tool call
    python test-rag.py --rag-only         # Just test RAG retrieval quality
    python test-rag.py --query "Am I allergic to ibuprofen?"

Prerequisites:
    cactus download openai/whisper-small --reconvert   (for --pipeline mode)
    Corpus files in test_corpus/ (created automatically)
"""

import ctypes
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, "cactus/python/src")

from cactus import (
    _lib,
    cactus_complete,
    cactus_destroy,
    cactus_init,
    cactus_reset,
    cactus_transcribe,
)

# ── Paths ─────────────────────────────────────────────────────────────

RAG_MODEL_PATH = "cactus/weights/lfm2-vl-450m"
FG_MODEL_PATH = "cactus/weights/functiongemma-270m-it"
WHISPER_MODEL_PATH = "cactus/weights/whisper-small"
CORPUS_DIR = "test_corpus"
AUDIO_DIR = "test_audio"

# ── Whisper prompt ────────────────────────────────────────────────────
WHISPER_PROMPT = "<|startoftranscript|><|en|><|transcribe|><|notimestamps|>"

# ── FunctionGemma knobs (optimal from sweep) ─────────────────────────
FG_SYSTEM_MSG = "You are a helpful assistant that can use tools."
FG_FORCE_TOOLS = True
FG_MAX_TOKENS = 256
FG_TOOL_RAG_TOP_K = 0
FG_TEMPERATURE = 0.0

# ── Tools — named to match FunctionGemma's training distribution ─────
# FG 270M is very literal. Use simple verbs it knows: "search", "create",
# "send", "set". Keep parameter schemas minimal.

TOOLS = [
    {
        "name": "search_records",
        "description": "Search personal records, notes, health logs, or documents",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "What to search for"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "create_note",
        "description": "Create a note or log entry",
        "parameters": {
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "Note title"},
                "content": {"type": "string", "description": "Note content"},
            },
            "required": ["title", "content"],
        },
    },
    {
        "name": "set_reminder",
        "description": "Set a reminder for a specific time",
        "parameters": {
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "What to be reminded about"},
                "time": {"type": "string", "description": "When to remind (e.g. 8:00 PM)"},
            },
            "required": ["title", "time"],
        },
    },
    {
        "name": "send_message",
        "description": "Send a message to a contact",
        "parameters": {
            "type": "object",
            "properties": {
                "recipient": {"type": "string", "description": "Name of the person"},
                "message": {"type": "string", "description": "The message content"},
            },
            "required": ["recipient", "message"],
        },
    },
    {
        "name": "make_call",
        "description": "Make a phone call to a contact",
        "parameters": {
            "type": "object",
            "properties": {
                "contact": {"type": "string", "description": "Name or number to call"},
                "reason": {"type": "string", "description": "Reason for the call"},
            },
            "required": ["contact"],
        },
    },
    {
        "name": "create_alert",
        "description": "Create an urgent alert or emergency notification",
        "parameters": {
            "type": "object",
            "properties": {
                "message": {"type": "string", "description": "Alert message"},
                "urgency": {"type": "string", "description": "low, medium, high, emergency"},
            },
            "required": ["message"],
        },
    },
]

# ── Test cases ───────────────────────────────────────────────────────
# Designed around FG 270M's strengths: direct verbs, literal phrasing.
# (user_query, expected_tool, description)

TEST_CASES = [
    # search_records — FG's strongest pattern with these tools
    (
        "Search my records for medication interactions with ibuprofen.",
        "search_records",
        "Med safety — RAG finds Warfarin/NSAID warning",
    ),
    (
        "Search my records for recent blood sugar readings.",
        "search_records",
        "Health lookup — RAG finds Feb 19: 142 mg/dL",
    ),
    (
        "Search my records for my cardiologist phone number.",
        "search_records",
        "Doctor lookup — RAG finds Dr. Chen 555-0198",
    ),
    (
        "Search my records for drug allergies.",
        "search_records",
        "Allergy check — RAG finds Penicillin, Sulfa, Codeine",
    ),
    (
        "Search my records for my emergency contact.",
        "search_records",
        "Emergency lookup — RAG finds Jane Doe 555-0100",
    ),
    # set_reminder — works with direct phrasing
    (
        "Set a reminder to take Warfarin at 8 PM.",
        "set_reminder",
        "Med reminder — RAG confirms: Warfarin 5mg, evening",
    ),
    (
        "Set a reminder for doctor appointment on March 5.",
        "set_reminder",
        "Appt reminder — RAG: Dr. Chen, cardiology follow-up",
    ),
    # send_message — sometimes works
    (
        "Send a message to Dr. Martinez saying I need to reschedule.",
        "send_message",
        "Doctor msg — RAG: Dr. Martinez, Family Medicine, 555-0142",
    ),
]


# ── Helpers ───────────────────────────────────────────────────────────


def banner(text):
    w = 64
    print(f"\n{'═' * w}")
    print(f"  {text}")
    print(f"{'═' * w}")


def section(text):
    print(f"\n{'─' * 64}")
    print(f"  {text}")
    print(f"{'─' * 64}")


def rag_query(model, query, top_k=3):
    """
    RAG query with workaround for Python wrapper bug.
    The C function returns bytes-written (not 0=success), so the wrapper
    incorrectly treats all results as errors. We call C directly.
    """
    buf = ctypes.create_string_buffer(65536)
    _lib.cactus_rag_query(
        model,
        query.encode() if isinstance(query, str) else query,
        buf,
        len(buf),
        top_k,
    )
    raw = buf.value.decode("utf-8", errors="ignore")
    if not raw:
        return []
    try:
        parsed = json.loads(raw)
        return parsed.get("chunks", [])
    except json.JSONDecodeError:
        return []


def clean_corpus_cache():
    """Remove cached index files so RAG rebuilds fresh."""
    for f in ["index.bin", "data.bin"]:
        p = os.path.join(CORPUS_DIR, f)
        if os.path.exists(p):
            os.remove(p)


def summarize_rag_context(chunks, max_chars=200):
    """Extract a very short context summary from RAG chunks.
    FunctionGemma 270M can't handle long context — keep it to 1-2 sentences max."""
    if not chunks:
        return ""
    # Just take the first chunk's content, trimmed hard
    text = chunks[0]["content"].replace("\n", " ").replace("  ", " ").strip()
    if len(text) > max_chars:
        text = text[:max_chars].rsplit(" ", 1)[0] + "..."
    return text


def call_functiongemma(fg_model, prompt, context=""):
    """Call FunctionGemma with optional short RAG context."""
    if context:
        # Keep it SHORT — FG 270M gets confused with long context
        full_prompt = f"{prompt}\n\nRelevant info: {context}"
    else:
        full_prompt = prompt

    messages = [
        {"role": "system", "content": FG_SYSTEM_MSG},
        {"role": "user", "content": full_prompt},
    ]
    cactus_tools = [{"type": "function", "function": t} for t in TOOLS]

    t0 = time.time()
    raw = cactus_complete(
        fg_model,
        messages,
        tools=cactus_tools,
        force_tools=FG_FORCE_TOOLS,
        max_tokens=FG_MAX_TOKENS,
        tool_rag_top_k=FG_TOOL_RAG_TOP_K,
        temperature=FG_TEMPERATURE,
        stop_sequences=["<|im_end|>", "<end_of_turn>"],
    )
    elapsed_ms = (time.time() - t0) * 1000

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        parsed = {"_raw": raw, "_parse_error": True}

    return parsed, elapsed_ms, full_prompt


def transcribe_audio(whisper_model, audio_path):
    """Transcribe a WAV file."""
    cactus_reset(whisper_model)
    t0 = time.time()
    raw = cactus_transcribe(whisper_model, audio_path, prompt=WHISPER_PROMPT)
    elapsed_ms = (time.time() - t0) * 1000
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        parsed = {"response": raw}
    return parsed.get("response", "").strip(), elapsed_ms


# ── Test modes ────────────────────────────────────────────────────────


def test_rag_only():
    """Test RAG retrieval quality — does it find the right documents?"""
    banner("RAG Retrieval Quality Test")
    clean_corpus_cache()

    print("Loading RAG model with corpus...")
    rag_model = cactus_init(RAG_MODEL_PATH, corpus_dir=CORPUS_DIR)
    if not rag_model:
        print("  ❌ Failed to load RAG model")
        return
    print("  ✅ RAG model loaded.\n")

    queries = [
        ("What medications am I taking?", "medications.md"),
        ("Am I allergic to penicillin?", "allergies.md"),
        ("Who is my cardiologist?", "doctors.md"),
        ("I have chest pain", "symptoms_log.md"),
        ("What was my blood sugar?", "symptoms_log.md"),
        ("What is my blood pressure goal?", "health_goals.md"),
        ("Who is my emergency contact?", "doctors.md"),
        ("Can I take ibuprofen?", "medications.md"),
    ]

    correct = 0
    for query, expected_source in queries:
        t0 = time.time()
        chunks = rag_query(rag_model, query, top_k=3)
        ms = (time.time() - t0) * 1000

        top_source = chunks[0]["source"] if chunks else "none"
        match = top_source == expected_source
        status = "✅" if match else "❌"
        if match:
            correct += 1

        print(f"  {status} [{ms:5.0f}ms] {query}")
        for i, chunk in enumerate(chunks[:3]):
            marker = "→" if i == 0 else " "
            text_preview = chunk["content"][:70].replace("\n", " ")
            print(
                f"      {marker} [{chunk['score']:.4f}] {chunk['source']}: {text_preview}..."
            )
        if not match:
            print(f"      Expected top: {expected_source}, got: {top_source}")
        print()

    cactus_destroy(rag_model)
    section(f"RAG Accuracy: {correct}/{len(queries)} top-1 correct")


def test_rag_plus_toolcall():
    """Test RAG context injection → FunctionGemma tool calling."""
    banner("RAG + FunctionGemma Tool Calling")
    clean_corpus_cache()

    print("Loading RAG model with corpus...")
    rag_model = cactus_init(RAG_MODEL_PATH, corpus_dir=CORPUS_DIR)
    print("  ✅ RAG model loaded.")

    print("Loading FunctionGemma model...")
    fg_model = cactus_init(FG_MODEL_PATH)
    print("  ✅ FunctionGemma loaded.\n")

    correct_no_rag = 0
    correct_with_rag = 0

    for query, expected_tool, desc in TEST_CASES:
        section(f"💬 {desc}")
        print(f'  User: "{query}"')

        # Step 1: RAG retrieval
        t0 = time.time()
        chunks = rag_query(rag_model, query, top_k=2)
        rag_ms = (time.time() - t0) * 1000
        context = summarize_rag_context(chunks)

        print(f"\n  📚 RAG ({rag_ms:.0f}ms) — top sources: {', '.join(set(c['source'] for c in chunks[:2]))}")
        if context:
            print(f"     Context: {context[:120]}...")

        # Step 2a: FunctionGemma WITHOUT context (baseline)
        cactus_reset(fg_model)
        fg_no_ctx, fg_no_ms, _ = call_functiongemma(fg_model, query, "")
        calls_no = fg_no_ctx.get("function_calls", [])
        tool_no = any(c["name"] == expected_tool for c in calls_no)
        if tool_no:
            correct_no_rag += 1

        # Step 2b: FunctionGemma WITH RAG context
        cactus_reset(fg_model)
        fg_ctx, fg_ctx_ms, _ = call_functiongemma(fg_model, query, context)
        calls_ctx = fg_ctx.get("function_calls", [])
        tool_ctx = any(c["name"] == expected_tool for c in calls_ctx)
        if tool_ctx:
            correct_with_rag += 1

        # Display comparison
        s_no = "✅" if tool_no else "❌"
        s_ctx = "✅" if tool_ctx else "❌"

        print(f"\n  🔧 Without RAG ({fg_no_ms:.0f}ms) {s_no}:")
        for c in calls_no:
            print(f"     → {c['name']}({json.dumps(c.get('arguments', {}))})")
        if not calls_no:
            print(f"     (no tool call — {fg_no_ctx.get('response', '')[:80]})")

        print(f"\n  🔧 With RAG ({fg_ctx_ms:.0f}ms) {s_ctx}:")
        for c in calls_ctx:
            print(f"     → {c['name']}({json.dumps(c.get('arguments', {}))})")
        if not calls_ctx:
            print(f"     (no tool call — {fg_ctx.get('response', '')[:80]})")

        total_ms = rag_ms + fg_ctx_ms
        print(f"\n  ⏱️  Total: {total_ms:.0f}ms (rag={rag_ms:.0f}ms + fg={fg_ctx_ms:.0f}ms) | 100% on-device")

    cactus_destroy(fg_model)
    cactus_destroy(rag_model)

    section(f"Tool Accuracy: no-RAG={correct_no_rag}/{len(TEST_CASES)}, with-RAG={correct_with_rag}/{len(TEST_CASES)}")


def test_full_pipeline():
    """Full pipeline: Voice → Whisper → RAG → FunctionGemma → Tool Call."""
    banner("Full Pipeline: Voice → Transcribe → RAG → Tool Call")
    clean_corpus_cache()

    # Generate test audio if needed — using phrasing FG 270M can handle
    os.makedirs(AUDIO_DIR, exist_ok=True)
    pipeline_audio = {
        "rag_ibuprofen.wav": "Search my records for medication interactions with ibuprofen.",
        "rag_bloodsugar.wav": "Search my records for recent blood sugar readings.",
        "rag_doctor.wav": "Search my records for my cardiologist phone number.",
        "rag_reminder.wav": "Set a reminder to take Warfarin at 8 PM.",
        "rag_allergies.wav": "Search my records for drug allergies.",
    }
    for fname, text in pipeline_audio.items():
        wav_path = os.path.join(AUDIO_DIR, fname)
        if not os.path.exists(wav_path):
            print(f"  Generating {fname}...")
            aiff = wav_path.replace(".wav", ".aiff")
            subprocess.run(["say", "-o", aiff, text], check=True)
            subprocess.run(
                ["ffmpeg", "-y", "-i", aiff, "-ar", "16000", "-ac", "1", "-sample_fmt", "s16", wav_path],
                capture_output=True, check=True,
            )
            os.remove(aiff)
    print("  ✅ Audio files ready.\n")

    # Load all three models
    print("Loading Whisper...")
    whisper = cactus_init(WHISPER_MODEL_PATH)
    print("  ✅ Whisper loaded.")

    print("Loading RAG model with corpus...")
    rag_model = cactus_init(RAG_MODEL_PATH, corpus_dir=CORPUS_DIR)
    print("  ✅ RAG model loaded.")

    print("Loading FunctionGemma...")
    fg_model = cactus_init(FG_MODEL_PATH)
    print("  ✅ FunctionGemma loaded.\n")

    expected_tools = {
        "rag_ibuprofen.wav": "search_records",
        "rag_bloodsugar.wav": "search_records",
        "rag_doctor.wav": "search_records",
        "rag_reminder.wav": "set_reminder",
        "rag_allergies.wav": "search_records",
    }

    for fname, expected_tool in expected_tools.items():
        wav_path = os.path.join(AUDIO_DIR, fname)
        if not os.path.exists(wav_path):
            continue

        section(f"🎤 {fname}")

        # Step 1: Whisper transcription
        transcript, whisper_ms = transcribe_audio(whisper, wav_path)
        print(f"  [Whisper  {whisper_ms:6.0f}ms] → {transcript!r}")

        if not transcript.strip():
            print("  ⚠️  Empty transcription, skipping.")
            continue

        # Step 2: RAG retrieval
        t0 = time.time()
        chunks = rag_query(rag_model, transcript, top_k=2)
        rag_ms = (time.time() - t0) * 1000

        context = summarize_rag_context(chunks)
        sources = ', '.join(set(c['source'] for c in chunks[:2])) if chunks else 'none'
        print(f"  [RAG      {rag_ms:6.0f}ms] → {len(chunks)} chunks from {sources}")
        if context:
            print(f"                         📋 {context[:100]}...")

        # Step 3: FunctionGemma tool call with RAG context
        cactus_reset(fg_model)
        fg_result, fg_ms, _ = call_functiongemma(fg_model, transcript, context)
        calls = fg_result.get("function_calls", [])

        tool_correct = any(c["name"] == expected_tool for c in calls)
        status = "✅" if tool_correct else "❌"

        print(f"  [FuncGem  {fg_ms:6.0f}ms] → {status} {len(calls)} call(s)")
        for c in calls:
            args_str = json.dumps(c.get("arguments", {}))
            print(f"                         → {c['name']}({args_str})")
        if not calls:
            resp = fg_result.get("response", "")[:100]
            print(f"                         (text: {resp})")

        total_ms = whisper_ms + rag_ms + fg_ms
        print(f"  ⏱️  Total: {total_ms:.0f}ms | whisper={whisper_ms:.0f} + rag={rag_ms:.0f} + fg={fg_ms:.0f} | 100% on-device 🔒")

    cactus_destroy(fg_model)
    cactus_destroy(rag_model)
    cactus_destroy(whisper)


def test_single_query(query):
    """Run a single query through RAG + FunctionGemma."""
    banner(f'Query: "{query}"')
    clean_corpus_cache()

    rag_model = cactus_init(RAG_MODEL_PATH, corpus_dir=CORPUS_DIR)
    fg_model = cactus_init(FG_MODEL_PATH)

    # RAG
    chunks = rag_query(rag_model, query, top_k=3)
    context = "\n---\n".join(c["content"] for c in chunks[:2])

    print("\n📚 RAG context:")
    for chunk in chunks[:3]:
        preview = chunk["content"][:100].replace("\n", " ")
        print(f"  [{chunk['score']:.4f}] {chunk['source']}: {preview}")

    # FunctionGemma
    fg_result, fg_ms, full_prompt = call_functiongemma(fg_model, query, context)
    calls = fg_result.get("function_calls", [])

    print(f"\n🔧 FunctionGemma ({fg_ms:.0f}ms):")
    for c in calls:
        print(f"  → {c['name']}({json.dumps(c.get('arguments', {}), indent=4)})")
    if not calls:
        print(f"  (text: {fg_result.get('response', 'none')[:150]})")

    print(f"\n📝 Full prompt sent to FG:")
    print(f"  {full_prompt[:300]}...")

    cactus_destroy(fg_model)
    cactus_destroy(rag_model)


# ── Main ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = sys.argv[1:]

    if not args:
        test_rag_only()
        test_rag_plus_toolcall()

    elif args[0] == "--rag-only":
        test_rag_only()

    elif args[0] == "--toolcall":
        test_rag_plus_toolcall()

    elif args[0] == "--pipeline":
        test_full_pipeline()

    elif args[0] == "--all":
        test_rag_only()
        test_rag_plus_toolcall()
        test_full_pipeline()

    elif args[0] == "--query" and len(args) > 1:
        test_single_query(" ".join(args[1:]))

    else:
        print("Usage:")
        print('  python test-rag.py                                  # RAG + tool call tests')
        print('  python test-rag.py --rag-only                       # Just RAG retrieval')
        print('  python test-rag.py --toolcall                       # RAG → FunctionGemma')
        print('  python test-rag.py --pipeline                       # Full voice→RAG→tool pipeline')
        print('  python test-rag.py --all                            # Everything')
        print('  python test-rag.py --query "Can I take aspirin?"    # Single query')
        sys.exit(1)
