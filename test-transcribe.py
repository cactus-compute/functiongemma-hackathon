#!/usr/bin/env python3
"""
Test script for cactus_transcribe (Whisper on-device speech-to-text).

Tests the voice-to-text pipeline using the Cactus Whisper model, then
optionally pipes the transcription into FunctionGemma for tool calling
(the full voice-to-action pipeline from Rubric 3).

Usage:
    source cactus/venv/bin/activate
    python test-transcribe.py                          # Run all built-in test cases
    python test-transcribe.py test_audio/weather.wav   # Transcribe a specific file
    python test-transcribe.py --record                 # Record from mic then transcribe
    python test-transcribe.py --pipeline               # Full voice→text→function-call pipeline

Prerequisites:
    cactus download openai/whisper-small --reconvert
    mkdir -p test_audio && (generate WAVs — see below)

Test WAV generation (macOS):
    say -o /tmp/t.aiff "What's the weather in Tokyo?" && ffmpeg -y -i /tmp/t.aiff -ar 16000 -ac 1 -sample_fmt s16 test_audio/weather.wav
"""

import json
import os
import subprocess
import sys
import time

sys.path.insert(0, "cactus/python/src")

from cactus import (
    cactus_complete,
    cactus_destroy,
    cactus_init,
    cactus_reset,
    cactus_transcribe,
    cactus_get_last_error,
)

# ── Paths ─────────────────────────────────────────────────────────────

WHISPER_MODEL_PATH = "cactus/weights/whisper-small"
FG_MODEL_PATH = "cactus/weights/functiongemma-270m-it"
AUDIO_DIR = "test_audio"

# ── Whisper prompt ────────────────────────────────────────────────────
# Standard Whisper prompt: English, transcribe mode, no timestamps
WHISPER_PROMPT = "<|startoftranscript|><|en|><|transcribe|><|notimestamps|>"

# ── Tool definitions (same as test.py / test-max.py) ─────────────────

TOOLS = [
    {
        "name": "get_weather",
        "description": "Get current weather for a location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "City name"},
            },
            "required": ["location"],
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
        "name": "set_alarm",
        "description": "Set an alarm for a given time",
        "parameters": {
            "type": "object",
            "properties": {
                "hour": {"type": "integer", "description": "Hour (0-23)"},
                "minute": {"type": "integer", "description": "Minute (0-59)"},
            },
            "required": ["hour", "minute"],
        },
    },
    {
        "name": "play_music",
        "description": "Play a song or playlist",
        "parameters": {
            "type": "object",
            "properties": {
                "song": {"type": "string", "description": "Song or playlist name"},
            },
            "required": ["song"],
        },
    },
    {
        "name": "set_timer",
        "description": "Set a countdown timer",
        "parameters": {
            "type": "object",
            "properties": {
                "minutes": {"type": "integer", "description": "Number of minutes"},
            },
            "required": ["minutes"],
        },
    },
    {
        "name": "create_reminder",
        "description": "Create a reminder with a title and time",
        "parameters": {
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "Reminder title"},
                "time": {
                    "type": "string",
                    "description": "Time for the reminder (e.g. 3:00 PM)",
                },
            },
            "required": ["title", "time"],
        },
    },
    {
        "name": "search_contacts",
        "description": "Search for a contact by name",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Name to search for"},
            },
            "required": ["query"],
        },
    },
]

# ── Built-in test cases ──────────────────────────────────────────────
# Each entry: (wav_filename, expected_transcript_substring, description)

TEST_CASES = [
    ("weather.wav", "weather", "Simple weather query"),
    ("alarm.wav", "alarm", "Set alarm command"),
    ("message.wav", "message", "Send message command"),
    ("timer.wav", "timer", "Set timer command"),
    ("music.wav", "stairway", "Play music command"),
    ("multi.wav", "weather", "Multi-action (weather + alarm)"),
]

# ── FunctionGemma knobs (optimal from test.md sweep) ─────────────────

FG_SYSTEM_MSG = "You are a helpful assistant that can use tools."
FG_FORCE_TOOLS = True
FG_MAX_TOKENS = 256
FG_TOOL_RAG_TOP_K = 0
FG_TEMPERATURE = 0.0


# ── Helpers ───────────────────────────────────────────────────────────


def banner(text):
    width = 60
    print(f"\n{'═' * width}")
    print(f"  {text}")
    print(f"{'═' * width}")


def section(text):
    print(f"\n{'─' * 60}")
    print(f"  {text}")
    print(f"{'─' * 60}")


def transcribe_audio(whisper_model, audio_path, prompt=WHISPER_PROMPT):
    """Transcribe a WAV file and return (transcript, time_ms, raw_json)."""
    # Reset model state to avoid KV cache contamination between files
    cactus_reset(whisper_model)

    t0 = time.time()
    raw = cactus_transcribe(whisper_model, audio_path, prompt=prompt)
    elapsed_ms = (time.time() - t0) * 1000

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        parsed = {"response": raw, "_parse_error": True}

    transcript = parsed.get("response", "").strip()

    # Check for errors
    err = cactus_get_last_error()
    if err:
        parsed["_cactus_error"] = err

    return transcript, elapsed_ms, parsed


def call_functiongemma(fg_model, prompt):
    """Call FunctionGemma with the transcribed text. Returns (parsed, time_ms)."""
    messages = [
        {"role": "system", "content": FG_SYSTEM_MSG},
        {"role": "user", "content": prompt},
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

    return parsed, elapsed_ms


def record_from_mic(output_path="test_audio/recorded.wav", duration=5):
    """Record audio from the default microphone using ffmpeg (macOS)."""
    print(f"🎙️  Recording {duration}s from microphone...")
    print("   Speak now!")
    cmd = [
        "ffmpeg", "-y",
        "-f", "avfoundation",
        "-i", ":0",             # default mic on macOS
        "-t", str(duration),
        "-ar", "16000",
        "-ac", "1",
        "-sample_fmt", "s16",
        output_path,
    ]
    try:
        subprocess.run(cmd, capture_output=True, timeout=duration + 5)
        print(f"   ✅ Saved to {output_path}")
        return output_path
    except Exception as e:
        print(f"   ❌ Recording failed: {e}")
        return None


def generate_test_audio():
    """Generate test WAV files using macOS 'say' command if they don't exist."""
    os.makedirs(AUDIO_DIR, exist_ok=True)
    phrases = {
        "weather.wav": "What's the weather in Tokyo?",
        "alarm.wav": "Set an alarm for 7:30 AM",
        "message.wav": "Send a message to Bob saying I'll be late",
        "timer.wav": "Set a timer for 10 minutes",
        "music.wav": "Play Stairway to Heaven",
        "multi.wav": "Check the weather in New York and set an alarm for 9 AM",
    }
    for fname, text in phrases.items():
        wav_path = os.path.join(AUDIO_DIR, fname)
        if os.path.exists(wav_path):
            continue
        print(f"  Generating {fname}...")
        aiff_path = wav_path.replace(".wav", ".aiff")
        subprocess.run(["say", "-o", aiff_path, text], check=True)
        subprocess.run([
            "ffmpeg", "-y", "-i", aiff_path,
            "-ar", "16000", "-ac", "1", "-sample_fmt", "s16",
            wav_path,
        ], capture_output=True, check=True)
        os.remove(aiff_path)
    print("  ✅ All test audio files ready.\n")


# ── Test modes ────────────────────────────────────────────────────────


def test_transcribe_only():
    """Test 1: Pure transcription — how well does Whisper understand each WAV?"""
    banner("TEST 1: Transcription Only (Whisper)")
    generate_test_audio()

    print("Loading Whisper model...")
    whisper = cactus_init(WHISPER_MODEL_PATH)
    print("  ✅ Whisper loaded.\n")

    results = []
    for wav_file, expected_substr, desc in TEST_CASES:
        wav_path = os.path.join(AUDIO_DIR, wav_file)
        if not os.path.exists(wav_path):
            print(f"  ⚠️  Skipping {wav_file} (not found)")
            continue

        transcript, ms, raw = transcribe_audio(whisper, wav_path)
        match = expected_substr.lower() in transcript.lower()
        status = "✅" if match else "❌"

        print(f"  {status} [{ms:6.0f}ms] {desc}")
        print(f"     File:       {wav_file}")
        print(f"     Transcript: {transcript!r}")
        if not match:
            print(f"     Expected:   substring {expected_substr!r}")
        print()

        results.append({
            "file": wav_file,
            "desc": desc,
            "transcript": transcript,
            "time_ms": ms,
            "match": match,
            "raw": raw,
        })

    cactus_destroy(whisper)

    # Summary
    passed = sum(1 for r in results if r["match"])
    total = len(results)
    avg_ms = sum(r["time_ms"] for r in results) / total if total else 0
    section(f"Transcription Summary: {passed}/{total} matched, avg {avg_ms:.0f}ms")

    return results


def test_pipeline():
    """Test 2: Full voice→text→function-call pipeline."""
    banner("TEST 2: Full Pipeline (Whisper → FunctionGemma)")
    generate_test_audio()

    print("Loading Whisper model...")
    whisper = cactus_init(WHISPER_MODEL_PATH)
    print("  ✅ Whisper loaded.")

    print("Loading FunctionGemma model...")
    fg = cactus_init(FG_MODEL_PATH)
    print("  ✅ FunctionGemma loaded.\n")

    for wav_file, expected_substr, desc in TEST_CASES:
        wav_path = os.path.join(AUDIO_DIR, wav_file)
        if not os.path.exists(wav_path):
            continue

        section(f"🎤 {desc}  ({wav_file})")

        # Step 1: Transcribe
        transcript, whisper_ms, _ = transcribe_audio(whisper, wav_path)
        print(f"  [Whisper {whisper_ms:.0f}ms] → {transcript!r}")

        if not transcript.strip():
            print("  ⚠️  Empty transcription, skipping FunctionGemma call.")
            continue

        # Step 2: Function calling
        cactus_reset(fg)
        fg_result, fg_ms = call_functiongemma(fg, transcript)

        calls = fg_result.get("function_calls", [])
        confidence = fg_result.get("confidence", "?")
        total_ms = whisper_ms + fg_ms

        print(f"  [FuncGemma {fg_ms:.0f}ms] → {len(calls)} call(s), confidence={confidence}")
        for c in calls:
            args_str = json.dumps(c.get("arguments", {}))
            print(f"     → {c['name']}({args_str})")
        if not calls:
            resp = fg_result.get("response", "")
            if resp:
                print(f"     (text response: {resp[:100]})")

        print(f"  ⏱️  Total: {total_ms:.0f}ms  (whisper={whisper_ms:.0f}ms + fg={fg_ms:.0f}ms)")

    cactus_destroy(fg)
    cactus_destroy(whisper)


def test_single_file(audio_path):
    """Transcribe a single audio file."""
    banner(f"Transcribe: {audio_path}")

    if not os.path.exists(audio_path):
        print(f"  ❌ File not found: {audio_path}")
        return

    print("Loading Whisper model...")
    whisper = cactus_init(WHISPER_MODEL_PATH)
    print("  ✅ Whisper loaded.\n")

    transcript, ms, raw = transcribe_audio(whisper, audio_path)
    print(f"  Transcript: {transcript!r}")
    print(f"  Time:       {ms:.0f}ms")
    print(f"  Raw JSON:   {json.dumps(raw, indent=2)}")

    cactus_destroy(whisper)
    return transcript


def test_record_and_transcribe():
    """Record from mic, transcribe, optionally run through FunctionGemma."""
    banner("Record → Transcribe → Function Call")

    os.makedirs(AUDIO_DIR, exist_ok=True)
    audio_path = record_from_mic()
    if not audio_path:
        return

    print("\nLoading Whisper model...")
    whisper = cactus_init(WHISPER_MODEL_PATH)

    transcript, whisper_ms, _ = transcribe_audio(whisper, audio_path)
    print(f"\n  🎤 Transcript: {transcript!r}  ({whisper_ms:.0f}ms)")
    cactus_destroy(whisper)

    if not transcript.strip():
        print("  ⚠️  Empty transcription.")
        return

    print("\nLoading FunctionGemma model...")
    fg = cactus_init(FG_MODEL_PATH)
    fg_result, fg_ms = call_functiongemma(fg, transcript)
    cactus_destroy(fg)

    calls = fg_result.get("function_calls", [])
    print(f"\n  🔧 FunctionGemma ({fg_ms:.0f}ms): {len(calls)} call(s)")
    for c in calls:
        args_str = json.dumps(c.get("arguments", {}))
        print(f"     → {c['name']}({args_str})")
    if not calls:
        print(f"     (response: {fg_result.get('response', 'none')[:100]})")

    total = whisper_ms + fg_ms
    print(f"\n  ⏱️  Total voice-to-action: {total:.0f}ms")


def test_whisper_prompts():
    """Test 3: Compare different Whisper prompts to see what works best."""
    banner("TEST 3: Whisper Prompt Variations")
    generate_test_audio()

    prompts = {
        "standard": "<|startoftranscript|><|en|><|transcribe|><|notimestamps|>",
        "empty": "",
        "with_context": "<|startoftranscript|><|en|><|transcribe|><|notimestamps|>Voice assistant command:",
    }

    # Use just 3 test files to keep it quick
    test_files = [
        ("weather.wav", "weather"),
        ("alarm.wav", "alarm"),
        ("message.wav", "message"),
    ]

    print("Loading Whisper model...")
    whisper = cactus_init(WHISPER_MODEL_PATH)
    print("  ✅ Whisper loaded.\n")

    for prompt_name, prompt_str in prompts.items():
        section(f"Prompt: {prompt_name!r}")
        print(f"  Value: {prompt_str!r}\n")

        for wav_file, expected in test_files:
            wav_path = os.path.join(AUDIO_DIR, wav_file)
            if not os.path.exists(wav_path):
                continue

            transcript, ms, _ = transcribe_audio(whisper, wav_path, prompt=prompt_str)
            match = expected.lower() in transcript.lower()
            status = "✅" if match else "❌"
            print(f"  {status} [{ms:5.0f}ms] {wav_file}: {transcript!r}")

        print()

    cactus_destroy(whisper)


# ── Main ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = sys.argv[1:]

    if not args:
        # Default: run transcription test + pipeline test
        test_transcribe_only()
        test_pipeline()

    elif args[0] == "--record":
        test_record_and_transcribe()

    elif args[0] == "--pipeline":
        test_pipeline()

    elif args[0] == "--transcribe":
        test_transcribe_only()

    elif args[0] == "--prompts":
        test_whisper_prompts()

    elif args[0] == "--all":
        test_transcribe_only()
        test_pipeline()
        test_whisper_prompts()

    elif os.path.isfile(args[0]):
        # Transcribe a specific file
        test_single_file(args[0])

    else:
        print(f"Unknown argument: {args[0]}")
        print()
        print("Usage:")
        print("  python test-transcribe.py                        # Transcription + pipeline tests")
        print("  python test-transcribe.py --transcribe            # Transcription only")
        print("  python test-transcribe.py --pipeline              # Full voice→function-call pipeline")
        print("  python test-transcribe.py --prompts               # Compare Whisper prompt variations")
        print("  python test-transcribe.py --record                # Record from mic → transcribe → call")
        print("  python test-transcribe.py --all                   # Run all tests")
        print("  python test-transcribe.py test_audio/weather.wav  # Transcribe a specific file")
        sys.exit(1)
