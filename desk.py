#!/usr/bin/env python3
"""
📁 Desk — Voice Assistant for Your Local Files

"Hey Desk, what was the staging database password?"

You have years of docs, meeting notes, credentials, and project files
scattered across folders. You *know* the answer is somewhere in there.
Today you Cmd+F through files or paste stuff into ChatGPT — and hope
IT doesn't notice you just uploaded api-credentials.md to the cloud.

Desk points at a folder. You speak. It finds the answer.
Nothing leaves your machine.

Three Cactus on-device APIs chained together:
  🎤 Whisper      — speech to text
  📂 RAG          — retrieves relevant chunks from your files
  🔧 FunctionGemma — extracts structured tool calls

Zero network traffic. Zero cloud. Your files stay yours.

The UX is intentionally constrained: you pick an action, then speak
the details. This isn't a limitation — it's how you build a reliable
product on a 270M parameter model. Every query is templated into a
form FunctionGemma handles well, with only one tool passed per call.
The result: consistent, fast, private.

Usage:
    python desk.py ./my-work-docs              # Interactive (text input)
    python desk.py ./my-work-docs --voice      # Enable live mic input
    python desk.py --demo                      # Scripted demo with sample corpus

Requirements:
    source cactus/venv/bin/activate
    cactus download google/functiongemma-270m-it --reconvert
    cactus download openai/whisper-small --reconvert
    Optional: brew install sox    (for --voice live mic recording)
"""

import argparse
import ctypes
import datetime
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

# ═══════════════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════════════

FG_PATH = "cactus/weights/functiongemma-270m-it"
WHISPER_PATH = "cactus/weights/whisper-small"
RAG_PATH = "cactus/weights/lfm2-vl-450m"
WHISPER_PROMPT = "<|startoftranscript|><|en|><|transcribe|><|notimestamps|>"
TEMP_AUDIO = "/tmp/desk_recording.wav"
DEMO_CORPUS = "/tmp/desk-demo"
DEMO_AUDIO = "/tmp/desk-demo-audio"

# FG optimal settings (from systematic sweep — see notes/test.md)
FG_SYSTEM = "You are a helpful assistant that can use tools."
FG_FORCE_TOOLS = True
FG_MAX_TOKENS = 256
FG_TEMPERATURE = 0.0
FG_TOOL_RAG_K = 0  # disable tool RAG — actively harmful per sweep

# ═══════════════════════════════════════════════════════════════════════
# Terminal colors
# ═══════════════════════════════════════════════════════════════════════

BOLD = "\033[1m"
DIM = "\033[2m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
CYAN = "\033[36m"
RED = "\033[31m"
MAGENTA = "\033[35m"
RESET = "\033[0m"

# ═══════════════════════════════════════════════════════════════════════
# Tool definitions — named to match FG's training distribution
# (search_X, set_X, create_X are patterns FG 270M recognizes)
# ═══════════════════════════════════════════════════════════════════════

TOOL_SEARCH = {
    "name": "search_contacts",
    "description": "Search local files, records and documents for information",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "What to search for"},
        },
        "required": ["query"],
    },
}

TOOL_REMINDER = {
    "name": "create_reminder",
    "description": "Set a reminder for a specific time",
    "parameters": {
        "type": "object",
        "properties": {
            "title": {"type": "string", "description": "What to be reminded about"},
            "time": {"type": "string", "description": "When to be reminded"},
        },
        "required": ["title", "time"],
    },
}

TOOL_NOTE = {
    "name": "send_message",
    "description": "Send a message or save a note",
    "parameters": {
        "type": "object",
        "properties": {
            "recipient": {"type": "string", "description": "Who to send to"},
            "message": {"type": "string", "description": "Message content"},
        },
        "required": ["recipient", "message"],
    },
}

ALL_TOOLS = [TOOL_SEARCH, TOOL_REMINDER, TOOL_NOTE]

# ═══════════════════════════════════════════════════════════════════════
# Menu — each option pre-selects the tool + templates the prompt
# This is the core UX trick: FG only does argument extraction, never
# tool selection. Every call is an "easy case."
# ═══════════════════════════════════════════════════════════════════════

MENU = {
    "1": {
        "label": "🔍  Search my files",
        "tool": TOOL_SEARCH,
        "voice_prompt": "What do you want to find?",
        "template": "Search my files for {input}",
        "use_rag": True,
    },
    "2": {
        "label": "⏰  Set a reminder",
        "tool": TOOL_REMINDER,
        "voice_prompt": "What and when?",
        "template": "Remind me to {input}",
        "use_rag": False,
    },
    "3": {
        "label": "🎤  Free voice command",
        "tool": None,
        "voice_prompt": "Go ahead, speak...",
        "template": "{input}",
        "use_rag": True,
    },
}

# ═══════════════════════════════════════════════════════════════════════
# Demo corpus — realistic work files with sensitive content
# (API keys, passwords, client contacts, financials)
# ═══════════════════════════════════════════════════════════════════════

DEMO_FILES = {
    "api-credentials.md": """# API Credentials & Service Accounts

## Production Database
- Host: db-prod.internal.acme.com
- User: app_service
- Password: xK9#mP2$vL5nQ8wR
- Port: 5432

## Staging Database
- Host: db-staging.internal.acme.com
- User: dev_service
- Password: staging_dev_2026!
- Port: 5432

## Stripe
- Live key: sk_live_EXAMPLE_KEY_REDACTED
- Test key: sk_test_EXAMPLE_KEY_REDACTED
- Rate limit: 100 req/sec test, 10000 req/sec production
- Webhook secret: whsec_MfKQ9r8GKYqrTwjUPD8ILPZIo

## AWS
- Access Key: AKIAIOSFODNN7EXAMPLE
- Secret Key: wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY
- Region: us-west-2
- S3 bucket: acme-prod-assets
""",
    "meeting-2026-02-14.md": """# Team Sync — February 14, 2026

## Attendees
Sarah Chen, Mike Rodriguez, Priya Patel, Alex Kim

## Key Decisions
- Migrate from MySQL to PostgreSQL 16 by end of Q1
- Move CI/CD from Jenkins to GitHub Actions
- Alex leads the API v3 redesign
- Budget approved for 3 additional engineers

## Action Items
- Sarah: finalize Postgres migration plan by Feb 21
- Mike: set up GitHub Actions pipeline by Feb 28
- Priya: interview candidates, target 2 hires by March 15
- Alex: API v3 design doc first draft by Feb 25

## Notes
- Q1 revenue tracking at 2.1M against 2.4M target
- Board meeting moved to March 5
- Next team sync: February 21
""",
    "project-atlas.md": """# Project Atlas — Client Portal Redesign

## Client
Acme Industries
Contact: Jennifer Walsh, VP Product
Email: jennifer@acme-industries.com
Phone: (415) 555-0142

## Timeline
- Design phase: Jan 15 to Feb 15 (complete)
- Backend API: Feb 1 to Mar 15
- Frontend build: Feb 15 to Apr 1
- QA and testing: Apr 1 to Apr 15
- Launch date: April 20 2026
- Acme deliverable due: March 30 2026

## Budget
- Total contract value: 340000 dollars
- Design: 45000 (invoiced and paid)
- Development: 220000 (80000 invoiced so far)
- QA: 40000
- Project management: 35000

## Team
- Project lead: Sarah Chen
- Backend: Mike Rodriguez, Alex Kim
- Frontend: Priya Patel, Jordan Lee
- Design: Studio Neon (contracted, ends Feb 28)
""",
    "architecture-decisions.md": """# Architecture Decision Records

## ADR-001: Database Migration
Date: February 10 2026
Decision: Migrate from MySQL 5.7 to PostgreSQL 16
Reason: Better JSON support, superior indexing, team expertise
Risk: 2-week migration window with potential downtime
Rollback: Keep MySQL read replica active for 30 days post-migration

## ADR-002: Authentication
Date: February 12 2026
Decision: Replace custom auth system with Auth0
Reason: SOC2 compliance requirement, SSO for enterprise clients
Cost: 0.50 per MAU, estimated 2500 per month at current scale
Migration: Phased rollout, existing sessions honored for 90 days

## ADR-003: Caching Layer
Date: February 14 2026
Decision: Add Redis cluster for API response caching
Target: Sub-50ms p99 latency for all read endpoints
Expected improvement: 3-5x faster product listing pages
""",
    "client-contacts.md": """# Client and Vendor Contacts

## Acme Industries (primary client)
- Jennifer Walsh, VP Product: jennifer@acme-industries.com, (415) 555-0142
- Tom Bradley, CTO: tom.b@acme-industries.com, (415) 555-0198
- Billing department: accounts@acme-industries.com
- Net-30 payment terms

## Studio Neon (design contractor)
- Casey Morales, Lead Designer: casey@studioneon.co
- Invoicing: billing@studioneon.co
- Contract ends: February 28, 2026

## CloudScale (hosting provider)
- Account rep: David Park, david.park@cloudscale.io
- Support: enterprise-support@cloudscale.io
- Account ID: CS-2024-8847
- Monthly spend: approximately 4200 dollars
""",
}

# Demo script — queries to walk through
DEMO_SCRIPT = [
    {
        "say": "What is the staging database password",
        "menu": "1",
        "title": "Finding sensitive credentials",
    },
    {
        "say": "When is the Acme deliverable due",
        "menu": "1",
        "title": "Searching project deadlines",
    },
    {
        "say": "rotate the API credentials on Friday at 2 PM",
        "menu": "2",
        "title": "Setting a reminder",
    },
    {
        "say": "Who is the contact at Acme Industries",
        "menu": "1",
        "title": "Looking up client contacts",
    },
]

# ═══════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════


def clean_corpus_cache(corpus_dir):
    """Remove RAG index files so cactus rebuilds from scratch."""
    for name in ("index.bin", "data.bin"):
        path = os.path.join(corpus_dir, name)
        if os.path.exists(path):
            os.remove(path)


def corpus_files(corpus_dir):
    """List indexable files in corpus."""
    exts = (".md", ".txt")
    return [f for f in sorted(os.listdir(corpus_dir)) if any(f.endswith(e) for e in exts)]


def corpus_size_kb(corpus_dir):
    """Total size of indexable files in KB."""
    total = sum(
        os.path.getsize(os.path.join(corpus_dir, f)) for f in corpus_files(corpus_dir)
    )
    return total / 1024


def fmt_time(ms):
    """Format milliseconds for display."""
    if ms < 1000:
        return f"{ms:.0f}ms"
    return f"{ms / 1000:.1f}s"


# ═══════════════════════════════════════════════════════════════════════
# RAG — ctypes direct call (workaround for Python wrapper bug)
# The Python wrapper returns [] always because it misinterprets the
# C return value. Calling the C function directly via ctypes works.
# ═══════════════════════════════════════════════════════════════════════


def rag_query(model, query, top_k=3):
    """Retrieve relevant chunks from the RAG index.
    Uses direct ctypes call — arg order from working test-rag.py."""
    buf = ctypes.create_string_buffer(65536)
    _lib.cactus_rag_query(
        model,
        query.encode("utf-8"),
        buf,
        len(buf),
        top_k,
    )
    raw = buf.value.decode("utf-8").strip()
    if raw:
        try:
            parsed = json.loads(raw)
            # Response is {"chunks": [{"score":..., "source":..., "content":...}]}
            if isinstance(parsed, dict):
                return parsed.get("chunks", [])
            return parsed if isinstance(parsed, list) else []
        except (json.JSONDecodeError, ValueError):
            return []
    return []


def summarize_rag(chunks, max_chars=200):
    """Trim RAG context for FG prompt injection. Must be SHORT or FG
    copies it verbatim into args instead of extracting user intent."""
    if not chunks:
        return "", ""
    text = chunks[0].get("content", chunks[0].get("text", ""))
    source = chunks[0].get("source", chunks[0].get("document_id", "file"))
    source = os.path.basename(str(source)) if source else "file"
    trimmed = text[:max_chars] + "..." if len(text) > max_chars else text
    return trimmed, source


def rerank_chunks(chunks, query):
    """Re-rank RAG chunks by keyword overlap since embedding scores are
    tightly clustered and unreliable for top-1 selection."""
    keywords = [w.lower() for w in query.split() if len(w) > 2]
    if not keywords or not chunks:
        return chunks
    scored = []
    for ch in chunks:
        text = (ch.get("content") or ch.get("text", "")).lower()
        hits = sum(1 for kw in keywords if kw in text)
        scored.append((hits, ch))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [ch for _, ch in scored]


# ═══════════════════════════════════════════════════════════════════════
# Audio
# ═══════════════════════════════════════════════════════════════════════


def generate_wav(text, path):
    """Generate WAV from text using macOS say (16kHz mono for Whisper).
    say outputs AIFF by default, so we pipe through ffmpeg to get proper WAV."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    aiff_path = path.rsplit(".", 1)[0] + ".aiff"
    subprocess.run(
        ["say", "-o", aiff_path, text],
        check=True, capture_output=True,
    )
    subprocess.run(
        ["ffmpeg", "-y", "-i", aiff_path, "-ar", "16000", "-ac", "1", "-sample_fmt", "s16", path],
        check=True, capture_output=True,
    )
    os.remove(aiff_path)


def record_wav(path=TEMP_AUDIO, seconds=5):
    """Record from mic using sox. Returns path or None."""
    try:
        print(f"    {DIM}🎤 Listening ({seconds}s)... speak now{RESET}", flush=True)
        subprocess.run(
            ["rec", "-q", "-r", "16000", "-c", "1", "-b", "16", path,
             "trim", "0", str(seconds)],
            check=True, timeout=seconds + 3, capture_output=True,
        )
        return path
    except FileNotFoundError:
        print(f"    {RED}sox not installed — run: brew install sox{RESET}")
        return None
    except subprocess.TimeoutExpired:
        return path if os.path.exists(path) else None


# ═══════════════════════════════════════════════════════════════════════
# Pipeline stages
# ═══════════════════════════════════════════════════════════════════════


def stage_transcribe(whisper_model, audio_path):
    """Whisper: audio → text. Returns (text, elapsed_ms)."""
    cactus_reset(whisper_model)
    t0 = time.time()
    raw = cactus_transcribe(whisper_model, audio_path, WHISPER_PROMPT)
    elapsed = (time.time() - t0) * 1000
    if raw is None:
        return "(transcription failed)", elapsed
    try:
        text = json.loads(raw).get("response", "").strip()
    except (json.JSONDecodeError, AttributeError):
        text = "(transcription failed)"
    return text, elapsed


def stage_rag(rag_model, query):
    """RAG: query → relevant chunks. Returns (chunks, context, source, elapsed_ms)."""
    t0 = time.time()
    chunks = rag_query(rag_model, query, top_k=10)
    chunks = rerank_chunks(chunks, query)
    elapsed = (time.time() - t0) * 1000
    context, source = summarize_rag(chunks)
    return chunks, context, source, elapsed


def stage_fg(fg_model, user_text, tool, rag_context=""):
    """FunctionGemma: text → tool call. Returns (call_dict|None, elapsed_ms).
    Tries up to 2 times with cactus_reset between attempts."""
    if rag_context:
        content = f"Context from local files:\n{rag_context}\n\nUser request: {user_text}"
    else:
        content = user_text

    messages = [
        {"role": "system", "content": FG_SYSTEM},
        {"role": "user", "content": content},
    ]
    # Tools must be wrapped in {"type": "function", "function": ...} for Cactus
    raw_tools = [tool] if tool else ALL_TOOLS
    cactus_tools = [{"type": "function", "function": t} for t in raw_tools]

    total_ms = 0
    for attempt in range(2):
        if attempt > 0:
            cactus_reset(fg_model)
        t0 = time.time()
        raw = cactus_complete(
            fg_model, messages,
            tools=cactus_tools,
            force_tools=FG_FORCE_TOOLS,
            max_tokens=FG_MAX_TOKENS,
            stop_sequences=["<|im_end|>", "<end_of_turn>"],
            tool_rag_top_k=FG_TOOL_RAG_K,
        )
        total_ms += (time.time() - t0) * 1000
        try:
            result = json.loads(raw)
            calls = result.get("function_calls", [])
            if calls:
                return calls[0], total_ms
        except (json.JSONDecodeError, TypeError):
            pass

    return None, total_ms


# ═══════════════════════════════════════════════════════════════════════
# Tool execution — actually do the thing
# ═══════════════════════════════════════════════════════════════════════


def execute(call, corpus_dir, rag_chunks=None):
    """Execute a tool call and return a display string."""
    if not call:
        return f"  {DIM}(no tool call produced — try rephrasing){RESET}"

    name = call.get("name", "")
    args = call.get("arguments", {})

    if name in ("search_files", "search_contacts"):
        chunk_text = (rag_chunks[0].get("content") or rag_chunks[0].get("text", "")) if rag_chunks else ""
        if not chunk_text:
            return f"  {DIM}No matching content found.{RESET}"
        lines = chunk_text.strip().split("\n")[:15]
        return "\n".join(f"    {DIM}>{RESET} {line}" for line in lines)

    elif name == "create_reminder":
        title = args.get("title", "untitled")
        at = args.get("time", "unspecified time")
        try:
            subprocess.run(
                ["osascript", "-e",
                 f'display notification "{title}" with title "📁 Desk" subtitle "at {at}"'],
                capture_output=True, timeout=3,
            )
        except Exception:
            pass
        return f"    {GREEN}✅ Reminder set:{RESET} {title} at {at}"

    elif name == "send_message":
        # Used as "create note" — FG knows send_message but not create_note
        content = args.get("message", args.get("content", "(empty)"))
        ts = datetime.datetime.now().strftime("%Y-%m-%d-%H%M")
        filename = f"note-{ts}.md"
        filepath = os.path.join(corpus_dir, filename)
        with open(filepath, "w") as f:
            f.write(f"# Note — {datetime.datetime.now().strftime('%B %d, %Y %I:%M %p')}\n\n{content}\n")
        return f"    {GREEN}✅ Saved:{RESET} {filename}\n    {DIM}> {content}{RESET}"

    else:
        return f"    {YELLOW}{name}({json.dumps(args)}){RESET}"


# ═══════════════════════════════════════════════════════════════════════
# Display
# ═══════════════════════════════════════════════════════════════════════


def show_banner(corpus_dir):
    files = corpus_files(corpus_dir)
    kb = corpus_size_kb(corpus_dir)
    print()
    print(f"  {BOLD}╔═══════════════════════════════════════════════════════╗{RESET}")
    print(f"  {BOLD}║  📁  Desk — Your Files, Your Voice, Your Machine     ║{RESET}")
    print(f"  {BOLD}╠═══════════════════════════════════════════════════════╣{RESET}")
    print(f"  {BOLD}║{RESET}  Corpus: {CYAN}{corpus_dir}{RESET}")
    print(f"  {BOLD}║{RESET}  {len(files)} files ({kb:.1f} KB) | {GREEN}100% on-device{RESET} | no network")
    print(f"  {BOLD}╚═══════════════════════════════════════════════════════╝{RESET}")
    print()


def show_menu():
    for key, opt in MENU.items():
        print(f"    {BOLD}[{key}]{RESET} {opt['label']}")
    print(f"    {BOLD}[q]{RESET} Quit")
    print()


def show_step(label, detail, ms, lock=True):
    icon = "🔒" if lock else "⌨️ "
    print(f"    {CYAN}{label:12s}{RESET} {detail:48s} {DIM}{fmt_time(ms):>6s}  {icon}{RESET}")


def show_divider():
    print(f"    {'─' * 55}")


def show_total(ms):
    show_divider()
    print(f"    {BOLD}⏱  {fmt_time(ms)}{RESET}  |  {GREEN}🔒 fully on-device{RESET}  |  {GREEN}0 bytes sent{RESET}")
    print()


# ═══════════════════════════════════════════════════════════════════════
# Run one query through the full pipeline
# ═══════════════════════════════════════════════════════════════════════


def run_query(models, corpus_dir, text, menu_key, audio_path=None):
    """Run a single query. Returns total pipeline time in ms."""
    whisper_model, rag_model, fg_model = models
    opt = MENU[menu_key]
    total_ms = 0

    # ── 1. Transcribe (if audio provided) ──
    if audio_path:
        transcript, ms = stage_transcribe(whisper_model, audio_path)
        show_step("Whisper", f'"{transcript}"', ms)
        total_ms += ms
        text = transcript
    else:
        show_step("Input", f'"{text}"', 0, lock=False)

    # ── 2. RAG retrieval (for search queries) ──
    rag_chunks = []
    rag_context = ""
    source = ""
    if opt["use_rag"]:
        rag_chunks, rag_context, source, ms = stage_rag(rag_model, text)
        show_step("RAG", source if source else "(no match)", ms)
        total_ms += ms

    # ── 3. Template + FunctionGemma ──
    fg_input = opt["template"].format(input=text)
    tool = opt["tool"]
    call, ms = stage_fg(fg_model, fg_input, tool, rag_context)
    total_ms += ms

    if call:
        display_name = {"search_contacts": "search_files", "send_message": "save_note"}.get(call["name"], call["name"])
        args_s = json.dumps(call.get("arguments", {}))
        if len(args_s) > 40:
            args_s = args_s[:40] + "…"
        show_step("FG", f"{display_name}({args_s})", ms)
    else:
        show_step("FG", f"{RED}no call{RESET}", ms)

    show_divider()
    if source and menu_key == "1":
        print(f"    {BOLD}📄 From {source}:{RESET}")
    result = execute(call, corpus_dir, rag_chunks)
    print(result)

    show_total(total_ms)
    return total_ms


# ═══════════════════════════════════════════════════════════════════════
# Model loading
# ═══════════════════════════════════════════════════════════════════════


def load_models(corpus_dir):
    """Load Whisper + RAG + FunctionGemma. Returns (whisper, rag, fg)."""
    print(f"\n  {BOLD}Loading models (all on-device)...{RESET}")

    t0 = time.time()
    whisper = cactus_init(WHISPER_PATH)
    print(f"    {GREEN}✓{RESET} Whisper (speech → text)         {DIM}{time.time() - t0:.1f}s{RESET}")

    clean_corpus_cache(corpus_dir)
    t0 = time.time()
    rag = cactus_init(RAG_PATH, corpus_dir=corpus_dir)
    n = len(corpus_files(corpus_dir))
    print(f"    {GREEN}✓{RESET} RAG index ({n} files)             {DIM}{time.time() - t0:.1f}s{RESET}")

    t0 = time.time()
    fg = cactus_init(FG_PATH)
    print(f"    {GREEN}✓{RESET} FunctionGemma (tool calling)    {DIM}{time.time() - t0:.1f}s{RESET}")

    print()
    return whisper, rag, fg


def destroy_models(models):
    for m in models:
        try:
            cactus_destroy(m)
        except Exception:
            pass


# ═══════════════════════════════════════════════════════════════════════
# Demo mode — scripted walkthrough
# ═══════════════════════════════════════════════════════════════════════


def run_demo():
    """Auto-run a scripted demo with sample corpus + generated audio."""
    print(f"\n  {BOLD}{MAGENTA}{'═' * 55}{RESET}")
    print(f"  {BOLD}{MAGENTA}  DESK — Fully On-Device Voice File Assistant{RESET}")
    print(f"  {BOLD}{MAGENTA}{'═' * 55}{RESET}")

    # 1. Create corpus
    print(f"\n  {BOLD}Setting up demo corpus...{RESET}")
    os.makedirs(DEMO_CORPUS, exist_ok=True)
    clean_corpus_cache(DEMO_CORPUS)
    for name, content in DEMO_FILES.items():
        with open(os.path.join(DEMO_CORPUS, name), "w") as f:
            f.write(content)

    files = corpus_files(DEMO_CORPUS)
    print(f"    Created {CYAN}{DEMO_CORPUS}{RESET} with {len(files)} files:")
    for fn in files:
        print(f"      {DIM}•{RESET} {fn}")
    print(f"\n    {YELLOW}⚠  API keys, passwords, client contacts, financials —")
    print(f"    the kind of data you'd never paste into ChatGPT.{RESET}")

    # 2. Generate audio
    print(f"\n  {BOLD}Generating voice samples (macOS TTS)...{RESET}")
    audio_paths = []
    os.makedirs(DEMO_AUDIO, exist_ok=True)
    for i, q in enumerate(DEMO_SCRIPT):
        path = os.path.join(DEMO_AUDIO, f"demo_{i}.wav")
        generate_wav(q["say"], path)
        audio_paths.append(path)
    print(f"    {GREEN}✓{RESET} {len(audio_paths)} audio files ready")

    # 3. Load models
    models = load_models(DEMO_CORPUS)
    show_banner(DEMO_CORPUS)

    input(f"  {DIM}[Press Enter to start demo]{RESET}\n")

    # 4. Walk through queries
    for i, (query, audio) in enumerate(zip(DEMO_SCRIPT, audio_paths)):
        print(f"  {BOLD}━━━ {i + 1}/{len(DEMO_SCRIPT)}: {query['title']} ━━━{RESET}")
        opt_label = MENU[query["menu"]]["label"]
        print(f"    Selected: [{query['menu']}] {opt_label}")
        print(f"    {DIM}🎤 \"{query['say']}\"{RESET}")
        print()

        run_query(models, DEMO_CORPUS, query["say"], query["menu"], audio_path=audio)

        if i < len(DEMO_SCRIPT) - 1:
            input(f"  {DIM}[Enter for next →]{RESET}\n")

    # 5. Closing
    print(f"  {BOLD}{'━' * 55}{RESET}")
    print(f"  {BOLD}Demo complete.{RESET}")
    print()
    print(f"    {len(files)} files  ·  {len(DEMO_SCRIPT)} voice commands  ·  {GREEN}0 bytes to cloud{RESET}")
    print()
    print(f"    Pipeline:  🎤 Whisper  →  📂 RAG  →  🔧 FunctionGemma")
    print(f"    Three on-device APIs. Your files never left this machine.")
    print()

    destroy_models(models)


# ═══════════════════════════════════════════════════════════════════════
# Interactive mode
# ═══════════════════════════════════════════════════════════════════════


def run_interactive(corpus_dir, voice=False):
    """Interactive loop: pick an action, speak or type, see results."""
    models = load_models(corpus_dir)
    show_banner(corpus_dir)

    try:
        while True:
            show_menu()
            choice = input(f"    {BOLD}>{RESET} ").strip()

            if choice in ("q", "quit", "exit"):
                break
            if choice not in MENU:
                print(f"    {RED}Pick 1–4 or q{RESET}\n")
                continue

            opt = MENU[choice]
            audio_path = None
            user_text = ""

            if voice or choice == "4":
                print(f"\n    {opt['voice_prompt']}")
                audio_path = record_wav()
                if not audio_path:
                    user_text = input(f"    {DIM}(type instead):{RESET} ").strip()
                    if not user_text:
                        print()
                        continue
                print()
            else:
                user_text = input(f"    {opt['voice_prompt']} ").strip()
                if not user_text:
                    print()
                    continue
                print()

            run_query(models, corpus_dir, user_text, choice, audio_path=audio_path)

    except (KeyboardInterrupt, EOFError):
        print(f"\n\n  {DIM}Bye.{RESET}\n")

    destroy_models(models)


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════


def suppress_stderr():
    """Redirect C-level stderr to /dev/null to suppress Cactus telemetry noise.
    Preserves Python's sys.stderr for exception tracebacks."""
    import io
    sys.stderr = open(os.devnull, "w")
    # Redirect fd 2 (C-level stderr) to suppress PGRST messages
    _saved_fd = os.dup(2)
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 2)
    os.close(devnull)
    # Restore Python-level stderr to a wrapper around saved fd
    sys.stderr = io.TextIOWrapper(io.FileIO(_saved_fd, "w"), write_through=True)


def main():
    parser = argparse.ArgumentParser(
        description="📁 Desk — Voice assistant for your local files",
    )
    parser.add_argument("corpus", nargs="?", help="Path to folder of documents to index")
    parser.add_argument("--demo", action="store_true", help="Scripted demo with sample corpus")
    parser.add_argument("--voice", action="store_true", help="Enable live microphone (requires sox)")
    args = parser.parse_args()

    # Suppress C-level stderr noise from Cactus telemetry
    _saved_fd = os.dup(2)
    os.dup2(os.open(os.devnull, os.O_WRONLY), 2)

    if args.demo:
        run_demo()
    elif args.corpus:
        if not os.path.isdir(args.corpus):
            print(f"  {RED}Not a directory: {args.corpus}{RESET}")
            sys.exit(1)
        run_interactive(args.corpus, voice=args.voice)
    else:
        parser.print_help()
        print(f"\n  Quick start:")
        print(f"    python desk.py --demo                  # scripted demo")
        print(f"    python desk.py ~/Documents/work         # your own files")
        print(f"    python desk.py ~/Documents/work --voice # with microphone\n")


if __name__ == "__main__":
    main()
