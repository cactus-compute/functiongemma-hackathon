"""Mingle AI Server — FastAPI endpoints that wrap generate_hybrid.

Every Mingle AI task is modelled as a tool call so generate_hybrid can
transparently route simple requests on-device (FunctionGemma) and fall back
to Gemini Cloud for complex/long-form generation.

Start:
    uvicorn ai_server:app --host 0.0.0.0 --port 8001 --reload
"""

import os
import sys
import json
import threading

# main.py is in the same directory; add it to path
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional

from main import generate_hybrid  # noqa: E402  (added after sys.path manipulation)

try:
    from cactus import cactus_init, cactus_destroy
    try:
        from cactus import cactus_rag_query
        _HAS_RAG = True
    except ImportError:
        _HAS_RAG = False
    _HAS_CACTUS = True
except ImportError:
    _HAS_CACTUS = False
    _HAS_RAG = False

# ---------------------------------------------------------------------------
# Globals
# ---------------------------------------------------------------------------
RAG_CORPUS_DIR = os.path.join(os.path.dirname(__file__), "rag_corpus")
_rag_model = None
_rag_lock = threading.Lock()

functiongemma_path = os.path.join(_HERE, "../../cactus/weights/functiongemma-270m-it")

app = FastAPI(title="Mingle AI Server", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# RAG model helpers
# ---------------------------------------------------------------------------

def _load_rag_model():
    global _rag_model
    if not _HAS_CACTUS:
        return
    with _rag_lock:
        if _rag_model is not None:
            try:
                cactus_destroy(_rag_model)
            except Exception:
                pass
        try:
            _rag_model = cactus_init(functiongemma_path, corpus_dir=RAG_CORPUS_DIR)
        except Exception:
            _rag_model = None


def _rag_query_profiles(query: str, top_k: int = 8) -> List[str]:
    """Return profile IDs from RAG retrieval (best-effort)."""
    if not _HAS_RAG or _rag_model is None:
        return []
    try:
        with _rag_lock:
            chunks = cactus_rag_query(_rag_model, query, top_k=top_k)
        profile_ids = []
        seen = set()
        for chunk in chunks:
            src = chunk.get("source", "")
            pid = os.path.splitext(os.path.basename(src))[0]
            if pid and pid not in seen:
                seen.add(pid)
                profile_ids.append(pid)
        return profile_ids
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class SummarizeBioRequest(BaseModel):
    name: str
    role: str
    company: str
    skills: List[str] = []
    looking_for: List[str] = []


class ContactProfile(BaseModel):
    id: str
    name: str
    role: str
    company: str
    bio: str = ""
    skills: List[str] = []
    looking_for: List[str] = []
    can_help_with: List[str] = []
    domains: List[str] = []


class RankContactRequest(BaseModel):
    query_looking_for: str
    query_domain: str
    query_help_type: str = ""
    urgency: str = "medium"
    candidates: List[ContactProfile]


class DraftOutreachRequest(BaseModel):
    sender: ContactProfile
    recipient: ContactProfile
    context: str = ""


class SyncProfileRequest(BaseModel):
    profile_id: str
    name: str
    role: str
    company: str
    bio: str
    skills: List[str] = []
    looking_for: List[str] = []
    can_help_with: List[str] = []
    domains: List[str] = []
    linkedin_url: str = ""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _profile_to_text(p: ContactProfile) -> str:
    return (
        f"Name: {p.name}\n"
        f"Role: {p.role}\n"
        f"Company: {p.company}\n"
        f"Bio: {p.bio}\n"
        f"Skills: {', '.join(p.skills)}\n"
        f"Looking For: {', '.join(p.looking_for)}\n"
        f"Can Help With: {', '.join(p.can_help_with)}\n"
        f"Domains: {', '.join(p.domains)}"
    )


def _safe_call(result: dict, field: str, fallback):
    try:
        return result["function_calls"][0]["arguments"][field]
    except (KeyError, IndexError, TypeError):
        return fallback


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/ai/health")
def health():
    return {
        "status": "ok",
        "cactus_available": _HAS_CACTUS,
        "rag_available": _HAS_RAG,
        "rag_model_loaded": _rag_model is not None,
    }


@app.post("/ai/summarize-bio")
def summarize_bio(req: SummarizeBioRequest):
    """Generate a one-liner professional bio summary (likely on-device)."""
    tools = [{
        "name": "generate_bio_summary",
        "description": "Generate a one-liner professional bio summary",
        "parameters": {
            "type": "object",
            "properties": {
                "bio_summary": {
                    "type": "string",
                    "description": "One sentence professional summary"
                }
            },
            "required": ["bio_summary"]
        }
    }]
    skills_str = ", ".join(req.skills) if req.skills else "N/A"
    looking_str = ", ".join(req.looking_for) if req.looking_for else "N/A"
    messages = [{
        "role": "user",
        "content": (
            f"Summarize: {req.name}, {req.role} at {req.company}. "
            f"Skills: {skills_str}. Looking for: {looking_str}."
        )
    }]
    result = generate_hybrid(messages, tools)
    bio = _safe_call(result, "bio_summary", f"{req.role} at {req.company}")
    return {"bio_summary": bio, "source": result.get("source", "unknown")}


@app.post("/ai/rank-contact")
def rank_contact(req: dict):
    """Rank a single contact against a query (on-device for clear matches)."""
    query_looking_for = req.get("query_looking_for", "")
    query_domain = req.get("query_domain", "")
    contact = req.get("contact", {})
    contact_id = contact.get("id", "unknown")
    contact_text = (
        f"Name: {contact.get('name', '')}, "
        f"Role: {contact.get('role', '')}, "
        f"Company: {contact.get('company', '')}, "
        f"Skills: {', '.join(contact.get('skills', []))}, "
        f"Looking For: {', '.join(contact.get('looking_for', []))}, "
        f"Domains: {', '.join(contact.get('domains', []))}"
    )
    tools = [{
        "name": "rank_contact",
        "description": "Score how well a contact matches a networking query",
        "parameters": {
            "type": "object",
            "properties": {
                "contact_id":     {"type": "string"},
                "match_score":    {"type": "number",  "description": "0.0-1.0"},
                "match_reason":   {"type": "string",  "description": "One sentence why they match"},
                "outreach_angle": {"type": "string",  "description": "One sentence suggested opening"}
            },
            "required": ["contact_id", "match_score", "match_reason", "outreach_angle"]
        }
    }]
    messages = [{
        "role": "user",
        "content": (
            f"I need a {query_looking_for} in {query_domain}. "
            f"Rate this contact: {contact_text}"
        )
    }]
    result = generate_hybrid(messages, tools)
    args = {}
    try:
        args = result["function_calls"][0]["arguments"]
    except (KeyError, IndexError):
        pass
    return {
        "contact_id":     args.get("contact_id", contact_id),
        "match_score":    float(args.get("match_score", 0.0)),
        "match_reason":   args.get("match_reason", ""),
        "outreach_angle": args.get("outreach_angle", ""),
        "source":         result.get("source", "unknown"),
    }


@app.post("/ai/rank-contacts")
def rank_contacts(req: RankContactRequest):
    """Rank multiple contacts. Uses Cactus RAG to pre-filter, then rank each."""
    query = f"{req.query_looking_for} {req.query_domain} {req.query_help_type}"

    # RAG pre-filter: get candidate profile IDs by semantic similarity
    rag_ids = _rag_query_profiles(query, top_k=8)

    # If RAG returned IDs, restrict candidates to those; otherwise use all
    if rag_ids:
        ordered_candidates = sorted(
            req.candidates,
            key=lambda c: (rag_ids.index(c.id) if c.id in rag_ids else len(rag_ids))
        )
        candidates_to_rank = ordered_candidates[:min(len(rag_ids), len(req.candidates))]
    else:
        candidates_to_rank = req.candidates

    tools = [{
        "name": "rank_contact",
        "description": "Score how well a contact matches a networking query",
        "parameters": {
            "type": "object",
            "properties": {
                "contact_id":     {"type": "string"},
                "match_score":    {"type": "number",  "description": "0.0-1.0"},
                "match_reason":   {"type": "string",  "description": "One sentence why they match"},
                "outreach_angle": {"type": "string",  "description": "One sentence suggested opening"}
            },
            "required": ["contact_id", "match_score", "match_reason", "outreach_angle"]
        }
    }]

    rankings = []
    for contact in candidates_to_rank:
        contact_text = _profile_to_text(contact)
        messages = [{
            "role": "user",
            "content": (
                f"I need a {req.query_looking_for} in {req.query_domain}. "
                f"Rate this contact: {contact_text}"
            )
        }]
        result = generate_hybrid(messages, tools)
        args = {}
        try:
            args = result["function_calls"][0]["arguments"]
        except (KeyError, IndexError):
            pass
        rankings.append({
            "contact_id":     args.get("contact_id", contact.id),
            "match_score":    float(args.get("match_score", 0.0)),
            "match_reason":   args.get("match_reason", ""),
            "outreach_angle": args.get("outreach_angle", ""),
            "source":         result.get("source", "unknown"),
        })

    rankings.sort(key=lambda r: r["match_score"], reverse=True)
    return {"rankings": rankings}


@app.post("/ai/draft-outreach")
def draft_outreach(req: DraftOutreachRequest):
    """Draft a warm, personalised outreach message (cloud for quality)."""
    tools = [{
        "name": "draft_outreach_message",
        "description": "Draft a warm, personalized 3-4 sentence outreach message",
        "parameters": {
            "type": "object",
            "properties": {
                "message": {
                    "type": "string",
                    "description": "The full outreach message text"
                }
            },
            "required": ["message"]
        }
    }]
    context = req.context or (
        f"{req.sender.name} wants to connect with {req.recipient.name} "
        f"about {', '.join(req.recipient.looking_for) or 'professional collaboration'}."
    )
    messages = [{
        "role": "user",
        "content": (
            f"Draft a warm networking message from {req.sender.name} "
            f"({req.sender.role} at {req.sender.company}) to "
            f"{req.recipient.name} ({req.recipient.role} at {req.recipient.company}). "
            f"Context: {context}"
        )
    }]
    result = generate_hybrid(messages, tools)
    message_text = _safe_call(result, "message", "")
    if not message_text:
        message_text = (
            f"Hi {req.recipient.name}, I came across your profile and was really impressed by "
            f"your work at {req.recipient.company}. I'm {req.sender.name}, "
            f"{req.sender.role} at {req.sender.company}, and I'd love to connect!"
        )
    return {"message": message_text, "source": result.get("source", "unknown")}


@app.post("/ai/sync-profile-rag")
def sync_profile_rag(req: SyncProfileRequest):
    """Write/overwrite the .txt for this profile and reload the RAG model."""
    os.makedirs(RAG_CORPUS_DIR, exist_ok=True)
    txt_path = os.path.join(RAG_CORPUS_DIR, f"{req.profile_id}.txt")
    content = (
        f"Name: {req.name}\n"
        f"Role: {req.role}\n"
        f"Company: {req.company}\n"
        f"Bio: {req.bio}\n"
        f"Skills: {', '.join(req.skills)}\n"
        f"Looking For: {', '.join(req.looking_for)}\n"
        f"Can Help With: {', '.join(req.can_help_with)}\n"
        f"Domains: {', '.join(req.domains)}\n"
        f"LinkedIn: {req.linkedin_url}\n"
    )
    with open(txt_path, "w") as f:
        f.write(content)

    # Reload RAG model with updated corpus (separate from generate_hybrid singleton)
    _load_rag_model()

    return {"status": "ok", "profile_id": req.profile_id, "path": txt_path}


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

@app.on_event("startup")
def on_startup():
    os.makedirs(RAG_CORPUS_DIR, exist_ok=True)
    _load_rag_model()


# Voice Note Processing

# ═══════════════════════════════════════════════════════════════════════════════
# Voice Note Processing - Hackathon Demo
# ═══════════════════════════════════════════════════════════════════════════════

class VoiceNoteRequest(BaseModel):
    audio_base64: str = ""
    mime_type: str = "audio/webm"
    transcript: str = None  # Allow passing transcript directly for demo

class VoiceNoteResponse(BaseModel):
    transcript: str
    contact_name: Optional[str] = None
    contact_email: Optional[str] = None
    contact_role: Optional[str] = None  
    contact_company: Optional[str] = None
    email_subject: Optional[str] = None
    email_body: Optional[str] = None
    source: str = "hybrid"
    tool_calls: list = []
    status: str = "success"
    error: Optional[str] = None


def _lookup_contact_in_db(name: str) -> dict:
    """Search for contact in Mingle database by name."""
    import requests
    try:
        response = requests.get("http://localhost:3001/api/profiles", timeout=5)
        profiles = response.json()
        
        name_lower = name.lower()
        for profile in profiles:
            if name_lower in profile.get("name", "").lower():
                # Generate email from name and company
                email = f"{profile['name'].lower().replace(' ', '.')}@{profile['company'].lower().replace(' ', '')}.com"
                return {
                    "found": True,
                    "name": profile["name"],
                    "role": profile["role"],
                    "company": profile["company"],
                    "email": email,
                    "bio": profile.get("bio", "")
                }
        return {"found": False, "name": name}
    except Exception as e:
        return {"found": False, "error": str(e)}


# Tools for FunctionGemma to call
VOICE_NOTE_TOOLS = [
    {
        "name": "lookup_contact",
        "description": "Look up a contact by name in the Mingle network database",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Name of the contact to look up"}
            },
            "required": ["name"]
        }
    },
    {
        "name": "draft_email",
        "description": "Draft a follow-up email after meeting someone",
        "parameters": {
            "type": "object", 
            "properties": {
                "to_name": {"type": "string", "description": "Recipient name"},
                "topic": {"type": "string", "description": "What you discussed"},
                "action": {"type": "string", "description": "Proposed next step"}
            },
            "required": ["to_name", "topic", "action"]
        }
    }
]


@app.post("/ai/process-voice-note")
def process_voice_note(req: VoiceNoteRequest):
    """
    Process voice note using FunctionGemma tool calling.
    
    Demo flow:
    1. Get transcript (passed in or mock)
    2. FunctionGemma extracts intent and calls tools
    3. Execute tool calls (lookup contact, draft email)
    4. Return structured response
    """
    # Step 1: Get transcript
    if req.transcript:
        transcript = req.transcript
    else:
        # Default demo transcript
        transcript = "Really enjoyed meeting Maya and talking about design systems. Send an email to schedule a follow up meeting."
    
    tool_calls_log = []
    contact_info = None
    email_draft = None
    source = "unknown"
    
    try:
        # Step 2: Use FunctionGemma to process the voice note
        messages = [{
            "role": "user",
            "content": f"""Process this voice note and help me follow up:

"{transcript}"

First, look up the contact mentioned. Then draft a follow-up email."""
        }]
        
        result = generate_hybrid(messages, VOICE_NOTE_TOOLS)
        source = result.get("source", "hybrid")
        
        # Step 3: Execute tool calls
        if result.get("tool_calls"):
            for tc in result["tool_calls"]:
                tool_name = tc.get("name", "")
                tool_args = tc.get("arguments", {})
                
                if tool_name == "lookup_contact":
                    name = tool_args.get("name", "")
                    contact_info = _lookup_contact_in_db(name)
                    tool_calls_log.append({"tool": "lookup_contact", "args": {"name": name}, "result": contact_info})
                    
                elif tool_name == "draft_email":
                    to_name = tool_args.get("to_name", contact_info.get("name") if contact_info else "Friend")
                    topic = tool_args.get("topic", "our conversation")
                    action = tool_args.get("action", "schedule a follow-up")
                    
                    email_draft = {
                        "subject": f"Great meeting you - {topic}",
                        "body": f"""Hi {to_name},

It was wonderful meeting you and discussing {topic}! I really enjoyed our conversation.

I'd love to {action}. Would you be available sometime next week?

Looking forward to staying in touch.

Best regards"""
                    }
                    tool_calls_log.append({"tool": "draft_email", "args": tool_args, "result": email_draft})
        
        # Fallback: Try to extract contact name from transcript if no tool calls
        if not contact_info:
            import re
            # Look for names after "meeting" or "met"
            match = re.search(r'(?:meeting|met|with)\s+([A-Z][a-z]+)', transcript)
            if match:
                name = match.group(1)
                contact_info = _lookup_contact_in_db(name)
                tool_calls_log.append({"tool": "lookup_contact", "args": {"name": name}, "result": contact_info, "source": "fallback"})
        
        # Generate email if we have contact but no draft
        if contact_info and contact_info.get("found") and not email_draft:
            email_draft = {
                "subject": f"Great meeting you, {contact_info['name']}!",
                "body": f"""Hi {contact_info['name']},

It was wonderful meeting you! I really enjoyed our conversation.

I'd love to schedule a follow-up meeting. Would you be available sometime next week?

Looking forward to staying in touch.

Best regards"""
            }
        
        return VoiceNoteResponse(
            transcript=transcript,
            contact_name=contact_info.get("name") if contact_info else None,
            contact_email=contact_info.get("email") if contact_info else None,
            contact_role=contact_info.get("role") if contact_info else None,
            contact_company=contact_info.get("company") if contact_info else None,
            email_subject=email_draft.get("subject") if email_draft else None,
            email_body=email_draft.get("body") if email_draft else None,
            source=source,
            tool_calls=tool_calls_log,
            status="success"
        )
        
    except Exception as e:
        # Fallback to demo mode when API fails
        try:
            from voice_demo import process_voice_note_demo
            demo_result = process_voice_note_demo(transcript)
            contact = demo_result.get("contact") or {}
            email = demo_result.get("email_draft") or {}
            
            contact = demo_result.get("contact") or {}
            output = demo_result.get("output") or {}
            intent = demo_result.get("intent") or {}
            
            return VoiceNoteResponse(
                transcript=transcript,
                contact_name=contact.get("name"),
                contact_email=contact.get("email"),
                contact_role=contact.get("role"),
                contact_company=contact.get("company"),
                email_subject=output.get("subject") if output.get("type") == "email" else None,
                email_body=output.get("body") if output.get("type") == "email" else output.get("message"),
                source=f"demo-mode ({intent.get('action_type', 'email')})",
                tool_calls=demo_result.get("tool_calls", []),
                status="success"
            )
        except Exception as demo_error:
            return VoiceNoteResponse(
                transcript=transcript,
                status="error",
                error=f"API: {str(e)}, Demo fallback: {str(demo_error)}",
                tool_calls=tool_calls_log
            )


