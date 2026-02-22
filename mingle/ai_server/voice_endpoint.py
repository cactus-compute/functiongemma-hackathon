"""
Voice Note Endpoint - Add this to ai_server.py
"""
from fastapi import APIRouter
from pydantic import BaseModel
import base64
import tempfile
import os

router = APIRouter()

class VoiceNoteRequest(BaseModel):
    audio_base64: str
    mime_type: str = "audio/webm"
    # For demo: allow passing transcript directly
    transcript: str = None

class VoiceNoteResponse(BaseModel):
    transcript: str
    contact_name: str = None
    contact_email: str = None
    contact_role: str = None
    contact_company: str = None
    email_subject: str = None
    email_body: str = None
    source: str = "hybrid"
    status: str = "success"
    error: str = None

@router.post("/ai/process-voice-note", response_model=VoiceNoteResponse)
def process_voice_note_endpoint(req: VoiceNoteRequest):
    """
    Process voice note with FunctionGemma tool calling.
    
    For hackathon demo:
    - If transcript provided, use it directly
    - Otherwise, use mock transcript (real transcription needs Whisper setup)
    - Use FunctionGemma to extract intent and call tools
    - Look up contact from database
    - Draft follow-up email
    """
    from voice_agent import process_voice_note, lookup_contact, draft_followup_email
    
    # Get transcript
    if req.transcript:
        transcript = req.transcript
    else:
        # For demo: use a mock transcript
        # In production: decode audio and use Cactus cactus_transcribe
        transcript = "Really enjoyed meeting Maya and talking about design systems. Send an email to schedule a follow up meeting."
    
    try:
        result = process_voice_note(transcript)
        
        contact = result.get("contact") or {}
        email = result.get("email_draft") or {}
        
        return VoiceNoteResponse(
            transcript=transcript,
            contact_name=contact.get("name"),
            contact_email=contact.get("email"),
            contact_role=contact.get("role"),
            contact_company=contact.get("company"),
            email_subject=email.get("subject"),
            email_body=email.get("body"),
            source=result.get("source", "hybrid"),
            status="success"
        )
    except Exception as e:
        return VoiceNoteResponse(
            transcript=transcript,
            status="error",
            error=str(e)
        )
