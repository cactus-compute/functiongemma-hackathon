"""
Voice Note Demo - Works without external API
For hackathon demonstration when API quota is exhausted
"""
import re
import requests

BACKEND_URL = "http://localhost:3001"

def lookup_contact_in_db(name: str) -> dict:
    """Search for contact in Mingle database by name."""
    try:
        response = requests.get(f"{BACKEND_URL}/api/profiles", timeout=5)
        profiles = response.json()
        
        name_lower = name.lower()
        for profile in profiles:
            if name_lower in profile.get("name", "").lower():
                email = f"{profile['name'].lower().replace(' ', '.')}@{profile['company'].lower().replace(' ', '')}.com"
                return {
                    "found": True,
                    "name": profile["name"],
                    "role": profile["role"],
                    "company": profile["company"],
                    "email": email,
                    "bio": profile.get("bio", ""),
                    "can_help_with": profile.get("can_help_with", [])
                }
        return {"found": False, "name": name}
    except Exception as e:
        return {"found": False, "error": str(e)}

def extract_intent_from_transcript(transcript: str) -> dict:
    """Extract contact name and action from transcript using regex."""
    # Find name - look for capitalized names after common patterns
    name_patterns = [
        r'(?:meeting|met|with|enjoyed meeting|talking to|spoke with|chatted with)\s+([A-Z][a-z]+)',
        r'([A-Z][a-z]+)\s+(?:and I|and we)\s+(?:talked|discussed|chatted)',
        r'(?:^|\s)([A-Z][a-z]+)\s+at\s+(?:the|a)',  # "Jordan at the meetup"
        r'Met\s+([A-Z][a-z]+)',  # Simple "Met Jordan"
    ]
    
    contact_name = None
    for pattern in name_patterns:
        match = re.search(pattern, transcript)
        if match:
            contact_name = match.group(1)
            break
    
    # Extract topic
    topic_patterns = [
        r'(?:about|discussing|talked about|chatted about|discussed)\s+([^.!?,]+)',
        r'(?:we talked|we discussed)\s+([^.!?]+)',
    ]
    topic = "our conversation"
    for pattern in topic_patterns:
        match = re.search(pattern, transcript, re.IGNORECASE)
        if match:
            topic = match.group(1).strip()
            # Clean up common trailing words
            topic = re.sub(r'\s+(and|to|with|for)\s*$', '', topic)
            break
    
    # Extract action
    action = "schedule a follow-up meeting"
    if "share" in transcript.lower():
        action = "share some resources"
    elif "intro" in transcript.lower():
        action = "make an introduction"
    elif "coffee" in transcript.lower():
        action = "grab coffee"
    elif "schedule" in transcript.lower() or "meeting" in transcript.lower():
        action = "schedule a follow-up meeting"
    elif "follow up" in transcript.lower():
        action = "follow up on our discussion"
    
    return {
        "contact_name": contact_name,
        "topic": topic,
        "action": action
    }

def process_voice_note_demo(transcript: str) -> dict:
    """Process voice note without external API - pure demo mode."""
    
    tool_calls = []
    
    # Step 1: Extract intent from transcript
    intent = extract_intent_from_transcript(transcript)
    tool_calls.append({
        "tool": "extract_intent",
        "source": "on-device (regex)",
        "result": intent
    })
    
    # Step 2: Look up contact
    contact_info = None
    if intent["contact_name"]:
        contact_info = lookup_contact_in_db(intent["contact_name"])
        tool_calls.append({
            "tool": "lookup_contact",
            "args": {"name": intent["contact_name"]},
            "source": "database",
            "result": contact_info
        })
    
    # Step 3: Generate email draft
    email_draft = None
    if contact_info and contact_info.get("found"):
        email_draft = {
            "to": contact_info["email"],
            "subject": f"Great meeting you - {intent['topic']}",
            "body": f"""Hi {contact_info['name']},

It was wonderful meeting you and discussing {intent['topic']}! I really enjoyed our conversation.

I'd love to {intent['action']}. Would you be available sometime next week?

Looking forward to staying in touch.

Best regards"""
        }
        tool_calls.append({
            "tool": "draft_email",
            "source": "template",
            "result": email_draft
        })
    
    return {
        "transcript": transcript,
        "contact": contact_info,
        "email_draft": email_draft,
        "tool_calls": tool_calls,
        "source": "demo-mode (no API)",
        "status": "success" if email_draft else "partial"
    }

if __name__ == "__main__":
    import json
    tests = [
        "Really enjoyed meeting Maya and talking about design systems. Send an email to schedule a follow up meeting.",
        "Met Jordan at the fintech meetup, we talked about Stripe and UX design. Want to follow up and share some resources.",
        "Had coffee with Chris and discussed Notion workflows. Let's schedule another chat.",
    ]
    for test in tests:
        print(f"\n=== Test: {test[:50]}... ===")
        result = process_voice_note_demo(test)
        print(f"Contact: {result.get('contact', {}).get('name', 'Not found')}")
        print(f"Status: {result['status']}")
