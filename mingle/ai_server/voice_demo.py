"""
Voice Note Demo - Works without external API
Supports multiple action types: email, linkedin, message
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
                    "linkedin": profile.get("linkedin_url", ""),
                    "bio": profile.get("bio", ""),
                    "can_help_with": profile.get("can_help_with", [])
                }
        return {"found": False, "name": name}
    except Exception as e:
        return {"found": False, "error": str(e)}

def extract_intent_from_transcript(transcript: str) -> dict:
    """Extract contact name, topic, and action type from transcript."""
    # Find name
    name_patterns = [
        r'(?:meeting|met|with|enjoyed meeting|talking to|spoke with|chatted with)\s+([A-Z][a-z]+)',
        r'([A-Z][a-z]+)\s+(?:and I|and we)\s+(?:talked|discussed|chatted)',
        r'(?:^|\s)([A-Z][a-z]+)\s+at\s+(?:the|a)',
        r'Met\s+([A-Z][a-z]+)',
        r'(?:coffee|lunch|dinner)\s+with\s+([A-Z][a-z]+)',
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
            topic = re.sub(r'\s+(and|to|with|for)\s*$', '', topic)
            break
    
    # Determine action type
    transcript_lower = transcript.lower()
    if "linkedin" in transcript_lower or "connect" in transcript_lower:
        action_type = "linkedin"
        action = "connect on LinkedIn"
    elif "text" in transcript_lower or "message" in transcript_lower or "dm" in transcript_lower:
        action_type = "message"
        action = "send a quick message"
    elif "call" in transcript_lower or "phone" in transcript_lower:
        action_type = "call"
        action = "schedule a call"
    elif "share" in transcript_lower:
        action_type = "email"
        action = "share some resources"
    elif "intro" in transcript_lower:
        action_type = "email"
        action = "make an introduction"
    else:
        action_type = "email"
        action = "schedule a follow-up meeting"
    
    return {
        "contact_name": contact_name,
        "topic": topic,
        "action": action,
        "action_type": action_type
    }

def generate_linkedin_message(contact: dict, topic: str) -> dict:
    """Generate a LinkedIn connection request message."""
    return {
        "type": "linkedin",
        "action": "Connect on LinkedIn",
        "message": f"Hi {contact['name'].split()[0]}, great meeting you! Would love to connect and continue our conversation about {topic}. - Looking forward to staying in touch!",
        "profile_url": contact.get("linkedin", f"https://linkedin.com/search?q={contact['name'].replace(' ', '+')}")
    }

def generate_quick_message(contact: dict, topic: str) -> dict:
    """Generate a quick text/DM message."""
    return {
        "type": "message",
        "action": "Send Message",
        "message": f"Hey {contact['name'].split()[0]}! 👋 Great meeting you earlier. Really enjoyed chatting about {topic}. Let's catch up again soon!",
        "to": contact.get("email", "")
    }

def generate_email(contact: dict, topic: str, action: str) -> dict:
    """Generate a follow-up email."""
    return {
        "type": "email",
        "action": "Send Email",
        "to": contact["email"],
        "subject": f"Great meeting you - {topic}",
        "body": f"""Hi {contact['name']},

It was wonderful meeting you and discussing {topic}! I really enjoyed our conversation.

I'd love to {action}. Would you be available sometime next week?

Looking forward to staying in touch.

Best regards"""
    }

def process_voice_note_demo(transcript: str) -> dict:
    """Process voice note and generate appropriate social action."""
    
    tool_calls = []
    
    # Step 1: Extract intent
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
    
    # Step 3: Generate output based on action type
    output = None
    if contact_info and contact_info.get("found"):
        action_type = intent["action_type"]
        
        if action_type == "linkedin":
            output = generate_linkedin_message(contact_info, intent["topic"])
        elif action_type == "message":
            output = generate_quick_message(contact_info, intent["topic"])
        else:  # email
            output = generate_email(contact_info, intent["topic"], intent["action"])
        
        tool_calls.append({
            "tool": f"generate_{action_type}",
            "source": "template",
            "result": output
        })
    
    return {
        "transcript": transcript,
        "contact": contact_info,
        "intent": intent,
        "output": output,
        "tool_calls": tool_calls,
        "source": "demo-mode (no API)",
        "status": "success" if output else "partial"
    }

if __name__ == "__main__":
    import json
    tests = [
        "Met Maya at the conference, want to connect on LinkedIn",
        "Had coffee with Jordan, send a quick message to say thanks",
        "Really enjoyed meeting Chris and talking about Notion. Send an email to follow up.",
    ]
    for test in tests:
        print(f"\n{'='*60}")
        print(f"Input: {test}")
        result = process_voice_note_demo(test)
        print(f"Action: {result['intent']['action_type']} → {result['output']['action'] if result['output'] else 'N/A'}")
        if result['output']:
            print(f"Message preview: {result['output'].get('message', result['output'].get('body', ''))[:80]}...")
