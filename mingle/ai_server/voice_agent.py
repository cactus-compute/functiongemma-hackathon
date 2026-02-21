"""
Voice Note Agent - Hackathon Demo
Uses FunctionGemma for on-device tool calling with Gemini fallback
"""
import os
import json
import requests
from typing import Optional
from main import generate_hybrid

BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:3001")

# Tool definitions for FunctionGemma
VOICE_TOOLS = [
    {
        "name": "lookup_contact",
        "description": "Look up a contact by name in the network database. Returns their profile info including email.",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "First name or full name of the contact to look up"
                }
            },
            "required": ["name"]
        }
    },
    {
        "name": "draft_followup_email",
        "description": "Draft a follow-up email to a contact after meeting them.",
        "parameters": {
            "type": "object",
            "properties": {
                "contact_name": {"type": "string", "description": "Name of the recipient"},
                "topic": {"type": "string", "description": "What you discussed or the topic"},
                "action": {"type": "string", "description": "What you want to do (schedule meeting, share resources, etc)"}
            },
            "required": ["contact_name", "topic", "action"]
        }
    }
]

def lookup_contact(name: str) -> dict:
    """Search for a contact in the database by name."""
    try:
        response = requests.get(f"{BACKEND_URL}/api/profiles")
        profiles = response.json()
        
        # Search by name (case-insensitive, partial match)
        name_lower = name.lower()
        for profile in profiles:
            if name_lower in profile.get("name", "").lower():
                return {
                    "found": True,
                    "name": profile["name"],
                    "role": profile["role"],
                    "company": profile["company"],
                    "email": f"{profile['name'].lower().replace(' ', '.')}@{profile['company'].lower().replace(' ', '')}.com",
                    "bio": profile["bio"]
                }
        
        return {"found": False, "message": f"No contact found matching '{name}'"}
    except Exception as e:
        return {"found": False, "error": str(e)}

def draft_followup_email(contact_name: str, topic: str, action: str) -> dict:
    """Generate a follow-up email draft."""
    email_subject = f"Great meeting you - {topic}"
    email_body = f"""Hi {contact_name},

It was wonderful meeting you and discussing {topic}! I really enjoyed our conversation.

I'd love to {action}. Would you be available sometime next week?

Looking forward to staying in touch.

Best regards"""
    
    return {
        "subject": email_subject,
        "body": email_body
    }

def execute_tool(tool_name: str, args: dict) -> dict:
    """Execute a tool and return the result."""
    if tool_name == "lookup_contact":
        return lookup_contact(args.get("name", ""))
    elif tool_name == "draft_followup_email":
        return draft_followup_email(
            args.get("contact_name", ""),
            args.get("topic", ""),
            args.get("action", "connect")
        )
    else:
        return {"error": f"Unknown tool: {tool_name}"}

def process_voice_note(transcript: str) -> dict:
    """
    Process a voice note transcript using FunctionGemma tool calling.
    
    Flow:
    1. Send transcript + tools to FunctionGemma
    2. Execute any tool calls
    3. Return results
    """
    messages = [{
        "role": "user",
        "content": f"""Process this voice note and take appropriate action:

\"{transcript}\"

Use the available tools to:
1. Look up the contact mentioned
2. Draft an appropriate follow-up email

Call the tools with the correct parameters."""
    }]
    
    # Call FunctionGemma/Gemini hybrid
    result = generate_hybrid(messages, VOICE_TOOLS)
    
    tool_results = []
    contact_info = None
    email_draft = None
    
    # Check if we got tool calls
    if result.get("tool_calls"):
        for tool_call in result["tool_calls"]:
            tool_name = tool_call.get("name")
            tool_args = tool_call.get("arguments", {})
            
            tool_result = execute_tool(tool_name, tool_args)
            tool_results.append({
                "tool": tool_name,
                "args": tool_args,
                "result": tool_result
            })
            
            if tool_name == "lookup_contact" and tool_result.get("found"):
                contact_info = tool_result
            elif tool_name == "draft_followup_email":
                email_draft = tool_result
    
    # If no tool calls, try to extract info manually
    if not contact_info:
        # Fallback: try to find any name mentioned
        words = transcript.split()
        for i, word in enumerate(words):
            if word.lower() in ["meeting", "met", "with", "enjoyed"]:
                if i + 1 < len(words):
                    potential_name = words[i + 1].strip(".,!")
                    if potential_name[0].isupper():
                        contact_info = lookup_contact(potential_name)
                        if contact_info.get("found"):
                            break
    
    # If we have contact but no email draft, create one
    if contact_info and contact_info.get("found") and not email_draft:
        email_draft = draft_followup_email(
            contact_info["name"],
            "our recent conversation",
            "schedule a follow-up meeting"
        )
    
    return {
        "transcript": transcript,
        "source": result.get("source", "unknown"),
        "tool_calls": tool_results,
        "contact": contact_info,
        "email_draft": email_draft,
        "raw_result": result
    }


if __name__ == "__main__":
    # Test
    test_transcript = "Really enjoyed meeting Maya and talking about design systems. Send an email to schedule a follow up meeting."
    result = process_voice_note(test_transcript)
    print(json.dumps(result, indent=2))
