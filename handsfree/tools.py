"""
HandsFree — Tool Registry
All tools available to the agent, tagged by whether they can run on-device.
"""

# ── Tool Definitions ─────────────────────────────────────────────────────────

TOOL_SEND_MESSAGE = {
    "name": "send_message",
    "description": "Send a text message or iMessage to a contact",
    "on_device": True,
    "parameters": {
        "type": "object",
        "properties": {
            "recipient": {"type": "string", "description": "Name of the contact to send the message to"},
            "message":   {"type": "string", "description": "The message content to send"},
        },
        "required": ["recipient", "message"],
    },
}

TOOL_SET_ALARM = {
    "name": "set_alarm",
    "description": "Set an alarm for a specific time",
    "on_device": True,
    "parameters": {
        "type": "object",
        "properties": {
            "hour":   {"type": "integer", "description": "Hour (0-23)"},
            "minute": {"type": "integer", "description": "Minute (0-59)"},
        },
        "required": ["hour", "minute"],
    },
}

TOOL_SET_TIMER = {
    "name": "set_timer",
    "description": "Set a countdown timer for a number of minutes",
    "on_device": True,
    "parameters": {
        "type": "object",
        "properties": {
            "minutes": {"type": "integer", "description": "Number of minutes for the timer"},
        },
        "required": ["minutes"],
    },
}

TOOL_CREATE_REMINDER = {
    "name": "create_reminder",
    "description": "Create a reminder with a title and time",
    "on_device": True,
    "parameters": {
        "type": "object",
        "properties": {
            "title": {"type": "string", "description": "Short reminder title"},
            "time":  {"type": "string", "description": "Time for the reminder (e.g. 3:00 PM)"},
        },
        "required": ["title", "time"],
    },
}

TOOL_PLAY_MUSIC = {
    "name": "play_music",
    "description": "Play a song, album, or playlist",
    "on_device": True,
    "parameters": {
        "type": "object",
        "properties": {
            "song": {"type": "string", "description": "Song, album, or playlist name"},
        },
        "required": ["song"],
    },
}

TOOL_SEARCH_CONTACTS = {
    "name": "search_contacts",
    "description": "Search for a contact by name",
    "on_device": True,
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Name to search for"},
        },
        "required": ["query"],
    },
}

TOOL_GET_WEATHER = {
    "name": "get_weather",
    "description": "Get current weather conditions for a location",
    "on_device": False,
    "parameters": {
        "type": "object",
        "properties": {
            "location": {"type": "string", "description": "City name or address"},
        },
        "required": ["location"],
    },
}

TOOL_GET_DIRECTIONS = {
    "name": "get_directions",
    "description": "Get driving or walking directions from one place to another",
    "on_device": False,
    "parameters": {
        "type": "object",
        "properties": {
            "origin":      {"type": "string", "description": "Starting location"},
            "destination": {"type": "string", "description": "Destination location"},
            "mode":        {"type": "string", "description": "Travel mode: driving, walking, transit"},
        },
        "required": ["origin", "destination"],
    },
}

TOOL_FIND_NEARBY = {
    "name": "find_nearby",
    "description": "Find nearby places of a given category (restaurants, gas stations, pharmacies, etc.)",
    "on_device": False,
    "parameters": {
        "type": "object",
        "properties": {
            "category": {"type": "string", "description": "Type of place (e.g. coffee shop, gas station, hospital)"},
            "location": {"type": "string", "description": "Center location to search around"},
        },
        "required": ["category", "location"],
    },
}

TOOL_SEARCH_ALONG_ROUTE = {
    "name": "search_along_route",
    "description": "Search for places of a given type along a driving route",
    "on_device": False,
    "parameters": {
        "type": "object",
        "properties": {
            "query":       {"type": "string", "description": "What to search for (e.g. gas station, coffee)"},
            "origin":      {"type": "string", "description": "Starting point of the route"},
            "destination": {"type": "string", "description": "End point of the route"},
        },
        "required": ["query", "origin", "destination"],
    },
}

TOOL_GET_CURRENT_LOCATION = {
    "name": "get_current_location",
    "description": "Get the user's current GPS location and return their address. Use when the user asks where they are, what their location is, or requests their current address.",
    "on_device": True,
    "parameters": {
        "type": "object",
        "properties": {
            "format": {"type": "string", "description": "Output format: 'full' for full address (default) or 'short' for city/neighborhood only"}
        },
        "required": [],
    },
}

# ── Grouped sets for different use contexts ───────────────────────────────────

# Full tool set available to the agent
ALL_TOOLS = [
    TOOL_SEND_MESSAGE,
    TOOL_SET_ALARM,
    TOOL_SET_TIMER,
    TOOL_CREATE_REMINDER,
    TOOL_PLAY_MUSIC,
    TOOL_SEARCH_CONTACTS,
    TOOL_GET_WEATHER,
    TOOL_GET_DIRECTIONS,
    TOOL_FIND_NEARBY,
    TOOL_SEARCH_ALONG_ROUTE,
    TOOL_GET_CURRENT_LOCATION,
]

# Subset for on-device-capable tasks
LOCAL_TOOLS = [t for t in ALL_TOOLS if t.get("on_device")]

TOOL_MAP = {t["name"]: t for t in ALL_TOOLS}
