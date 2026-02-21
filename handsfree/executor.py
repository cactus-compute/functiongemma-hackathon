"""
HandsFree — Function Executor
Real API integrations:
  - Weather  : Open-Meteo (free, no key)
  - Maps     : Google Maps Platform (GOOGLE_MAPS_API_KEY env var)
  - Others   : simulated (iMessage, alarms, music)
"""

import os
import time
from datetime import datetime

import requests

# ── Google Maps client (lazy-initialised) ─────────────────────────────────────
_gmaps = None

def _get_gmaps():
    global _gmaps
    if _gmaps is None:
        key = os.environ.get("GOOGLE_MAPS_API_KEY", "")
        if not key:
            raise RuntimeError("GOOGLE_MAPS_API_KEY is not set")
        import googlemaps
        _gmaps = googlemaps.Client(key=key)
    return _gmaps


# Phrases that mean "use my current GPS location"
_HERE_PHRASES = {
    "near me", "my location", "my current location", "current location",
    "here", "where i am", "where i'm at", "my position",
}

def _resolve_location(loc_str: str) -> str:
    """
    If loc_str is a 'near me' style phrase, replace it with the user's
    real GPS coordinates (lat,lng string) suitable for geocoding/Maps APIs.
    Otherwise return loc_str unchanged.
    """
    if loc_str.strip().lower() in _HERE_PHRASES:
        from handsfree.location import get_gps_location
        loc = get_gps_location()
        if loc:
            return f"{loc['lat']},{loc['lon']}"
    return loc_str


def execute(function_calls: list[dict]) -> list[dict]:
    """Execute a list of function calls and return results."""
    results = []
    for call in function_calls:
        fn   = call.get("name", "unknown")
        args = call.get("arguments", {})
        handler = _HANDLERS.get(fn, _unknown)
        try:
            result = handler(args)
        except Exception as e:
            result = {"status": "error", "error": str(e)}
        results.append({
            "function": fn,
            "arguments": args,
            "result": result,
        })
    return results


# ── Handlers ──────────────────────────────────────────────────────────────────

def _send_message(args):
    recipient = args.get("recipient", "Unknown")
    message   = args.get("message", "")
    return {
        "status": "sent",
        "to": recipient,
        "preview": message[:60] + ("…" if len(message) > 60 else ""),
        "timestamp": datetime.now().strftime("%I:%M %p"),
        "icon": "💬",
    }


def _set_alarm(args):
    hour   = args.get("hour", 0)
    minute = args.get("minute", 0)
    period = "AM" if hour < 12 else "PM"
    display_hour = hour if hour <= 12 else hour - 12
    display_hour = display_hour or 12
    return {
        "status": "set",
        "time": f"{display_hour}:{minute:02d} {period}",
        "icon": "⏰",
    }


def _set_timer(args):
    minutes = args.get("minutes", 0)
    return {
        "status": "running",
        "duration": f"{minutes} minute{'s' if minutes != 1 else ''}",
        "ends_at": f"{minutes}m from now",
        "icon": "⏱️",
    }


def _create_reminder(args):
    title = args.get("title", "Reminder")
    time_str = args.get("time", "")
    return {
        "status": "created",
        "title": title.capitalize(),
        "time": time_str,
        "icon": "📌",
    }


def _play_music(args):
    song = args.get("song", "")
    return {
        "status": "playing",
        "track": song,
        "icon": "🎵",
    }


def _search_contacts(args):
    query = args.get("query", "")
    # Simulate finding a contact
    return {
        "status": "found",
        "query": query,
        "results": [
            {"name": query, "phone": "+1 (555) 000-0000", "email": f"{query.lower()}@example.com"},
        ],
        "icon": "👤",
    }


# WMO weather code → human label
_WMO = {
    0: "Clear Sky", 1: "Mainly Clear", 2: "Partly Cloudy", 3: "Overcast",
    45: "Foggy", 48: "Icy Fog",
    51: "Light Drizzle", 53: "Moderate Drizzle", 55: "Heavy Drizzle",
    61: "Light Rain", 63: "Moderate Rain", 65: "Heavy Rain",
    71: "Light Snow", 73: "Moderate Snow", 75: "Heavy Snow",
    80: "Rain Showers", 81: "Moderate Showers", 82: "Violent Showers",
    95: "Thunderstorm", 96: "Thunderstorm w/ Hail",
}

def _get_weather(args):
    location = _resolve_location(args.get("location", ""))
    try:
        # If location is already "lat,lon" (from 'near me' resolution), reverse geocode it
        if location.count(",") == 1 and all(c in "0123456789.-, " for c in location):
            parts = location.split(",")
            lat, lon = float(parts[0].strip()), float(parts[1].strip())
            rev = requests.get(
                "https://nominatim.openstreetmap.org/reverse",
                params={"lat": lat, "lon": lon, "format": "json"},
                headers={"User-Agent": "HandsFreeApp/1.0"},
                timeout=5,
            ).json()
            display = rev.get("address", {}).get("city") or rev.get("display_name", location).split(",")[0]
        else:
            # 1. Geocode city name via Nominatim (free, no key)
            geo = requests.get(
                "https://nominatim.openstreetmap.org/search",
                params={"q": location, "format": "json", "limit": 1},
                headers={"User-Agent": "HandsFreeApp/1.0"},
                timeout=5,
            ).json()
            if not geo:
                raise ValueError(f"Location not found: {location}")
            lat, lon = float(geo[0]["lat"]), float(geo[0]["lon"])
            display = geo[0].get("display_name", location).split(",")[0]

        # 2. Fetch weather from Open-Meteo (free, no key)
        wx = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": lat, "longitude": lon,
                "current": "temperature_2m,relative_humidity_2m,wind_speed_10m,weathercode",
                "temperature_unit": "fahrenheit",
                "wind_speed_unit": "mph",
                "forecast_days": 1,
            },
            timeout=5,
        ).json()
        cur = wx["current"]
        code = cur.get("weathercode", 0)
        condition = _WMO.get(code, "Unknown")
        temp_f = cur["temperature_2m"]
        temp_c = round((temp_f - 32) * 5 / 9, 1)
        humidity = cur["relative_humidity_2m"]
        wind = cur["wind_speed_10m"]
        return {
            "status": "ok",
            "location": display,
            "condition": condition,
            "temp_f": round(temp_f, 1),
            "temp_c": temp_c,
            "humidity": f"{humidity}%",
            "wind": f"{wind} mph",
            "icon": "⛅",
        }
    except Exception as e:
        return {"status": "error", "error": str(e), "icon": "⛅"}


def _get_directions(args):
    origin      = _resolve_location(args.get("origin", "") or "Current location")
    destination = _resolve_location(args.get("destination", ""))
    mode        = args.get("mode", "driving")
    try:
        gmaps = _get_gmaps()
        result = gmaps.directions(origin, destination, mode=mode)
        if not result:
            raise ValueError("No route found")
        leg = result[0]["legs"][0]
        duration = leg["duration"]["text"]
        distance = leg["distance"]["text"]
        start    = leg["start_address"]
        end      = leg["end_address"]
        import re as _re
        def _strip_html(h):
            h = h.replace("<wbr/>", "").replace("<wbr>", "")
            h = h.replace('<div style="font-size:0.9em">', " — ").replace("</div>", "")
            return _re.sub(r"<[^>]+>", "", h).strip()
        steps = [_strip_html(s["html_instructions"]) for s in leg["steps"][:6]]
        maps_url = (
            f"https://www.google.com/maps/dir/?api=1"
            f"&origin={requests.utils.quote(start)}"
            f"&destination={requests.utils.quote(end)}"
            f"&travelmode={mode}"
        )
        return {
            "status": "ok",
            "from": start,
            "to": end,
            "mode": mode,
            "duration": duration,
            "distance": distance,
            "steps": steps,
            "maps_url": maps_url,
            "icon": "🗺️",
        }
    except Exception as e:
        return {"status": "error", "error": str(e), "icon": "🗺️"}


def _find_nearby(args):
    category = args.get("category", "")
    location = _resolve_location(args.get("location", ""))
    try:
        gmaps = _get_gmaps()
        # If already lat,lng from _resolve_location, pass directly; else geocode
        if location.count(",") == 1 and all(c in "0123456789.-, " for c in location):
            parts = location.split(",")
            latlng = {"lat": float(parts[0].strip()), "lng": float(parts[1].strip())}
        else:
            geo = gmaps.geocode(location)
            if not geo:
                raise ValueError(f"Cannot geocode: {location}")
            latlng = geo[0]["geometry"]["location"]

        places = gmaps.places_nearby(
            location=latlng,
            radius=1500,
            keyword=category,
        )
        results = []
        for p in places.get("results", [])[:5]:
            name    = p.get("name", "")
            rating  = p.get("rating", "N/A")
            address = p.get("vicinity", "")
            open_now = p.get("opening_hours", {}).get("open_now", None)
            status  = "Open" if open_now else ("Closed" if open_now is False else "Hours unknown")
            results.append({"name": name, "rating": rating, "address": address, "status": status})

        return {
            "status": "ok",
            "category": category,
            "near": location,
            "results": results,
            "icon": "📍",
        }
    except Exception as e:
        return {"status": "error", "error": str(e), "icon": "📍"}


def _search_along_route(args):
    query       = args.get("query", "")
    origin      = _resolve_location(args.get("origin", ""))
    destination = _resolve_location(args.get("destination", ""))
    try:
        gmaps = _get_gmaps()
        # Get route polyline
        route = gmaps.directions(origin, destination, mode="driving")
        if not route:
            raise ValueError("No route found")

        # Sample waypoints along the route (every ~5 steps)
        steps = route[0]["legs"][0]["steps"]
        sample_points = [
            steps[i]["end_location"]
            for i in range(0, len(steps), max(1, len(steps) // 5))
        ][:3]

        results = []
        seen = set()
        for pt in sample_points:
            nearby = gmaps.places_nearby(
                location=pt,
                radius=800,
                keyword=query,
            )
            for p in nearby.get("results", [])[:2]:
                name = p.get("name", "")
                if name in seen:
                    continue
                seen.add(name)
                results.append({
                    "name": name,
                    "address": p.get("vicinity", ""),
                    "rating": p.get("rating", "N/A"),
                })
            if len(results) >= 4:
                break

        total_duration = route[0]["legs"][0]["duration"]["text"]
        total_distance = route[0]["legs"][0]["distance"]["text"]
        return {
            "status": "ok",
            "query": query,
            "route": f"{origin} → {destination}",
            "route_duration": total_duration,
            "route_distance": total_distance,
            "results": results,
            "icon": "🛣️",
        }
    except Exception as e:
        return {"status": "error", "error": str(e), "icon": "🛣️"}


def _get_current_location(args):
    fmt = args.get("format", "full")
    try:
        # 1. Get GPS coordinates from CoreLocation
        from handsfree.location import get_gps_location
        loc = get_gps_location()
        if not loc:
            raise RuntimeError("Could not determine location — CoreLocation denied and IP lookup failed")

        lat, lon = loc["lat"], loc["lon"]

        # 2. Reverse-geocode via Google Maps for a clean, accurate address
        try:
            gmaps = _get_gmaps()
            results = gmaps.reverse_geocode((lat, lon))
            if results:
                full_address = results[0]["formatted_address"]
                # Extract neighbourhood/city for short format
                components = results[0].get("address_components", [])
                neighbourhood = next(
                    (c["long_name"] for c in components
                     if "sublocality" in c["types"] or "neighborhood" in c["types"]),
                    None
                )
                city = next(
                    (c["long_name"] for c in components if "locality" in c["types"]),
                    None
                )
                short_address = neighbourhood or city or full_address.split(",")[0]
            else:
                full_address = loc.get("address", f"{lat:.5f}, {lon:.5f}")
                short_address = full_address.split(",")[0]
        except Exception:
            # Fall back to CoreLocation address if Maps key unavailable
            full_address = loc.get("address", f"{lat:.5f}, {lon:.5f}")
            short_address = full_address.split(",")[0]

        display = short_address if fmt == "short" else full_address
        maps_link = f"https://maps.google.com/?q={lat:.6f},{lon:.6f}"

        return {
            "status": "ok",
            "address": display,
            "full_address": full_address,
            "latitude": round(lat, 6),
            "longitude": round(lon, 6),
            "source": loc.get("source", "GPS"),
            "maps_link": maps_link,
            "icon": "📍",
        }
    except Exception as e:
        return {"status": "error", "error": str(e), "icon": "📍"}


def _unknown(args):
    return {"status": "error", "error": "Unknown function"}


_HANDLERS = {
    "send_message":          _send_message,
    "set_alarm":             _set_alarm,
    "set_timer":             _set_timer,
    "create_reminder":       _create_reminder,
    "play_music":            _play_music,
    "search_contacts":       _search_contacts,
    "get_weather":           _get_weather,
    "get_directions":        _get_directions,
    "find_nearby":           _find_nearby,
    "search_along_route":    _search_along_route,
    "get_current_location":  _get_current_location,
}
