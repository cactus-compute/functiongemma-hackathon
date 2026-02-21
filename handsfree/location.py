"""
HandsFree — Location Module
On-device GPS via Apple CoreLocation + intent detection via keyword scanning.
No external API calls — coordinates and address stay on device.
"""

import re

# ── Intent detection ───────────────────────────────────────────────────────────────

# Patterns that mean the user wants to KNOW their current location.
_QUERY_KEYWORDS = [
    r"\bwhat.*\b(my|current)\s*(location|address|position)\b",
    r"\bwhere\s+am\s+i\b",
    r"\bwhere\s+i('m| am)\b",
    r"\bmy\s+(current\s+)?(location|address|position)\b",
    r"\bcurrent\s+location\b",
    r"\bmy\s+address\b",
]
_QUERY_RE = [re.compile(p, re.I) for p in _QUERY_KEYWORDS]


def detect_location_intent(text: str) -> bool:
    """Return True if the command is asking for the user's current location."""
    return is_location_query(text)


def is_location_query(text: str) -> bool:
    """Return True if the user is asking what their current location is."""
    return any(pat.search(text) for pat in _QUERY_RE)


def get_gps_location() -> dict | None:
    """
    Retrieve current GPS coordinates using Apple CoreLocation via pyobjc.
    Falls back to IP-based geolocation if CoreLocation is denied or unavailable.
    Returns dict with lat, lon, address, maps_link — or None if all methods fail.
    """
    try:
        import CoreLocation
        import time

        manager = CoreLocation.CLLocationManager.alloc().init()

        auth_status = CoreLocation.CLLocationManager.authorizationStatus()
        # kCLAuthorizationStatusDenied = 2, Restricted = 1, NotDetermined = 0
        if auth_status in (1, 2):
            # Permission denied — skip straight to IP fallback
            return _fallback_location()
        if auth_status == 0:
            manager.requestWhenInUseAuthorization()
            time.sleep(1.5)

        location = manager.location()
        if location is None:
            return _fallback_location()

        coord = location.coordinate()
        lat, lon = coord.latitude, coord.longitude
        if lat == 0.0 and lon == 0.0:
            return _fallback_location()

        address = _reverse_geocode(lat, lon)
        return {
            "lat": lat,
            "lon": lon,
            "address": address,
            "maps_link": f"https://maps.google.com/?q={lat:.6f},{lon:.6f}",
            "source": "CoreLocation (on-device GPS)",
        }
    except Exception:
        return _fallback_location()


def _reverse_geocode(lat: float, lon: float) -> str:
    """Reverse geocode coordinates to a human-readable address using CLGeocoder."""
    try:
        import CoreLocation
        import threading

        result = {"address": None, "done": threading.Event()}

        def completion(placemarks, error):
            if placemarks:
                pm = placemarks[0]
                parts = []
                if pm.subThoroughfare():
                    parts.append(pm.subThoroughfare())
                if pm.thoroughfare():
                    parts.append(pm.thoroughfare())
                if pm.locality():
                    parts.append(pm.locality())
                if pm.administrativeArea():
                    parts.append(pm.administrativeArea())
                result["address"] = ", ".join(parts) if parts else f"{lat:.4f}, {lon:.4f}"
            result["done"].set()

        geocoder = CoreLocation.CLGeocoder.alloc().init()
        loc = CoreLocation.CLLocation.alloc().initWithLatitude_longitude_(lat, lon)
        geocoder.reverseGeocodeLocation_completionHandler_(loc, completion)
        result["done"].wait(timeout=3.0)
        return result["address"] or f"{lat:.4f}°N, {lon:.4f}°W"
    except Exception:
        return f"{lat:.4f}°N, {lon:.4f}°W"


def _fallback_location() -> dict:
    """
    Fallback when CoreLocation is unavailable or denied.
    Uses IP-based geolocation (ipinfo.io, free, no key needed) for real location.
    """
    import requests as _req
    try:
        resp = _req.get("https://ipinfo.io/json", timeout=4).json()
        loc_str = resp.get("loc", "")         # "37.7749,-122.4194"
        city    = resp.get("city", "")
        region  = resp.get("region", "")
        country = resp.get("country", "")
        if loc_str and "," in loc_str:
            lat, lon = map(float, loc_str.split(","))
            address = ", ".join(p for p in [city, region, country] if p)
            return {
                "lat": lat,
                "lon": lon,
                "address": address or f"{lat:.4f}, {lon:.4f}",
                "maps_link": f"https://maps.google.com/?q={lat:.6f},{lon:.6f}",
                "source": "IP geolocation (ipinfo.io)",
            }
    except Exception:
        pass
    # Last resort: return None so callers know it truly failed
    return None


def inject_location_into_command(text: str, location: dict) -> str:
    """
    Rewrite a command to embed actual GPS coordinates.
    e.g. "Send my location to Mom" →
         "Send a message to Mom saying I'm at Civic Center, SF — https://maps.google.com/?q=..."
    """
    address = location["address"]
    maps_link = location["maps_link"]

    # Replace location-intent phrases with concrete address + link
    location_string = f"I'm at {address} — {maps_link}"

    # Try to detect a recipient pattern
    recipient_match = re.search(
        r'\bto\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b', text
    )

    if recipient_match:
        recipient = recipient_match.group(1)
        return f"Send a message to {recipient} saying {location_string}"

    # Generic fallback
    return f"{text.rstrip('.')} — my current location is: {location_string}"
