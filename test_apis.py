"""Test real API integrations in executor.py"""
import sys, os
sys.path.insert(0, "cactus/python/src")
sys.path.insert(0, ".")

os.environ.setdefault("GOOGLE_MAPS_API_KEY", "AIzaSyA5IPZCbVqBCbvOK24erpnIShSRltBWgYE")

from handsfree.executor import execute

print("=== 1. Weather (Open-Meteo, free, no key) ===")
r = execute([{"name": "get_weather", "arguments": {"location": "San Francisco"}}])[0]["result"]
if r["status"] == "ok":
    print(f"  {r['icon']} {r['location']}: {r['condition']}, {r['temp_f']}°F / {r['temp_c']}°C, {r['humidity']} humidity, wind {r['wind']}")
else:
    print(f"  ❌ {r['error']}")

print("\n=== 2. Directions (Google Maps) ===")
r = execute([{"name": "get_directions", "arguments": {
    "origin": "Civic Center San Francisco",
    "destination": "Golden Gate Bridge",
    "mode": "driving",
}}])[0]["result"]
if r["status"] == "ok":
    print(f"  {r['icon']} {r['from']} → {r['to']}")
    print(f"  Duration: {r['duration']} | Distance: {r['distance']}")
    for s in r.get("steps", [])[:3]:
        print(f"    • {s}")
    print(f"  URL: {r['maps_url']}")
else:
    print(f"  ❌ {r['error']}")

print("\n=== 3. Find Nearby (Google Places) ===")
r = execute([{"name": "find_nearby", "arguments": {
    "category": "coffee",
    "location": "Union Square, San Francisco",
}}])[0]["result"]
if r["status"] == "ok":
    print(f"  {r['icon']} {r['category']} near {r['near']}")
    for p in r["results"]:
        print(f"    • {p['name']} — {p['rating']}⭐ — {p['address']} ({p['status']})")
else:
    print(f"  ❌ {r['error']}")

print("\n=== 4. Search Along Route (Google Places + Directions) ===")
r = execute([{"name": "search_along_route", "arguments": {
    "query": "gas station",
    "origin": "San Jose, CA",
    "destination": "San Francisco, CA",
}}])[0]["result"]
if r["status"] == "ok":
    print(f"  {r['icon']} {r['query']} along {r['route']} ({r['route_duration']}, {r['route_distance']})")
    for p in r["results"][:4]:
        print(f"    • {p['name']} — {p['address']} — {p['rating']}⭐")
else:
    print(f"  ❌ {r['error']}")
