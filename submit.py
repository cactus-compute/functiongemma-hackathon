"""
Submit your solution to the FunctionGemma Hackathon leaderboard.

Usage:
    python submit.py --team "YourTeamName"
"""

import sys
sys.path.insert(0, "cactus/python/src")
import os
os.environ["CACTUS_NO_CLOUD_TELE"] = "1"

import argparse
import requests
from example import generate_hybrid

SERVER_URL = "https://khalilah-unbibulous-carlena.ngrok-free.dev"

HEADERS = {"ngrok-skip-browser-warning": "true"}


def submit(team):
    print("=" * 60)
    print("  Held-out evaluation")
    print("=" * 60)

    resp = requests.post(f"{SERVER_URL}/eval/start", json={"team": team}, headers=HEADERS)
    if resp.status_code != 200:
        try:
            msg = resp.json().get("error", resp.text)
        except requests.exceptions.JSONDecodeError:
            msg = resp.text[:200]
        print(f"Error starting eval: {msg}")
        return
    session = resp.json()
    token = session["token"]
    total = session["total_cases"]
    print(f"Session started for team '{team}'. {total} held-out cases to evaluate.\n")

    completed = 0
    while True:
        resp = requests.get(f"{SERVER_URL}/eval/next", params={"token": token}, headers=HEADERS)
        if resp.status_code != 200:
            try:
                msg = resp.json().get("error", resp.text)
            except requests.exceptions.JSONDecodeError:
                msg = resp.text[:200]
            print(f"Error getting next case: {msg}")
            return
        data = resp.json()

        if data.get("done"):
            break

        case_num = data["case_number"]
        case_id = data["id"]
        difficulty = data["difficulty"]
        print(f"[{case_num}/{total}] Evaluating: {case_id} ({difficulty})...", end=" ", flush=True)

        result = generate_hybrid(data["messages"], data["tools"])

        resp = requests.post(
            f"{SERVER_URL}/eval/submit",
            params={"token": token},
            json={
                "function_calls": result["function_calls"],
                "total_time_ms": result["total_time_ms"],
                "source": result.get("source", "unknown"),
            },
            headers=HEADERS,
        )
        if resp.status_code != 200:
            try:
                msg = resp.json().get("error", resp.text)
            except requests.exceptions.JSONDecodeError:
                msg = resp.text[:200]
            print(f"Error submitting: {msg}")
            return

        sub = resp.json()
        print(f"F1={sub['f1']:.2f} | {result['total_time_ms']:.0f}ms | {result.get('source', 'unknown')}")
        completed += 1

    print(f"\nAll {completed} cases submitted. Fetching final score...")
    resp = requests.get(f"{SERVER_URL}/eval/finish", params={"token": token}, headers=HEADERS)
    if resp.status_code != 200:
        try:
            msg = resp.json().get("error", resp.text)
        except requests.exceptions.JSONDecodeError:
            msg = resp.text[:200]
        print(f"Error finishing: {msg}")
        return

    final = resp.json()
    print(f"\n{'=' * 50}")
    print(f"  RESULTS for team '{final['team']}'")
    print(f"{'=' * 50}")
    print(f"  Total Score : {final['score']:.1f}%")
    print(f"  Avg F1      : {final['f1']:.4f}")
    print(f"  Avg Time    : {final['avg_time_ms']:.0f}ms")
    print(f"  On-Device   : {final['on_device_pct']:.0f}%")
    print(f"  Leaderboard : {'Updated!' if final['leaderboard_updated'] else 'Not updated'}")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Submit to FunctionGemma Hackathon Leaderboard")
    parser.add_argument("--team", type=str, required=True, help="Your team name")
    args = parser.parse_args()
    submit(args.team)
