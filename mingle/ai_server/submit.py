"""
Submit your solution to the Cactus Evals leaderboard.

Usage:
    python submit.py --team "YourTeamName" --location "SF"
"""

import argparse
import os
import time
import requests

_HERE = os.path.dirname(os.path.abspath(__file__))

SERVER_URL = "https://cactusevals.ngrok.app"
HEADERS = {"ngrok-skip-browser-warning": "true"}


def submit(team, location):
    print("=" * 60)
    print(f"  Submitting main.py for team '{team}' ({location})")
    print("=" * 60)

    try:
        with open(os.path.join(_HERE, "main.py"), "rb") as f:
            resp = requests.post(
                f"{SERVER_URL}/eval/submit",
                data={"team": team, "location": location},
                files={"file": ("main.py", f, "text/x-python")},
                headers=HEADERS,
                timeout=15,
            )
    except requests.exceptions.ConnectionError:
        print("The Leaderboard is not accepting submissions at this time.")
        return
    except requests.exceptions.Timeout:
        print("The Leaderboard is not accepting submissions at this time.")
        return

    if resp.status_code != 200:
        try:
            msg = resp.json().get("error", resp.text)
        except (requests.exceptions.JSONDecodeError, ValueError):
            print("The Leaderboard is not accepting submissions at this time.")
            return
        print(f"Error: {msg}")
        return

    data = resp.json()
    submission_id = data["submission_id"]
    print(f"Queued! Position: #{data['position_in_queue']}")
    print(f"Submission ID: {submission_id}")
    print(f"\nWaiting for evaluation to complete...\n")

    last_progress = ""
    while True:
        time.sleep(3)
        resp = requests.get(
            f"{SERVER_URL}/eval/status",
            params={"id": submission_id},
            headers=HEADERS,
        )
        if resp.status_code != 200:
            print("Error polling status. Retrying...")
            continue

        status = resp.json()

        if status["progress"] and status["progress"] != last_progress:
            last_progress = status["progress"]
            print(f"  [{status['progress']}]", flush=True)

        if status["status"] == "complete":
            result = status["result"]
            print(f"\n{'=' * 50}")
            print(f"  RESULTS for team '{result['team']}'")
            print(f"{'=' * 50}")
            print(f"  Total Score : {result['score']:.1f}%")
            print(f"  Avg F1      : {result['f1']:.4f}")
            print(f"  Avg Time    : {result['avg_time_ms']:.0f}ms")
            print(f"  On-Device   : {result['on_device_pct']:.0f}%")
            print(f"  Leaderboard : Updated!")
            print(f"{'=' * 50}")
            return

        if status["status"] == "error":
            print(f"\nError: {status.get('error', 'Unknown error')}")
            return

        if status["status"] == "queued":
            print(f"  Queued (queue size: {status['queue_size']})...", end="\r", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Submit to Cactus Evals Leaderboard")
    parser.add_argument("--team", type=str, required=True, help="Your team name")
    parser.add_argument("--location", type=str, required=True, help="Your location (e.g. SF, NYC, London)")
    args = parser.parse_args()
    submit(args.team, args.location)
