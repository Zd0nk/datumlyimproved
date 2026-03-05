"""
FPL Data Fetcher
Runs via GitHub Actions to fetch data from the FPL API (no CORS issues server-side)
and saves it as JSON files that the static site can read.
"""
import json
import urllib.request
import os
from datetime import datetime

FPL_BASE = "https://fantasy.premierleague.com/api"
DATA_DIR = "data"

def fetch_json(url):
    """Fetch JSON from a URL."""
    req = urllib.request.Request(url, headers={
        'User-Agent': 'Mozilla/5.0 (compatible; FPL-Optimizer/1.0)'
    })
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode())

def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    print(f"[{datetime.utcnow().isoformat()}] Fetching FPL bootstrap data...")
    bootstrap = fetch_json(f"{FPL_BASE}/bootstrap-static/")

    # Save full bootstrap (elements, teams, events)
    with open(f"{DATA_DIR}/bootstrap.json", "w") as f:
        json.dump({
            "elements": bootstrap.get("elements", []),
            "teams": bootstrap.get("teams", []),
            "events": bootstrap.get("events", []),
            "element_types": bootstrap.get("element_types", []),
        }, f, separators=(",", ":"))
    print(f"  -> Saved {len(bootstrap.get('elements', []))} players, {len(bootstrap.get('teams', []))} teams")

    print("Fetching fixtures...")
    fixtures = fetch_json(f"{FPL_BASE}/fixtures/")
    with open(f"{DATA_DIR}/fixtures.json", "w") as f:
        json.dump(fixtures, f, separators=(",", ":"))
    print(f"  -> Saved {len(fixtures)} fixtures")

    # Save a metadata file with last update timestamp
    meta = {
        "last_updated": datetime.utcnow().isoformat() + "Z",
        "player_count": len(bootstrap.get("elements", [])),
        "fixture_count": len(fixtures),
    }
    with open(f"{DATA_DIR}/meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Done! Data saved to {DATA_DIR}/")

if __name__ == "__main__":
    main()
