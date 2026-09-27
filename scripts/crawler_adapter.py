import json
import requests
from typing import List, Dict, Any
import argparse
from datetime import datetime

# Simulating a crawler export (StyloPost format from DarkWeb-Deanonymization/src/models.py)
CRAWLER_DUMP = [
    {
        "session_id": "CRAWL-20261010-001",
        "username": "shadow_broker",
        "platform": "darkweb",
        "source_url": "http://shadow555xyz.onion/thread/1",
        "content": "tbh we need to reconsider the architecture. FYI the previous build was highly unstable and I noticed recurring bugs.",
        "timestamp_parsed": "2026-10-10T02:00:00Z"
    },
    {
        "session_id": "CRAWL-20261010-001",
        "username": "shadow_broker",
        "platform": "darkweb",
        "source_url": "http://shadow555xyz.onion/thread/2",
        "content": "tbh I agree. FYI the latency is unacceptable. We need an alternative approach to this architecture issue.",
        "timestamp_parsed": "2026-10-11T02:15:00Z"
    },
    {
        "session_id": "CRAWL-20261010-001",
        "username": "shadow_broker",
        "platform": "darkweb",
        "source_url": "http://shadow555xyz.onion/thread/3",
        "content": "tbh testing is required. FYI deploying without security checks is a bad idea. Anyone found a workaround?",
        "timestamp_parsed": "2026-10-12T02:30:00Z"
    },
    {
        "session_id": "CRAWL-20261010-001",
        "username": "shadow_broker",
        "platform": "darkweb",
        "source_url": "http://shadow555xyz.onion/thread/4",
        "content": "tbh networking stack is failing. FYI I cannot connect to the backend server. Reconsider this deployment.",
        "timestamp_parsed": "2026-10-13T02:45:00Z"
    },
    {
        "session_id": "CRAWL-20261010-001",
        "username": "shadow_broker",
        "platform": "darkweb",
        "source_url": "http://shadow555xyz.onion/thread/5",
        "content": "tbh databases are out of sync. FYI the replica is lagging by 5 hours. We need imperative action now.",
        "timestamp_parsed": "2026-10-14T03:00:00Z"
    },
    {
        "session_id": "CRAWL-20261010-001",
        "username": "shadow_broker",
        "platform": "darkweb",
        "source_url": "http://shadow555xyz.onion/thread/6",
        "content": "tbh CPU is maxed out. FYI we hit 100% on all nodes. Scale up the architecture.",
        "timestamp_parsed": "2026-10-15T03:15:00Z"
    }
]

def adapt_crawler_posts(crawler_posts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Converts crawler StyloPost format to Persona API Post format."""
    adapted = []
    for idx, p in enumerate(crawler_posts):
        adapted.append({
            "post_id": f"crawl_{idx}",
            "author_id": p.get("username") or p.get("handle") or "unknown",
            "platform": p.get("platform", "darkweb"),
            "text": p.get("content", ""),
            "timestamp": p.get("timestamp_parsed"),
            "source_url": p.get("source_url")
        })
    return adapted

def send_to_persona_api(posts: List[Dict[str, Any]], persona_id: str):
    """Sends adapted posts to the running Persona API."""
    url = "http://127.0.0.1:8000/analyze"
    payload = {
        "persona_id": persona_id,
        "posts": posts
    }
    
    print(f"[+] Sending {len(posts)} adapted posts to {url}...")
    try:
        response = requests.post(url, json=payload, timeout=10)
        response.raise_for_status()
        result = response.json()
        print(f"[+] Success! Received {len(json.dumps(result))} bytes of analysis data.")
        
        # Save to file
        out_file = f"{persona_id}_analysis.json"
        with open(out_file, "w") as f:
            json.dump(result, f, indent=2)
        print(f"[+] Saved analysis to {out_file}")
        
    except requests.exceptions.RequestException as e:
        print(f"[-] API connection failed. Ensure the Persona API is running (uvicorn src.main:app --port 8000). Error: {e}")

if __name__ == "__main__":
    print("--- PRALAYX Crawler to Persona Adapter ---")
    adapted = adapt_crawler_posts(CRAWLER_DUMP)
    send_to_persona_api(adapted, "shadow_broker")
