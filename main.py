"""Single entry point: builds the index if needed, parses the sample log,
and runs every incident through the multi-agent analysis pipeline.

Run with: python main.py
"""
import os
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from src.log_parser import extract_error_events
from src.incident_store import IncidentStore
from src.agent_graph import build_graph, analyze_event

LOG_PATH = "data/sample_pipeline.log"
INDEX_DIR = "faiss_incident_store"
KB_PATH = "data/incident_kb.jsonl"


def main():
    store = IncidentStore(INDEX_DIR)
    if not os.path.exists(os.path.join(INDEX_DIR, "faiss.index")):
        print("[INFO] No index found, building it first...")
        store.build_from_jsonl(KB_PATH)
    else:
        store.load()

    graph = build_graph(store)

    events = extract_error_events(LOG_PATH)
    print(f"[INFO] Found {len(events)} error event(s) in {LOG_PATH}\n")

    for i, event in enumerate(events, 1):
        print(f"{'='*70}\nEVENT {i}/{len(events)}\n{'='*70}")
        result = analyze_event(graph, event)
        print(result["report"])
        print()


if __name__ == "__main__":
    main()
