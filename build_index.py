"""Builds the FAISS index over the historical incident knowledge base."""
from src.incident_store import IncidentStore

if __name__ == "__main__":
    store = IncidentStore("faiss_incident_store")
    store.build_from_jsonl("data/incident_kb.jsonl")
