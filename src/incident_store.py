"""Vector store over the historical incident knowledge base."""
import json
import os
import faiss
import numpy as np
import pickle
from typing import List, Dict
from sentence_transformers import SentenceTransformer


class IncidentStore:
    def __init__(self, persist_dir: str = "faiss_incident_store", embedding_model: str = "all-MiniLM-L6-v2"):
        self.persist_dir = persist_dir
        os.makedirs(self.persist_dir, exist_ok=True)
        self.model = SentenceTransformer(embedding_model)
        self.index = None
        self.incidents: List[Dict] = []

    def _embed_text(self, incident: Dict) -> str:
        return f"{incident['system']} {incident['error_signature']} {incident['symptom']}"

    def build_from_jsonl(self, path: str):
        incidents = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    incidents.append(json.loads(line))
        texts = [self._embed_text(inc) for inc in incidents]
        embeddings = self.model.encode(texts).astype("float32")
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(embeddings)
        self.incidents = incidents
        self.save()
        print(f"[INFO] Built incident store with {len(incidents)} incidents.")

    def save(self):
        faiss.write_index(self.index, os.path.join(self.persist_dir, "faiss.index"))
        with open(os.path.join(self.persist_dir, "incidents.pkl"), "wb") as f:
            pickle.dump(self.incidents, f)

    def load(self):
        self.index = faiss.read_index(os.path.join(self.persist_dir, "faiss.index"))
        with open(os.path.join(self.persist_dir, "incidents.pkl"), "rb") as f:
            self.incidents = pickle.load(f)

    def search(self, error_text: str, top_k: int = 3) -> List[Dict]:
        query_emb = self.model.encode([error_text]).astype("float32")
        distances, indices = self.index.search(query_emb, top_k)
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if 0 <= idx < len(self.incidents):
                results.append({**self.incidents[idx], "distance": float(dist)})
        return results
