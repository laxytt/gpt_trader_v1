import os
import json
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

CASES_FILE = "data/cases.json"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

class TradeMemoryRAG:
    def __init__(self, max_cases=300):
        self.model = SentenceTransformer(MODEL_NAME)
        self.max_cases = max_cases
        self.cases = []
        self.embeddings = None
        self.index = None
        self._load_cases()

    def _load_cases(self):
        if os.path.exists(CASES_FILE):
            with open(CASES_FILE, "r", encoding="utf-8") as f:
                try:
                    self.cases = json.load(f)
                except Exception:
                    self.cases = []
        if self.cases:
            self._build_index()

    def _save_cases(self):
        with open(CASES_FILE, "w", encoding="utf-8") as f:
            json.dump(self.cases, f, indent=2)

    def _build_index(self):
        # Use 'context' field for semantic embedding
        texts = [case.get("context", "") for case in self.cases]
        self.embeddings = np.array(self.model.encode(texts, show_progress_bar=False)).astype(np.float32)
        self.index = faiss.IndexFlatL2(self.embeddings.shape[1])
        self.index.add(self.embeddings)

    def add_case(self, case):
        # Prevent duplicates: require 'id' or fallback to hash of context+entry
        case_id = case.get("id") or f"{case.get('timestamp', '')}_{case.get('entry', '')}"
        if any(existing.get("id") == case_id for existing in self.cases):
            return
        # Keep only most recent cases
        if len(self.cases) >= self.max_cases:
            self.cases = self.cases[-self.max_cases + 1 :]
        case["id"] = case_id
        self.cases.append(case)
        self._save_cases()
        self._build_index()

    def query(self, context, k=3):
        """
        Returns the k most similar trade cases for the given context string.
        """
        if not self.cases or not self.index or not context:
            return []
        emb = np.array(self.model.encode([context])).astype(np.float32)
        D, I = self.index.search(emb, k)
        results = []
        for idx in I[0]:
            if 0 <= idx < len(self.cases):
                results.append(self.cases[idx])
        return results
