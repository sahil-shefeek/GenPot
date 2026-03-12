import json
import pickle
import time
from pathlib import Path
from typing import List

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


# In server/rag_system.py

class RAGSystem:
    def __init__(
        self,
        index_path: Path | str | None = None,
        mapping_path: Path | str | None = None,
        meta_path: Path | str | None = None,  # Added meta_path
        model_name: str = "BAAI/bge-small-en-v1.5",  # FIXED model mismatch
        top_k: int = 3,
    ):
        base_dir = Path(__file__).resolve().parents[1]
        kb_dir = base_dir / "knowledge_base"

        self.index_path = (
            Path(index_path) if index_path else (kb_dir / "api_index.faiss")
        )
        self.mapping_path = (
            Path(mapping_path) if mapping_path else (kb_dir / "index_to_chunk.pkl")
        )
        self.meta_path = (
            Path(meta_path) if meta_path else (kb_dir / "index_metadata.pkl")
        )
        self.top_k = max(1, top_k)

        print(f"[RAG] Loading SentenceTransformer: {model_name}")
        self.encoder = SentenceTransformer("BAAI/bge-small-en-v1.5")

        # --- Load HTTP Knowledge Base ---
        print(f"[RAG] Reading HTTP FAISS index from: {kb_dir / 'api_index.faiss'}")
        self.http_index = faiss.read_index(str(kb_dir / "api_index.faiss"))
        print(f"[RAG] Loading HTTP index-to-chunk mapping from: {kb_dir / 'index_to_chunk.pkl'}")
        with open(kb_dir / "index_to_chunk.pkl", "rb") as f:
            self.http_chunk = pickle.load(f)
        print(f"[RAG] Loading HTTP metadata from: {kb_dir / 'index_metadata.pkl'}")
        with open(kb_dir / "index_metadata.pkl", "rb") as f:
            self.http_meta = pickle.load(f)

        # --- Load SMTP Knowledge Base ---
        print(f"[RAG] Reading SMTP FAISS index from: {kb_dir / 'smtp_index.faiss'}")
        self.smtp_index = faiss.read_index(str(kb_dir / "smtp_index.faiss"))
        print(f"[RAG] Loading SMTP index-to-chunk mapping from: {kb_dir / 'smtp_index_to_chunk.pkl'}")
        with open(kb_dir / "smtp_index_to_chunk.pkl", "rb") as f:
            self.smtp_chunk = pickle.load(f)
        print(f"[RAG] Loading SMTP metadata from: {kb_dir / 'smtp_index_metadata.pkl'}")
        with open(kb_dir / "smtp_index_metadata.pkl", "rb") as f:
            self.smtp_meta = pickle.load(f)

        print("[RAG] System ready")

    def get_context(self, query: str, protocol: str = "http", similarity_threshold: float = 0.45) -> str:
        vec = self.encoder.encode([query], normalize_embeddings=True).astype("float32")

        # Select the correct protocol index
        if protocol.lower() == "smtp":
            index, chunk_map, meta_map = self.smtp_index, self.smtp_chunk, self.smtp_meta
        else:
            index, chunk_map, meta_map = self.http_index, self.http_chunk, self.http_meta

        distances, indices = index.search(vec, self.top_k)
        
        chunks = []
        for dist, idx in zip(distances[0], indices[0]):
            idx = int(idx)
            if dist >= similarity_threshold and idx in chunk_map:
                meta = meta_map.get(idx, {})
                
                # Format dynamically based on available metadata
                header = f"--- {'SMTP COMMAND' if protocol == 'smtp' else 'ENDPOINT'} ---"
                identifier = meta.get("command") or f"{meta.get('method', 'GET')} {meta.get('path', '/')}"
                
                formatted_chunk = (
                    f"{header}\n"
                    f"Target: {identifier}\n"
                    f"Similarity Score: {dist:.4f}\n"
                    f"Details:\n{chunk_map[idx]}"
                )
                chunks.append(formatted_chunk)

        if not chunks:
            return "NO_RELEVANT_CONTEXT_FOUND"

        return "\n\n".join(chunks)

    def inspect_query(self, query: str, protocol: str = "http", top_k: int = None) -> dict:
        start_time = time.perf_counter()
        k = max(1, top_k) if top_k is not None else self.top_k

        vec = self.encoder.encode([query], normalize_embeddings=True)
        if not isinstance(vec, np.ndarray):
            vec = np.array(vec)
        vec = vec.astype("float32")

        if protocol.lower() == "smtp":
            index, chunk_map = self.smtp_index, self.smtp_chunk
        else:
            index, chunk_map = self.http_index, self.http_chunk

        distances, indices = index.search(vec, k)

        chunks = []
        for idx, dist in zip(indices[0], distances[0]):
            idx = int(idx)
            if idx in chunk_map:
                chunks.append(
                    {
                        "chunk_index": idx,
                        "faiss_distance": float(dist),
                        "text": str(chunk_map[idx]),
                    }
                )

        end_time = time.perf_counter()
        latency_ms = (end_time - start_time) * 1000

        return {
            "provided_query": query,
            "latency_ms": round(latency_ms, 2),
            "chunks": chunks,
        }

    def _extract_schema(self, data) -> str:
        """
        Recursively extracts the schema from a JSON-like dictionary or list,
        replacing concrete values with their type representations (e.g., 'str', 'int').
        """

        def extract(obj):
            if isinstance(obj, dict):
                return {k: extract(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                if len(obj) > 0:
                    return [extract(obj[0])]  # Capture the schema of the first item
                return []
            else:
                return type(obj).__name__

        schema_dict = extract(data)
        return json.dumps(schema_dict, indent=2)

    def compute_similarity(self, doc_text: str, response_json: dict) -> float:
        """
        Computes the cosine similarity between the retrieved documentation context
        and the structural schema of the generated JSON response.
        """
        if not doc_text or doc_text == "NO_RELEVANT_CONTEXT_FOUND":
            return 0.0

        schema_str = self._extract_schema(response_json)

        doc_vec = self.encoder.encode([doc_text], normalize_embeddings=True)
        resp_vec = self.encoder.encode([schema_str], normalize_embeddings=True)

        if not isinstance(doc_vec, np.ndarray):
            doc_vec = np.array(doc_vec)
        if not isinstance(resp_vec, np.ndarray):
            resp_vec = np.array(resp_vec)

        doc_vec = doc_vec.astype("float32")
        resp_vec = resp_vec.astype("float32")

        # Compute cosine similarity (dot product of normalized vectors)
        similarity = float(np.dot(doc_vec[0], resp_vec[0]))

        return max(0.0, min(1.0, similarity))
