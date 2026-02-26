import json
import pickle
import time
from pathlib import Path
from typing import List

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


class RAGSystem:
    def __init__(
        self,
        index_path: Path | str | None = None,
        mapping_path: Path | str | None = None,
        meta_path: Path | str | None = None, # Added meta_path
        model_name: str = "sentence-transformers/multi-qa-MiniLM-L6-cos-v1", # FIXED model mismatch
        top_k: int = 3,
    ):
        base_dir = Path(__file__).resolve().parents[1]
        kb_dir = base_dir / "knowledge_base"

        self.index_path = Path(index_path) if index_path else (kb_dir / "api_index.faiss")
        self.mapping_path = Path(mapping_path) if mapping_path else (kb_dir / "index_to_chunk.pkl")
        self.meta_path = Path(meta_path) if meta_path else (kb_dir / "index_metadata.pkl")
        self.top_k = max(1, top_k)

        print(f"[RAG] Loading SentenceTransformer: {model_name}")
        self.encoder = SentenceTransformer(model_name)

        print(f"[RAG] Reading FAISS index from: {self.index_path}")
        self.index = faiss.read_index(str(self.index_path))

        print(f"[RAG] Loading index-to-chunk mapping from: {self.mapping_path}")
        with open(self.mapping_path, "rb") as f:
            self.index_to_chunk: dict = pickle.load(f)

        # NEW: Load the metadata mapping
        print(f"[RAG] Loading metadata from: {self.meta_path}")
        with open(self.meta_path, "rb") as f:
            self.index_metadata: dict = pickle.load(f)

        print("[RAG] System ready")

    def get_context(self, query: str, similarity_threshold: float = 0.45) -> str:
        vec = self.encoder.encode([query], normalize_embeddings=True)
        if not isinstance(vec, np.ndarray):
            vec = np.array(vec)
        vec = vec.astype("float32")

        distances, indices = self.index.search(vec, self.top_k)
        
        chunks: List[str] = []
        # Zip distances and indices together so we can check the score
        for dist, idx in zip(distances[0], indices[0]):
            idx = int(idx)
            
            # ONLY include the chunk if the cosine similarity is above our threshold
            if dist >= similarity_threshold:
                if idx in self.index_to_chunk and idx in self.index_metadata:
                    chunk_text = self.index_to_chunk[idx]
                    chunk_meta = self.index_metadata[idx]
                    
                    api_path = chunk_meta.get("path", "Unknown Path")
                    api_method = chunk_meta.get("method", "Unknown Method")
                    
                    formatted_chunk = (
                        f"--- ENDPOINT ---\n"
                        f"Path: {api_path}\n"
                        f"Method: {api_method}\n"
                        f"Similarity Score: {dist:.4f}\n" # Helpful for debugging
                        f"Details:\n{chunk_text}"
                    )
                    chunks.append(formatted_chunk)

        # If no chunks met the threshold, return a specific flag or empty string
        if not chunks:
            return "NO_RELEVANT_CONTEXT_FOUND"

        return "\n\n".join(chunks)
    def inspect_query(self, query: str, top_k: int = None) -> dict:
        start_time = time.perf_counter()
        k = max(1, top_k) if top_k is not None else self.top_k

        vec = self.encoder.encode([query], normalize_embeddings=True)
        if not isinstance(vec, np.ndarray):
            vec = np.array(vec)
        vec = vec.astype("float32")

        distances, indices = self.index.search(vec, k)
        
        chunks = []
        for idx, dist in zip(indices[0], distances[0]):
            if 0 <= idx < len(self.index_to_chunk):
                chunks.append({
                    "chunk_index": int(idx),
                    "faiss_distance": float(dist),
                    "text": str(self.index_to_chunk[idx])
                })

        end_time = time.perf_counter()
        latency_ms = (end_time - start_time) * 1000

        return {
            "provided_query": query,
            "latency_ms": round(latency_ms, 2),
            "chunks": chunks
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