import pickle
from pathlib import Path
from typing import List

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


class RAGSystem:
    """
    Loads a pre-built FAISS index and corresponding chunk mapping, and retrieves
    the most relevant documentation context for a given query.
    """

    def __init__(
        self,
        index_path: Path | str | None = None,
        mapping_path: Path | str | None = None,
        model_name: str = "all-MiniLM-L6-v2",
        top_k: int = 3,
    ):
        base_dir = Path(__file__).resolve().parents[1]  # project root
        kb_dir = base_dir / "knowledge_base"

        self.index_path = (
            Path(index_path) if index_path else (kb_dir / "api_index.faiss")
        )
        self.mapping_path = (
            Path(mapping_path)
            if mapping_path
            else (kb_dir / "index_to_chunk.pkl")
        )
        self.top_k = max(1, top_k)

        print(f"[RAG] Loading SentenceTransformer: {model_name}")
        self.encoder = SentenceTransformer(model_name)

        print(f"[RAG] Reading FAISS index from: {self.index_path}")
        self.index = faiss.read_index(str(self.index_path))

        print(f"[RAG] Loading index-to-chunk mapping from: {self.mapping_path}")
        with open(self.mapping_path, "rb") as f:
            self.index_to_chunk: List[str] = pickle.load(f)

        print("[RAG] System ready")

    def get_context(self, query: str) -> str:
        # Encode and ensure float32 for FAISS
        vec = self.encoder.encode([query], normalize_embeddings=True)
        if not isinstance(vec, np.ndarray):
            vec = np.array(vec)
        vec = vec.astype("float32")

        distances, indices = self.index.search(vec, self.top_k)
        top_indices = indices[0]

        chunks: List[str] = []
        for idx in top_indices:
            if 0 <= idx < len(self.index_to_chunk):
                chunks.append(self.index_to_chunk[idx])

        # Join the best chunks as the context block
        return "\n\n".join(chunks) if chunks else ""
