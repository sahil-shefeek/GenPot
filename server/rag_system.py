import pickle
import faiss
import os
import time
import torch
import numpy as np
from pathlib import Path
from optimum.onnxruntime import ORTModelForFeatureExtraction
from transformers import AutoTokenizer

# Disable heavy parallelism which hurts latency on single queries
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"


class RAGSystem:
    def __init__(self):
        # Get the directory where this script (rag_system.py) is located
        current_dir = Path(__file__).resolve().parent

        # ASSUMPTION: 'knowledge_base' is inside 'server/'
        self.kb_dir = current_dir / "knowledge_base"
        if not self.kb_dir.exists():
            self.kb_dir = current_dir.parent / "knowledge_base"

        # CHECK 1: Look for onnx_model inside 'server/'
        self.onnx_path = current_dir / "onnx_model"
        # CHECK 2: If not found, look one level up (in case it's in project root)
        if not self.onnx_path.exists():
            self.onnx_path = current_dir.parent / "onnx_model"

        # FINAL CHECK: Did we find it?
        if not self.onnx_path.exists():
            raise FileNotFoundError(
                f"❌ Could not find 'onnx_model' folder.\n"
                f"Checked: {current_dir / 'onnx_model'}\n"
                f"Checked: {current_dir.parent / 'onnx_model'}\n"
                "Please run optimize_model.py first!"
            )
        print("[RAG] Initializing ONNX Runtime...")
        # Load Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.onnx_path)

        # Load ONNX Model (Quantized)
        self.model = ORTModelForFeatureExtraction.from_pretrained(
            self.onnx_path, file_name="model_quantized.onnx"
        )
        # 🚀 LOAD THE TRAINED REASONER
        reasoner_path = self.kb_dir / "reasoner_model.pkl"
        if reasoner_path.exists():
            with open(reasoner_path, "rb") as f:
                data = pickle.load(f)
                self.reasoner_model = data["pipeline"]
                self.concept_map = data["concepts"]
            print("[RAG] Trained Query Reasoner Loaded.")
        else:
            print(
                "⚠️ Reasoner model not found. Run train_reasoner.py! Falling back to raw mode."
            )
            self.reasoner_model = None

        # Load FAISS
        self.index = faiss.read_index(str(self.kb_dir / "api.faiss"))

        # Load Mapping
        with open(self.kb_dir / "mapping.pkl", "rb") as f:
            self.chunks = pickle.load(f)

        self._warmup()
        print("[RAG] System Ready.")

    def _warmup(self):
        """
        Forces the ONNX Runtime to initialize all optimizations and
        memory allocations so the first real user request is fast.
        """
        print("[RAG] Warming up inference engine...")
        dummy_queries = [
            "login",  # Short query
            "admin access to database",  # Medium query
            "select * from users where id=1 OR 1=1 AND admin='true'",  # Long query
        ]

        # Run inference on dummy data.
        # We don't need the result, just the computation.
        for q in dummy_queries:
            self.get_context(q)

    def _encode_onnx(self, text):
        """
        Optimized ONNX encoding.
        """
        inputs = self.tokenizer(
            text, padding=True, truncation=True, return_tensors="pt"
        )
        outputs = self.model(**inputs)

        # Mean Pooling logic
        token_embeddings = outputs.last_hidden_state
        attention_mask = inputs["attention_mask"]
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        embeddings = sum_embeddings / sum_mask
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

        return embeddings.numpy()

    def reason_query(self, query: str) -> str:
        if not self.reasoner_model:
            return query

        try:
            # 1. Predict Intent (e.g., "SQLI")
            intent = self.reasoner_model.predict([query])[0]

            # 2. Retrieve Description (e.g., "SQL Injection...")
            enrichment = self.concept_map.get(intent, "")

            # 3. Combine
            if enrichment:
                return f"{query} {enrichment}"
            return query

        except Exception as e:
            # Fail safe to return original query if ML fails
            return query

    def get_context(self, query: str, top_k: int = 3, threshold: float = None):
        t0 = time.perf_counter()

        # 1. Reason (CPU: negligible)
        enriched_query = self.reason_query(query)

        # 2. Embed (ONNX Quantized: < 20ms)
        t_embed_start = time.perf_counter()
        vector = self._encode_onnx(enriched_query)
        t_embed_end = time.perf_counter()

        if top_k is None:
            top_k = 3

        # 3. Search (FAISS: < 1ms)
        t_search_start = time.perf_counter()
        scores, indices = self.index.search(vector, top_k)
        t_search_end = time.perf_counter()

        # 4. Context Assembly
        retrieved_chunks = []
        contexts = []

        if len(indices) > 0:
            for i, idx in enumerate(indices[0]):
                if idx == -1:
                    continue

                score = float(scores[0][i])

                # Check threshold if provided (Higher score = better for Inner Product)
                if threshold is not None and score < threshold:
                    continue

                chunk_text = self.chunks[idx]["text"]
                contexts.append(chunk_text)
                retrieved_chunks.append(
                    {"text": chunk_text, "score": score, "id": int(idx)}
                )

        final_context = "\n---\n".join(contexts)

        t_total = (time.perf_counter() - t0) * 1000

        print(
            f"[RAG LATENCY] "
            f"Embed: {(t_embed_end - t_embed_start) * 1000:.2f}ms | "
            f"Search: {(t_search_end - t_search_start) * 1000:.2f}ms | "
            f"TOTAL: {t_total:.2f}ms | "
            f"Chunks: {len(retrieved_chunks)}"
        )

        metadata = {
            "latency_ms": t_total,
            "retrieved_chunks": retrieved_chunks,
            "scores": [c["score"] for c in retrieved_chunks],
        }

        return final_context, metadata


# -------------------- TEST --------------------
if __name__ == "__main__":
    rag = RAGSystem()

    # Warmup
    rag.get_context("warmup")

    print("\n--- TEST RUN ---")
    rag.get_context("GET /api/users?id=1' OR 1=1")
