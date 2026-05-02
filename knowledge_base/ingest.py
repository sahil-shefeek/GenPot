from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import os
import yaml
import pickle

API_SPEC_PATH = 'data/github_api.yaml'
INDEX_PATH = 'knowledge_base/api_index.faiss'
MAPPING_PATH = 'knowledge_base/index_to_chunk.pkl'
META_PATH = 'knowledge_base/index_metadata.pkl'  # new

def parse_and_chunk(spec_path: str) -> list[dict]:
    """Return list of {text, path, method, tags, operation_id} chunks."""
    with open(spec_path, 'r') as f:
        spec = yaml.safe_load(f)

    chunks: list[dict] = []
    for path, methods in spec.get('paths', {}).items():
        for method, details in methods.items():
            endpoint_block = {path: {method: details}}
            text = yaml.dump(endpoint_block, default_flow_style=False)
            chunks.append({
                "text": text,
                "path": path,
                "method": method.upper(),
                "operation_id": details.get("operationId"),
                "tags": details.get("tags", []),
            })
    print(f"DEBUG: Found {len(chunks)} chunks in {spec_path}")
    return chunks

def create_knowledge_base():
    os.makedirs(os.path.dirname(INDEX_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(API_SPEC_PATH), exist_ok=True)

    print(f"Parsing and chunking API specification from {API_SPEC_PATH}...")
    chunks = parse_and_chunk(API_SPEC_PATH)
    print(f"Created {len(chunks)} chunks.")

    print("Generating embeddings...")
    model = SentenceTransformer("sentence-transformers/multi-qa-MiniLM-L6-cos-v1")
    texts = [c["text"] for c in chunks]
    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True  # important for cosine/IP
    ).astype("float32")

    print("Building FAISS index (cosine via inner product)...")
    d = embeddings.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(embeddings)

    print("Saving FAISS index and chunk mapping...")
    faiss.write_index(index, INDEX_PATH)

    index_to_chunk = {i: chunks[i]["text"] for i in range(len(chunks))}
    with open(MAPPING_PATH, "wb") as f:
        pickle.dump(index_to_chunk, f)

    # save metadata for better filtering / logging later
    meta = {
        i: {
            "path": chunks[i]["path"],
            "method": chunks[i]["method"],
            "operation_id": chunks[i]["operation_id"],
            "tags": chunks[i]["tags"],
        }
        for i in range(len(chunks))
    }
    with open(META_PATH, "wb") as f:
        pickle.dump(meta, f)

    print("✅ Knowledge base successfully created.")

if __name__ == '__main__':
    create_knowledge_base()
