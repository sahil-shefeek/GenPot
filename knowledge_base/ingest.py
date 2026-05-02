from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import os
import yaml
import pickle

API_SPEC_PATH = 'data/api.github.com.2022-11-28.deref.yaml'
INDEX_PATH = 'knowledge_base/api_index.faiss'
MAPPING_PATH = 'knowledge_base/index_to_chunk.pkl'
META_PATH = 'knowledge_base/index_metadata.pkl'

def parse_and_chunk(spec_path: str) -> tuple[list[dict], int]:
    with open(spec_path, 'r') as f:
        spec = yaml.safe_load(f)

    chunks: list[dict] = []
    allowed_methods = {"get", "post", "put", "patch", "delete", "options", "head"}
    
    paths_dict = spec.get('paths', {})
    unique_paths = len(paths_dict)
    
    for path, methods in paths_dict.items():
        if not isinstance(methods, dict):
            continue
        for method, details in methods.items():
            if method.lower() not in allowed_methods:
                continue
            
            try:
                # summary_or_description
                summary = details.get("summary")
                if summary and isinstance(summary, str) and summary.strip():
                    summary_or_description = summary.strip()
                else:
                    summary_or_description = details.get("description", "No description.")
                    if isinstance(summary_or_description, str):
                        summary_or_description = summary_or_description.strip()
                    else:
                        summary_or_description = str(summary_or_description).strip()
                
                # Strip newlines and extra whitespace
                summary_or_description = " ".join(summary_or_description.split())
                
                # Truncate to 300 characters
                summary_or_description = summary_or_description[:300]
                
                # param_names
                param_list = details.get("parameters", [])
                p_names = []
                if isinstance(param_list, list):
                    for p in param_list:
                        if isinstance(p, dict) and "name" in p:
                            p_names.append(str(p["name"]))
                
                if p_names:
                    param_names = ", ".join(p_names)
                else:
                    param_names = "none"
                    
                # operationId
                operation_id = details.get("operationId", "unknown")
                
                # format text
                text = f"{method.upper()} {path}: {summary_or_description}. Parameters: {param_names}. OperationId: {operation_id}."
                
                chunks.append({
                    "text": text,
                    "path": path,
                    "method": method.lower(),
                    "operation_id": operation_id,
                    "tags": details.get("tags", []),
                })
            except Exception as e:
                print(f"Warning: skipping operation {method.upper()} {path} due to error: {e}")
                continue
                
    return chunks, unique_paths

def create_knowledge_base():
    os.makedirs(os.path.dirname(INDEX_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(API_SPEC_PATH), exist_ok=True)

    print(f"Parsing and chunking API specification from {API_SPEC_PATH}...")
    chunks, unique_paths = parse_and_chunk(API_SPEC_PATH)
    print(f"Created {len(chunks)} chunks.")

    print("Generating embeddings...")
    model = SentenceTransformer("BAAI/bge-small-en-v1.5")
    texts = [c["text"] for c in chunks]
    embeddings = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=False
    )
    if not isinstance(embeddings, np.ndarray):
        embeddings = np.array(embeddings)
    embeddings = embeddings.astype("float32")
    faiss.normalize_L2(embeddings)

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

    print(f"Indexed {len(chunks)} operations from {unique_paths} paths → knowledge_base/")

if __name__ == '__main__':
    create_knowledge_base()
