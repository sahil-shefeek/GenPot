# scripts/build_smtp_index.py
import yaml
import pickle
import faiss
from pathlib import Path
from sentence_transformers import SentenceTransformer

def build_smtp_index():
    base_dir = Path(__file__).resolve().parents[1]
    data_path = base_dir / "data" / "smtp_rfc5321.yaml"
    kb_dir = base_dir / "knowledge_base"
    kb_dir.mkdir(exist_ok=True)

    with open(data_path, "r") as f:
        data = yaml.safe_load(f)

    encoder = SentenceTransformer("BAAI/bge-small-en-v1.5")
    
    index_to_chunk = {}
    index_metadata = {}
    embeddings = []

    for idx, item in enumerate(data.get("commands", [])):
        command = item["command"]
        # Format the text chunk for the LLM
        chunk_text = (
            f"Command: {command}\n"
            f"Summary: {item['summary']}\n"
            f"Description: {item['description']}\n"
            f"Expected Response: {item['expected_response']}"
        )
        
        # Create embedding based on the command and summary
        vec = encoder.encode(f"{command} {item['summary']}", normalize_embeddings=True)
        embeddings.append(vec)
        
        index_to_chunk[idx] = chunk_text
        index_metadata[idx] = {"protocol": "smtp", "command": command}

    # Build FAISS index
    dimension = len(embeddings[0])
    index = faiss.IndexFlatIP(dimension) # Inner product = cosine sim for normalized vectors
    index.add(faiss.np.array(embeddings).astype("float32"))

    # Save files
    faiss.write_index(index, str(kb_dir / "smtp_index.faiss"))
    with open(kb_dir / "smtp_index_to_chunk.pkl", "wb") as f:
        pickle.dump(index_to_chunk, f)
    with open(kb_dir / "smtp_index_metadata.pkl", "wb") as f:
        pickle.dump(index_metadata, f)

    print("✅ SMTP Knowledge Base built successfully!")

if __name__ == "__main__":
    build_smtp_index()