import faiss
import sys
from pathlib import Path

try:
    # Try multiple locations
    possible_paths = [
        Path("knowledge_base/api.faiss"),
        Path("server/knowledge_base/api.faiss"),
    ]

    index_path = None
    for p in possible_paths:
        if p.exists():
            index_path = p
            break

    if not index_path:
        print(f"Index not found in: {[str(p) for p in possible_paths]}")
        sys.exit(1)

    print(f"Found index at: {index_path}")
    index = faiss.read_index(str(index_path))
    print(f"Metric Type: {index.metric_type}")

    if index.metric_type == faiss.METRIC_L2:
        print("Metric: L2 (Euclidean) -> Lower is better")
    elif index.metric_type == faiss.METRIC_INNER_PRODUCT:
        print("Metric: Inner Product (Cosine-like) -> Higher is better")
    else:
        print(f"Metric Code: {index.metric_type}")

except Exception as e:
    print(f"Error: {e}")
