import json
import pytest
import numpy as np
from server.rag_system import RAGSystem


@pytest.fixture
def mock_rag_dependencies(mocker, tmp_path):
    # Mock file paths
    kb_dir = tmp_path / "knowledge_base"
    kb_dir.mkdir()
    index_path = kb_dir / "api_index.faiss"
    mapping_path = kb_dir / "index_to_chunk.pkl"
    meta_path = kb_dir / "index_metadata.pkl"

    # Mock faiss.read_index
    mock_index = mocker.MagicMock()
    mocker.patch("faiss.read_index", return_value=mock_index)

    # Mock pickle.load for the mapping and metadata files
    mock_mapping = {0: "dummy documentation chunk 0", 1: "dummy documentation chunk 1"}
    mock_meta = {
        0: {"path": "/api/test", "method": "GET"},
        1: {"path": "/api/other", "method": "POST"},
    }

    # Create an iterator for the side_effect to return mapping then metadata
    mock_pickle_load = mocker.patch(
        "pickle.load", side_effect=[mock_mapping, mock_meta]
    )

    # Mock SentenceTransformer
    mock_encoder = mocker.MagicMock()
    mocker.patch("server.rag_system.SentenceTransformer", return_value=mock_encoder)

    # We also need to mock builtins.open so pickle.load works without real files
    mocker.patch("builtins.open", mocker.mock_open())

    return {
        "index_path": index_path,
        "mapping_path": mapping_path,
        "meta_path": meta_path,
        "mock_index": mock_index,
        "mock_encoder": mock_encoder,
        "mock_mapping": mock_mapping,
        "mock_meta": mock_meta,
    }


def test_initialization(mock_rag_dependencies):
    deps = mock_rag_dependencies
    rag = RAGSystem(
        index_path=deps["index_path"],
        mapping_path=deps["mapping_path"],
        meta_path=deps["meta_path"],
    )

    assert rag.index == deps["mock_index"]
    assert rag.encoder == deps["mock_encoder"]
    assert rag.index_to_chunk == deps["mock_mapping"]
    assert rag.index_metadata == deps["mock_meta"]


def test_get_context(mock_rag_dependencies):
    deps = mock_rag_dependencies

    # Setup mock returns
    mock_vec = np.array([[0.1, 0.2, 0.3]], dtype="float32")
    deps["mock_encoder"].encode.return_value = mock_vec

    # Return distances and indices (shape: 1 x top_k)
    # distance index 0 is high (above threshold), index 1 is low (below threshold)
    deps["mock_index"].search.return_value = (
        np.array([[0.8, 0.3]]),
        np.array([[0, 1]]),
    )

    rag = RAGSystem(
        index_path=deps["index_path"],
        mapping_path=deps["mapping_path"],
        meta_path=deps["meta_path"],
        top_k=2,
    )

    # Test with threshold 0.5
    # Should only return index 0
    context = rag.get_context("query returning match")
    print(context)  # Let's see it during dev
    # Due to a bug in rag_system this function isn't complete, it doesn't return anything

    # Setup mock returns again, we need to adapt the mock to the zip() bug
    # Wait, rag_system get_context logic is:
    #         for dist, idx in zip(distances[0], indices[0]):
    #             ...
    #             formatted_chunk = ...
    #             chunks.append(formatted_chunk)
    #         if not chunks: return "NO_RELEVANT_CONTEXT_FOUND"
    #         return "\n\n".join(chunks)

    # Let's test again properly
    context = rag.get_context("query returning match", similarity_threshold=0.5)

    assert "dummy documentation chunk 0" in context
    assert "dummy documentation chunk 1" not in context
    assert "/api/test" in context

    # Test no match
    context_no_match = rag.get_context(
        "query returning no match", similarity_threshold=0.9
    )
    assert context_no_match == "NO_RELEVANT_CONTEXT_FOUND"


def test_inspect_query(mock_rag_dependencies):
    deps = mock_rag_dependencies
    deps["mock_encoder"].encode.return_value = np.array([[0.1, 0.1]])
    deps["mock_index"].search.return_value = (np.array([[0.9]]), np.array([[0]]))

    rag = RAGSystem(
        index_path=deps["index_path"],
        mapping_path=deps["mapping_path"],
        meta_path=deps["meta_path"],
    )
    result = rag.inspect_query("test query", top_k=1)

    assert result["provided_query"] == "test query"
    assert "latency_ms" in result
    assert len(result["chunks"]) == 1
    assert result["chunks"][0]["chunk_index"] == 0
    assert result["chunks"][0]["faiss_distance"] == 0.9
    assert result["chunks"][0]["text"] == "dummy documentation chunk 0"


def test_extract_schema(mock_rag_dependencies):
    rag = RAGSystem(index_path="i", mapping_path="m", meta_path="meta")

    data = {"name": "Alice", "age": 30, "tags": ["a", "b"], "nested": {"b": True}}
    schema_str = rag._extract_schema(data)
    schema = json.loads(schema_str)

    assert schema["name"] == "str"
    assert schema["age"] == "int"
    assert schema["tags"] == ["str"]
    assert schema["nested"]["b"] == "bool"


def test_compute_similarity(mock_rag_dependencies):
    deps = mock_rag_dependencies
    rag = RAGSystem(index_path="i", mapping_path="m", meta_path="meta")

    # Test identical vectors (similarity 1.0)
    def mock_encode(texts, **kwargs):
        # We need to return normalized vectors
        vec = np.array([[1.0, 0.0]])
        return vec

    deps["mock_encoder"].encode.side_effect = mock_encode

    sim = rag.compute_similarity("doc text", {"response": "json"})
    assert sim == 1.0

    # Test empty doc text
    sim_empty = rag.compute_similarity("", {"response": "json"})
    assert sim_empty == 0.0

    sim_no_found = rag.compute_similarity(
        "NO_RELEVANT_CONTEXT_FOUND", {"response": "json"}
    )
    assert sim_no_found == 0.0
