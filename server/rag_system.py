import faiss
import pickle
from sentence_transformers import SentenceTransformer

class RAGSystem:
    def __init__(self,index_path="api_index.faiss",
        mapping_path="index_to_chunk.pkl"):
        print("Loading Rag System..")
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.model = faiss.read_index(index_path)
        with open(mapping_path,"rb") as f:
            self.index_to_chunk = pickle.load(f)
        print("RAG System ready")

    def get_context(self,query:str) -> str:
        query_embedding = self.model.encode([query])
        _distances,indices = self.index.search(query_embedding,1)
        best_match_index = indices[0][0]
        return self.index_to_chunk[best_match_index]