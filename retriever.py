import numpy as np
from sentence_transformers import SentenceTransformer

class Retriever:
    def __init__(self, chunks, model_name="all-MiniLM-L6-v2"):
        self.chunks = chunks
        self.model = SentenceTransformer(model_name)
        self.embeddings = self.model.encode(
            chunks, normalize_embeddings=True, show_progress_bar=True
        )

    def search(self, query: str, k: int = 3):
        q = self.model.encode([query], normalize_embeddings=True)[0]
        scores = self.embeddings @ q          # cosine similarity
        idx = np.argsort(scores)[::-1][:k]
        return [self.chunks[i] for i in idx]