import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from data import smart_home_data

class SmartHomeVectorDB:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

        self.dim = 384
        self.index = faiss.IndexFlatL2(self.dim)

        self.texts = []
        self.metadata = []

        self._build_db()

    def _build_db(self):
        for item in smart_home_data:
            text = f"{item['name']} {item['category']} {item['description']} {item['features']} {item['url']}"
            embedding = self.model.encode(text)

            self.index.add(np.array([embedding], dtype=np.float32))
            self.texts.append(text)
            self.metadata.append(item)

    def search(self, query: str, k: int = 2):
        query_vec = self.model.encode(query)
        distances, indices = self.index.search(
            np.array([query_vec], dtype=np.float32),
            k
        )

        results = []
        for idx in indices[0]:
            if idx < len(self.metadata):
                results.append(self.metadata[idx])

        return results
