import chromadb
from sentence_transformers import SentenceTransformer


class MedicalQARetriever:
    def __init__(self, persist_dir="chroma_db", collection="taiwan_ehospital_qa", embedding_model="BAAI/bge-m3"):
        self.embedder = SentenceTransformer(embedding_model, device="cuda")
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_collection(collection)

    def search(self, query: str, top_k: int = 5):
        query_embedding = self.embedder.encode(
            [query],
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0].tolist()
        result = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        hits = []
        for doc, meta, dist in zip(result["documents"][0], result["metadatas"][0], result["distances"][0]):
            hits.append({
                "document": doc,
                "metadata": meta,
                "distance": dist,
                "score": 1 - dist,
            })
        return hits
