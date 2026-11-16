from __future__ import annotations

import hashlib
import re
from typing import Any, Protocol, Sequence

import chromadb


class Embedder(Protocol):
    def encode(self, texts: Sequence[str]) -> Any: ...


class LocalSentenceEmbedder:
    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)

    def encode(self, texts: Sequence[str]) -> Any:
        return self.model.encode(list(texts), normalize_embeddings=True).tolist()


class DeterministicTestEmbedder:
    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        vectors = []
        for text in texts:
            vector = [0.0] * 64
            for token in re.findall(r"[a-z0-9]+", text.lower()):
                index = int(hashlib.sha256(token.encode()).hexdigest()[:8], 16) % len(vector)
                vector[index] += 1.0
            norm = sum(value * value for value in vector) ** 0.5 or 1.0
            vectors.append([value / norm for value in vector])
        return vectors


class ChromaVectorStore:
    def __init__(self, path: str, collection_name: str, embedder: Embedder):
        self.client = chromadb.PersistentClient(path=path) if path != ":memory:" else chromadb.Client()
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self.embedder = embedder

    def upsert(self, chunks: list[dict[str, Any]], batch_size: int = 64) -> None:
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            self.collection.upsert(
                ids=[item["id"] for item in batch],
                documents=[item["text"] for item in batch],
                metadatas=[item["metadata"] for item in batch],
                embeddings=self.embedder.encode([item["text"] for item in batch]),
            )

    def query(self, text: str, where: dict[str, Any], limit: int = 5) -> list[dict[str, Any]]:
        result = self.collection.query(
            query_embeddings=self.embedder.encode([text]),
            n_results=limit,
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        ids = result.get("ids", [[]])[0]
        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]
        return [
            {"id": chunk_id, "text": document, "metadata": metadata, "distance": distance}
            for chunk_id, document, metadata, distance in zip(ids, documents, metadatas, distances)
        ]