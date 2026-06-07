"""
rag_engine.py  —  Drop this in the same folder as app.py
Loads the persisted ChromaDB vector store and exposes a single function: ask(query) -> str
"""

import os
import numpy as np
import chromadb
import uuid
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
from groq import Groq
from pathlib import Path

# ── Resolve vector store path relative to this file ──────────────────────────
_HERE = Path(__file__).parent
VECTOR_STORE_PATH = str(_HERE / "data" / "vector_store")


# ─────────────────────────────────────────────────────────────────────────────
class EmbeddingManager:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        return self.model.encode(texts, show_progress_bar=False)


class VectorStore:
    def __init__(
        self,
        collection_name: str = "pdf_documents",
        persist_directory: str = VECTOR_STORE_PATH,
    ):
        self.collection_name = collection_name
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={
                "description": "PDF document embeddings for RAG",
                "hnsw:space": "cosine",
            },
            embedding_function=None,
        )

    def count(self) -> int:
        return self.collection.count()


class RAGRetriever:
    def __init__(self, vector_store: VectorStore, embedding_manager: EmbeddingManager):
        self.vector_store = vector_store
        self.embedding_manager = embedding_manager

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        query_embedding = self.embedding_manager.generate_embeddings([query])[0]
        n = min(top_k, self.vector_store.count())
        if n == 0:
            return []
        results = self.vector_store.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=n,
        )
        docs = []
        if results["documents"] and results["documents"][0]:
            for doc_id, document, metadata, distance in zip(
                results["ids"][0],
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            ):
                docs.append(
                    {
                        "id": doc_id,
                        "content": document,
                        "metadata": metadata,
                        "similarity_score": 1 - distance,
                        "distance": distance,
                    }
                )
        return docs


class RAGPipeline:
    def __init__(
        self,
        retriever: RAGRetriever,
        groq_api_key: str,
        model: str = "llama-3.1-8b-instant",
    ):
        self.retriever = retriever
        self.client = Groq(api_key=groq_api_key)
        self.model = model

    def ask(self, query: str, top_k: int = 5) -> str:
        docs = self.retriever.retrieve(query, top_k=top_k)
        if not docs:
            return "I can only answer questions related to crop diseases."
        if docs[0]["similarity_score"] < 0.3:
            return (
                "I can only answer questions related to crop diseases and agriculture. "
                "Please ask something relevant."
            )
        context = "\n\n".join([d["content"] for d in docs])
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an expert agricultural assistant specializing in crop diseases. "
                        "Answer the question directly and concisely based on the provided context. "
                        "If the context doesn't contain enough information, say so clearly. "
                        "Always mention the specific crop and disease name when available."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Context:\n{context}\n\nQuestion: {query}",
                },
            ],
        )
        return response.choices[0].message.content


# ── Singleton loader (cached by Streamlit via st.cache_resource) ─────────────
def load_rag_pipeline(groq_api_key: str) -> RAGPipeline:
    em = EmbeddingManager()
    vs = VectorStore()
    retriever = RAGRetriever(vs, em)
    return RAGPipeline(retriever, groq_api_key)
