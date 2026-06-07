import os
from pathlib import Path
from typing import List, Dict, Any

import chromadb
import numpy as np

from sentence_transformers import SentenceTransformer
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from groq import Groq
from faster_whisper import WhisperModel
import tempfile
import os


# ==========================================================
# PDF PROCESSING
# ==========================================================

def process_all_pdfs(pdf_directory):
    all_documents = []

    pdf_dir = Path(pdf_directory)
    pdf_files = list(pdf_dir.glob("**/*.pdf"))

    print(f"Found {len(pdf_files)} PDF files")

    for pdf_file in pdf_files:
        try:
            loader = PyMuPDFLoader(str(pdf_file))
            docs = loader.load()

            for doc in docs:
                doc.metadata["source_file"] = pdf_file.name
                doc.metadata["file_type"] = "pdf"

            all_documents.extend(docs)

        except Exception as e:
            print(f"Error loading {pdf_file}: {e}")

    return all_documents


def split_documents(documents,
                    chunk_size=1000,
                    chunk_overlap=200):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )

    return splitter.split_documents(documents)

#===========================================================
# Whisper
#===========================================================
class WhisperManager:

    def __init__(self):

        self.model = WhisperModel(
            "base",
            device="cpu",
            compute_type="int8"
        )

    def transcribe(self, audio_bytes):

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".wav"
        ) as tmp:

            tmp.write(audio_bytes)

            temp_path = tmp.name

        segments, info = self.model.transcribe(
            temp_path
        )

        text = " ".join(
            segment.text
            for segment in segments
        )

        os.remove(temp_path)

        return text.strip()


# ==========================================================
# EMBEDDINGS
# ==========================================================


class EmbeddingManager:

    def __init__(self,
                 model_name="all-MiniLM-L6-v2"):

        self.model = SentenceTransformer(model_name)

    def generate_embeddings(self, texts):

        return self.model.encode(
            texts,
            show_progress_bar=False
        )


# ==========================================================
# VECTOR STORE
# ==========================================================

class VectorStore:

    def __init__(
        self,
        collection_name="pdf_documents",
        persist_directory="./data/vector_store"
    ):

        self.client = chromadb.PersistentClient(
            path=persist_directory
        )

        self.collection = (
            self.client.get_or_create_collection(
                name=collection_name,
                embedding_function=None
            )
        )

    def add_documents(
        self,
        documents,
        embeddings
    ):

        if self.collection.count() > 0:
            print("Vector DB already exists")
            return

        ids = []
        metadatas = []
        docs_text = []
        embeds = []

        for i, (doc, emb) in enumerate(
            zip(documents, embeddings)
        ):

            ids.append(str(i))

            metadatas.append(
                dict(doc.metadata)
            )

            docs_text.append(
                doc.page_content
            )

            embeds.append(
                emb.tolist()
            )

        self.collection.add(
            ids=ids,
            documents=docs_text,
            metadatas=metadatas,
            embeddings=embeds
        )


# ==========================================================
# RETRIEVER
# ==========================================================

class RAGRetriever:

    def __init__(
        self,
        vector_store,
        embedding_manager
    ):

        self.vector_store = vector_store
        self.embedding_manager = embedding_manager

    def retrieve(
        self,
        query,
        top_k=5
    ):

        query_embedding = (
            self.embedding_manager
            .generate_embeddings([query])[0]
        )

        results = (
            self.vector_store.collection.query(
                query_embeddings=[
                    query_embedding.tolist()
                ],
                n_results=min(
                    top_k,
                    self.vector_store.collection.count()
                )
            )
        )

        docs = []

        if results["documents"]:

            for doc,meta,distance in zip(
                    results["documents"][0],
                    results["metadatas"][0],
                    results["distances"][0]
                ):

                docs.append(
                    {
                        "content": doc,
                        "metadata": meta,
                        "similarity": 1 - distance
                    }
                )

        return docs


# ==========================================================
# RAG PIPELINE
# ==========================================================

class RAGPipeline:

    def __init__(
        self,
        retriever,
        groq_api_key,
        model="llama-3.3-70b-versatile"
    ):

        self.retriever = retriever
        self.whisper = WhisperManager()
        self.client = Groq(
            api_key=groq_api_key
        )

        self.model = model


    def ask(self, query):

        docs = self.retriever.retrieve(query)

        if not docs:
            return (
                "I couldn't find anything "
                "relevant in my agriculture "
                "knowledge base."
            )


        context = "\n\n".join(
            [d["content"] for d in docs]
        )

        response = (
            self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content":
                        """
                        You are AgriMind.

                        Answer only agriculture,
                        crop disease, fertilizer,
                        irrigation, pest management,
                        farming and plant related
                        questions.

                        Use the provided context.
                        """
                    },
                    {
                        "role": "user",
                        "content":
                        f"""
                        Context:
                        {context}

                        Question:
                        {query}
                        """
                    }
                ]
            )
        )

        return (
            response
            .choices[0]
            .message
            .content
        )
    def ask_voice(self, audio_bytes):

        query = self.whisper.transcribe(
            audio_bytes
        )
        answer  = self.ask(query)
        return query, answer


# ==========================================================
# BUILD PIPELINE
# ==========================================================

def build_rag_pipeline():

    embedding_manager = EmbeddingManager()

    vector_store = VectorStore()

    if vector_store.collection.count() == 0:

        print("Building vector database...")

        pdf_docs = process_all_pdfs("./pdfs")

        chunks = split_documents(pdf_docs)

        texts = [
            doc.page_content
            for doc in chunks
        ]

        embeddings = (
            embedding_manager
            .generate_embeddings(texts)
        )

        vector_store.add_documents(
            chunks,
            embeddings
        )

    retriever = RAGRetriever(
        vector_store,
        embedding_manager
    )

    pipeline = RAGPipeline(
        retriever=retriever,
        groq_api_key=os.getenv("GROQ_API_KEY")
    )

    return pipeline