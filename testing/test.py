# =========================
# 0. Setup
# =========================

import os
from dotenv import load_dotenv
from typing import List, Dict, Any

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

# =========================
# 1. Embeddings + Vector DB
# =========================

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-base-en",
    model_kwargs={"device": "cpu"}
)

vectorstore = FAISS.load_local(
    "faiss_ragdb_500",
    embeddings,
    allow_dangerous_deserialization=True
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 8})

# =========================
# 2. Optional reranker (purely retrieval quality boost)
# =========================

try:
    from sentence_transformers import CrossEncoder
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
except Exception:
    reranker = None

# =========================
# 3. Core RAG function (DUMB LAYER)
# =========================

def retrieve(query: str, top_k: int = 4) -> Dict[str, Any]:
    """
    Dumb RAG core:
    - no grading
    - no rewriting
    - no LLM calls
    - only retrieval + optional rerank
    """

    docs = retriever.invoke(query)

    # -------------------------
    # Optional reranking
    # -------------------------
    if reranker:
        pairs = [(query, d.page_content) for d in docs]
        scores = reranker.predict(pairs)

        docs = [
            d for _, d in sorted(zip(scores, docs), reverse=True)
        ]

    # -------------------------
    # Format output
    # -------------------------
    context_blocks = []
    sources = []

    for d in docs[:top_k]:
        source = d.metadata.get("source", "unknown")

        context_blocks.append(
            f"[SOURCE: {source}]\n{d.page_content}"
        )

        sources.append(source)

    return {
        "query": query,
        "context": "\n\n".join(context_blocks),
        "sources": sources
    }