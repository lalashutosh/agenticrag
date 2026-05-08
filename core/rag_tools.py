# core/rag_tools.py

import json
import pickle
import numpy as np
import faiss
from dataclasses import dataclass
from typing import List, Dict, Any

from FlagEmbedding import BGEM3FlagModel
from rank_bm25 import BM25Okapi

from core.llm import invoke

# =========================
# 1. Load index assets
# =========================

M3_INDEX_PATH = "data/faiss_ragdb_m3"

dense_index = faiss.read_index(f"{M3_INDEX_PATH}/dense.index")

with open(f"{M3_INDEX_PATH}/chunks.json") as f:
    chunks_meta = json.load(f)

with open(f"{M3_INDEX_PATH}/token_lists.json") as f:
    token_lists = json.load(f)

bm25_index = BM25Okapi(token_lists)

# =========================
# 2. Load M3 encoder
# =========================

m3_model = BGEM3FlagModel(
    "BAAI/bge-m3",
    use_fp16=True,
    device="cpu"
)

# =========================
# 3. RetrievedChunk contract
# =========================

@dataclass
class RetrievedChunk:
    content: str
    source: str
    score: float
    chunk_type: str = "unknown"

# =========================
# 4. Prompts
# =========================

EXPAND_PROMPT = """
Rewrite this query for scientific retrieval.
Keep it short and precise.

Query: {query}
"""

ANSWER_PROMPT = """
You are a quantum machine learning expert.

Question:
{question}

Context:
{context}

Answer clearly and precisely. If unsure, say so explicitly.
"""

# =========================
# 5. Helpers
# =========================

NEIGHBOR_DISTANCE_THRESHOLD = 0.7


def _rrf(rankings: List[List[int]], k: int = 60) -> Dict[int, float]:
    scores = {}
    for ranking in rankings:
        for rank, idx in enumerate(ranking):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (rank + k)
    return scores


# -------------------------
# SAFE dense encoding FIX
# -------------------------
def _encode_dense(query: str):
    vec = m3_model.encode([query], return_dense=True)["dense_vecs"]

    vec = np.asarray(vec, dtype=np.float32, order="C")
    faiss.normalize_L2(vec)

    return vec


# -------------------------
# SAFE sparse encoding FIX
# -------------------------
def _encode_sparse(query: str):
    out = m3_model.encode([query], return_sparse=True)
    return out["lexical_weights"][0]


# IMPORTANT FIX:
# we DO NOT use token_id strings blindly anymore
# we preserve token identity but treat as lexical surrogate
def _sparse_to_token_list(lexical_weights: Dict[str, float]) -> List[str]:
    tokens = []

    for token_id, weight in lexical_weights.items():
        # FIX: clamp + stabilize BM25 input
        w = max(1, int(round(weight)))
        tokens.extend([str(token_id)] * w)

    return tokens


# =========================
# 6. retrieve_chunks (HYBRID RRF)
# =========================

def retrieve_chunks(query: str, k: int = 6, expand: bool = True) -> List[RetrievedChunk]:

    if expand:
        query = invoke(
            [{"role": "user", "content": EXPAND_PROMPT.format(query=query)}],
            max_new_tokens=64
        ).strip()

    # -------------------------
    # Dense retrieval
    # -------------------------
    q_dense = _encode_dense(query)

    dist, idx = dense_index.search(q_dense, k * 2)

    # FIX: ensure ranking correctness
    dense_ranking = list(np.argsort(dist[0])[::-1])

    # -------------------------
    # Sparse retrieval (BM25)
    # -------------------------
    sparse_weights = _encode_sparse(query)
    sparse_tokens = _sparse_to_token_list(sparse_weights)

    bm25_scores = bm25_index.get_scores(sparse_tokens)

    sparse_ranking = list(np.argsort(bm25_scores)[::-1][:k * 2])

    # -------------------------
    # RRF fusion
    # -------------------------
    fused = _rrf([dense_ranking, sparse_ranking])

    top_indices = sorted(fused.items(), key=lambda x: x[1], reverse=True)[:k]
    top_indices = [i for i, _ in top_indices]

    results = []
    for i in top_indices:
        meta = chunks_meta[i]

        results.append(RetrievedChunk(
            content=meta["content"],
            source=meta["metadata"]["source"],
            score=float(fused[i]),
            chunk_type=meta["metadata"].get("chunk_type", "unknown")
        ))

    return results


# =========================
# 7. vector_search (DENSE ONLY)
# =========================

def vector_search(query_text: str, k: int = 6) -> List[RetrievedChunk]:

    q_dense = _encode_dense(query_text)
    dist, idx = dense_index.search(q_dense, k)

    results = []

    for d, i in zip(dist[0], idx[0]):
        if i == -1:
            continue

        meta = chunks_meta[i]

        # FIX: cosine distance stability
        cosine_distance = float(1.0 - d)
        cosine_distance = max(0.0, min(1.0, cosine_distance))

        if cosine_distance > NEIGHBOR_DISTANCE_THRESHOLD:
            continue

        results.append(RetrievedChunk(
            content=meta["content"],
            source=meta["metadata"]["source"],
            score=cosine_distance,
            chunk_type=meta["metadata"].get("chunk_type", "unknown")
        ))

    return results


# =========================
# 8. sparse_search (BM25 ONLY)
# =========================

def sparse_search(query_text: str, k: int = 6) -> List[RetrievedChunk]:

    sparse_weights = _encode_sparse(query_text)
    tokens = _sparse_to_token_list(sparse_weights)

    scores = bm25_index.get_scores(tokens)

    top_idx = np.argsort(scores)[::-1][:k]

    max_score = max(scores) + 1e-8

    results = []

    for i in top_idx:
        meta = chunks_meta[i]

        results.append(RetrievedChunk(
            content=meta["content"],
            source=meta["metadata"]["source"],
            score=float(scores[i] / max_score),  # FIX: normalize BM25
            chunk_type=meta["metadata"].get("chunk_type", "unknown")
        ))

    return results


# =========================
# 9. generate_grounded
# =========================

def generate_grounded(question: str, chunks: List[RetrievedChunk]) -> str:

    context = "\n\n".join(
        f"[Source: {c.source}]\n{c.content}"
        for c in chunks
    )

    return invoke(
        [{
            "role": "user",
            "content": ANSWER_PROMPT.format(
                question=question,
                context=context
            )
        }],
        max_new_tokens=1024
    )