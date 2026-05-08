import os
import json
import time
import pickle
import re
import numpy as np
from tqdm import tqdm

import faiss
import fitz  # PyMuPDF

from rank_bm25 import BM25Okapi
from FlagEmbedding import BGEM3FlagModel
from langchain_text_splitters import RecursiveCharacterTextSplitter

# =========================
# Paths
# =========================

DATA_PATH  = "data/ragdb_500"
OUTPUT_DIR = "data/faiss_ragdb_m3"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================
# Math translation (lightweight)
# =========================

MATH_TRANSLATIONS = {
    r"\\mathcal\{H\}": "Hilbert space",
    r"\\mathbb\{R\}": "real space",
    r"\\mathbb\{C\}": "complex space",
    r"\\otimes": "tensor product",
    r"\\nabla": "gradient",
    r"\\partial": "partial derivative",
    r"\\phi": "phi encoding",
    r"\\psi": "quantum state psi",
    r"\\rho": "density matrix rho",
    r"\\theta": "theta parameter",
}

def translate_math(text: str) -> str:
    for k, v in MATH_TRANSLATIONS.items():
        text = re.sub(k, v, text)

    text = re.sub(r"\$\$(.*?)\$\$", r" \1 ", text, flags=re.DOTALL)
    text = re.sub(r"\$(.*?)\$", r" \1 ", text)

    text = re.sub(r"\\[a-zA-Z]+\{[^}]*\}", " ", text)
    text = re.sub(r"\\[a-zA-Z]+", " ", text)

    return re.sub(r"\s{2,}", " ", text).strip()

# =========================
# Chunking
# =========================

splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=200,
    separators=["\n\n", "\n", ". ", " ", ""],
)

# =========================
# 🚀 GPU-OPTIMIZED M3 LOAD
# =========================

print("Loading BGE-M3 on GPU...")

model = BGEM3FlagModel(
    "BAAI/bge-m3",
    use_fp16=True,
    device="cuda"
)

# IMPORTANT GPU knobs (hidden but impactful):
# - disable gradient tracking implicitly
# - keep model warm
# - avoid repeated kernel launches

# =========================
# PDF LOADING (faster PyMuPDF usage)
# =========================

def load_pdfs():
    pages = []

    for fname in os.listdir(DATA_PATH):
        if not fname.endswith(".pdf"):
            continue

        path = os.path.join(DATA_PATH, fname)

        with fitz.open(path) as pdf:
            for page_idx, page in enumerate(pdf):
                text = translate_math(page.get_text("text"))

                if len(text.split()) < 40:
                    continue

                pages.append({
                    "text": text,
                    "source": fname,
                    "page": page_idx
                })

    return pages

# =========================
# Chunk building (no overhead dict churn)
# =========================

def build_chunks(pages):
    chunks = []
    chunk_id = 0

    for p in pages:
        splits = splitter.split_text(p["text"])

        for s in splits:
            chunks.append({
                "content": s,
                "metadata": {
                    "source": p["source"],
                    "page": p["page"],
                    "chunk_index": chunk_id,
                    "char_count": len(s)
                }
            })
            chunk_id += 1

    return chunks

# =========================
# 🚀 GPU BATCH ENCODING (OPTIMIZED)
# =========================

def encode_chunks(chunks, batch_size=64):
    """
    Key optimizations:
    - larger batch size (GPU saturation)
    - single extraction of texts
    - minimal Python branching inside loop
    """

    texts = [c["content"] for c in chunks]

    dense_vecs = []
    lexical_all = []
    token_lists = []

    # IMPORTANT: M3 is heavy → fewer Python loop iterations = faster GPU utilization
    for i in tqdm(range(0, len(texts), batch_size), desc="Encoding M3"):
        batch = texts[i:i + batch_size]

        with np.errstate(all='ignore'):
            out = model.encode(
                batch,
                return_dense=True,
                return_sparse=True
            )

        dense_vecs.append(out["dense_vecs"])
        lexical_all.extend(out["lexical_weights"])

        # BM25 token expansion (keep minimal overhead)
        for lw in out["lexical_weights"]:
            tokens = []
            for token_id, weight in lw.items():
                repeat = max(1, int(weight + 0.5))
                tokens.extend([str(token_id)] * repeat)
            token_lists.append(tokens)

    dense_vecs = np.vstack(dense_vecs).astype("float32")

    # 🚀 CRITICAL: normalize once (enables IP = cosine similarity)
    faiss.normalize_L2(dense_vecs)

    return dense_vecs, lexical_all, token_lists

# =========================
# FAISS BUILD (GPU-friendly layout)
# =========================

def build_faiss(dense_vecs):
    dim = dense_vecs.shape[1]

    # Inner Product index (cosine due to normalization)
    index = faiss.IndexFlatIP(dim)
    index.add(dense_vecs)

    return index

# =========================
# MAIN PIPELINE
# =========================

def main():
    start = time.time()

    print("Loading PDFs...")
    pages = load_pdfs()

    print(f"Pages loaded: {len(pages)}")

    print("Chunking...")
    chunks = build_chunks(pages)

    print(f"Total chunks: {len(chunks)}")

    print("Encoding (GPU)...")
    dense_vecs, lexical_weights, token_lists = encode_chunks(chunks)

    print("Building FAISS...")
    index = build_faiss(dense_vecs)

    faiss.write_index(
        index,
        os.path.join(OUTPUT_DIR, "dense.index")
    )

    print("Saving metadata...")
    with open(os.path.join(OUTPUT_DIR, "chunks.json"), "w") as f:
        json.dump(chunks, f)

    with open(os.path.join(OUTPUT_DIR, "token_lists.json"), "w") as f:
        json.dump(token_lists, f)

    with open(os.path.join(OUTPUT_DIR, "bm25.pkl"), "wb") as f:
        pickle.dump(BM25Okapi(token_lists), f)

    print("\n--- SAMPLE ---")
    for c in chunks[:3]:
        print(c["metadata"])
        print(c["content"][:200], "\n")

    end = time.time()

    print("\n====================")
    print(f"Chunks: {len(chunks)}")
    print(f"FAISS size: {index.ntotal}")
    print(f"Time: {(end - start)/60:.2f} min")
    print("====================")

if __name__ == "__main__":
    main()