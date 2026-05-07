# =========================
# ingest.py
# =========================

import os
import re
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

DATA_PATH = "ragdb_500"
INDEX_PATH = "faiss_ragdb_500"


# =========================
# 1. Math Cleaner
# =========================

def remove_math(text: str) -> str:
    """
    Replace LaTeX/math with placeholder to stabilize embeddings.
    """

    # Inline math $...$
    text = re.sub(r"\$.*?\$", " [MATH] ", text)

    # Display math $$...$$
    text = re.sub(r"\$\$.*?\$\$", " [MATH] ", text, flags=re.DOTALL)

    # Common LaTeX commands
    text = re.sub(r"\\[a-zA-Z]+", " ", text)

    return text


# =========================
# 2. Load PDFs
# =========================

def load_documents():
    docs = []

    for file in os.listdir(DATA_PATH):
        if file.endswith(".pdf"):
            path = os.path.join(DATA_PATH, file)
            loader = PyMuPDFLoader(path)
            pages = loader.load()

            # attach metadata
            for p in pages:
                p.metadata["source"] = file

            docs.extend(pages)

    print(f"Loaded {len(docs)} pages")
    return docs


# =========================
# 3. Clean Documents
# =========================

def clean_documents(docs):
    for doc in docs:
        doc.page_content = remove_math(doc.page_content)
    return docs


# =========================
# 4. Chunking
# =========================

def split_documents(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", " ", ""],
    )

    return splitter.split_documents(docs)


# =========================
# 5. Build Vector DB
# =========================

def build_vectorstore(doc_splits):
    embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-base-en",
    model_kwargs={"device": "cuda"},
    encode_kwargs={"batch_size": 64}
)

    vectorstore = FAISS.from_documents(doc_splits, embeddings)
    return vectorstore


# =========================
# 6. Main
# =========================

def main():
    docs = load_documents()
    docs = clean_documents(docs)

    doc_splits = split_documents(docs)
    print(f"Total chunks: {len(doc_splits)}")

    vectorstore = build_vectorstore(doc_splits)

    vectorstore.save_local(INDEX_PATH)
    print(f"Saved FAISS index to {INDEX_PATH}")


if __name__ == "__main__":
    main()