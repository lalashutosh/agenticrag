# =========================
# 0. Setup
# =========================

import os
from dotenv import load_dotenv
from typing import TypedDict, List

from langchain_core.messages import HumanMessage, AIMessage
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from langgraph.graph import StateGraph, START, END

from core.llm import invoke  # your local LLM wrapper

load_dotenv()

# =========================
# 1. Embeddings + Vector DB
# =========================

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-base-en",
    model_kwargs={"device": "cpu"}
)

vectorstore = FAISS.load_local(
    "data/faiss_ragdb_500",
    embeddings,
    allow_dangerous_deserialization=True
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 8})

# =========================
# 2. Optional: Cross-encoder reranker
# =========================

try:
    from sentence_transformers import CrossEncoder
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
except Exception:
    reranker = None

# =========================
# 3. State Schema (IMPORTANT FIX)
# =========================

class RAGState(TypedDict):
    messages: list
    context: str
    loops: int

# =========================
# 4. Helpers
# =========================

def get_text(msg):
    if isinstance(msg, dict):
        return msg.get("content", "")
    if hasattr(msg, "content"):
        return msg.content
    return str(msg)

# =========================
# 5. Query Expansion (cheap + effective)
# =========================

def expand_query(query: str) -> str:
    prompt = f"""
Rewrite this query for better scientific retrieval.
Keep it short.

Query:
{query}

Return only rewritten query.
"""
    return invoke([{"role": "user", "content": prompt}]).strip()

# =========================
# 6. Retrieval (Deterministic)
# =========================

def retrieve(state: RAGState):
    query = get_text(state["messages"][-1])

    expanded = expand_query(query)

    docs = retriever.invoke(expanded)

    # rerank if available
    if reranker:
        pairs = [(query, d.page_content) for d in docs]
        scores = reranker.predict(pairs)

        docs = [d for _, d in sorted(zip(scores, docs), reverse=True)]

    context = "\n\n".join(
        f"[Source: {d.metadata.get('source','unknown')}]\n{d.page_content}"
        for d in docs[:4]
    )

    return {
        "messages": state["messages"],
        "context": context,
        "loops": state.get("loops", 0)
    }

# =========================
# 7. Grade Step (deterministic routing)
# =========================

GRADE_PROMPT = """
You are grading retrieval quality.

Question:
{question}

Context:
{context}

Answer only:
yes or no
"""

def grade(state: RAGState):
    question = get_text(state["messages"][0])
    context = state["context"]

    response = invoke([{
        "role": "user",
        "content": GRADE_PROMPT.format(
            question=question,
            context=context
        )
    }])

    decision = "yes" if "yes" in response.lower() else "no"

    return {"decision": decision}

# =========================
# 8. Rewrite (bounded loops)
# =========================

REWRITE_PROMPT = """
Improve this quantum ML research query:

{question}

Return only improved query.
"""
def safe_return(state, extra=None):
    out = {
        "messages": state.get("messages", []),
        "context": state.get("context", ""),
        "loops": state.get("loops", 0)
    }
    if extra:
        out.update(extra)
    return out

def rewrite(state):
    loops = state.get("loops", 0)

    if loops >= 2:
        return {
            "messages": state["messages"],
            "context": state.get("context", ""),
            "loops": loops
        }

    question = get_text(state["messages"][0])

    new_q = invoke([{
        "role": "user",
        "content": REWRITE_PROMPT.format(question=question)
    }])

    return safe_return(state, {
        "messages": [HumanMessage(content=new_q)],
        "loops": loops + 1
    })
# =========================
# 9. Answer Generation
# =========================

ANSWER_PROMPT = """
You are a quantum machine learning expert.

Question:
{question}

Context:
{context}

Answer clearly and precisely.
If uncertain, say "I don't know".
"""

def generate(state: RAGState):
    question = get_text(state["messages"][0])

    response = invoke([{
        "role": "user",
        "content": ANSWER_PROMPT.format(
            question=question,
            context=state["context"]
        )
    }])

    return {
        "messages": state["messages"] + [
            AIMessage(content=response)
        ]
    }

# =========================
# 10. Routing
# =========================

def route(state: dict):
    return state.get("decision", "no")

# =========================
# 11. Build Graph (CLEAN + DETERMINISTIC)
# =========================

workflow = StateGraph(RAGState)

workflow.add_node("retrieve", retrieve)
workflow.add_node("grade", grade)
workflow.add_node("rewrite", rewrite)
workflow.add_node("generate", generate)

workflow.add_edge(START, "retrieve")
workflow.add_edge("retrieve", "grade")

workflow.add_conditional_edges(
    "grade",
    route,
    {
        "yes": "generate",
        "no": "rewrite"
    }
)

workflow.add_edge("rewrite", "retrieve")
workflow.add_edge("generate", END)

rag_core = workflow.compile()

# =========================
# 12. Run
# =========================

if __name__ == "__main__":

    query = "What is a quantum kernel method?"

    result = rag_core.invoke({
        "messages": [HumanMessage(content=query)],
        "context": "",
        "loops": 0
    })

    print("\nFINAL ANSWER:\n")
    print(get_text(result["messages"][-1]))

    rag_core.get_graph().draw_mermaid()