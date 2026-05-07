from core.rag_sys import rag_core, retriever

from langchain_core.messages import HumanMessage


# =========================
# 1. Retrieval Tests
# =========================

def test_retrieval():
    q = "What is a quantum kernel method?"

    docs = retriever.invoke(q)

    assert len(docs) > 0, "No docs retrieved"

    text = " ".join(
        d.page_content.lower()
        for d in docs
    )

    assert "kernel" in text, "Retrieval not relevant"

    print("✅ test_retrieval passed")

#use semantic similarity scoring for better relevance testing (optional, requires sentence-transformers)


def test_topk_relevance():
    q = "quantum kernel method"

    docs = retriever.invoke(q)

    hits = sum(
        1 for d in docs
        if "kernel" in d.page_content.lower()
    )

    assert hits >= 2, "Weak retrieval quality"

    print("✅ test_topk_relevance passed")


# =========================
# 2. RAG Core Tests
# =========================

def test_context_formatting():
    q = "What is a quantum kernel method?"

    result = rag_core.invoke({
        "messages": [
            HumanMessage(content=q)
        ]
    })

    msg = result["messages"][-1]

    assert hasattr(msg, "content")
    assert isinstance(msg.content, str)
    assert len(msg.content) > 50

    print("✅ test_context_formatting passed")


def test_grounding():
    q = "What is a quantum kernel method?"

    result = rag_core.invoke({
        "messages": [
            HumanMessage(content=q)
        ]
    })

    answer = result["messages"][-1].content.lower()

    assert (
        "kernel" in answer or
        "quantum" in answer
    ), "Answer not grounded in retrieval"

    print("✅ test_grounding passed")


# =========================
# 3. Determinism Test
# =========================

def test_determinism():
    q = "What is a quantum kernel method?"

    r1 = rag_core.invoke({
        "messages": [
            HumanMessage(content=q)
        ]
    })

    r2 = rag_core.invoke({
        "messages": [
            HumanMessage(content=q)
        ]
    })

    a1 = r1["messages"][-1].content
    a2 = r2["messages"][-1].content

    assert a1 == a2, "Non-deterministic output"

    print("✅ test_determinism passed")

    # again use semantic similarity scoring for better testing (optional, requires sentence-transformers)


# =========================
# 4. Streaming / Graph Test
# =========================

def test_streaming():
    q = "What is a variational quantum circuit?"

    steps = 0

    for step in rag_core.stream({
        "messages": [
            HumanMessage(content=q)
        ]
    }):

        if step is None:
            continue

        steps += 1

    assert steps > 0, "Graph produced no steps"

    print("✅ test_streaming passed")


# =========================
# 5. Manual Trace Utility
# =========================

def trace_rag(question):
    print(f"\n=== TRACE: {question} ===\n")

    for step in rag_core.stream({
        "messages": [
            HumanMessage(content=question)
        ]
    }):

        if step is None:
            continue

        for node, update in step.items():

            print(f"\n--- NODE: {node} ---")

            if update is None:
                continue

            msgs = update.get("messages", [])

            if not msgs:
                continue

            msg = msgs[-1]

            if hasattr(msg, "content"):
                print(msg.content[:500])


# =========================
# MAIN
# =========================

if __name__ == "__main__":

    print("\n=== RUNNING RETRIEVAL TESTS ===")
    test_retrieval()
    test_topk_relevance()

    print("\n=== RUNNING RAG CORE TESTS ===")
    test_context_formatting()
    test_grounding()
    test_determinism()
    test_streaming()

    print("\n=== TRACE EXAMPLE ===")
    trace_rag("What is a variational quantum circuit?")

    print("\n✅ ALL TESTS PASSED")