from core.rag_tools import (
    retrieve_chunks,
    vector_search,
    sparse_search,
    generate_grounded,
    RetrievedChunk
)

from core.llm import invoke


# =========================
# Helpers
# =========================

def print_pass(name):
    print(f"\n✅ PASS: {name}")


def print_fail(name, err):
    print(f"\n❌ FAIL: {name}")
    print(f"   Reason: {err}")


# =========================
# 1. retrieve_chunks
# =========================

def test_retrieve_chunks():
    name = "retrieve_chunks"

    try:
        chunks = retrieve_chunks(
            "quantum kernel methods",
            k=4,
            expand=False
        )

        assert isinstance(chunks, list)
        assert len(chunks) == 4

        for i, c in enumerate(chunks):
            assert isinstance(c, RetrievedChunk)
            assert c.content and len(c.content.strip()) > 0
            assert isinstance(c.source, str)
            assert isinstance(c.score, float)

        print("\nHYBRID SOURCES:")
        for c in chunks:
            print(f"- {c.source} | score={c.score:.4f}")

        print_pass(name)
        return chunks

    except Exception as e:
        print_fail(name, e)
        raise


# =========================
# 2. vector_search (dense only)
# =========================

def test_vector_search():
    name = "vector_search"

    try:
        results = vector_search(
            "variational quantum eigensolver",
            k=6
        )

        assert isinstance(results, list)
        assert len(results) <= 6

        distances = []

        for r in results:
            assert isinstance(r, RetrievedChunk)
            assert 0.0 <= r.score <= 1.0
            distances.append(r.score)

        print("\nDENSE DISTANCES:")
        print([round(d, 3) for d in distances])

        assert any(d < 0.6 for d in distances), (
            "Embedding space may be misaligned (no close neighbors)"
        )

        print_pass(name)

    except Exception as e:
        print_fail(name, e)
        raise


# =========================
# 3. generate_grounded (NO MOCKING ANYMORE)
# =========================

def test_generate_grounded(chunks):
    name = "generate_grounded"

    try:
        answer = generate_grounded(
            "What is a quantum kernel?",
            chunks
        )

        assert isinstance(answer, str)
        assert len(answer.strip()) > 0

        print("\nGENERATED ANSWER (truncated):")
        print(answer[:400])

        print_pass(name)

    except Exception as e:
        print_fail(name, e)
        raise


# =========================
# 4. LLM sanity (Qwen via invoke)
# =========================

def test_invoke():
    name = "invoke"

    try:
        response = invoke([
            {
                "role": "user",
                "content": "Reply with the single word: READY"
            }
        ])

        assert isinstance(response, str)
        assert "READY" in response.upper()

        print("\nLLM OUTPUT:")
        print(response)

        print_pass(name)

    except Exception as e:
        print_fail(name, e)
        raise


# =========================
# 5. Hybrid vs Dense divergence
# =========================

def test_hybrid_vs_dense_difference():
    name = "hybrid_vs_dense"

    try:
        query = "quantum kernel classification"

        hybrid = retrieve_chunks(query, k=6, expand=False)
        dense = vector_search(query, k=6)

        assert len(hybrid) > 0
        assert len(dense) > 0

        hybrid_sources = [c.source for c in hybrid]
        dense_sources = [c.source for c in dense]

        print("\nHYBRID TOP:")
        print(hybrid_sources[:3])

        print("DENSE TOP:")
        print(dense_sources[:3])

        assert hybrid_sources != dense_sources, (
            "Hybrid retrieval not diverging from dense (BM25 not contributing)"
        )

        print_pass(name)

    except Exception as e:
        print_fail(name, e)
        raise


# =========================
# 6. Sparse lexical correctness
# =========================

def test_sparse_search():
    name = "sparse_search"

    try:
        results = sparse_search(
            "SWAP test fidelity estimation",
            k=4
        )

        assert len(results) > 0

        keywords = ["swap", "fidelity", "estimation"]

        for r in results:
            text = r.content.lower()
            assert any(k in text for k in keywords)

        print("\nSPARSE RESULTS:")
        for r in results:
            print("-", r.source)

        print_pass(name)

    except Exception as e:
        print_fail(name, e)
        raise


# =========================
# 7. M3 embedding sanity
# =========================

def test_m3_distances():
    name = "m3_distances"

    try:
        results = vector_search(
            "variational quantum eigensolver",
            k=6
        )

        distances = [r.score for r in results]

        print("\nDISTANCES:", [round(d, 3) for d in distances])

        for d in distances:
            assert 0.0 <= d <= 1.0

        assert any(d < 0.6 for d in distances), (
            "All distances too high — embedding space may be broken"
        )

        print_pass(name)

    except Exception as e:
        print_fail(name, e)
        raise


# =========================
# MAIN
# =========================

if __name__ == "__main__":

    print("\n==============================")
    print("RAG TOOL TEST SUITE (M3)")
    print("==============================")

    chunks = test_retrieve_chunks()
    test_vector_search()
    test_generate_grounded(chunks)
    test_invoke()
    test_hybrid_vs_dense_difference()
    test_sparse_search()
    test_m3_distances()

    print("\n==============================")
    print("✅ ALL TESTS PASSED")
    print("==============================")