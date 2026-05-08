# testing/test_agents_unit.py

import sys
sys.path.insert(0, ".")

from agents.state import AgentState, Triple, NeighborResult, BridgeCandidate, EvidenceRecord
from core.rag_tools import RetrievedChunk

# ── helpers ──────────────────────────────────────────────────────────────────

def make_base_state(**overrides) -> AgentState:
    base = AgentState(
        original_query="How do quantum kernels relate to barren plateaus?",
        triples=[],
        neighbor_map={},
        bridge_candidates=[],
        evidence=[],
        final_answer="",
        phase="init",
        degraded_mode=False,
        errors=[],
        llm_calls_used=0,
        retrieved_chunks_log=[]
    )
    base.update(overrides)
    return base

def make_triple(subj, rel, obj, src="retrieved") -> Triple:
    return Triple(subject=subj, relation=rel, object=obj, source=src)

def make_chunk(content, source="paper_a.pdf", score=0.2) -> RetrievedChunk:
    return RetrievedChunk(content=content, source=source, score=score)

def passed(name): print(f"  PASS  {name}")
def failed(name, reason): print(f"  FAIL  {name}: {reason}")

# ── Decomposition ─────────────────────────────────────────────────────────────

def test_decomposition_happy_path():
    from agents.decomposition import decompose
    state = make_base_state()
    result = decompose(state)

    assert isinstance(result["triples"], list),         "triples must be a list"
    assert result["phase"] == "neighbor",               "phase must advance to neighbor"
    assert result["llm_calls_used"] == 1,               "must use exactly 1 LLM call"
    assert len(result["retrieved_chunks_log"]) > 0,     "must log chunk sources"
    passed("decomposition_happy_path")

def test_decomposition_empty_query():
    # If the query produces no usable triples, errors should be logged
    # but phase must still advance — no crash
    from agents.decomposition import decompose
    state = make_base_state(original_query="asdkjhasd nonsense xyz")
    result = decompose(state)

    assert result["phase"] == "neighbor",   "phase must advance even on empty triples"
    # triples may be [] — that's fine, just must not crash
    passed("decomposition_empty_query")

# ── Neighbor ──────────────────────────────────────────────────────────────────

def test_neighbor_happy_path():
    from agents.neighbor import find_neighbors
    state = make_base_state(
        phase="neighbor",
        triples=[
            make_triple("quantum kernel", "maps to", "Hilbert space"),
            make_triple("VQE", "minimizes", "expectation value")
        ]
    )
    result = find_neighbors(state)

    assert isinstance(result["neighbor_map"], dict),    "neighbor_map must be a dict"
    assert result["phase"] == "bridge",                 "phase must advance to bridge"
    assert result["llm_calls_used"] == 0,               "neighbor agent must make 0 LLM calls"

    for key, neighbors in result["neighbor_map"].items():
        for n in neighbors:
            assert n.cosine_distance >= 0.0,            "distances must be non-negative"
            assert n.cosine_distance <= 0.4,            "distances must pass threshold filter"
    passed("neighbor_happy_path")

def test_neighbor_empty_triples():
    from agents.neighbor import find_neighbors
    state = make_base_state(phase="neighbor", triples=[])
    result = find_neighbors(state)

    assert result["neighbor_map"] == {},    "empty triples → empty neighbor_map"
    assert result["phase"] == "bridge",     "must still advance"
    assert any("skipped" in e for e in result["errors"]), "must log skip error"
    passed("neighbor_empty_triples")

# ── Bridge ────────────────────────────────────────────────────────────────────

def test_bridge_happy_path():
    """Inject a neighbor_map where we know a bridge exists."""
    from agents.bridge import find_bridges

    shared_chunk = make_chunk(
        "Parameterized quantum circuits form the basis of both kernel methods and variational algorithms",
        source="shared_paper.pdf"
    )

    t1 = make_triple("quantum kernel", "uses", "feature map")
    t2 = make_triple("VQE", "uses", "ansatz")

    neighbor_map = {
        "quantum kernel_feature map": [
            NeighborResult(triple_key="quantum kernel_feature map", chunk=shared_chunk, cosine_distance=0.15),
            NeighborResult(triple_key="quantum kernel_feature map", chunk=make_chunk("kernel alignment methods", source="paper_b.pdf"), cosine_distance=0.25),
        ],
        "VQE_ansatz": [
            NeighborResult(triple_key="VQE_ansatz", chunk=shared_chunk, cosine_distance=0.18),
            NeighborResult(triple_key="VQE_ansatz", chunk=make_chunk("barren plateau landscape", source="paper_c.pdf"), cosine_distance=0.3),
        ]
    }

    state = make_base_state(
        phase="bridge",
        triples=[t1, t2],
        neighbor_map=neighbor_map
    )
    result = find_bridges(state)

    assert isinstance(result["bridge_candidates"], list),   "must return a list"
    assert result["phase"] == "validation",                 "must advance to validation"
    assert result["llm_calls_used"] == 0,                   "bridge agent must make 0 LLM calls"

    if result["bridge_candidates"]:
        c = result["bridge_candidates"][0]
        assert 0.0 <= c.novelty_score <= 1.0,   "novelty score must be in [0,1]"
        # Verify sorted descending
        scores = [x.novelty_score for x in result["bridge_candidates"]]
        assert scores == sorted(scores, reverse=True), "must be sorted by novelty descending"

    passed("bridge_happy_path")

def test_bridge_no_shared_neighbors():
    from agents.bridge import find_bridges

    t1 = make_triple("quantum kernel", "uses", "feature map")
    t2 = make_triple("VQE", "uses", "ansatz")

    neighbor_map = {
        "quantum kernel_feature map": [
            NeighborResult(triple_key="quantum kernel_feature map", chunk=make_chunk("kernel text", source="paper_a.pdf"), cosine_distance=0.2)
        ],
        "VQE_ansatz": [
            NeighborResult(triple_key="VQE_ansatz", chunk=make_chunk("vqe text", source="paper_b.pdf"), cosine_distance=0.2)
        ]
    }

    state = make_base_state(phase="bridge", triples=[t1, t2], neighbor_map=neighbor_map)
    result = find_bridges(state)

    assert result["bridge_candidates"] == [],   "no shared neighbors → no candidates"
    assert result["phase"] == "validation",     "must still advance"
    assert any("no_candidates" in e for e in result["errors"]), "must log no_candidates"
    passed("bridge_no_shared_neighbors")

# ── Validation ────────────────────────────────────────────────────────────────

def test_validation_happy_path():
    from agents.validation import validate_bridges

    candidate = BridgeCandidate(
        triple_a=make_triple("quantum kernel", "uses", "feature map"),
        triple_b=make_triple("barren plateau", "affects", "gradient"),
        bridge_concept="parameterized quantum circuit expressibility",
        co_occurrence_count=0,
        novelty_score=0.85
    )

    state = make_base_state(phase="validation", bridge_candidates=[candidate])
    result = validate_bridges(state)

    assert isinstance(result["evidence"], list),    "must return evidence list"
    assert len(result["evidence"]) == 1,            "one candidate → one evidence record"
    assert result["phase"] == "synthesis",          "must advance to synthesis"

    rec = result["evidence"][0]
    assert 0.0 <= rec.confidence <= 1.0,            "confidence must be in [0,1]"
    assert rec.verdict in {"supported", "speculative", "contradicted"}, "verdict must be valid"
    assert isinstance(rec.supporting_chunks, list), "supporting_chunks must be a list"
    assert isinstance(rec.opposing_chunks, list),   "opposing_chunks must be a list"
    passed("validation_happy_path")

def test_validation_empty_candidates():
    from agents.validation import validate_bridges
    state = make_base_state(phase="validation", bridge_candidates=[])
    result = validate_bridges(state)

    assert result["evidence"] == [],    "no candidates → no evidence"
    assert result["phase"] == "synthesis"
    passed("validation_empty_candidates")

# ── Synthesis ─────────────────────────────────────────────────────────────────

def test_synthesis_happy_path():
    from agents.synthesis import synthesize

    candidate = BridgeCandidate(
        triple_a=make_triple("quantum kernel", "uses", "feature map"),
        triple_b=make_triple("barren plateau", "affects", "gradient"),
        bridge_concept="parameterized circuit expressibility",
        co_occurrence_count=0,
        novelty_score=0.85
    )

    evidence = [EvidenceRecord(
        candidate=candidate,
        supporting_chunks=[make_chunk("Quantum kernels use feature maps derived from parameterized circuits")],
        opposing_chunks=[make_chunk("Barren plateaus make gradient-based training of deep circuits intractable")],
        confidence=0.72,
        verdict="speculative"
    )]

    state = make_base_state(
        phase="synthesis",
        bridge_candidates=[candidate],
        evidence=evidence
    )
    result = synthesize(state)

    assert isinstance(result["final_answer"], str),     "final_answer must be a string"
    assert len(result["final_answer"]) > 100,           "answer must have substance"
    assert result["phase"] == "done",                   "must set phase to done"

    answer = result["final_answer"]
    for section in ["ESTABLISHED", "PROBABLE BRIDGES", "OPEN QUESTION", "LIMITATIONS"]:
        assert section in answer, f"missing section: {section}"

    passed("synthesis_happy_path")

def test_synthesis_degraded_mode():
    from agents.synthesis import synthesize
    state = make_base_state(
        phase="synthesis",
        degraded_mode=True,
        errors=["decomposition: json_parse_failed", "bridge: no_candidates"]
    )
    result = synthesize(state)

    assert "Note:" in result["final_answer"] or "degraded" in result["final_answer"].lower(), \
        "degraded mode must surface in the answer"
    passed("synthesis_degraded_mode")

def test_synthesis_fallback_no_evidence():
    # When evidence=[] and bridge_candidates=[], synthesis should fall back to pure RAG
    from agents.synthesis import synthesize
    state = make_base_state(phase="synthesis")
    result = synthesize(state)

    assert isinstance(result["final_answer"], str)
    assert len(result["final_answer"]) > 50
    passed("synthesis_fallback_no_evidence")

# ── Run all ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        test_decomposition_happy_path,
        test_decomposition_empty_query,
        test_neighbor_happy_path,
        test_neighbor_empty_triples,
        test_bridge_happy_path,
        test_bridge_no_shared_neighbors,
        test_validation_happy_path,
        test_validation_empty_candidates,
        test_synthesis_happy_path,
        test_synthesis_degraded_mode,
        test_synthesis_fallback_no_evidence,
    ]

    print(f"\nRunning {len(tests)} unit tests...\n")
    passed_count = 0
    for t in tests:
        try:
            t()
            passed_count += 1
        except AssertionError as e:
            failed(t.__name__, str(e))
        except Exception as e:
            failed(t.__name__, f"CRASHED: {type(e).__name__}: {e}")

    print(f"\n{passed_count}/{len(tests)} passed")