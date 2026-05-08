# testing/test_integration.py

import sys
sys.path.insert(0, ".")

from agents.state import AgentState, Triple
from agents.decomposition import decompose
from agents.neighbor import find_neighbors
from agents.bridge import find_bridges
from agents.validation import validate_bridges
from agents.synthesis import synthesize

def run_pipeline_partial(query: str, stop_after: str) -> AgentState:
    """
    Run agents in sequence up to stop_after.
    stop_after: "decomposition" | "neighbor" | "bridge" | "validation" | "synthesis"
    Returns state at that point for inspection.
    """
    state = AgentState(
        original_query=query,
        triples=[], neighbor_map={}, bridge_candidates=[],
        evidence=[], final_answer="", phase="init",
        degraded_mode=False, errors=[], llm_calls_used=0,
        retrieved_chunks_log=[]
    )

    stages = ["decomposition", "neighbor", "bridge", "validation", "synthesis"]
    fns    = [decompose, find_neighbors, find_bridges, validate_bridges, synthesize]

    for stage, fn in zip(stages, fns):
        print(f"  → Running {stage}...")
        state.update(fn(state))
        print(f"    phase={state['phase']}  llm_calls={state['llm_calls_used']}  errors={state['errors']}")
        if stop_after == stage:
            break

    return state

# ── Handoff tests ─────────────────────────────────────────────────────────────

def test_decomp_to_neighbor_handoff():
    """Triples written by decompose must be consumable by find_neighbors."""
    print("\ntest_decomp_to_neighbor_handoff")
    query = "How do quantum kernels avoid barren plateaus?"
    state = run_pipeline_partial(query, stop_after="neighbor")

    assert state["phase"] == "bridge", f"expected bridge, got {state['phase']}"
    # Every key in neighbor_map must correspond to a triple from decomposition
    triple_keys = {f"{t.subject}_{t.object}" for t in state["triples"]}
    for key in state["neighbor_map"].keys():
        assert key in triple_keys, f"neighbor_map key {key!r} has no matching triple"
    print("  PASS")

def test_neighbor_to_bridge_handoff():
    """neighbor_map structure must satisfy bridge agent's access pattern."""
    print("\ntest_neighbor_to_bridge_handoff")
    query = "Can variational quantum circuits learn classical kernels?"
    state = run_pipeline_partial(query, stop_after="bridge")

    assert state["phase"] == "validation"
    for c in state["bridge_candidates"]:
        # Both triples must have been in the original triple list
        orig_subjects = {t.subject for t in state["triples"]}
        assert c.triple_a.subject in orig_subjects, "bridge candidate references unknown triple_a"
        assert c.triple_b.subject in orig_subjects, "bridge candidate references unknown triple_b"
        assert 0.0 <= c.novelty_score <= 1.0
    print("  PASS")

def test_bridge_to_validation_handoff():
    """EvidenceRecord.candidate must reference the BridgeCandidate passed in."""
    print("\ntest_bridge_to_validation_handoff")
    query = "What is the relationship between QAOA and quantum approximate optimization?"
    state = run_pipeline_partial(query, stop_after="validation")

    assert state["phase"] == "synthesis"
    candidate_concepts = {c.bridge_concept for c in state["bridge_candidates"]}
    for rec in state["evidence"]:
        assert rec.candidate.bridge_concept in candidate_concepts, \
            "evidence record references a candidate not in bridge_candidates"
    print("  PASS")

def test_full_pipeline_no_crash():
    """
    The most important integration test: does the full pipeline complete
    without crashing on a real QML query?
    Checks structural integrity only — not answer quality.
    """
    print("\ntest_full_pipeline_no_crash")
    query = "How might quantum kernel methods be affected by barren plateaus in variational circuits?"
    state = run_pipeline_partial(query, stop_after="synthesis")

    assert state["phase"] == "done",                        "pipeline did not reach done"
    assert isinstance(state["final_answer"], str),          "final_answer missing"
    assert len(state["final_answer"]) > 100,                "final_answer suspiciously short"
    assert state["llm_calls_used"] > 0,                     "no LLM calls were made"

    print(f"\n  llm_calls_used  : {state['llm_calls_used']}")
    print(f"  errors          : {state['errors']}")
    print(f"  bridge_count    : {len(state['bridge_candidates'])}")
    print(f"  evidence_count  : {len(state['evidence'])}")
    print(f"  answer_length   : {len(state['final_answer'].split())} words")
    print("  PASS")

def test_degraded_mode_propagates():
    """
    Simulate decomposition failure — verify degraded_mode reaches synthesis
    and surfaces in final_answer.
    """
    print("\ntest_degraded_mode_propagates")
    from agents.neighbor import find_neighbors
    from agents.bridge import find_bridges
    from agents.validation import validate_bridges
    from agents.synthesis import synthesize

    # Manually set a broken state as if decomposition failed
    state = AgentState(
        original_query="quantum error correction thresholds",
        triples=[],
        neighbor_map={}, bridge_candidates=[], evidence=[],
        final_answer="", phase="neighbor",
        degraded_mode=True,
        errors=["decomposition: json_parse_failed"],
        llm_calls_used=1, retrieved_chunks_log=[]
    )

    for fn in [find_neighbors, find_bridges, validate_bridges, synthesize]:
        state.update(fn(state))

    assert state["degraded_mode"] == True,  "degraded_mode must stay True once set"
    assert "Note:" in state["final_answer"] or "degraded" in state["final_answer"].lower(), \
        "degraded state must surface in answer"
    print("  PASS")

if __name__ == "__main__":
    tests = [
        test_decomp_to_neighbor_handoff,
        test_neighbor_to_bridge_handoff,
        test_bridge_to_validation_handoff,
        test_full_pipeline_no_crash,
        test_degraded_mode_propagates,
    ]

    print(f"\nRunning {len(tests)} integration tests...\n")
    passed_count = 0
    for t in tests:
        try:
            t()
            passed_count += 1
        except AssertionError as e:
            print(f"  FAIL: {e}")
        except Exception as e:
            print(f"  CRASH: {type(e).__name__}: {e}")
            import traceback; traceback.print_exc()

    print(f"\n{passed_count}/{len(tests)} passed")