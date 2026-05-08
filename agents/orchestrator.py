# agents/orchestrator.py

from agents.state import AgentState


PHASE_SEQUENCE = [
    "init",
    "decomposition",
    "neighbor",
    "bridge",
    "validation",
    "synthesis",
    "done"
]


def _next_phase(current: str) -> str:
    """
    Deterministic phase advancement.
    """

    if current not in PHASE_SEQUENCE:
        return "synthesis"

    idx = PHASE_SEQUENCE.index(current)

    if idx + 1 >= len(PHASE_SEQUENCE):
        return "done"

    return PHASE_SEQUENCE[idx + 1]


def orchestrator_node(state: AgentState) -> AgentState:
    """
    Pure routing supervisor.

    Responsibilities:
    - Advance phase
    - Detect repeated failures
    - Enable degraded mode
    - Never performs LLM or retrieval calls
    """

    current_phase = state.get("phase", "init")

    errors = state.get("errors", [])

    degraded_mode = state.get("degraded_mode", False)

    # -----------------------------------------------------
    # Failure bookkeeping
    # -----------------------------------------------------

    retry_counts = state.get("_retry_counts", {})

    current_failures = len(errors)

    previous_failures = state.get("_previous_error_count", 0)

    error_grew = current_failures > previous_failures

    # -----------------------------------------------------
    # Retry once if new error appeared
    # -----------------------------------------------------

    if error_grew:

        retries = retry_counts.get(current_phase, 0)

        # First failure → retry same phase once
        if retries == 0:

            retry_counts[current_phase] = 1

            return {
                **state,
                "_retry_counts": retry_counts,
                "_previous_error_count": current_failures
            }

        # Second failure → degrade and jump to synthesis
        return {
            **state,
            "phase": "synthesis",
            "degraded_mode": True,
            "_retry_counts": retry_counts,
            "_previous_error_count": current_failures
        }

    # -----------------------------------------------------
    # Normal phase advancement
    # -----------------------------------------------------

    next_phase = _next_phase(current_phase)

    return {
        **state,
        "phase": next_phase,
        "_previous_error_count": current_failures
    }


def route(state: AgentState) -> str:
    """
    LangGraph conditional router.

    Uses ONLY the phase field as source of truth.
    """

    phase = state.get("phase", "init")

    if phase == "decomposition":
        return "decomposition"

    if phase == "neighbor":
        return "neighbor"

    if phase == "bridge":
        return "bridge"

    if phase == "validation":
        return "validation"

    if phase == "synthesis":
        return "synthesis"

    if phase == "done":
        return "done"

    return "decomposition"