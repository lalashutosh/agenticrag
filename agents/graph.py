# agents/graph.py

from langgraph.graph import StateGraph, START, END

from agents.state import AgentState

from agents.orchestrator import (
    orchestrator_node,
    route
)

# ---------------------------------------------------------
# Import all agent nodes
# ONLY graph.py imports every agent
# ---------------------------------------------------------

from agents.decomposition import decomposition_node
from agents.neighbor import neighbor_node
from agents.bridge import bridge_node
from agents.validation import validation_node
from agents.synthesis import synthesis_node


# =========================================================
# Build graph
# =========================================================

workflow = StateGraph(AgentState)

# ---------------------------------------------------------
# Nodes
# ---------------------------------------------------------

workflow.add_node("orchestrator", orchestrator_node)

workflow.add_node("decomposition", decomposition_node)

workflow.add_node("neighbor", neighbor_node)

workflow.add_node("bridge", bridge_node)

workflow.add_node("validation", validation_node)

workflow.add_node("synthesis", synthesis_node)

# ---------------------------------------------------------
# Entry
# ---------------------------------------------------------

workflow.add_edge(START, "orchestrator")

# ---------------------------------------------------------
# Orchestrator routing
# ---------------------------------------------------------

workflow.add_conditional_edges(
    "orchestrator",
    route,
    {
        "decomposition": "decomposition",
        "neighbor": "neighbor",
        "bridge": "bridge",
        "validation": "validation",
        "synthesis": "synthesis",
        "done": END
    }
)

# ---------------------------------------------------------
# Every worker returns to orchestrator
# ---------------------------------------------------------

workflow.add_edge("decomposition", "orchestrator")

workflow.add_edge("neighbor", "orchestrator")

workflow.add_edge("bridge", "orchestrator")

workflow.add_edge("validation", "orchestrator")

workflow.add_edge("synthesis", "orchestrator")

# ---------------------------------------------------------
# Compile
# ---------------------------------------------------------

research_graph = workflow.compile()


# =========================================================
# Manual smoke test
# =========================================================

if __name__ == "__main__":

    initial_state = {
        "original_query":
            "Can quantum kernels improve molecular property prediction?",

        "phase": "init",

        "degraded_mode": False,

        "errors": [],

        "llm_calls_used": 0,

        "triples": [],

        "neighbor_map": {},

        "bridge_candidates": [],

        "evidence": [],

        "final_answer": "",

        "retrieved_chunks_log": []
    }

    result = research_graph.invoke(initial_state)

    print("\n===== FINAL ANSWER =====\n")

    print(result.get("final_answer", ""))

    print("\n===== DEGRADED MODE =====\n")

    print(result.get("degraded_mode"))