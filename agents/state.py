"""
agents/state.py

Shared inter-agent contracts and graph state for the QML multi-agent system.

================================================================================================================
| Agent          | Reads                                              | Writes                               |
|----------------|----------------------------------------------------|--------------------------------------|
| Orchestrator   | original_query, errors, degraded_mode             | phase, degraded_mode, errors         |
| Decomposition  | original_query                                    | triples, retrieved_chunks_log        |
| Neighbor       | triples                                           | neighbor_map, retrieved_chunks_log   |
| Bridge         | triples, neighbor_map                             | bridge_candidates                    |
| Validation     | bridge_candidates                                 | evidence, errors                     |
| Synthesis      | original_query, evidence, bridge_candidates       | final_answer                         |
| All agents     | llm_calls_used                                    | llm_calls_used                       |
================================================================================================================
"""

from typing import Dict, List, Literal, TypedDict
from pydantic import BaseModel, Field, ConfigDict

from core.rag_tools import RetrievedChunk


# =========================================================
# 1. Triple
# =========================================================

class Triple(BaseModel):
    subject: str
    relation: str
    object: str
    source: Literal["query", "retrieved"]

    model_config = ConfigDict(
        frozen=False,
        arbitrary_types_allowed=True
    )


# =========================================================
# 2. NeighborResult
# =========================================================

class NeighborResult(BaseModel):
    triple_key: str
    chunk: RetrievedChunk
    cosine_distance: float

    model_config = ConfigDict(
        frozen=False,
        arbitrary_types_allowed=True
    )


# =========================================================
# 3. BridgeCandidate
# =========================================================

class BridgeCandidate(BaseModel):
    triple_a: Triple
    triple_b: Triple

    bridge_concept: str

    co_occurrence_count: int

    novelty_score: float

    model_config = ConfigDict(
        frozen=False,
        arbitrary_types_allowed=True
    )


# =========================================================
# 4. EvidenceRecord
# =========================================================

class EvidenceRecord(BaseModel):
    candidate: BridgeCandidate

    supporting_chunks: List[RetrievedChunk] = Field(default_factory=list)

    opposing_chunks: List[RetrievedChunk] = Field(default_factory=list)

    confidence: float

    verdict: Literal[
        "supported",
        "speculative",
        "contradicted"
    ]

    model_config = ConfigDict(
        frozen=False,
        arbitrary_types_allowed=True
    )


# =========================================================
# 5. AgentState
# =========================================================

class AgentState(TypedDict, total=False):

    # -----------------------------------------------------
    # Input query
    # -----------------------------------------------------

    original_query: str

    # -----------------------------------------------------
    # Semantic decomposition
    # -----------------------------------------------------

    triples: List[Triple]

    # -----------------------------------------------------
    # Neighbor exploration
    # key = "{subject}_{object}"
    # -----------------------------------------------------

    neighbor_map: Dict[
        str,
        List[NeighborResult]
    ]

    # -----------------------------------------------------
    # Hypothesis generation
    # -----------------------------------------------------

    bridge_candidates: List[BridgeCandidate]

    # -----------------------------------------------------
    # Validation outputs
    # -----------------------------------------------------

    evidence: List[EvidenceRecord]

    # -----------------------------------------------------
    # Final synthesis
    # -----------------------------------------------------

    final_answer: str

    # -----------------------------------------------------
    # Runtime / orchestration
    # -----------------------------------------------------

    phase: str

    degraded_mode: bool

    errors: List[str]

    llm_calls_used: int

    retrieved_chunks_log: List[str]


# =========================================================
# 6. Default state factory
# =========================================================

def make_initial_state(query: str) -> AgentState:
    return AgentState(
        original_query=query,

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