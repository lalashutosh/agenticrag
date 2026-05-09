# Agentic RAG System for Quantum ML research

A hybrid retrieval + reasoning system for scientific discovery over ~500 quantum machine learning papers.

The system combines deterministic graph-based retrieval with LLM-guided hypothesis generation and validation.

---

# Overview

This project is a **state-aware agentic RAG system** designed to:
- discover non-obvious relationships in quantum machine learning literature
- generate and validate scientific bridge hypotheses
- operate under strict cost and retrieval constraints

It is not a chat system — it is a **structured reasoning engine over a semantic graph of research papers**.

---

## Architecture

### Retrieval Layer — Hybrid BFS RAG Engine

![Hybrid BFS RAG Engine]<img width="1536" height="1024" alt="hybridrag" src="https://github.com/user-attachments/assets/19d06a98-ed33-4eda-90fd-0e1ae31bc26f" />


Fully deterministic — no LLM calls during graph construction.

The engine performs k-hop BFS traversal over paper chunks, running dense (BGE-M3 + FAISS)
and sparse (BM25) retrieval in parallel. Results are fused via Reciprocal Rank Fusion and
pruned to a compact, grounded candidate subgraph. A second LLM-guided expansion phase
activates only if coverage is insufficient.

---

### Reasoning Layer — Agentic Orchestration Stack

![Agentic Orchestration Stack]<img width="1024" height="660" alt="agentstack" src="https://github.com/user-attachments/assets/f35186b8-7b3d-42e1-8dd4-a4e5ac1acd9f" />


LLM agents operate only on the retrieved subgraph — no retrieval decisions are made here.

A state machine Orchestrator manages execution flow and enforces budgets. Agents run in
a fixed pipeline: Decomposition → Bridge → Validation → Synthesis. Re-retrieval is
triggered by the Orchestrator only when evidence coverage is insufficient.

**Max 2–3 LLM calls per query. No recursive loops.**

---

# System Design

## Key Principles

- **Retrieval before reasoning**
- **Graph construction is deterministic**
- **LLM is a hypothesis evaluator, not a search engine**
- **All exploration is budgeted and bounded**
- **Hybrid IR (dense + sparse) ensures coverage + precision**

---

## Constraints

- Max BFS depth: 2
- Max retrieved nodes: ~100–150 per query
- Max LLM calls per query: 2–3
- Fully deterministic retrieval layer
- No recursive agent loops

---

## Why this design works

This system separates concerns into:

- **Control Plane** → orchestrator (state machine)
- **Data Plane** → BFS hybrid retrieval engine
- **Reasoning Plane** → LLM agents

This avoids:
- exponential agent loops
- redundant reasoning over unfiltered context
- uncontrolled retrieval recursion

---

# Reproduce on your system

## 1. Install dependencies

```bash
pip install -r requirements.txt
# Reproduce

## 1. Install

```bash
git clone <repo>
cd agenticrag

python -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```

---

## 2. Add corpus

Place PDFs here:

```text
data/ragdb_500/
```

(~500 QML papers expected)

---

## 3. Build hybrid index

```bash
python -m data.reingest_m3
```

Creates:

```text
data/faiss_ragdb_m3/
├── dense.index
├── chunks.json
├── token_lists.json
└── bm25.pkl
```

---

## 4. Run tests

```bash
python -m testing.test_tools
python -m testing.test_integration
python -m testing.test_quality
```

---

## 5. Run system

```python
from core.rag_sys import ask

ask("Can barren plateaus be mitigated using kernel initialization?")
```

---

# Why this project

Most “agentic” systems focus on orchestration.

This one focuses on **research quality**:
- retrieval correctness
- evidence grounding
- explicit uncertainty
- controllable multi-agent reasoning

Built as an engineering exercise in making LLM systems **more trustworthy for scientific work**.
