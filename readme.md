# Quantum ML Agentic RAG System

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

# Architecture

## Retrieval Layer (Deterministic Core)

This system builds a **bounded semantic graph before any reasoning happens**.

👉 Insert Diagram Here: *Hybrid BFS Retrieval Engine*

Key components:
- BGE-M3 dense embeddings (FAISS IndexFlatIP)
- BM25 sparse lexical retrieval (M3 lexical weights)
- Reciprocal Rank Fusion (RRF)
- BFS-style k-hop expansion over paper chunks
- strict deduplication + per-source pruning

Output:
> A pruned, structured **candidate subgraph of scientific context**

---

## Reasoning Layer (LLM-Guided)

👉 Insert Diagram Here: *Agent Reasoning Stack*

Sequential pipeline:
- Decomposition Agent → extracts scientific triples
- Bridge Agent → proposes cross-paper hypotheses
- Validation Agent → evaluates evidence support vs contradiction
- Synthesis Agent → generates final structured response

Important:
- LLM operates ONLY on retrieved subgraph
- No retrieval decisions are made during reasoning

---

## Control Layer

👉 Insert Diagram Here: *Orchestrator State Machine*

A deterministic controller that:
- manages execution flow
- enforces BFS expansion limits
- tracks LLM + retrieval budgets
- triggers bounded re-retrieval only when coverage is insufficient

No reasoning logic is performed here — only system control.

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

# Architecture

```text
User Question
    ↓
Decomposition Agent
(extract triples)
    ↓
Neighbor Agent
(vector-space expansion)
    ↓
Bridge Agent
(generate cross-paper hypotheses)
    ↓
Validation Agent
(adversarial retrieval + LLM grading)
    ↓
Synthesis Agent
(ESTABLISHED / BRIDGES / OPEN QUESTION)
```

---

# Design Choices

### Hybrid retrieval for scientific corpora
Pure dense retrieval misses exact technical phrases.

Used:
- **BGE-M3 dense vectors** → semantic similarity
- **BGE-M3 lexical weights → BM25** → exact notation / method names
- **RRF fusion** → stable hybrid ranking

This improved retrieval diversity substantially over dense-only FAISS.

---

### Native BGE-M3 ingestion
Avoided LangChain embedding wrappers.

Reason:
LangChain hides:
- sparse lexical outputs
- ColBERT outputs
- batching control

Using `FlagEmbedding` directly preserves full M3 capability.

---

### Version-locked dependencies
`requirements-lock.txt` included for reproducibility.

Important because:
- `faiss`
- `transformers`
- `bitsandbytes`
- `FlagEmbedding`

can silently break across versions.

---

### Quantized local inference
Qwen2.5-7B runs locally in **4-bit NF4** on a 12GB GPU.

Tradeoff:
- slightly slower than API inference
- fully local
- reproducible
- no external dependency

---

### Evaluation set versioning
Evaluation artifacts are versioned:

```text
testing/eval_sets/
testing/eval_bundle.json
```

This prevents “benchmark drift” during iteration.

---

# Why this project

Most “agentic” systems focus on orchestration.

This one focuses on **research quality**:
- retrieval correctness
- evidence grounding
- explicit uncertainty
- controllable multi-agent reasoning

Built as an engineering exercise in making LLM systems **more trustworthy for scientific work**.
