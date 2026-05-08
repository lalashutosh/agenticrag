# Agentic RAG for Quantum ML Research

A multi-agent retrieval-augmented research assistant for exploring **~500 quantum machine learning papers**.

Built to answer scientific questions with:
- grounded evidence retrieval,
- multi-hop reasoning over retrieved literature,
- and explicit separation between **known facts** vs **novel/speculative bridges**.

Current focus: **Quantum Machine Learning (QML)**, but architecture is designed to be **domain-portable** by swapping the corpus.

---

# Current Status (v2)

✅ Hybrid retrieval upgraded:
- dense semantic search: **BGE-M3**
- sparse lexical search: **BM25**
- fusion: **Reciprocal Rank Fusion (RRF)**

✅ LLM upgraded:
- **Qwen2.5-7B-Instruct**
- 4-bit quantized for 12GB GPU deployment

✅ Multi-agent pipeline operational:
- decomposition
- neighbor expansion
- bridge discovery
- adversarial validation
- synthesis

✅ Evaluation suite added:
- unit tests
- integration tests
- quality probes
- reusable eval bundles

Known limitations:
- bridge discovery still overproduces some weak/self-loop hypotheses
- decomposition occasionally emits generic triples
- synthesis can repeat grounded facts in speculative sections

---

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