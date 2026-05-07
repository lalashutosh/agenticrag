from datetime import datetime
import json
from pathlib import Path
from core.rag_sys import graph
import time

BASE_DIR = Path(__file__).resolve().parent


# -----------------------------
# Run agent
# -----------------------------
def run_agent(graph, question, retries=2):
    for i in range(retries):
        try:
            result = graph.invoke({
                "messages": [{"role": "user", "content": question}]
            })
            return result["messages"][-1].content
        except Exception:
            time.sleep(1)
    return "ERROR"

# -----------------------------
# Create run directory
# -----------------------------
def create_run_dir():
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = BASE_DIR / "runs" / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


# -----------------------------
# Main eval pipeline
# -----------------------------
def build_eval_bundle(graph, eval_version="v1.json"):
    run_dir = create_run_dir()

    eval_path = BASE_DIR / "eval_sets" / eval_version

    if not eval_path.exists():
        raise FileNotFoundError(f"Eval set not found: {eval_path}")

    with open(eval_path) as f:
        test_data = json.load(f)

    outputs = []
    judge_inputs = []

    print("\n=== RUNNING EVAL ===\n")

    for item in test_data:
        q = item["question"]
        q_id = item.get("id", None)

        answer = run_agent(graph, q)

        outputs.append({
            "id": q_id,
            "question": q,
            "answer": answer
        })

        judge_inputs.append({
            "id": q_id,
            "question": q,
            "answer": answer
        })

        print(f"Processed: {q}")

    # -----------------------------
    # Save outputs (raw model results)
    # -----------------------------
    with open(run_dir / "outputs.json", "w") as f:
        json.dump(outputs, f, indent=2)

    # -----------------------------
    # Save judge input bundle
    # -----------------------------
    bundle = {
        "instructions": """
You are an expert evaluator in quantum machine learning and physics.

Evaluate each answer based on:

1. Technical correctness
2. Conceptual completeness
3. Grounding (no hallucinations)
4. Proper use of terminology

Scoring:
5 = fully correct, precise, complete
4 = mostly correct, minor gaps
3 = partially correct, missing key ideas
2 = mostly incorrect
1 = incorrect or hallucinated

Return STRICT JSON:
{
  "id": "...",
  "score": 1-5,
  "verdict": "correct | partially_correct | incorrect",
  "reason": "brief explanation"
}

Be strict and avoid generosity bias.
""",
        "eval_set": eval_version,
        "examples": judge_inputs
    }

    with open(run_dir / "eval_bundle.json", "w") as f:
        json.dump(bundle, f, indent=2)

    # -----------------------------
    # Save config (CRITICAL for reproducibility)
    # -----------------------------
    config = {
    "model": "Qwen2.5-7B-Instruct (vLLM local)",
    "backend": "vllm",
    "retriever_k": 5,
    "eval_file": eval_version,
    "timestamp": datetime.now().isoformat()
}

    with open(run_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"\n✅ Saved run to: {run_dir}")


# -----------------------------
# Entry point
# -----------------------------
if __name__ == "__main__":
    build_eval_bundle(graph, eval_version="v1.json")