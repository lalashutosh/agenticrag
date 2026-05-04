import json

from torch.cuda import graph

# assumes you already have:
# - graph (your LangGraph agent)
# - test_set.json

def run_agent(graph, question):
    result = graph.invoke({
        "messages": [{"role": "user", "content": question}]
    })
    return result["messages"][-1].content


def build_eval_bundle(graph, test_file="test_set.json", output_file="eval_bundle.json"):
    with open(test_file) as f:
        test_data = json.load(f)

    examples = []

    for item in test_data:
        q = item["question"]
        answer = run_agent(graph, q)

        examples.append({
            "question": q,
            "answer": answer
        })

        print(f"Processed: {q}")

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

Return STRICT JSON for each item:
{
  "score": 1-5,
  "verdict": "correct" | "partially_correct" | "incorrect",
  "reason": "brief explanation"
}

Be strict. Penalize vague or generic answers.
""",
        "schema": {
            "score": "1-5",
            "verdict": "correct | partially_correct | incorrect",
            "reason": "string"
        },
        "examples": examples
    }

    with open(output_file, "w") as f:
        json.dump(bundle, f, indent=2)

    print(f"\nSaved eval bundle to {output_file}")


# run it
build_eval_bundle(graph)