from rag_sys import graph, retriever

def test_retrieval(retriever, question):
    docs = retriever.invoke(question)

    print(f"\nQuestion: {question}")
    for i, doc in enumerate(docs):
        print(f"\n--- Chunk {i} ---")
        print(doc.page_content[:300])

def run_agent(graph, question):
    result = graph.invoke({
        "messages": [{"role": "user", "content": question}]
    })

    return result["messages"][-1].content

def trace_graph(graph, question):
    print(f"\n=== TRACE: {question} ===\n")

    for step in graph.stream({
        "messages": [{"role": "user", "content": question}]
    }):
        for node, update in step.items():
            print(f"\n--- Node: {node} ---")

            msg = update["messages"][-1]

            # pretty print content
            if hasattr(msg, "content"):
                print("Content:", msg.content[:300])

            # tool calls
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                print("Tool Calls:", msg.tool_calls)


def used_tool(graph, question):
    for step in graph.stream({
        "messages": [{"role": "user", "content": question}]
    }):
        for node, update in step.items():
            msg = update["messages"][-1]
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                return True
    return False


def used_rewrite(graph, question):
    for step in graph.stream({
        "messages": [{"role": "user", "content": question}]
    }):
        for node, _ in step.items():
            if node == "rewrite_question":
                return True
    return False

def count_steps(graph, question):
    steps = 0
    for _ in graph.stream({
        "messages": [{"role": "user", "content": question}]
    }):
        steps += 1
    return steps

def run_trace(graph, question):
    trace = {
        "nodes": [],
        "tool_calls": False,
        "rewrite_hit": False,
        "steps": 0,
        "final_answer": None
    }

    result = None

    for step in graph.stream({
        "messages": [{"role": "user", "content": question}]
    }):
        trace["steps"] += 1

        for node, update in step.items():
            trace["nodes"].append(node)

            msg = update["messages"][-1]

            if hasattr(msg, "tool_calls") and msg.tool_calls:
                trace["tool_calls"] = True

            if node == "rewrite_question":
                trace["rewrite_hit"] = True

            result = update

    if result:
        trace["final_answer"] = result["messages"][-1].content

    return trace
def test_graph_behavior(graph):
    tests = [
        {
            "q": "hello",
            "should_retrieve": False,
            "should_rewrite": False
        },
        {
            "q": "What is a quantum kernel method?",
            "should_retrieve": True,
            "should_rewrite": False
        },
        {
            "q": "quantum kernel weird thing",
            "should_retrieve": True,
            "should_rewrite": True
        }
    ]

    for t in tests:
        trace = run_trace(graph, t["q"])

        print("\n====================")
        print("Q:", t["q"])
        print("Nodes:", trace["nodes"])
        print("Tool used:", trace["tool_calls"])
        print("Rewrite used:", trace["rewrite_hit"])
        print("Steps:", trace["steps"])
        print("Answer:", trace["final_answer"][:200])

        assert trace["tool_calls"] == t["should_retrieve"], "Retrieval mismatch"
        assert trace["rewrite_hit"] == t["should_rewrite"], "Rewrite mismatch"
        assert trace["steps"] < 10, "Possible infinite loop"


if __name__ == "__main__":

    print("\n=== RUNNING RETRIEVAL TESTS ===")
    test_retrieval(retriever, "What is a quantum kernel method?")

    print("\n=== RUNNING GRAPH BEHAVIOR TESTS ===")
    test_graph_behavior(graph)

    print("\n=== RUNNING TRACE EXAMPLE ===")
    trace_graph(graph, "What is a variational quantum circuit?")