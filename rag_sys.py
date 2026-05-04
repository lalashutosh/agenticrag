# =========================
# 0. Setup
# =========================

import os
from dotenv import load_dotenv
from IPython.display import Image, display


load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-base-en",
    model_kwargs={"device": "cpu"} 
)

vectorstore = FAISS.load_local(
    "faiss_ragdb_500",
    embeddings,
    allow_dangerous_deserialization=True
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# =========================
# 4. LLM (OpenRouter)
# =========================

from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
    model="nvidia/nemotron-3-super-120b-a12b:free",
    temperature=0,
)

# =========================
# 5. Retriever Tool
# =========================

from langchain.tools import tool

from langchain.tools import tool

@tool(description="Retrieve quantum ML papers from FAISS vector database")
def retrieve_papers(query: str) -> str:
    docs = retriever.invoke(query)

    formatted = []
    for doc in docs:
        source = doc.metadata.get("source", "unknown")
        content = doc.page_content
        formatted.append(f"[Source: {source}]\n{content}")

    return "\n\n".join(formatted)

# =========================
# 6. Prompts (Quantum ML tuned)
# =========================

GRADE_PROMPT = """
You are evaluating relevance of retrieved context for a quantum machine learning question.

Document:
{context}

Question:
{question}

Check:
- quantum concepts (circuits, kernels, Hamiltonians, etc.)
- mathematical alignment
- terminology match

Answer ONLY:
yes
or
no
"""

REWRITE_PROMPT = """
You are improving a research query in quantum machine learning.

Original question:
{question}

Rewrite it to:
- include precise terminology
- include mathematical intent if relevant
- optimize for retrieval

Return ONLY the improved question.
"""

GENERATE_PROMPT = """
You are a quantum machine learning research assistant.

Use the context to answer.

Rules:
- Preserve LaTeX exactly
- Be precise
- If uncertain, say "I don't know"

Question:
{question}

Context:
{context}
"""

# =========================
# 7. Graph Components
# =========================

from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition

from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field
from typing import Literal

# ---- Agent Node ----

def generate_query_or_respond(state: MessagesState):
    response = llm.bind_tools([retrieve_papers]).invoke(state["messages"])
    return {"messages": [response]}

# ---- Grader ----

class GradeDocuments(BaseModel):
    binary_score: str = Field(description="'yes' or 'no'")

def grade_documents(state: MessagesState) -> Literal["generate_answer", "rewrite_question"]:
    question = state["messages"][0].content
    context = state["messages"][-1].content

    prompt = GRADE_PROMPT.format(question=question, context=context)

    result = llm.with_structured_output(GradeDocuments).invoke(
        [{"role": "user", "content": prompt}]
    )

    return "generate_answer" if result.binary_score == "yes" else "rewrite_question"

# ---- Rewrite ----

def rewrite_question(state: MessagesState):
    question = state["messages"][0].content
    prompt = REWRITE_PROMPT.format(question=question)

    response = llm.invoke([{"role": "user", "content": prompt}])

    return {"messages": [HumanMessage(content=response.content)]}

# ---- Answer ----

def generate_answer(state: MessagesState):
    question = state["messages"][0].content
    context = state["messages"][-1].content

    prompt = GENERATE_PROMPT.format(question=question, context=context)

    response = llm.invoke([{"role": "user", "content": prompt}])

    return {"messages": [response]}

# =========================
# 8. Build Graph
# =========================

workflow = StateGraph(MessagesState)

workflow.add_node("generate_query_or_respond", generate_query_or_respond)
workflow.add_node("retrieve", ToolNode([retrieve_papers]))
workflow.add_node("rewrite_question", rewrite_question)
workflow.add_node("generate_answer", generate_answer)
workflow.add_node("grade_documents", grade_documents)


workflow.add_edge(START, "generate_query_or_respond")

workflow.add_conditional_edges(
    "generate_query_or_respond",
    tools_condition,
    {
        "tools": "retrieve",
        END: END,
    },
)


workflow.add_conditional_edges(
    "retrieve",
    grade_documents,
    {
        "generate_answer": "generate_answer",
        "rewrite_question": "rewrite_question"
    }
)

workflow.add_edge("generate_answer", END)
workflow.add_edge("rewrite_question", "generate_query_or_respond")

graph = workflow.compile()

# =========================
# 9. Run Test
# =========================

query = "What is a quantum kernel method?"

result = graph.invoke({
    "messages": [
        {"role": "user", "content": query}
    ]
})

print(result["messages"][-1].content)


display(Image(graph.get_graph().draw_mermaid_png()))