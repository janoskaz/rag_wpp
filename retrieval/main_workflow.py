# workflow.py
"""
Main RAG Workflow

This file defines the main LangGraph workflow that:
1. Triages the incoming query (decides if it's in-scope or not).
2. Runs document retrieval (via a subgraph).
3. Generates the final answer using an LLM.
"""

import sys
sys.path.append("../")  # Allow imports from parent directory

from dotenv import load_dotenv
from typing import TypedDict
from langgraph.graph import StateGraph, END
from langchain.schema import Document
from prompt_templates import prompt_templates
from query_gemini import query_ai
from .retrieval_workflow import retrieval_app  # Import subgraph

# Load environment variables (API keys, config, etc.)
load_dotenv()


# -------------------
# State Definitions
# -------------------
class InputState(TypedDict, total=False):
    """
    State structure for the workflow.
    Keys are passed between nodes.
    """
    query: str
    query_type: str  # Either "DATA_QUESTION" or "OUT_OF_SCOPE"
    retrieved_documents: list[Document]
    answer: str


class OutputState(TypedDict):
    """Final output of the workflow."""
    answer: str


# -------------------
# Nodes
# -------------------
def triage_node(state: InputState) -> InputState:
    """
    Classify the query into:
    - "DATA_QUESTION" if it's about demography / UN WPP
    - "OUT_OF_SCOPE" otherwise
    """
    query = state["query"]

    triage_result = query_ai(
        system_prompt=prompt_templates["triage_prompt"]["system"],
        user_prompt=f"Evaluate user query: {query}"
    )

    state["query_type"] = triage_result
    return state


def out_of_scope_node(state: InputState) -> OutputState:
    """
    Respond to queries that are out of scope.
    """
    return {"answer": "Your query is out of scope."}


def retrieval(state: InputState) -> InputState:
    """
    Run the retrieval subgraph.
    This will enrich the state with 'retrieved_documents'.
    """
    subgraph_output = retrieval_app.invoke(state)
    return subgraph_output


def generate_answer_node(state: InputState) -> OutputState:
    """
    Generate the final answer using retrieved documents and the query.
    """
    query = state["query"]
    context = "\n\n".join(doc.page_content for doc in state["retrieved_documents"])

    answer_result = query_ai(
        system_prompt=prompt_templates["answer_generation_prompt"]["system"],
        user_prompt=f"Evaluate user query: {query} with context: {context}"
    )

    return {"answer": answer_result}


# -------------------
# Build Workflow
# -------------------
workflow = StateGraph(InputState, OutputState)

# Register nodes
workflow.add_node("triage", triage_node)
workflow.add_node("retrieval", retrieval)  # Subgraph
workflow.add_node("out_of_scope", out_of_scope_node)
workflow.add_node("answer", generate_answer_node)

# Entry point
workflow.set_entry_point("triage")

# Conditional branching after triage
workflow.add_conditional_edges(
    "triage",
    lambda s: s["query_type"],
    {
        "OUT_OF_SCOPE": "out_of_scope",
        "DATA_QUESTION": "retrieval"
    }
)

# Path after retrieval
workflow.add_edge("retrieval", "answer")

# End states
workflow.add_edge("out_of_scope", END)
workflow.add_edge("answer", END)

# Compile the app
main_rag_app = workflow.compile()
