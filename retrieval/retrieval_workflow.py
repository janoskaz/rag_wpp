"""
retrieval_workflow.py
---------------------
This module defines a LangGraph sub-graph for multi-strategy document retrieval and reranking.

Retrieval steps:
1. **Semantic Search** via ChromaDB vector store
2. **BM25 Search** using lexical matching
3. **CrossEncoder Reranking** to combine results and select top documents

Outputs:
- A list of top-ranked `Document` objects in `retrieved_documents`.
"""

from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from chromadb import PersistentClient
from langchain.schema import Document
from langchain_community.retrievers import BM25Retriever
from sentence_transformers import CrossEncoder
import numpy as np

# =============================================================================
# --- CONFIGURATION ---
# =============================================================================
CHROMA_PATH = "chroma"                              # Path to local ChromaDB storage
COLLECTION_NAME = "un_population_report_2024"       # Name of the ChromaDB collection
NR_RETRIEVED_DOCS = 25                              # How many docs to fetch per method
NR_FINAL_DOCS = 5                                   # How many docs to keep after reranking

# =============================================================================
# --- INITIALIZATION ---
# =============================================================================
# ChromaDB client & collection
chroma_client = PersistentClient(path=CHROMA_PATH)
collection = chroma_client.get_collection(COLLECTION_NAME)

# CrossEncoder model for reranking
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# =============================================================================
# --- STATE DEFINITIONS ---
# =============================================================================
class RetrievalInputState(TypedDict):
    query: str
    semantic_docs: list[Document]
    bm25_docs: list[Document]
    retrieved_documents: list[Document]

class RetrievalOutputState(TypedDict):
    query: str
    retrieved_documents: list[Document]

# =============================================================================
# --- NODE FUNCTIONS ---
# =============================================================================
def semantic_search_node(state: RetrievalInputState) -> RetrievalInputState:
    """
    Perform semantic search using the ChromaDB vector store.
    Stores results in state["semantic_docs"].
    """
    results = collection.query(
        query_texts=[state["query"]],
        n_results=NR_RETRIEVED_DOCS,
        include=["documents", "metadatas", "distances"]
    )

    docs = []
    for doc, meta, dist in zip(results["documents"][0],
                               results["metadatas"][0],
                               results["distances"][0]):
        # Convert distance to similarity score (0 1)
        vector_score = max(0, 1 - dist)
        docs.append(Document(
            page_content=doc,
            metadata={
                **(meta or {}),
                "vector_score": vector_score,
                "bm25_score": 0.0,
                "colbert_score": 0.0,
                "context_score": 0.0,
                "retrieval_method": "vector"
            }
        ))

    state["semantic_docs"] = docs
    return state


def bm25_search_node(state: RetrievalInputState) -> RetrievalInputState:
    """
    Perform BM25 lexical search across all documents in ChromaDB.
    Stores results in state["bm25_docs"].
    """
    try:
        # Fetch all documents from Chroma
        all_results = collection.get(include=["documents", "metadatas"])
        all_docs = [
            Document(page_content=doc, metadata=meta or {})
            for doc, meta in zip(all_results["documents"], all_results["metadatas"])
        ]

        # Run BM25 retrieval
        bm25_retriever = BM25Retriever.from_documents(all_docs)
        bm25_retriever.k = NR_RETRIEVED_DOCS
        results = bm25_retriever.invoke(state["query"])

        docs = []
        for i, doc in enumerate(results):
            # Simple descending score based on rank position
            bm25_score = 1.0 - (i / len(results))
            docs.append(Document(
                page_content=doc.page_content,
                metadata={
                    **doc.metadata,
                    "bm25_score": bm25_score,
                    "vector_score": 0.0,
                    "colbert_score": 0.0,
                    "context_score": 0.0,
                    "retrieval_method": "bm25"
                }
            ))

        state["bm25_docs"] = docs

    except Exception as e:
        print(f"[ERROR] BM25 retrieval failed: {e}")
        state["bm25_docs"] = []

    return state


def merge_and_rerank_node(state: RetrievalInputState) -> RetrievalOutputState:
    """
    Merge semantic and BM25 results, rerank using CrossEncoder,
    and return top NR_FINAL_DOCS in `retrieved_documents`.
    """
    all_docs = state["semantic_docs"] + state["bm25_docs"]
    if not all_docs:
        return {"query": state["query"], "retrieved_documents": []}

    # Prepare query document pairs for CrossEncoder
    doc_texts = [doc.page_content for doc in all_docs]
    pairs = [(state["query"], doc_text) for doc_text in doc_texts]

    # Get CrossEncoder scores
    scores = np.array(reranker.predict(pairs))

    # Normalize scores to 0 1 range
    normalized_scores = (
        (scores - scores.min()) / (scores.max() - scores.min() + 1e-6)
        if scores.max() > scores.min()
        else np.ones_like(scores) * 0.5
    )

    # Store scores in metadata
    for i, doc in enumerate(all_docs):
        doc.metadata["crossencoder_score"] = float(normalized_scores[i])

    # Sort documents by raw CrossEncoder score (not normalized)
    sorted_docs = [
        doc for _, doc in sorted(zip(scores, all_docs),
                                 key=lambda x: x[0],
                                 reverse=True)
    ]

    # Take top documents
    top_docs = sorted_docs[:NR_FINAL_DOCS]

    return {"query": state["query"], "retrieved_documents": top_docs}

# =============================================================================
# --- SUB-GRAPH DEFINITION ---
# =============================================================================
retrieval_graph = StateGraph(RetrievalInputState, RetrievalOutputState)
retrieval_graph.add_node("semantic_search", semantic_search_node)
retrieval_graph.add_node("bm25_search", bm25_search_node)
retrieval_graph.add_node("merge_and_rerank", merge_and_rerank_node)

retrieval_graph.add_edge(START, "semantic_search")
retrieval_graph.add_edge("semantic_search", "bm25_search")
retrieval_graph.add_edge("bm25_search", "merge_and_rerank")
retrieval_graph.add_edge("merge_and_rerank", END)

# Compiled retrieval pipeline
retrieval_app = retrieval_graph.compile()
