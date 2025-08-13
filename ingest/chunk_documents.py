from pathlib import Path
import os
from dotenv import load_dotenv

import markdown
from bs4 import BeautifulSoup

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
import chromadb

# -------------------
# Environment Setup
# -------------------
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in .env")

# -------------------
# Initialize Gemini LLM
# -------------------
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    google_api_key=GEMINI_API_KEY
)

# -------------------
# Utility Functions
# -------------------
def clean_text(text: str) -> str:
    """
    Remove everything after the 'images={' substring in the text.
    """
    split_token = "images={"
    if split_token in text:
        return text.split(split_token)[0]
    return text


def markdown_to_text(md: str) -> str:
    """
    Convert Markdown content to plain text.
    """
    html = markdown.markdown(md)
    soup = BeautifulSoup(html, features="html.parser")
    return soup.get_text(separator="\n")


def generate_summary(text: str) -> str:
    """
    Generate a concise high-level summary for a document using the Gemini LLM.
    """
    messages = [
        (
            "system",
            (
                "You are an AI assistant tasked with creating a concise, high-level contextual summary for a document. "
                "This summary will be prepended to smaller text chunks from the document to provide context for retrieval and understanding.\n\n"
                "Instructions:\n"
                "- Clearly state the name/title of the document.\n"
                "- Describe the overall purpose or intent of the document.\n"
                "- Highlight the main themes or scope in broad terms.\n"
                "- Do NOT repeat specific details, examples, or points from the document.\n"
                "- Keep the summary focused, clear, and informative.\n"
                "- Limit the summary length to approximately 50 to 100 tokens."
            ),
        ),
        ("human", f"Please summarize the following text: {text}"),
    ]

    response = llm.invoke(messages)
    return response.content if response.content else ""


# -------------------
# Main Processing
# -------------------
def main():
    input_dir = Path("../docs/markdown")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
    )

    # Initialize Chroma client and collection
    client = chromadb.PersistentClient(path="../")
    collection = client.get_or_create_collection(name="un_population_report_2024")

    total_docs_added = 0

    for file_path in input_dir.glob("*.md"):
        print(f"\nProcessing {file_path.name}...")

        raw_text = file_path.read_text(encoding="utf-8")
        cleaned_text = clean_text(raw_text)
        plain_text = markdown_to_text(cleaned_text)

        # Generate summary for context
        summary = generate_summary(plain_text)

        # Split text into chunks
        chunks = splitter.split_text(plain_text)

        # Prepare documents for Chroma
        documents = [f"{summary}\n\n{chunk}" for chunk in chunks]
        metadatas = [{"source": file_path.name, "chunk_index": i} for i in range(len(chunks))]
        ids = [f"{file_path.stem}_chunk_{i}" for i in range(len(chunks))]

        # Add chunks to collection
        collection.add(documents=documents, metadatas=metadatas, ids=ids)
        total_docs_added += len(chunks)

        print(f"  Added {len(chunks)} chunks from {file_path.name}.")

    # Print summary
    total_docs_in_collection = collection.count()
    print(f"\nTotal documents in 'un_population_report_2024' collection: {total_docs_in_collection}")


# -------------------
# Entry Point
# -------------------
if __name__ == "__main__":
    main()
