# query_gemini.py
"""
Utility to send prompts to Google Gemini via LangChain.

Functions:
- query_ai(system_prompt, user_prompt) → str
"""

from pathlib import Path
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
import os

# -------------------
# Environment Setup
# -------------------
load_dotenv()

gemini_api_key = os.getenv("GEMINI_API_KEY")
if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY not found in .env")

# -------------------
# Model Initialization
# -------------------
chat = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    max_retries=2,
    google_api_key=gemini_api_key
)

# -------------------
# Query Function
# -------------------
def query_ai(system_prompt: str, user_prompt: str) -> str:
    """
    Sends a system prompt and user prompt to the Gemini model.

    Args:
        system_prompt (str): The system-level instructions for the model.
        user_prompt (str): The user’s question or request.

    Returns:
        str: The model's response (cleaned), or empty string if no content.
    """
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ]

    response = chat.invoke(
        input=messages,
        config={"temperature": 0.7}
    )

    return response.content.strip() if response.content else ""
