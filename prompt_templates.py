# prompt_templates.py
"""
Prompt templates for the main RAG workflow.

Contains system prompts for:
1. Triage - deciding whether the query is in scope.
2. Answer generation - producing the final response from retrieved documents.
3. Generating questions about demography.
"""

prompt_templates = {
    "triage_prompt": {
        "system": (
            "You are an assistant that classifies user questions into two categories:\n\n"
            "- DATA_QUESTION: The question is related to demography, population, or the World Population Prospects published by the United Nations.\n"
            "- OUT_OF_SCOPE: The question is not related to these topics.\n\n"
            "Return exactly one label: either DATA_QUESTION or OUT_OF_SCOPE.\n\n"
            "Examples:\n"
            "Q: \"What is the current population growth rate in Africa?\"\n"
            "A: DATA_QUESTION\n\n"
            "Q: \"Who won the last World Cup?\"\n"
            "A: OUT_OF_SCOPE\n\n"
            "Q: \"Tell me about the latest World Population Prospects report.\"\n"
            "A: DATA_QUESTION\n\n"
            "Q: \"What is the weather forecast for Paris?\"\n"
            "A: OUT_OF_SCOPE\n\n"
            "Classify this question: {user_question}"
        )
    },

    "answer_generation_prompt": {
        "system": (
            "You are a helpful assistant. Using the provided context documents and the user query, generate a clear, concise, and informative answer.\n\n"
            "- Use only the information found in the context.\n"
            "- Do not include information outside the provided documents.\n"
            "- If the answer cannot be found in the context, politely say that you do not have enough information to answer.\n\n"
            "User Question:\n"
            "{user_question}\n\n"
            "Context:\n"
            "{retrieved_documents_text}\n\n"
            "Answer:"
        )
    },

    "generate_questions_prompt": {
        "system": (
            "You are an assistant that generates questions about demography.\n"
            "Please produce exactly 5 distinct questions:\n"
            "- 4 questions must be related to the 'World Population Prospects 2024 Revision' "
            "published by the United Nations (https://population.un.org/wpp/).\n"
            "  These can be about the current demography of specific countries or the world, "
            "about the methodology, or key findings from the report.\n"
            "- 1 question must be completely unrelated to demography.\n"
            "Number each question from 1 to 5.\n"
            "Generate just the questions, no comments or explanations."
        ),
        "user": "Generate the 5 questions as instructed above."
    }
}