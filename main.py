# main.py
from retrieval.main_workflow import main_rag_app
from query_gemini import query_ai
from prompt_templates import prompt_templates

# -------------------
# Step 1: Ask Gemini to generate 5 questions
# -------------------
questions_text = query_ai(prompt_templates["generate_questions_prompt"]["system"], prompt_templates["generate_questions_prompt"]["user"])

# Convert the LLM output into a list of questions
questions = [q.strip("12345. ").strip() for q in questions_text.split("\n") if q.strip()]

# -------------------
# Step 2: Pass each question to main_rag_app and print nicely
# -------------------
print("\n=== Running RAG Workflow on Generated Questions ===\n")

for idx, question in enumerate(questions, start=1):
    result = main_rag_app.invoke({"query": question})
    answer = result.get("answer", "[No answer returned]")

    print(f"Q{idx}: {question}")
    print(f"A{idx}: {answer}\n" + "-"*60 + "\n")
    