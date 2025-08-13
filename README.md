# Document Ingestion & Retrieval with Gemini LLM

This project provides a full pipeline to convert PDF documents into Markdown, chunk and store them in a vector database (Chroma), and perform retrieval using Google’s Gemini LLM.

---

## Project Structure

```
.
├── ingest
│   ├── convert_pdfs_to_markdown.py   # Convert PDFs to Markdown
│   └── chunk_documents.py            # Split Markdown into chunks & add to Chroma
├── retrieval
│   ├── main_workflow.py              # Main retrieval workflow
│   └── retrieval_workflow.py         # Helper workflow for queries
├── prompt_templates.py               # Prompts used for Gemini LLM
├── query_gemini.py                   # Gemini LLM query wrapper
├── main.py                           # Entry point for retrieval
├── .env                              # Environment variables (GEMINI_API_KEY)
├── docs
│   ├── *.pdf                          # Original PDF documents
│   └── markdown                       # Markdown files converted from PDFs
└── chroma                             # Chroma database storage
```

---

## Setup

1. **Clone the repository:**

```bash
git clone <repository_url>
cd <repository_name>
```

2. **Set up your **``** file:**

Create a `.env` file in the project root with the following:

```
GEMINI_API_KEY=your_gemini_api_key
```

---

## Workflow

### 1. Ingest PDFs

The `ingest` scripts handle PDF conversion and chunking:

- **Convert PDFs to Markdown:**

  ```bash
  python ingest/convert_pdfs_to_markdown.py
  ```

  This reads PDFs from `docs/` and writes Markdown files to `docs/markdown/`.

- **Chunk Markdown & add to Chroma:**

  ```bash
  python ingest/chunk_documents.py
  ```

  This splits the Markdown files into smaller chunks, generates summaries, and stores them in the Chroma vector database (`chroma/`).

### 2. Retrieval

Once the database is ready, you can run the retrieval workflow:

```bash
python main.py
```

This will:

- Load the Chroma database
- Accept user queries
- Retrieve relevant chunks using the Gemini LLM
- Provide answers based on the documents

---

## Notes

- Make sure all PDFs are in `docs/` before running the ingest scripts.
- The Gemini API key is required for both chunk summarization and retrieval.
- The database is stored persistently in the `chroma/` folder.
- Markdown files generated during ingestion are saved in `docs/markdown/` for reference.

---

## License

This project is licensed under the MIT License.

