# Basic QA Project

> A FastAPI service for retrieval-augmented question answering — upload documents **or crawl any website**, then ask questions grounded in the retrieved context.

This project accepts documents and URLs, converts them into searchable chunks, stores their embeddings in Qdrant, and uses Ollama to answer questions from the retrieved context. It ships with a minimal web frontend for one-click website analysis.

## How It Works

```text
                          ┌─ Document upload → Docling parsing ─┐
                          │                                      ├─ Chunking → Ollama embeddings → Qdrant
URL (web crawl) → Crawler ┘→ Text extraction ──────────────────┘                                     │
                                                                                                      │
Question → Ollama LLM ← Retrieved context ←──────────────────────────────────────────────────────────+
```

## Stack

- **API:** FastAPI + Uvicorn
- **Web crawling:** [web-crawler](../web-crawler) (local library)
- **Parsing and chunking:** Docling + LangChain
- **Embeddings and generation:** Ollama
- **Vector search:** Qdrant with hybrid retrieval
- **Evaluation:** Ragas
- **Frontend:** Vanilla HTML/CSS/JS (single file, served by FastAPI)
- **Package management:** uv

## Prerequisites

- Python 3.13+
- [uv](https://docs.astral.sh/uv/)
- [Ollama](https://ollama.com/)
- A local or hosted Qdrant instance
- The [web-crawler](../web-crawler) project cloned as a sibling directory

Pull the models used by the default configuration:

```bash
ollama pull nomic-embed-text:latest
ollama pull llama3.2:latest
ollama pull llama3.1:latest
```

Make sure Ollama is running before using document upload or query endpoints:

```bash
ollama serve
```

## Setup

Install the locked dependencies:

```bash
uv sync
```

Create a `.env` file in the project root:

```env
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=
```

For a hosted Qdrant instance, replace `QDRANT_URL` and provide its API key.

## Run

Start the development server:

```bash
uv run uvicorn app.main:app --host 127.0.0.1 --port 3000
```

- **Frontend:** [http://127.0.0.1:3000](http://127.0.0.1:3000)
- **API docs:** [http://127.0.0.1:3000/docs](http://127.0.0.1:3000/docs)

## Frontend

The built-in web UI lets you:

1. **Enter any URL** and set a page limit
2. **Crawl the site** — the crawler fetches pages and extracts text
3. **Get an auto-generated summary** of the website content
4. **Ask follow-up questions** in a chat interface, with source citations

No build step required — it's a single HTML file served at `/`.

## API Quick Start

Check that the service is alive:

```bash
curl http://127.0.0.1:3000/health
```

Check Qdrant readiness:

```bash
curl http://127.0.0.1:3000/health/ready
```

Upload a supported PDF, CSV, TXT, or DOCX file:

```bash
curl -X POST http://127.0.0.1:3000/documents/upload \
	-F "file=@/path/to/document.pdf"
```

Crawl a website:

```bash
curl -X POST http://127.0.0.1:3000/documents/crawl \
	-H "Content-Type: application/json" \
	-d '{"url":"https://example.com","page_limit":20}'
```

Ask a question:

```bash
curl -X POST http://127.0.0.1:3000/query \
	-H "Content-Type: application/json" \
	-d '{"question":"What is this about?","include_sources":true}'
```

View collection info:

```bash
curl http://127.0.0.1:3000/documents/info
```

Delete all stored data:

```bash
curl -X DELETE http://127.0.0.1:3000/documents/collection
```

## Project Layout

```text
app/
├── api/
│   ├── routes/
│   │   ├── document.py      Upload, crawl, collection management
│   │   ├── query.py         Question answering + streaming
│   │   └── health.py        Health and readiness checks
│   └── schemas.py           Pydantic request/response models
├── core/
│   ├── document_processor.py  Docling-based file parsing + chunking
│   ├── web_processor.py       Web crawl text → LangChain documents
│   ├── embeddings.py          Ollama embedding setup
│   ├── vector_store.py        Qdrant hybrid vector store
│   ├── rag_chain.py           LangChain RAG pipeline
│   └── ragas_evaluator.py     Answer quality evaluation
├── static/
│   └── index.html             Web frontend (HTML + CSS + JS)
├── utils/                     Logging helpers
├── config.py                  Environment-backed settings
└── main.py                    FastAPI application entrypoint
```

## Debugging

The repository includes a VS Code launch configuration. Open **Run and Debug**, choose **Debug FastAPI**, and start the debugger. Breakpoints are kept reliable by launching Uvicorn without auto-reload.

## Development Notes

- Keep `.env` local; it is ignored by Git.
- The `web-crawler` dependency is resolved from `../web-crawler` via `[tool.uv.sources]` in `pyproject.toml`.
- The first document upload can take longer while Docling downloads its parsing models.
- The first embedding or generation request can take longer while Ollama loads a model.
- `uv.lock` is committed to keep dependency resolution reproducible.
