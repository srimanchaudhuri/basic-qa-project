# Basic QA Project

> A focused FastAPI service for experimenting with retrieval-augmented question answering.

This project accepts documents, converts them into searchable chunks, stores their embeddings in Qdrant, and uses Ollama to answer questions from the retrieved context. It is intentionally small enough to study while still exposing the moving parts of a practical RAG pipeline.

## How It Works

```text
Document upload -> Docling parsing -> Chunking -> Ollama embeddings -> Qdrant
																																		 |
Question -> Ollama LLM <- Retrieved context <-------------------------+
```

## Stack

- **API:** FastAPI + Uvicorn
- **Parsing and chunking:** Docling + LangChain
- **Embeddings and generation:** Ollama
- **Vector search:** Qdrant with hybrid retrieval
- **Evaluation:** Ragas
- **Package management:** uv

## Prerequisites

- Python 3.13+
- [uv](https://docs.astral.sh/uv/)
- [Ollama](https://ollama.com/)
- A local or hosted Qdrant instance

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

## Run the API

Start the development server:

```bash
uv run uvicorn app.main:app --host 127.0.0.1 --port 3000
```

The interactive API documentation is available at [http://127.0.0.1:3000/docs](http://127.0.0.1:3000/docs).

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

Ask a question:

```bash
curl -X POST http://127.0.0.1:3000/query \
	-H "Content-Type: application/json" \
	-d '{"question":"What is this document about?","include_sources":true}'
```

## Project Layout

```text
app/
├── api/              HTTP routes and Pydantic schemas
├── core/             Parsing, embeddings, retrieval, and evaluation
├── utils/             Logging helpers
├── config.py         Environment-backed settings
└── main.py           FastAPI application entrypoint
```

## Debugging

The repository includes a VS Code launch configuration. Open **Run and Debug**, choose **Debug FastAPI**, and start the debugger. Breakpoints are kept reliable by launching Uvicorn without auto-reload.

## Development Notes

- Keep `.env` local; it is ignored by Git.
- The first document upload can take longer while Docling downloads its parsing models.
- The first embedding or generation request can take longer while Ollama loads a model.
- `uv.lock` is committed to keep dependency resolution reproducible.
