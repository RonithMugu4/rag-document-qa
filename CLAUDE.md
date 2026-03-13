# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Setup

Activate the virtual environment before running anything:
```bash
source venv/Scripts/activate  # Windows Git Bash
# or
venv\Scripts\activate         # Windows cmd/PowerShell
```

Requires a `.env` file with `OPENAI_API_KEY` set. The project uses `python-dotenv` to load it automatically.

## Commands

**Ingest a PDF** (chunks it and builds the FAISS vector store):
```bash
python ingest.py
```

**Test retrieval** (loads the FAISS index and runs a similarity search):
```bash
python retriever.py
```

## Architecture

This is a RAG (Retrieval-Augmented Generation) pipeline built with LangChain and OpenAI. The pipeline has three phases:

1. **Ingestion** ([ingest.py](ingest.py)): Loads a PDF from `data/sample.pdf`, splits it into overlapping chunks (500 chars, 50 overlap) using `RecursiveCharacterTextSplitter`, embeds them with `text-embedding-3-small`, and persists the FAISS index to `faiss_index/`.

2. **Retrieval** ([retriever.py](retriever.py)): Loads the persisted FAISS index from disk and performs cosine similarity search to return the top-k most relevant chunks for a query.

3. **Generation** ([generator.py](generator.py)): Intended to take retrieved chunks and generate an answer using an LLM (currently empty — next phase to implement).

4. **Entrypoint** ([main.py](main.py)): Top-level orchestration (currently empty).

The FAISS index (`faiss_index/index.faiss` + `faiss_index/index.pkl`) is persisted to disk so re-ingestion is only needed when the source document changes.
