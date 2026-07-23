---
title: Automated Support RAG API
emoji: 🤖
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# Production RAG with LangChain & Vector Databases

This is a complete, production-ready implementation of a Retrieval-Augmented Generation (RAG) system using LangChain, Vector Databases, and FastAPI.

## Project Structure

The project has been structured for professional production deployment:

- `src/production_rag/`: Core application package containing:
  - `api/`: FastAPI application and routes.
  - `core/`: Core RAG logic, LLM management, and vector store operations.
  - `agents/`: LangGraph agents and state management.
  - `security/`: Rate limiting, prompt injection prevention, and output filtering.
  - `ui/`: Streamlit frontend for the chat interface.
- `tests/`: Comprehensive test suite for the pipeline and API.
- `tutorials/`: Educational notebooks and scripts detailing fundamental to advanced RAG concepts (chunking, hybrid search, caching, etc.).

## Setup & Execution

1. Copy `.env.example` to `.env` and configure your API keys.
2. Install dependencies (recommended using `uv`):
   ```bash
   uv pip install -e .
   ```
3. Run the tests to verify functionality:
   ```bash
   uv run pytest tests/
   ```
4. Start the FastAPI backend:
   ```bash
   cd src/production_rag
   uv run uvicorn api.main:app --reload
   ```
5. Open the application in your browser:
   Navigate to [http://localhost:8000](http://localhost:8000) to view the landing page and access the RAG tool.

## AI Provider Configuration
The application uses a cascading fallback chain for LLMs: Google Gemini → Local Ollama → Anthropic Claude → OpenAI.
Configure these in your `.env` file:
- `GOOGLE_API_KEY`, `GOOGLE_LLM_MODEL`
- `OLLAMA_BASE_URL`, `LOCAL_LLM_MODEL`
- `ANTHROPIC_API_KEY`, `ANTHROPIC_LLM_MODEL`
- `OPENAI_API_KEY`, `OPENAI_LLM_MODEL`
