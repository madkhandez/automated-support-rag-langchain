# RAG Automated Support with LangChain & Vector Databases

Turn scattered company knowledge into instant, accurate AI answers with Retrieval-Augmented Generation (RAG).

An enterprise-grade AI assistant capable of retrieving information from PDFs, Office documents, databases, APIs, websites, Git repositories, and other internal knowledge sources—providing grounded, context-aware answers with citations.

🔴 **[Test the Live Demo Here](https://madkhan-rag.hf.space)** *(Hosted on Hugging Face)*


## Project Structure

The project has been structured for professional production deployment:

- `src/production_rag/`: Core application package containing:
  - `api/`: FastAPI application and routes.
  - `core/`: Core RAG logic, LLM management, and vector store operations.
  - `agents/`: LangGraph agents and state management.
  - `security/`: Rate limiting, prompt injection prevention, and output filtering.
  - `ui/`: Frontend for the chat interface.
- `tests/`: Comprehensive test suite for the pipeline and API.

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
