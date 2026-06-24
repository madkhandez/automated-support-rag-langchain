# Production RAG with LangChain & Vector Databases

This is the complete implementation for the "Production RAG with LangChain & Vector Databases" course.

## Project Structure
- `part1_foundation`: Core RAG concepts, chunking, embeddings.
- `part2_debugging`: Failure modes, LangSmith tracing.
- `part3_optimization`: Hybrid search, reranking, multi-query.
- `part4_scaling`: Supabase PGVector, caching, monitoring.
- `part5_production`: Production FastAPI and Streamlit UI.
- `part6_advanced`: Agentic RAG, GraphRAG, Multimodal RAG.

## Setup
1. Copy `.env.example` to `.env` and configure your keys.
2. Run `uv run pytest tests/` to verify functionality.
3. Start FastAPI server: `cd part5_production && uv run uvicorn api.main:app --reload`
4. Start Streamlit UI: `cd part5_production && uv run streamlit run ui/streamlit_app.py`
