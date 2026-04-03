# 🔬 arXiv Paper Curator — Production RAG System

> A production-grade **Retrieval-Augmented Generation (RAG)** system that automatically ingests arXiv CS.AI research papers, indexes them with hybrid search, and answers research questions using a local LLM — implemented week by week following production engineering best practices.

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/FastAPI-0.115+-green.svg" alt="FastAPI">
  <img src="https://img.shields.io/badge/OpenSearch-2.19-orange.svg" alt="OpenSearch">
  <img src="https://img.shields.io/badge/Docker-Compose-blue.svg" alt="Docker">
  <img src="https://img.shields.io/badge/LLM-Ollama%20llama3.2-purple.svg" alt="LLM">
  <img src="https://img.shields.io/badge/Embeddings-Jina%20AI%201024dim-red.svg" alt="Embeddings">
  <img src="https://img.shields.io/badge/LangGraph-Agentic%20RAG-ff69b4.svg" alt="LangGraph">
  <img src="https://img.shields.io/badge/Telegram-Bot%20Integrated-26A5E4.svg" alt="Telegram">
  <img src="https://img.shields.io/badge/Status-Week%207%20Complete-brightgreen.svg" alt="Status">
</p>

<p align="center">
  <img src="static/mother_of_ai_project_rag_architecture.gif" alt="RAG Architecture" width="700">
</p>

---

## 📖 Table of Contents

- [The Problem](#-the-problem)
- [My Approach](#-my-approach)
- [System Architecture](#-system-architecture)
- [Live Demo](#-live-demo)
- [API Endpoints](#-api-endpoints)
- [Automated Ingestion Pipeline](#-automated-ingestion-pipeline)
- [Tech Stack](#-tech-stack)
- [Build Journey — Week by Week](#-build-journey--week-by-week)
- [Quick Start](#-quick-start)
- [Telegram Bot Setup Guide](#-telegram-bot-setup-guide)
- [Project Structure](#-project-structure)
- [Configuration](#-configuration)
- [Performance](#-performance)
- [Development Guide](#-development-guide)
- [Troubleshooting](#-troubleshooting)

---

## 🧠 The Problem

Researchers and engineers are overwhelmed by the volume of AI papers published daily on arXiv — hundreds of new cs.AI papers every weekday. There is no easy way to:

- **Stay updated** — automatically track the latest papers in your area
- **Search semantically** — find relevant work beyond exact keyword matching
- **Ask questions** — get answers grounded in actual research, not hallucinated facts
- **Cite sources** — know exactly which papers an answer comes from

Most RAG tutorials jump straight to vector search and ignore the production realities: pipeline reliability, search quality, observability, and performance at scale.

---

## 💡 My Approach

I implemented a complete **production-grade RAG pipeline** week by week, following a structured course that mirrors professional AI engineering workflows:

> **The Professional Difference:** Build keyword search foundations first, then enhance with vectors for hybrid retrieval — not the other way around.

| Layer | What I Implemented |
|---|---|
| **Infrastructure** | Docker Compose stack with 7 production services |
| **Data Ingestion** | Airflow DAG — daily arXiv fetch, Docling PDF parse, PostgreSQL store |
| **Search Foundation** | OpenSearch BM25 keyword search with field boosting |
| **Semantic Layer** | Jina AI 1024-dim embeddings + RRF hybrid fusion |
| **LLM Integration** | Ollama llama3.2:1b — fully local, no API costs |
| **Production** | Redis exact-match cache + Langfuse end-to-end tracing |
| **Agentic RAG** | LangGraph workflow with guardrails, grading, and adaptive retrieval |
| **Mobile Access** | Telegram bot for conversational AI on any device |
| **UI** | Gradio web interface for natural language Q&A |

---

## 🏗️ System Architecture

### Services Running

![Services Healthy](assets/screenshots/01_services_healthy.png)

| Service | Purpose | Port |
|---|---|---|
| **FastAPI** | REST API — papers, search, ask, agentic-ask, stream | 8000 |
| **PostgreSQL 16** | Paper metadata + content storage | 5432 |
| **OpenSearch 2.19** | Hybrid search engine (BM25 + KNN vectors) | 9200 |
| **OpenSearch Dashboards** | Search UI and index management | 5601 |
| **Apache Airflow** | Daily paper ingestion pipeline scheduler | 8080 |
| **Ollama** | Local LLM inference (llama3.2:1b) | 11434 |
| **Redis** | Response caching layer | 6379 |

### Architecture Diagrams

<p align="center">
  <img src="static/week1_infra_setup.png" alt="Week 1 Infrastructure" width="800">
  <br><em>Week 1 — Infrastructure Foundation</em>
</p>

<p align="center">
  <img src="static/week2_data_ingestion_flow.png" alt="Week 2 Data Ingestion" width="800">
  <br><em>Week 2 — Data Ingestion Flow</em>
</p>

<p align="center">
  <img src="static/week3_opensearch_flow.png" alt="Week 3 BM25 Search" width="800">
  <br><em>Week 3 — BM25 Keyword Search</em>
</p>

<p align="center">
  <img src="static/week4_hybrid_opensearch.png" alt="Week 4 Hybrid Search" width="800">
  <br><em>Week 4 — Hybrid Search Architecture</em>
</p>

<p align="center">
  <img src="static/week5_complete_rag.png" alt="Week 5 Complete RAG" width="800">
  <br><em>Week 5 — Complete RAG System</em>
</p>

<p align="center">
  <img src="static/week6_monitoring_and_caching.png" alt="Week 6 Monitoring" width="800">
  <br><em>Week 6 — Monitoring & Caching</em>
</p>

<p align="center">
  <img src="static/week7_telegram_and_agentic_ai.png" alt="Week 7 Agentic RAG" width="800">
  <br><em>Week 7 — Agentic RAG with LangGraph & Telegram Bot</em>
</p>

### RAG Pipeline Flow

```
User Query
    ↓
Redis Cache Check ──────────────────→ [HIT: return in <100ms]
    ↓ [MISS]
Jina AI Embeddings (1024-dim vectors)
    ↓
OpenSearch Hybrid Search
├── BM25 keyword search (exact + fuzzy matching)
└── KNN vector similarity search (semantic)
    ↓
RRF Fusion (Reciprocal Rank Fusion)
    ↓
Top-K document chunks retrieved
    ↓
Prompt Builder (system prompt + context + query)
    ↓
Ollama llama3.2:1b (local LLM — no API cost)
    ↓
Answer with cited arXiv sources
    ↓
Store in Redis cache + Langfuse trace logged
```

### Agentic RAG Flow (Week 7)

```
User Query
    ↓
Guardrail Validation (LLM scores 0-100)
    ├── Score < 60 → Out-of-scope rejection
    └── Score ≥ 60 → Continue
         ↓
    Retrieve Documents (via tool call)
         ↓
    Grade Documents (LLM relevance check)
    ├── Relevant → Generate Answer
    └── Not relevant → Rewrite Query → Retry Retrieval
         ↓
    Generate Answer with Citations
         ↓
    Return answer + reasoning steps + sources
```

---

## 🎯 Live Demo

### Simple Query — "What is machine learning?"
![Simple Question](assets/screenshots/05_gradio_simple.png)

### Medium Complexity — "How do transformer models use attention mechanisms?"
![Medium Question](assets/screenshots/06_gradio_medium.png)

### Complex Research Question — "What are the current challenges in AI safety?"
![Complex Question](assets/screenshots/07_gradio_complex.png)

### Redis Cache in Action — 150x+ Speedup
![Cache Demo](assets/screenshots/08_gradio_cache.png)

> **First query:** ~16 seconds (full RAG pipeline — embedding → search → LLM generation)
>
> **Same query repeated:** <100ms (Redis exact-match cache hit)
>
> **Speedup: 150-400x**

### Telegram Bot — Mobile AI Research Assistant (Week 7)
![Telegram Bot](assets/screenshots/09_Telegram.png)

> **Ask research questions directly from Telegram** — the bot searches your indexed arXiv papers, generates answers with citations, and responds conversationally. Supports `/start`, `/help`, `/search`, and free-text questions.

---

## 📡 API Endpoints

![API Documentation](assets/screenshots/02_api_docs.png)
![API Schemas](assets/screenshots/03_api_schemas.png)

| Method | Endpoint | Description | Week Added |
|---|---|---|---|
| `GET` | `/api/v1/ping` | Simple ping | Week 1 |
| `GET` | `/api/v1/health` | Full service health check | Week 1 |
| `GET` | `/api/v1/papers/` | List all ingested papers | Week 2 |
| `GET` | `/api/v1/papers/{arxiv_id}` | Get specific paper | Week 2 |
| `POST` | `/api/v1/search/` | BM25 keyword search | Week 3 |
| `POST` | `/api/v1/hybrid-search/` | Hybrid BM25 + vector search with RRF | Week 4 |
| `POST` | `/api/v1/ask` | RAG question answering (blocking) | Week 5 |
| `POST` | `/api/v1/stream` | Streaming RAG response via SSE | Week 5 |
| `POST` | `/api/v1/ask-agentic` | Agentic RAG with guardrails + reasoning | Week 7 |
| `POST` | `/api/v1/feedback` | Submit feedback on agentic responses | Week 7 |

### Example API Calls

```bash
# Health check
curl http://localhost:8000/api/v1/health

# BM25 search
curl -X POST http://localhost:8000/api/v1/search/ \
  -H "Content-Type: application/json" \
  -d '{"query": "reinforcement learning", "size": 5}'

# Hybrid search (BM25 + vector + RRF)
curl -X POST http://localhost:8000/api/v1/hybrid-search/ \
  -H "Content-Type: application/json" \
  -d '{"query": "transformer attention mechanisms", "use_hybrid": true, "size": 3}'

# RAG question answering
curl -X POST http://localhost:8000/api/v1/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the challenges in AI safety?", "top_k": 3, "use_hybrid": true}'

# Agentic RAG (Week 7) — with guardrails, grading, and reasoning transparency
curl -X POST http://localhost:8000/api/v1/ask-agentic \
  -H "Content-Type: application/json" \
  -d '{"query": "What are transformer architectures in deep learning?"}'
```

---

## 🔄 Automated Ingestion Pipeline

![Airflow Pipeline](assets/screenshots/04_airflow_pipeline.png)

Apache Airflow runs a **daily DAG** (`arxiv_paper_ingestion`) on weekdays:

```
[setup_environment]
       ↓
[fetch_daily_papers] ──── arXiv API → PDF Download → Docling Parse → PostgreSQL Store
       ↓
[process_failed_pdfs] ─── Retry logic for failed PDF parsing
       ↓
[create_opensearch_placeholders] ── Prepare papers for search indexing
       ↓
[generate_daily_report] ── Log statistics: fetched / parsed / stored / errors
       ↓
[cleanup_temp_files] ──── Remove temporary PDF files
```

**Key pipeline features:**
- 3-second rate limiting between arXiv API calls (respects API etiquette)
- Automatic weekend skip — targets Friday papers when scheduled on Monday
- 3 retry attempts per task with exponential backoff
- Graceful continuation despite individual paper failures
- 80-90% PDF parse success rate, ~20 papers/minute

---

## 🛠️ Tech Stack

| Category | Technology | Version | Why Chosen |
|---|---|---|---|
| **API Framework** | FastAPI | 0.115+ | Async support, automatic OpenAPI docs, type safety |
| **Database** | PostgreSQL + SQLAlchemy | 16 | Reliable relational storage, JSONB metadata, ORM |
| **Search Engine** | OpenSearch | 2.19 | BM25 + KNN vector search in one index, RRF pipeline |
| **Embeddings** | Jina AI | v3 (1024-dim) | Free 10M tokens, retrieval-optimized, high quality |
| **LLM** | Ollama + llama3.2:1b | 0.11.2 | Fully local — zero API costs, complete privacy |
| **PDF Parsing** | Docling | 2.43+ | State-of-the-art scientific PDF extraction |
| **Orchestration** | Apache Airflow | 2.10 | Production DAG scheduling, retry, monitoring |
| **Caching** | Redis | 7 | Sub-millisecond exact-match cache with TTL |
| **Observability** | Langfuse | 3.0+ | End-to-end RAG pipeline tracing |
| **Agentic Framework** | LangGraph + LangChain | 1.1+ | State-based agent orchestration with decision nodes |
| **Bot Integration** | python-telegram-bot | 21.x | Conversational AI access on mobile |
| **UI** | Gradio | 4.0+ | Rapid ML interface with streaming support |
| **Containerization** | Docker Compose | Latest | Full stack orchestration, service networking |
| **Package Manager** | uv | Latest | 10-100x faster than pip, lockfile support |
| **Code Quality** | Ruff + MyPy | Latest | Fast linting, static type checking |

---

## 📅 Build Journey — Week by Week

### Week 1: Infrastructure Foundation ✅

**Goal:** Set up the complete production infrastructure stack.

**What I implemented:**
- Docker Compose stack with 6 services, health checks, and Docker networking
- FastAPI application with async lifespan management and dependency injection
- PostgreSQL database with SQLAlchemy models and connection pooling
- OpenSearch cluster configured for local development
- Apache Airflow with SQLite backend and admin user
- Ollama service with model volume persistence

**Key learnings:**
- Container networking — `postgres:5432` not `localhost:5432` inside Docker
- Health check chaining — API waits for postgres + opensearch healthy before starting
- FastAPI lifespan pattern — startup/shutdown hooks for service initialization
- Environment variable management with Pydantic BaseSettings nested settings

---

### Week 2: Data Ingestion Pipeline ✅

**Goal:** Build the automated pipeline that keeps the knowledge base fresh daily.

**What I implemented:**
- `ArxivClient` — async HTTP client with 3-second rate limiting, date filtering, retry logic
- `PDFParserService` — Docling integration for structured scientific PDF extraction
- `MetadataFetcher` — orchestrator coordinating arXiv → PDF → PostgreSQL pipeline
- `PaperRepository` — SQLAlchemy repository with upsert (no duplicate papers)
- Airflow DAG with 6 tasks, retry config, and weekend date handling fix
- `GET /api/v1/papers/` and `GET /api/v1/papers/{arxiv_id}` endpoints

**Key learnings:**
- Async API clients with `httpx` and timeout handling
- Airflow execution date vs actual date — backfill behavior
- Upsert patterns in SQLAlchemy for idempotent pipeline runs
- arXiv API quirks — 503 on weekends, `YYYYMMDD` date format
- Docling model loading — lazy initialization prevents startup OOM

---

### Week 3: BM25 Keyword Search ✅

**Goal:** Build the keyword search foundation every RAG system needs.

**What I implemented:**
- `OpenSearchClient` — factory-pattern service with health monitoring
- `QueryBuilder` — BM25 multi-field search (title 3x boost, abstract 2x)
- Strict index mapping for type safety
- `POST /api/v1/search/` — pagination, filtering, result highlighting
- RRF search pipeline configured for future hybrid search

**Key learnings:**
- BM25 algorithm — TF-IDF with document length normalization
- OpenSearch strict mappings — field types, analyzers, `keyword` vs `text`
- Query DSL — `multi_match`, `bool`, `filter`, `highlight`
- Why keyword first — exact terms, speed, interpretability before semantics

---

### Week 4: Hybrid Search + Jina Embeddings ✅

**Goal:** Add semantic understanding on top of the keyword foundation.

**What I implemented:**
- `TextChunker` — section-aware chunking (600 words target, 100-word overlap)
- `JinaEmbeddingsClient` — async Jina AI client for 1024-dimensional embeddings
- `HybridIndexingService` — chunk → embed → OpenSearch KNN index pipeline
- Manual RRF fusion: `score = Σ 1/(k + rank)` where k=60
- `POST /api/v1/hybrid-search/` — BM25, vector, or hybrid mode via single endpoint

**Key learnings:**
- Section-based chunking preserves document structure vs fixed-size splits
- Overlapping chunks prevent information loss at section boundaries
- RRF outperforms both BM25 and vector search individually
- Jina v3 is retrieval-optimized — asymmetric encode for query vs passage
- `knn_vector` field in OpenSearch — cosine similarity, 1024 dimensions

---

### Week 5: Complete RAG Pipeline ✅

**Goal:** Add the LLM that turns search results into intelligent answers.

**What I implemented:**
- `OllamaClient` — async LLM client with `generate` and streaming methods
- `RAGPromptBuilder` — system prompt + retrieved chunks + user query template
- `POST /api/v1/ask` — blocking RAG (15-20s), returns answer + sources
- `POST /api/v1/stream` — Server-Sent Events streaming (2-3s first token)
- `gradio_app.py` — web UI with search mode controls and source citations

**Key learnings:**
- RAG prompt engineering — system role, context window limits, answer constraints
- SSE in FastAPI — `StreamingResponse` with `text/event-stream` content type
- Source deduplication — multiple chunks from same paper → single citation
- Gradio layout — `gr.Blocks`, streaming output, advanced parameter accordion

---

### Week 6: Production Monitoring & Caching ✅

**Goal:** Make the system production-ready with observability and 150x+ speedup.

**What I implemented:**
- `LangfuseTracer` — span → generation → completion tracing for each RAG call
- `CacheClient` — SHA-256 parameter-aware cache keys in Redis
- Updated `/api/v1/ask` — cache hit check before pipeline, store result after
- TTL management (6-hour default), graceful cache failure handling
- Langfuse cloud dashboard for latency analytics and query patterns

**Key learnings:**
- Cache key design — hash(query + top_k + use_hybrid + model) for exact matching
- Graceful degradation — Redis failure → bypass cache, still serve response
- Langfuse v3 API — `trace()`, `span()`, `generation()` hierarchy
- Redis `allkeys-lru` eviction policy for bounded memory usage

---

### Week 7: Agentic RAG with LangGraph & Telegram Bot ✅

**Goal:** Transform the static RAG pipeline into an intelligent agent that reasons, adapts, and makes decisions — with mobile access via Telegram.

**What I implemented:**
- `AgenticRAGService` — LangGraph `StateGraph` with 6 specialized nodes wired via conditional edges
- **Guardrail node** — LLM scores query relevance (0-100), rejects out-of-scope queries below threshold
- **Retrieve node** — creates tool calls with attempt tracking and max-retry fallback
- **Grade documents node** — LLM evaluates retrieved document relevance, routes to generate or rewrite
- **Rewrite query node** — LLM rewrites query for better retrieval when docs are insufficient
- **Generate answer node** — produces final answer with citations from graded context
- `POST /api/v1/ask-agentic` — endpoint exposing full reasoning transparency
- `POST /api/v1/feedback` — Langfuse feedback submission for response scoring
- `TelegramBot` — conversational AI access with `/start`, `/help`, `/search` commands
- `get_langchain_model()` method on `OllamaClient` for LangChain/LangGraph integration
- `AgenticRAGDep` — FastAPI dependency injection creating service on-demand

**Key learnings:**
- LangGraph `StateGraph` with `context_schema` for type-safe dependency injection via `Runtime[Context]`
- Conditional edges — routing based on guardrail scores and grading results
- `ToolNode` + `tools_condition` — LangGraph's prebuilt tool execution pattern
- Structured output — `llm.with_structured_output(PydanticModel)` for typed LLM responses
- Graceful fallbacks — every node has try/except with heuristic fallback if LLM fails
- Agent transparency — reasoning steps tracked and returned to the user
- Telegram bot lifecycle — async start/stop tied to FastAPI lifespan, single-worker requirement

---

## 🚀 Quick Start

### Prerequisites
- **Docker Desktop** — [Install Guide](https://docs.docker.com/get-docker/)
- **Python 3.12+**
- **uv** — `pip install uv` or `curl -LsSf https://astral.sh/uv/install.sh | sh`
- **Jina AI API key** — Free at [jina.ai](https://jina.ai) (10M tokens, no credit card)
- **8GB+ RAM**, **20GB+ free disk**

### Setup

```bash
# 1. Clone the repository
git clone https://github.com/sherurox/arxiv-paper-curator-rag.git
cd arxiv-paper-curator-rag

# 2. Install Python dependencies
uv sync

# 3. Configure environment
cp .env.example .env
# Add your JINA_API_KEY to .env

# 4. Start core services (wait ~2 min for OpenSearch)
docker compose up -d postgres opensearch ollama

# 5. Start application services
docker compose up -d api airflow redis

# 6. Pull LLM model (1.3GB, one-time)
docker compose exec ollama ollama pull llama3.2:1b

# 7. Verify
docker compose ps
curl http://localhost:8000/api/v1/health
```

### Trigger Paper Ingestion

```bash
# Open http://localhost:8080 (admin/admin)
# Toggle ON arxiv_paper_ingestion
# Click ▶ to trigger
# All 6 tasks should turn green in ~10 minutes

# Verify papers stored
curl http://localhost:8000/api/v1/papers
```

### Launch Gradio UI

```bash
docker compose cp gradio_launcher.py api:/app/gradio_launcher.py
docker compose exec api python /app/gradio_launcher.py
# Open http://localhost:7861
```

### Access Points

| Interface | URL | Credentials |
|---|---|---|
| **Gradio UI** | http://localhost:7861 | — |
| **API Docs** | http://localhost:8000/docs | — |
| **Airflow** | http://localhost:8080 | admin / admin |
| **OpenSearch Dashboards** | http://localhost:5601 | — |
| **Langfuse** | https://cloud.langfuse.com | Your account |

---

## 🤖 Telegram Bot Setup Guide

The Telegram bot lets you ask research questions, search papers, and get AI-generated answers — all from your phone. Here's how to set it up from scratch.

### Step 1: Create a Telegram Bot via BotFather

1. Open **Telegram** (phone or desktop)
2. Search for **@BotFather** and start a chat
3. Send `/newbot`
4. When asked for a **name**, enter: `arXiv Paper Curator` (or any display name you like)
5. When asked for a **username**, enter something unique ending in `bot`, e.g.: `your_arxiv_rag_bot`
6. BotFather will respond with a **bot token** like: `7123456789:AAH...` — **copy this token**

### Step 2: Add the Token to Your Environment

Add these two lines to your `.env` file in the project root:

```bash
TELEGRAM__ENABLED=true
TELEGRAM__BOT_TOKEN=your_bot_token_here
```

Then add them to the `api` service environment in `compose.yml`. Find the `environment` section under the `api` service and add:

```yaml
      - TELEGRAM__ENABLED=true
      - TELEGRAM__BOT_TOKEN=your_bot_token_here
```

### Step 3: Ensure Single Worker Mode

The Telegram bot requires a single uvicorn worker (multiple workers each try to poll Telegram, causing conflicts). In your `Dockerfile`, make sure the CMD uses `--workers 1`:

```dockerfile
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
```

### Step 4: Rebuild and Start

```bash
# Rebuild the API container with the new config
docker compose build api

# Start all services
docker compose up -d

# Wait for startup and verify
sleep 15
docker compose logs api --tail=15
```

You should see these lines in the logs:

```
Telegram bot created successfully
Starting Telegram bot...
Application started
Telegram bot started successfully
Telegram bot started
API ready
```

### Step 5: Test Your Bot

Open Telegram and search for your bot username (the one you chose in Step 1). Then try:

| Command | What it does |
|---|---|
| `/start` | Shows welcome message with available commands |
| `/help` | Lists example questions and usage instructions |
| `/search transformer attention` | Searches indexed papers by keywords, returns top 5 with arXiv links |
| `What are the main challenges in reinforcement learning?` | Free-text question — runs the full RAG pipeline and returns an answer with source citations |

### Troubleshooting Telegram

| Issue | Solution |
|---|---|
| `Conflict: terminated by other getUpdates request` | Multiple workers running. Ensure `--workers 1` in Dockerfile and rebuild |
| `Telegram bot token not configured` | Check that `TELEGRAM__BOT_TOKEN` is set in both `.env` and `compose.yml` |
| Bot doesn't respond | Check `docker compose logs api --tail=30` for errors. Ensure `TELEGRAM__ENABLED=true` |
| Stale session after restart | Run: `curl "https://api.telegram.org/bot<YOUR_TOKEN>/deleteWebhook?drop_pending_updates=true"` then restart the API |

---

## 📂 Project Structure

```
arxiv-paper-curator-rag/
│
├── src/
│   ├── main.py                         # FastAPI app + lifespan
│   ├── config.py                       # All settings via Pydantic BaseSettings
│   ├── database.py                     # SQLAlchemy engine + session
│   ├── dependencies.py                 # FastAPI dependency injection
│   ├── exceptions.py                   # Custom exception classes
│   ├── middlewares.py                  # CORS middleware
│   │
│   ├── routers/
│   │   ├── ping.py                     # GET /ping, /health
│   │   ├── papers.py                   # GET /papers/
│   │   ├── search.py                   # POST /search/ (BM25)
│   │   ├── hybrid_search.py            # POST /hybrid-search/
│   │   ├── ask.py                      # POST /ask, /stream
│   │   └── agentic_ask.py             # POST /ask-agentic, /feedback (Week 7)
│   │
│   ├── services/
│   │   ├── arxiv/client.py             # arXiv API + rate limiting
│   │   ├── pdf_parser/                 # Docling PDF extraction
│   │   ├── opensearch/                 # BM25 + KNN search client
│   │   ├── embeddings/jina_client.py   # Jina AI 1024-dim embeddings
│   │   ├── indexing/                   # Chunking + hybrid indexer
│   │   ├── ollama/                     # LLM client + prompt builder
│   │   ├── cache/client.py             # Redis exact-match cache
│   │   ├── langfuse/                   # Tracing client
│   │   ├── agents/                     # LangGraph agentic RAG (Week 7)
│   │   │   ├── agentic_rag.py         # AgenticRAGService — graph builder
│   │   │   ├── state.py               # AgentState TypedDict
│   │   │   ├── context.py             # Runtime dependency injection
│   │   │   ├── config.py              # GraphConfig settings
│   │   │   ├── models.py              # Pydantic models (guardrail, grading)
│   │   │   ├── prompts.py             # All agent prompt templates
│   │   │   ├── tools.py               # Retriever tool wrapping OpenSearch
│   │   │   ├── factory.py             # make_agentic_rag_service
│   │   │   └── nodes/                 # Individual graph nodes
│   │   │       ├── guardrail_node.py  # Query scope validation
│   │   │       ├── retrieve_node.py   # Document retrieval with retry
│   │   │       ├── grade_documents_node.py  # LLM relevance grading
│   │   │       ├── rewrite_query_node.py    # Adaptive query rewriting
│   │   │       ├── generate_answer_node.py  # Answer generation
│   │   │       ├── out_of_scope_node.py     # Rejection handler
│   │   │       └── utils.py           # Message parsing helpers
│   │   ├── telegram/                   # Telegram bot (Week 7)
│   │   │   ├── bot.py                 # TelegramBot — handlers + RAG pipeline
│   │   │   └── factory.py            # make_telegram_service factory
│   │   └── metadata_fetcher.py         # Pipeline orchestrator
│   │
│   ├── models/paper.py                 # SQLAlchemy Paper model
│   ├── repositories/paper.py           # PaperRepository with upsert
│   ├── schemas/                        # Pydantic request/response models
│   └── db/                             # Database factory + interfaces
│
├── airflow/
│   └── dags/
│       ├── arxiv_paper_ingestion.py    # Main DAG definition
│       └── arxiv_ingestion/
│           └── tasks.py               # All task functions
│
├── notebooks/
│   ├── week1/week1_setup.ipynb
│   ├── week2/week2_arxiv_integration.ipynb
│   ├── week3/week3_opensearch.ipynb
│   ├── week4/week4_hybrid_search.ipynb
│   ├── week5/week5_complete_rag_system.ipynb
│   └── week6/week6_cache_testing.ipynb
│
├── assets/screenshots/                 # Demo screenshots
├── static/                             # Architecture diagrams
├── tests/                              # Unit + integration + API tests
├── gradio_launcher.py                  # Gradio UI launcher
├── compose.yml                         # Full Docker orchestration
├── Dockerfile                          # API container image
├── pyproject.toml                      # Python dependencies
└── .env.example                        # Environment template
```

---

## ⚙️ Configuration

```bash
# ── Application ─────────────────────────────────────
APP_VERSION=0.1.0
DEBUG=true
ENVIRONMENT=development

# ── PostgreSQL ───────────────────────────────────────
POSTGRES_DATABASE_URL=postgresql+psycopg2://rag_user:rag_password@localhost:5432/rag_db

# ── OpenSearch ───────────────────────────────────────
OPENSEARCH__HOST=http://localhost:9200
OPENSEARCH__INDEX_NAME=arxiv-papers
OPENSEARCH__VECTOR_DIMENSION=1024

# ── Ollama ───────────────────────────────────────────
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=llama3.2:1b
OLLAMA_TIMEOUT=300

# ── arXiv API ────────────────────────────────────────
ARXIV__RATE_LIMIT_DELAY=3.0
ARXIV__TIMEOUT_SECONDS=120
ARXIV__MAX_RESULTS=15
ARXIV__SEARCH_CATEGORY=cs.AI

# ── PDF Parser ───────────────────────────────────────
PDF_PARSER__MAX_PAGES=30
PDF_PARSER__MAX_FILE_SIZE_MB=20

# ── Chunking ─────────────────────────────────────────
CHUNKING__CHUNK_SIZE=600
CHUNKING__OVERLAP_SIZE=100

# ── Jina AI (Week 4+) — FREE at jina.ai ─────────────
JINA_API_KEY=jina_your_key_here

# ── Redis (Week 6+) ──────────────────────────────────
REDIS__HOST=localhost
REDIS__TTL_HOURS=6

# ── Langfuse (Week 6+) — FREE at cloud.langfuse.com ─
LANGFUSE__PUBLIC_KEY=pk-lf-your-key
LANGFUSE__SECRET_KEY=sk-lf-your-key
LANGFUSE__HOST=https://cloud.langfuse.com

# ── Telegram Bot (Week 7, optional) ──────────────────
TELEGRAM__ENABLED=false
TELEGRAM__BOT_TOKEN=
```

**Required by week:**

| Week | Required |
|---|---|
| 1-3 | Defaults only |
| 4+ | `JINA_API_KEY` |
| 6+ | `LANGFUSE__PUBLIC_KEY`, `LANGFUSE__SECRET_KEY` |
| 7 (optional) | `TELEGRAM__BOT_TOKEN`, `TELEGRAM__ENABLED=true` |

---

## 📊 Performance

| Metric | Value |
|---|---|
| Papers indexed | 30+ cs.AI papers |
| BM25 search latency | ~50ms |
| Hybrid search latency | ~200ms |
| RAG first response | ~15-20s |
| RAG cached response | **<100ms** |
| Cache speedup | **150-400x** |
| Agentic RAG response | ~30-60s (multi-step reasoning) |
| Guardrail rejection | ~5-10s |
| Telegram bot response | ~15-25s (full RAG pipeline) |
| PDF parse success rate | 80-90% |
| Embedding dimensions | 1024 (Jina v3) |
| arXiv fetch rate | ~20 papers/min |

---

## 🔧 Development Guide

```bash
# Makefile commands
make start         # Start all services
make stop          # Stop all services
make health        # Check service health
make test          # Run test suite
make test-cov      # Tests with coverage
make format        # Ruff format
make lint          # Ruff lint + mypy
make clean         # Full teardown

# Docker commands
docker compose up -d --build api        # Rebuild + restart API
docker compose logs api --tail=30       # API logs
docker compose exec api env             # Check env vars in container
docker compose exec ollama ollama list  # List LLM models

# Testing
uv run pytest tests/unit/
uv run pytest tests/api/
uv run pytest --cov=src --cov-report=html
```

---

## 🛠️ Troubleshooting

| Issue | Solution |
|---|---|
| Services won't start | Wait 2-3 min for OpenSearch. Check `docker compose logs opensearch` |
| API returns 500 | Check `docker compose logs api --tail=50` for startup errors |
| Jina 401 Unauthorized | Verify no duplicate key: `docker compose exec api env \| grep JINA` |
| Redis connection refused | Ensure `REDIS__HOST=redis` (not localhost) in compose env |
| Airflow UI not loading | `docker compose exec airflow rm -f /opt/airflow/airflow-webserver.pid` then restart |
| OpenSearch unhealthy | Increase Docker Desktop memory to 4GB+ |
| Agentic endpoint slow | Expected — multiple LLM calls (guardrail + grade + generate). Use `/ask` for cached fast responses |
| Telegram conflict error | Ensure `--workers 1` in Dockerfile. Run `deleteWebhook` API call then restart |

**Full reset:**
```bash
docker compose down --volumes
docker compose up --build -d
```

---

## 💰 Cost

| Component | Cost |
|---|---|
| Infrastructure (Docker) | **$0** |
| Jina AI embeddings | **$0** — 10M free tokens |
| LLM inference (Ollama) | **$0** — local CPU/GPU |
| Langfuse tracing | **$0** — free tier |
| Telegram bot | **$0** — free via BotFather |
| **Total** | **$0** |

---

> **📚 Course Attribution:** This project was built by following the
> [Mother of AI Project — Production Agentic RAG Course](https://github.com/jamwithai/production-agentic-rag-course)
> by [Shirin Khosravi Jam](https://www.linkedin.com/in/shirin-khosravi-jam/)
> and [Shantanu Ladhwe](https://www.linkedin.com/in/shantanuladhwe/).
> The system architecture, weekly progression, and tech stack decisions
> come from their course. This repo is my hands-on implementation,
> including my own code, notes, and learnings from completing all 7 weeks.

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

<div align="center">
  <p>Built by <a href="https://github.com/sherurox"><strong>Shreyas Khandale</strong></a></p>
  <p>MS Computer Science (AI Track) — Binghamton University</p>
  <p>
    <a href="https://github.com/sherurox">GitHub</a> •
    <a href="https://github.com/sherurox/arxiv-paper-curator-rag">Repository</a>
  </p>
  <br>
  <p>📚 Inspired from the <a href="https://github.com/jamwithai/production-agentic-rag-course"><strong>Mother of AI Project — Production Agentic RAG Course</strong></a></p>
  <p><em>Course created by <a href="https://www.linkedin.com/in/shirin-khosravi-jam/">Shirin Khosravi Jam</a> & <a href="https://www.linkedin.com/in/shantanuladhwe/">Shantanu Ladhwe</a></em></p>
</div>
