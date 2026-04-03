import logging
import os
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI

from src.config import get_settings
from src.db.factory import make_database
from src.routers import papers, ping
from src.routers.ask import ask_router, stream_router
from src.routers.hybrid_search import router as hybrid_search_router
from src.routers.search import router as search_router
from src.routers.agentic_ask import router as agentic_ask_router
from src.services.arxiv.factory import make_arxiv_client
from src.services.cache.factory import make_cache_client
from src.services.embeddings.factory import make_embeddings_service
from src.services.langfuse.factory import make_langfuse_tracer
from src.services.ollama.factory import make_ollama_client
from src.services.opensearch.factory import make_opensearch_client
from src.services.pdf_parser.factory import make_pdf_parser_service
from src.services.telegram.factory import make_telegram_service

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan for the API.
    """
    logger.info("Starting RAG API...")

    settings = get_settings()
    app.state.settings = settings

    database = make_database()
    app.state.database = database
    logger.info("Database connected")

    # Initialize OpenSearch client
    opensearch_client = make_opensearch_client()
    app.state.opensearch_client = opensearch_client

    if opensearch_client.health_check():
        logger.info("OpenSearch connected successfully")
        try:
            setup_results = opensearch_client.setup_indices(force=False)
            if setup_results.get("hybrid_index"):
                logger.info("OpenSearch index created")
            else:
                logger.info("OpenSearch index already exists")
        except Exception as e:
            logger.warning(f"OpenSearch index setup failed (search may be limited): {e}")

        try:
            stats = opensearch_client.client.count(index=opensearch_client.index_name)
            logger.info(f"OpenSearch ready: {stats['count']} documents indexed")
        except Exception:
            logger.info("OpenSearch index ready (stats unavailable)")
    else:
        logger.warning("OpenSearch connection failed - search features will be limited")

    # Initialize embeddings service
    app.state.embeddings_service = make_embeddings_service(settings)
    logger.info("Embeddings service initialized (Jina AI)")

    # Initialize Ollama client
    app.state.ollama_client = make_ollama_client()
    logger.info("Ollama client initialized")

    # Initialize Langfuse tracer
    app.state.langfuse_tracer = make_langfuse_tracer()
    logger.info("Langfuse tracer initialized")

    # Initialize Redis cache client (optional - API works without cache)
    try:
        app.state.cache_client = make_cache_client(settings)
        logger.info("Cache client initialized")
    except Exception as e:
        logger.warning(f"Cache client initialization failed (caching disabled): {e}")
        app.state.cache_client = None

    # Initialize services (kept for future endpoints and notebook demos)
    app.state.arxiv_client = make_arxiv_client()
    app.state.pdf_parser = make_pdf_parser_service()
    logger.info("Services initialized: arXiv API client, PDF parser, OpenSearch, Embeddings, Ollama")

    # Initialize Telegram bot (Week 7 - optional)
    telegram_service = make_telegram_service(
        opensearch_client=app.state.opensearch_client,
        embeddings_client=app.state.embeddings_service,
        ollama_client=app.state.ollama_client,
        cache_client=app.state.cache_client,
        langfuse_tracer=app.state.langfuse_tracer,
    )

    if telegram_service:
        app.state.telegram_service = telegram_service
        try:
            await telegram_service.start()
            logger.info("Telegram bot started")
        except Exception as e:
            logger.warning(f"Telegram bot failed to start: {e}")
    else:
        logger.info("Telegram bot disabled (set TELEGRAM__ENABLED=true and TELEGRAM__BOT_TOKEN to enable)")

    logger.info("API ready")
    yield

    # Cleanup
    telegram = getattr(app.state, "telegram_service", None)
    if telegram:
        try:
            await telegram.stop()
        except Exception as e:
            logger.warning(f"Error stopping Telegram bot: {e}")

    langfuse_tracer = getattr(app.state, "langfuse_tracer", None)
    if langfuse_tracer:
        langfuse_tracer.shutdown()
    database.teardown()
    logger.info("API shutdown complete")


app = FastAPI(
    title="arXiv Paper Curator API",
    description="Personal arXiv CS.AI paper curator with RAG capabilities",
    version=os.getenv("APP_VERSION", "0.1.0"),
    lifespan=lifespan,
)

# Include routers
app.include_router(ping.router, prefix="/api/v1")
app.include_router(papers.router, prefix="/api/v1")
app.include_router(search_router, prefix="/api/v1")
app.include_router(hybrid_search_router, prefix="/api/v1")
app.include_router(ask_router, prefix="/api/v1")
app.include_router(stream_router, prefix="/api/v1")
app.include_router(agentic_ask_router)  # Already has /api/v1 prefix


if __name__ == "__main__":
    uvicorn.run(app, port=8000, host="0.0.0.0")
