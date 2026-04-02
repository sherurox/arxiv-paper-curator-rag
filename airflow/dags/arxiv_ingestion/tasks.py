import asyncio
import logging
import sys
from datetime import datetime, timedelta
from functools import lru_cache
from typing import Any, Tuple

# Add project root to Python path for imports
sys.path.insert(0, "/opt/airflow")

# All imports at the top
from sqlalchemy import text
from src.db.factory import make_database
from src.services.arxiv.factory import make_arxiv_client
from src.services.metadata_fetcher import make_metadata_fetcher
from src.services.opensearch.factory import make_opensearch_client_fresh
from src.services.pdf_parser.factory import make_pdf_parser_service

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_cached_services() -> Tuple[Any, Any, Any, Any]:
    """
    Get cached service instances using lru_cache for automatic memoization.

    Returns:
        Tuple of (arxiv_client, pdf_parser, database, metadata_fetcher)
    """
    logger.info("Initializing services (cached with lru_cache)")

    # Initialize core services
    arxiv_client = make_arxiv_client()
    pdf_parser = make_pdf_parser_service()
    database = make_database()

    # Create metadata fetcher with dependencies
    metadata_fetcher = make_metadata_fetcher(arxiv_client, pdf_parser)

    logger.info("All services initialized and cached with lru_cache")
    return arxiv_client, pdf_parser, database, metadata_fetcher


async def run_paper_ingestion_pipeline(
    target_date: str,
    max_results: int = 5,
    process_pdfs: bool = True,
) -> dict:
    """
    Async wrapper for the paper ingestion pipeline.

    Args:
        target_date: Date to fetch papers for (YYYYMMDD format)
        max_results: Maximum number of papers to fetch
        process_pdfs: Whether to process PDFs

    Returns:
        Dictionary with processing results
    """
    _arxiv_client, _pdf_parser, database, metadata_fetcher = get_cached_services()

    with database.get_session() as session:
        return await metadata_fetcher.fetch_and_process_papers(
            max_results=max_results,
            from_date=target_date,
            to_date=target_date,
            process_pdfs=process_pdfs,
            store_to_db=True,
            db_session=session,
        )


def setup_environment():
    """Setup environment and verify dependencies."""
    logger.info("Setting up environment for arXiv paper ingestion")

    try:
        # Get cached services (initialized once)
        arxiv_client, _pdf_parser, database, _metadata_fetcher = get_cached_services()

        # Test database connection
        with database.get_session() as session:
            session.execute(text("SELECT 1"))
            logger.info("Database connection verified")

        logger.info(f"arXiv client ready: {arxiv_client.base_url}")
        logger.info("PDF parser service ready (Docling models cached)")

        return {"status": "success", "message": "Environment setup completed"}

    except Exception as e:
        error_msg = f"Environment setup failed: {str(e)}"
        logger.error(error_msg)
        raise Exception(error_msg)


def fetch_daily_papers(**context):
    """
    Fetch CS.AI papers from the last 24 hours.

    This function:
    1. Calculates the date range for yesterday
    2. Fetches papers using the MetadataFetcher
    3. Returns processing statistics
    """
    logger.info("Starting daily arXiv paper fetch")

    try:
        # Calculate date range (yesterday - execution_date - 1)
        execution_date = context["ds"]  # YYYY-MM-DD format
        execution_dt = datetime.strptime(execution_date, "%Y-%m-%d")
        target_dt = execution_dt - timedelta(days=1)  # Get papers from day before
        # Skip weekends - arXiv has no submissions on Saturday/Sunday
        if target_dt.weekday() == 6:  # Sunday -> go to Friday
            target_dt = target_dt - timedelta(days=2)
        elif target_dt.weekday() == 5:  # Saturday -> go to Friday
            target_dt = target_dt - timedelta(days=1)
        target_date = target_dt.strftime("%Y%m%d")

        logger.info(f"Fetching papers for date: {target_date}")

        # Execute paper ingestion pipeline
        results = asyncio.run(
            run_paper_ingestion_pipeline(
                target_date=target_date,
                max_results=10,
                process_pdfs=True,
            )
        )
        logger.info(f"Daily paper fetch completed: {results}")

        # Store results for downstream tasks
        context["task_instance"].xcom_push(key="fetch_results", value=results)

        return results

    except Exception as e:
        error_msg = f"Daily paper fetch failed: {str(e)}"
        logger.error(error_msg)
        raise Exception(error_msg)


def process_failed_pdfs(**context):
    """
    Retry processing of PDFs that failed in the main fetch task.

    This function:
    1. Gets failed PDF list from the main task
    2. Retries processing with different settings
    3. Reports final success/failure statistics
    """
    logger.info("Processing failed PDFs")

    try:
        fetch_results = context["task_instance"].xcom_pull(task_ids="fetch_daily_papers", key="fetch_results")

        if not fetch_results or not fetch_results.get("errors"):
            logger.info("No failed PDFs to retry")
            return {"status": "skipped", "message": "No failures to retry"}

        logger.info(f"Found {len(fetch_results['errors'])} errors to investigate")

        for error in fetch_results["errors"]:
            # TODO: Implement retry logic
            logger.warning(f"Error to investigate: {error}")

        return {
            "status": "analyzed",
            "errors_logged": len(fetch_results["errors"]),
            "message": "Errors logged for investigation",
        }

    except Exception as e:
        error_msg = f"Failed PDF processing error: {str(e)}"
        logger.error(error_msg)
        raise Exception(error_msg)


def create_opensearch_placeholders(**context):
    """
    Index papers into OpenSearch for BM25 keyword search (Week 3).

    This task:
    1. Gets successfully stored papers from the fetch task
    2. Indexes them as text chunks into OpenSearch for BM25 search
    """
    logger.info("Indexing papers into OpenSearch for BM25 search (Week 3)")

    try:
        from src.models.paper import Paper

        fetch_results = context["task_instance"].xcom_pull(task_ids="fetch_daily_papers", key="fetch_results")

        if not fetch_results:
            logger.warning("No fetch results available for OpenSearch indexing")
            return {"status": "skipped", "message": "No papers to index"}

        papers_stored = fetch_results.get("papers_stored", 0)
        if papers_stored == 0:
            logger.info("No papers were stored - skipping OpenSearch indexing")
            return {"status": "skipped", "papers_indexed": 0}

        _arxiv_client, _pdf_parser, database, _metadata_fetcher = get_cached_services()
        opensearch_client = make_opensearch_client_fresh()

        with database.get_session() as session:
            papers = (
                session.query(Paper)
                .order_by(Paper.created_at.desc())
                .limit(papers_stored)
                .all()
            )

            if not papers:
                logger.info("No papers found in database for indexing")
                return {"status": "skipped", "papers_indexed": 0}

            indexed = 0
            failed = 0

            for paper in papers:
                try:
                    categories = paper.categories if isinstance(paper.categories, list) else (
                        [paper.categories] if paper.categories else []
                    )
                    chunk_text = f"{paper.title}\n\n{paper.abstract or ''}"
                    chunk_data = {
                        "arxiv_id": paper.arxiv_id,
                        "paper_id": str(paper.id),
                        "chunk_index": 0,
                        "chunk_text": chunk_text,
                        "chunk_word_count": len(chunk_text.split()),
                        "start_char": 0,
                        "end_char": len(chunk_text),
                        "title": paper.title,
                        "authors": paper.authors or "",
                        "abstract": paper.abstract or "",
                        "categories": categories,
                        "published_date": paper.published_date.isoformat() if paper.published_date else None,
                    }

                    opensearch_client.client.index(
                        index=opensearch_client.index_name,
                        body=chunk_data,
                        refresh=True,
                    )
                    indexed += 1
                    logger.info(f"Indexed paper: {paper.arxiv_id}")

                except Exception as e:
                    logger.error(f"Failed to index paper {paper.arxiv_id}: {e}")
                    failed += 1

        result = {
            "status": "success",
            "papers_indexed": indexed,
            "papers_failed": failed,
            "message": f"Week 3: {indexed} papers indexed into OpenSearch for BM25 search",
        }

        logger.info(f"OpenSearch BM25 indexing complete: {result}")
        context["task_instance"].xcom_push(key="opensearch_results", value=result)

        return result

    except Exception as e:
        error_msg = f"OpenSearch BM25 indexing failed: {str(e)}"
        logger.error(error_msg)
        raise Exception(error_msg)


def generate_daily_report(**context):
    """
    Generate a daily processing report.

    This function:
    1. Collects results from all tasks
    2. Generates summary statistics
    3. Logs the daily report
    """
    logger.info("Generating daily processing report")

    try:
        fetch_results = context["task_instance"].xcom_pull(task_ids="fetch_daily_papers", key="fetch_results")

        failed_pdf_results = context["task_instance"].xcom_pull(task_ids="process_failed_pdfs")

        opensearch_results = context["task_instance"].xcom_pull(task_ids="create_opensearch_placeholders")

        report = {
            "date": context["ds"],
            "execution_time": datetime.now().isoformat(),
            "papers": {
                "fetched": fetch_results.get("papers_fetched", 0) if fetch_results else 0,
                "pdfs_downloaded": fetch_results.get("pdfs_downloaded", 0) if fetch_results else 0,
                "pdfs_parsed": fetch_results.get("pdfs_parsed", 0) if fetch_results else 0,
                "stored": fetch_results.get("papers_stored", 0) if fetch_results else 0,
            },
            "processing": {
                "processing_time_seconds": fetch_results.get("processing_time", 0) if fetch_results else 0,
                "errors": len(fetch_results.get("errors", [])) if fetch_results else 0,
                "failed_pdf_retries": failed_pdf_results.get("errors_logged", 0) if failed_pdf_results else 0,
            },
            "opensearch": {
                "papers_indexed": opensearch_results.get("papers_indexed", 0) if opensearch_results else 0,
                "status": opensearch_results.get("status", "unknown") if opensearch_results else "unknown",
            },
        }

        logger.info("=== DAILY ARXIV PROCESSING REPORT ===")
        logger.info(f"Date: {report['date']}")
        logger.info(f"Papers fetched: {report['papers']['fetched']}")
        logger.info(f"PDFs downloaded: {report['papers']['pdfs_downloaded']}")
        logger.info(f"PDFs parsed: {report['papers']['pdfs_parsed']}")
        logger.info(f"Papers stored: {report['papers']['stored']}")
        logger.info(f"Processing time: {report['processing']['processing_time_seconds']:.1f}s")
        logger.info(f"Errors encountered: {report['processing']['errors']}")
        logger.info(f"OpenSearch papers indexed: {report['opensearch']['papers_indexed']}")
        logger.info("=== END REPORT ===")

        return report

    except Exception as e:
        error_msg = f"Report generation failed: {str(e)}"
        logger.error(error_msg)
        raise Exception(error_msg)
