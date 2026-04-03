import json
import logging
import time

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from src.dependencies import CacheDep, EmbeddingsDep, LangfuseDep, OllamaDep, OpenSearchDep
from src.schemas.api.ask import AskRequest, AskResponse

logger = logging.getLogger(__name__)

# Two separate routers - one for regular ask, one for streaming
ask_router = APIRouter(tags=["ask"])
stream_router = APIRouter(tags=["stream"])


@ask_router.post("/ask", response_model=AskResponse)
async def ask_question(
    request: AskRequest,
    opensearch_client: OpenSearchDep,
    embeddings_service: EmbeddingsDep,
    ollama_client: OllamaDep,
    langfuse_tracer: LangfuseDep,
    cache_client: CacheDep,
) -> AskResponse:
    """RAG endpoint: retrieve relevant chunks and generate an answer with Ollama."""
    try:
        # Check exact cache first
        if cache_client:
            try:
                cached_response = await cache_client.find_cached_response(request)
                if cached_response:
                    logger.info("Returning cached response for exact query match")
                    return cached_response
            except Exception as e:
                logger.warning(f"Cache check failed, proceeding with normal flow: {e}")

        # Generate query embedding for hybrid search
        query_embedding = None
        if request.use_hybrid:
            with langfuse_tracer.start_span(
                name="query_embedding",
                input_data={"query": request.query, "query_length": len(request.query)},
            ) as embedding_span:
                try:
                    query_embedding = await embeddings_service.embed_query(request.query)
                    logger.info("Generated query embedding for hybrid search")
                    langfuse_tracer.update_span(embedding_span, output={"success": True})
                except Exception as e:
                    logger.warning(f"Failed to generate embeddings, falling back to BM25: {e}")
                    langfuse_tracer.update_span(embedding_span, output={"success": False, "error": str(e)})

        # Search for relevant chunks
        with langfuse_tracer.start_span(
            name="search_retrieval",
            input_data={"query": request.query, "top_k": request.top_k},
        ) as search_span:
            search_results = opensearch_client.search_unified(
                query=request.query,
                query_embedding=query_embedding,
                size=request.top_k,
                from_=0,
                categories=request.categories,
                use_hybrid=request.use_hybrid and query_embedding is not None,
                min_score=0.0,
            )

            chunks = []
            sources_set = set()
            arxiv_ids = []

            for hit in search_results.get("hits", []):
                arxiv_id = hit.get("arxiv_id", "")
                chunks.append({
                    "arxiv_id": arxiv_id,
                    "chunk_text": hit.get("chunk_text", hit.get("abstract", "")),
                })
                if arxiv_id:
                    arxiv_ids.append(arxiv_id)
                    arxiv_id_clean = arxiv_id.split("v")[0] if "v" in arxiv_id else arxiv_id
                    sources_set.add(f"https://arxiv.org/pdf/{arxiv_id_clean}.pdf")

            langfuse_tracer.update_span(
                search_span,
                output={
                    "chunks_returned": len(chunks),
                    "unique_papers": len(set(arxiv_ids)),
                    "total_hits": search_results.get("total", 0),
                },
            )

        if not chunks:
            return AskResponse(
                query=request.query,
                answer="I couldn't find any relevant information in the papers to answer your question.",
                sources=[],
                chunks_used=0,
                search_mode="bm25" if not request.use_hybrid else "hybrid",
            )

        # Generate answer with Ollama
        start_gen = time.time()
        with langfuse_tracer.start_generation(
            name="llm_generation",
            model=request.model,
            input_data={"query": request.query, "chunks_count": len(chunks)},
        ) as gen_span:
            rag_response = await ollama_client.generate_rag_answer(
                query=request.query,
                chunks=chunks,
                model=request.model,
            )
            answer = rag_response.get("answer", "Unable to generate answer")
            langfuse_tracer.update_generation(
                gen_span,
                output=answer,
                usage_metadata={"latency_ms": round((time.time() - start_gen) * 1000, 2)},
            )

        response = AskResponse(
            query=request.query,
            answer=answer,
            sources=list(sources_set),
            chunks_used=len(chunks),
            search_mode="bm25" if not request.use_hybrid else "hybrid",
        )

        # Store response in exact match cache
        if cache_client:
            try:
                await cache_client.store_response(request, response)
            except Exception as e:
                logger.warning(f"Failed to store response in cache: {e}")

        if langfuse_tracer:
            langfuse_tracer.flush()

        return response

    except Exception as e:
        logger.error(f"Error processing request: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@stream_router.post("/stream")
async def ask_question_stream(
    request: AskRequest,
    opensearch_client: OpenSearchDep,
    embeddings_service: EmbeddingsDep,
    ollama_client: OllamaDep,
    langfuse_tracer: LangfuseDep,
    cache_client: CacheDep,
) -> StreamingResponse:
    """Streaming RAG endpoint."""

    async def generate_stream():
        try:
            # Check exact cache first
            if cache_client:
                try:
                    cached_response = await cache_client.find_cached_response(request)
                    if cached_response:
                        logger.info("Returning cached response for exact streaming query match")

                        metadata_response = {
                            "sources": cached_response.sources,
                            "chunks_used": cached_response.chunks_used,
                            "search_mode": cached_response.search_mode,
                        }
                        yield f"data: {json.dumps(metadata_response)}\n\n"

                        for word in cached_response.answer.split():
                            yield f"data: {json.dumps({'chunk': word + ' '})}\n\n"

                        yield f"data: {json.dumps({'answer': cached_response.answer, 'done': True})}\n\n"
                        return
                except Exception as e:
                    logger.warning(f"Cache check failed, proceeding with normal flow: {e}")

            # Generate query embedding for hybrid search
            query_embedding = None
            if request.use_hybrid:
                try:
                    query_embedding = await embeddings_service.embed_query(request.query)
                    logger.info("Generated query embedding for hybrid search")
                except Exception as e:
                    logger.warning(f"Failed to generate embeddings, falling back to BM25: {e}")

            # Search for relevant chunks
            search_results = opensearch_client.search_unified(
                query=request.query,
                query_embedding=query_embedding,
                size=request.top_k,
                from_=0,
                categories=request.categories,
                use_hybrid=request.use_hybrid and query_embedding is not None,
                min_score=0.0,
            )

            chunks = []
            sources_set = set()

            for hit in search_results.get("hits", []):
                arxiv_id = hit.get("arxiv_id", "")
                chunks.append({
                    "arxiv_id": arxiv_id,
                    "chunk_text": hit.get("chunk_text", hit.get("abstract", "")),
                })
                if arxiv_id:
                    arxiv_id_clean = arxiv_id.split("v")[0] if "v" in arxiv_id else arxiv_id
                    sources_set.add(f"https://arxiv.org/pdf/{arxiv_id_clean}.pdf")

            if not chunks:
                yield f"data: {json.dumps({'answer': 'No relevant information found.', 'sources': [], 'done': True})}\n\n"
                return

            sources = list(sources_set)
            search_mode = "bm25" if not request.use_hybrid else "hybrid"

            # Send metadata first
            metadata_response = {"sources": sources, "chunks_used": len(chunks), "search_mode": search_mode}
            yield f"data: {json.dumps(metadata_response)}\n\n"

            # Stream generation
            full_response = ""
            async for chunk in ollama_client.generate_rag_answer_stream(
                query=request.query,
                chunks=chunks,
                model=request.model,
            ):
                if chunk.get("response"):
                    text_chunk = chunk["response"]
                    full_response += text_chunk
                    yield f"data: {json.dumps({'chunk': text_chunk})}\n\n"

                if chunk.get("done", False):
                    yield f"data: {json.dumps({'answer': full_response, 'done': True})}\n\n"
                    break

            # Store response in cache
            if cache_client and full_response:
                try:
                    response_to_cache = AskResponse(
                        query=request.query,
                        answer=full_response,
                        sources=sources,
                        chunks_used=len(chunks),
                        search_mode=search_mode,
                    )
                    await cache_client.store_response(request, response_to_cache)
                except Exception as e:
                    logger.warning(f"Failed to store streaming response in cache: {e}")

        except Exception as e:
            logger.error(f"Streaming error: {e}")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(
        generate_stream(),
        media_type="text/plain",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )
