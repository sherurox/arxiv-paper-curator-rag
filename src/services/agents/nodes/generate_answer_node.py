import logging
import time
from typing import Dict, List

from langchain_core.messages import AIMessage
from langgraph.runtime import Runtime

from ..context import Context
from ..prompts import GENERATE_ANSWER_PROMPT
from ..state import AgentState
from .utils import get_latest_context, get_latest_query

logger = logging.getLogger(__name__)


async def ainvoke_generate_answer_step(
    state: AgentState,
    runtime: Runtime[Context],
) -> Dict[str, List[AIMessage]]:
    """Generate final answer using retrieved documents."""
    logger.info("NODE: generate_answer")
    start_time = time.time()

    question = get_latest_query(state["messages"])
    context = get_latest_context(state["messages"])
    sources_count = len(state.get("relevant_sources", []))

    if not context:
        context = "No relevant documents found."
        logger.warning("No context available for answer generation")

    # Create span
    span = None
    if runtime.context.langfuse_enabled and runtime.context.trace:
        try:
            span = runtime.context.langfuse_tracer.create_span(
                trace=runtime.context.trace,
                name="answer_generation",
                input_data={"query": question, "context_length": len(context), "sources_count": sources_count},
                metadata={"node": "generate_answer", "model": runtime.context.model_name, "temperature": runtime.context.temperature},
            )
        except Exception as e:
            logger.warning(f"Failed to create span for generate_answer node: {e}")

    try:
        answer_prompt = GENERATE_ANSWER_PROMPT.format(context=context, question=question)

        llm = runtime.context.ollama_client.get_langchain_model(
            model=runtime.context.model_name,
            temperature=runtime.context.temperature,
        )

        logger.info("Invoking LLM for answer generation")
        response = await llm.ainvoke(answer_prompt)

        answer = response.content if hasattr(response, "content") else str(response)
        logger.info(f"Generated answer of length: {len(answer)} characters")

        if span:
            execution_time = (time.time() - start_time) * 1000
            runtime.context.langfuse_tracer.end_span(
                span,
                output={"answer_length": len(answer), "sources_used": sources_count},
                metadata={"execution_time_ms": execution_time, "context_length": len(context)},
            )

    except Exception as e:
        logger.error(f"LLM answer generation failed: {e}, falling back to error message")
        answer = f"I apologize, but I encountered an error while generating the answer: {str(e)}\n\nPlease try again or rephrase your question."

        if span:
            execution_time = (time.time() - start_time) * 1000
            runtime.context.langfuse_tracer.update_span(
                span,
                output={"error": str(e), "fallback": True},
                metadata={"execution_time_ms": execution_time},
                level="ERROR",
            )
            runtime.context.langfuse_tracer.end_span(span)

    return {"messages": [AIMessage(content=answer)]}
