from typing import Annotated, Any, Dict, List, Optional, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

from .models import GradingResult, GuardrailScoring, RoutingDecision, SourceItem, ToolArtefact


class AgentState(TypedDict):
    """State class for the Agentic RAG workflow.

    TypedDict-based state following LangGraph best practices.
    Tracks all data that needs to be passed between nodes.
    """

    messages: Annotated[list[AnyMessage], add_messages]
    original_query: Optional[str]
    rewritten_query: Optional[str]
    retrieval_attempts: int
    guardrail_result: Optional[GuardrailScoring]
    routing_decision: Optional[RoutingDecision]
    sources: Optional[Dict[str, Any]]
    relevant_sources: List[SourceItem]
    relevant_tool_artefacts: Optional[List[ToolArtefact]]
    grading_results: List[GradingResult]
    metadata: Dict[str, Any]
