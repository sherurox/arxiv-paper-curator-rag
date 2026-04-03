from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class GuardrailScoring(BaseModel):
    """Scoring result of a user query for guardrail validation."""

    score: int = Field(ge=0, le=100, description="Relevance score between 0 and 100")
    reason: str = Field(description="Brief reason for the score")


class GradeDocuments(BaseModel):
    """Binary score for document relevance check."""

    binary_score: Literal["yes", "no"] = Field(description="Document relevance: 'yes' or 'no'")
    reasoning: str = Field(default="", description="Explanation for the decision")


class SourceItem(BaseModel):
    """Source item from retrieved documents."""

    arxiv_id: str = Field(description="arXiv paper ID")
    title: str = Field(description="Paper title")
    authors: List[str] = Field(default_factory=list, description="List of authors")
    url: str = Field(description="Link to paper")
    relevance_score: float = Field(default=0.0, description="Relevance score from search")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "arxiv_id": self.arxiv_id,
            "title": self.title,
            "authors": self.authors,
            "url": self.url,
            "relevance_score": self.relevance_score,
        }


class ToolArtefact(BaseModel):
    """Artifact returned by tool calls with metadata."""

    tool_name: str = Field(description="Name of the tool")
    tool_call_id: str = Field(description="Unique tool call ID")
    content: Any = Field(description="Tool result content")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class RoutingDecision(BaseModel):
    """Routing decision for graph navigation."""

    route: Literal["retrieve", "out_of_scope", "generate_answer", "rewrite_query"] = Field(
        description="Next node to route to"
    )
    reason: str = Field(default="", description="Reason for routing decision")


class GradingResult(BaseModel):
    """Result of document grading with details."""

    document_id: str = Field(description="Document identifier")
    is_relevant: bool = Field(description="Relevance flag")
    score: float = Field(default=0.0, description="Relevance score")
    reasoning: str = Field(default="", description="Grading reasoning")


class ReasoningStep(BaseModel):
    """A reasoning step in the agent workflow."""

    step_name: str = Field(description="Name of the reasoning step")
    description: str = Field(description="Human-readable description")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Step metadata")
