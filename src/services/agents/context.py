from dataclasses import dataclass
from typing import Any, Optional

from src.services.embeddings.jina_client import JinaEmbeddingsClient
from src.services.langfuse.client import LangfuseTracer
from src.services.ollama.client import OllamaClient
from src.services.opensearch.client import OpenSearchClient


@dataclass
class Context:
    """Runtime context for agent dependencies.

    Contains immutable dependencies that nodes need but don't modify.
    """

    ollama_client: OllamaClient
    opensearch_client: OpenSearchClient
    embeddings_client: JinaEmbeddingsClient
    langfuse_tracer: Optional[LangfuseTracer] = None
    trace: Optional[Any] = None
    langfuse_enabled: bool = False
    model_name: str = "llama3.2:1b"
    temperature: float = 0.0
    top_k: int = 3
    max_retrieval_attempts: int = 2
    guardrail_threshold: int = 60
