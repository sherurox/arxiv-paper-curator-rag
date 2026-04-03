from typing import List

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class DefaultSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
        frozen=True,
        env_nested_delimiter="__",
    )


class ArxivSettings(DefaultSettings):
    """arXiv API client settings."""

    base_url: str = "https://export.arxiv.org/api/query"
    namespaces: dict = Field(
        default={
            "atom": "http://www.w3.org/2005/Atom",
            "opensearch": "http://a9.com/-/spec/opensearch/1.1/",
            "arxiv": "http://arxiv.org/schemas/atom",
        }
    )
    pdf_cache_dir: str = "./data/arxiv_pdfs"
    rate_limit_delay: float = 3.0  # seconds between requests
    timeout_seconds: int = 30
    max_results: int = 100
    search_category: str = "cs.AI"  # Default category to search


class PDFParserSettings(DefaultSettings):
    """PDF parser service settings."""

    max_pages: int = 30
    max_file_size_mb: int = 20
    do_ocr: bool = False
    do_table_structure: bool = True


class ChunkingSettings(DefaultSettings):
    """Text chunking settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="CHUNKING__",
        extra="ignore",
        frozen=True,
        env_nested_delimiter="__",
    )

    chunk_size: int = 600  # Target words per chunk
    overlap_size: int = 100  # Words to overlap between chunks
    min_chunk_size: int = 100  # Minimum words for a valid chunk
    section_based: bool = True  # Use section-based chunking when available


class OpenSearchSettings(DefaultSettings):
    """OpenSearch client settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="OPENSEARCH__",
        extra="ignore",
        frozen=True,
        env_nested_delimiter="__",
    )

    host: str = "http://localhost:9200"
    index_name: str = "arxiv-papers"
    chunk_index_suffix: str = "chunks"  # Creates single hybrid index: {index_name}-{suffix}
    max_text_size: int = 1000000

    # Vector search settings (used in Week 5+)
    vector_dimension: int = 1024
    vector_space_type: str = "cosinesimil"

    # Hybrid search settings (used in Week 5+)
    rrf_pipeline_name: str = "hybrid-rrf-pipeline"
    hybrid_search_size_multiplier: int = 2


class LangfuseSettings(DefaultSettings):
    """Langfuse tracing settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="LANGFUSE__",
        extra="ignore",
        frozen=True,
        env_nested_delimiter="__",
    )

    public_key: str = ""
    secret_key: str = ""
    host: str = "http://localhost:3000"
    enabled: bool = True
    flush_at: int = 15
    flush_interval: float = 1.0
    max_retries: int = 3
    timeout: int = 30
    debug: bool = False


class RedisSettings(DefaultSettings):
    """Redis cache settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="REDIS__",
        extra="ignore",
        frozen=True,
        env_nested_delimiter="__",
    )

    host: str = "localhost"
    port: int = 6379
    password: str = ""
    db: int = 0
    decode_responses: bool = True
    socket_timeout: int = 30
    socket_connect_timeout: int = 30
    ttl_hours: int = 6


class TelegramSettings(DefaultSettings):
    """Telegram bot settings (Week 7)."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="TELEGRAM__",
        extra="ignore",
        frozen=True,
        env_nested_delimiter="__",
    )

    bot_token: str = ""
    enabled: bool = False


class Settings(DefaultSettings):
    """Application settings."""

    app_version: str = "0.1.0"
    debug: bool = True
    environment: str = "development"
    service_name: str = "rag-api"

    # PostgreSQL configuration
    postgres_database_url: str = "postgresql://rag_user:rag_password@localhost:5432/rag_db"
    postgres_echo_sql: bool = False
    postgres_pool_size: int = 20
    postgres_max_overflow: int = 0

    # Jina AI embeddings
    jina_api_key: str = ""

    # OpenSearch configuration
    opensearch: OpenSearchSettings = Field(default_factory=OpenSearchSettings)

    # Chunking configuration
    chunking: ChunkingSettings = Field(default_factory=ChunkingSettings)

    # Ollama configuration (used in Week 1 notebook)
    ollama_host: str = "http://localhost:11434"
    ollama_models: List[str] = Field(default=["llama3.2:1b"])
    ollama_default_model: str = "llama3.2:1b"
    ollama_timeout: int = 300  # 5 minutes for LLM operations

    # arXiv settings
    arxiv: ArxivSettings = Field(default_factory=ArxivSettings)

    # PDF parser settings
    pdf_parser: PDFParserSettings = Field(default_factory=PDFParserSettings)

    # Langfuse tracing
    langfuse: LangfuseSettings = Field(default_factory=LangfuseSettings)

    # Redis caching
    redis: RedisSettings = Field(default_factory=RedisSettings)

    # Telegram bot (Week 7)
    telegram: TelegramSettings = Field(default_factory=TelegramSettings)

    @field_validator("ollama_models", mode="before")
    @classmethod
    def parse_ollama_models(cls, v):
        """Parse comma-separated string into list of models."""
        if isinstance(v, str):
            return [model.strip() for model in v.split(",") if model.strip()]
        return v


def get_settings() -> Settings:
    """Get application settings."""
    return Settings()
