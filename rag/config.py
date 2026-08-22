import os
from pathlib import Path
from pydantic import BaseModel, Field
from typing import Optional, Literal

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

class Settings(BaseModel):
    # App Settings
    app_name: str = "Voice-Enabled RAG System (HH Goa 2026)"
    app_version: str = "1.0.0"
    debug: bool = False
    port: int = 8000
    host: str = "0.0.0.0"
    
    # STT Provider Settings
    stt_provider: Literal["sarvam", "elevenlabs", "local"] = "sarvam"
    sarvam_api_key: Optional[str] = Field(default_factory=lambda: os.getenv("SARVAM_API_KEY", ""))
    elevenlabs_api_key: Optional[str] = Field(default_factory=lambda: os.getenv("ELEVENLABS_API_KEY", ""))
    sarvam_model: str = "saaras:v1"
    sarvam_language_code: str = "hi-IN"  # hi-IN, en-IN, or auto
    elevenlabs_model: str = "scribe_v1"
    
    # LLM & Generation Settings
    llm_provider: Literal["fast_local", "gemini", "groq", "openai"] = "fast_local"
    gemini_api_key: Optional[str] = Field(default_factory=lambda: os.getenv("GEMINI_API_KEY", ""))
    groq_api_key: Optional[str] = Field(default_factory=lambda: os.getenv("GROQ_API_KEY", ""))
    openai_api_key: Optional[str] = Field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    
    # Embedding Model Settings
    embedding_model_name: str = "BAAI/bge-small-en-v1.5"  # Fast ONNX local embedding
    embedding_dim: int = 384
    
    # Retrieval & RRF Settings
    default_chunking_strategy: str = "hierarchical"  # hierarchical, semantic, metadata_aware, sentence_window
    top_k: int = 3
    hybrid_dense_weight: float = 0.7
    hybrid_sparse_weight: float = 0.3
    rrf_k: int = 60
    
    # Latency Budget & Caching
    latency_target_ms: float = 200.0
    enable_semantic_cache: bool = True
    semantic_cache_similarity_threshold: float = 0.95
    semantic_cache_max_size: int = 1000
    
    # Guardrails Thresholds
    guardrail_input_safety_enabled: bool = True
    guardrail_ood_enabled: bool = True
    guardrail_retrieval_min_similarity: float = 0.38
    guardrail_grounding_min_confidence: float = 0.50
    refusal_message: str = (
        "I am unable to answer this question because it is either outside the indexed "
        "dataset domain or cannot be safely verified against the retrieved context."
    )

settings = Settings()
