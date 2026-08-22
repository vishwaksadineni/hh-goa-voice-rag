from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Literal

class DocumentPassage(BaseModel):
    doc_id: str
    query_id: Optional[int] = None
    query_type: Optional[str] = None
    text: str
    translated_text: Optional[str] = None
    source_lang: str = "en"
    target_lang: Optional[str] = "hi"
    is_gold: bool = False
    metadata: Dict[str, Any] = Field(default_factory=dict)

class Chunk(BaseModel):
    chunk_id: str
    doc_id: str
    text: str
    parent_text: Optional[str] = None
    strategy: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    token_count: int = 0
    embedding: Optional[List[float]] = None

class RetrievalResult(BaseModel):
    chunk: Chunk
    score: float
    dense_score: float = 0.0
    sparse_score: float = 0.0
    rank: int = 0

class LatencyBreakdown(BaseModel):
    stt_ms: float = 0.0
    input_guard_ms: float = 0.0
    cache_lookup_ms: float = 0.0
    embedding_ms: float = 0.0
    retrieval_ms: float = 0.0
    generation_ttft_ms: float = 0.0
    generation_total_ms: float = 0.0
    grounding_check_ms: float = 0.0
    total_ms: float = 0.0

class GuardrailVerdict(BaseModel):
    passed: bool
    input_safe: bool = True
    in_domain: bool = True
    retrieval_sufficient: bool = True
    grounded: bool = True
    safety_flags: List[str] = Field(default_factory=list)
    ood_score: float = 0.0
    retrieval_confidence: float = 0.0
    grounding_confidence: float = 0.0
    action: Literal["allow", "refuse", "retry"] = "allow"
    reason: Optional[str] = None

class VoiceRAGRequest(BaseModel):
    query_text: Optional[str] = None
    audio_base64: Optional[str] = None
    audio_format: Optional[str] = "webm"
    language_code: Optional[str] = "hi-IN"
    stt_provider: Optional[str] = None
    chunking_strategy: Optional[str] = None
    llm_provider: Optional[str] = None
    top_k: Optional[int] = 3

class VoiceRAGResponse(BaseModel):
    query: str
    transcribed_from_audio: bool = False
    stt_provider: Optional[str] = None
    language_detected: Optional[str] = None
    answer: str
    is_refusal: bool = False
    refusal_reason: Optional[str] = None
    chunking_strategy_used: str
    retrieved_contexts: List[RetrievalResult] = Field(default_factory=list)
    guardrails: GuardrailVerdict
    latency: LatencyBreakdown
    cache_hit: bool = False
    model_provider: str = "fast_local"

class BenchmarkQueryResult(BaseModel):
    query_id: int
    query: str
    language: str
    latency_ms: float
    breakdown: LatencyBreakdown
    cache_hit: bool
    status: str
    guardrail_passed: bool

class BenchmarkSummary(BaseModel):
    total_queries: int
    successful_queries: int
    failed_queries: int
    cache_hits: int
    p50_latency_ms: float
    p70_latency_ms: float
    p90_latency_ms: float
    p99_latency_ms: float
    p100_max_latency_ms: float
    min_latency_ms: float
    avg_latency_ms: float
    sub_200ms_compliance_pct: float
    stage_averages_ms: Dict[str, float]
    results: List[BenchmarkQueryResult] = Field(default_factory=list)
