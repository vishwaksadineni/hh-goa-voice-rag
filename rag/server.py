import os
import time
import base64
import logging
from pathlib import Path
from typing import Optional, Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from pydantic import BaseModel

from rag.config import settings, BASE_DIR
from rag.schemas import VoiceRAGRequest, VoiceRAGResponse, BenchmarkSummary
from rag.harness.pipeline_harness import rag_harness
from rag.chunking.registry import chunking_registry
from rag.benchmark import LatencyBenchmarkSuite

logger = logging.getLogger("uvicorn.error")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Warm up in-memory index
    logger.info("Initializing Voice RAG Pipeline indexes...")
    rag_harness.initialize_indexes()
    logger.info("Voice RAG Pipeline ready on port 8000.")
    yield
    # Shutdown
    logger.info("Shutting down Voice RAG Pipeline.")

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Voice-Enabled RAG System with Sarvam/ElevenLabs STT, vast chunking, sub-200ms latency, and multi-tier guardrails.",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ConfigUpdateRequest(BaseModel):
    sarvam_api_key: Optional[str] = None
    elevenlabs_api_key: Optional[str] = None
    gemini_api_key: Optional[str] = None
    groq_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None
    stt_provider: Optional[str] = None
    llm_provider: Optional[str] = None
    default_chunking_strategy: Optional[str] = None

@app.get("/api/rag/health")
async def health():
    return {
        "status": "healthy",
        "app": settings.app_name,
        "version": settings.app_version,
        "initialized": rag_harness._initialized,
        "strategies_available": [s["id"] for s in chunking_registry.list_strategies()],
        "stt_provider": settings.stt_provider,
        "llm_provider": settings.llm_provider
    }

@app.get("/api/rag/strategies")
async def get_strategies():
    return {
        "strategies": chunking_registry.list_strategies(),
        "default": settings.default_chunking_strategy
    }

@app.get("/api/rag/config")
async def get_config():
    return {
        "stt_provider": settings.stt_provider,
        "llm_provider": settings.llm_provider,
        "default_chunking_strategy": settings.default_chunking_strategy,
        "latency_target_ms": settings.latency_target_ms,
        "has_sarvam_key": bool(settings.sarvam_api_key and len(settings.sarvam_api_key) > 5),
        "has_elevenlabs_key": bool(settings.elevenlabs_api_key and len(settings.elevenlabs_api_key) > 5),
        "has_gemini_key": bool(settings.gemini_api_key and len(settings.gemini_api_key) > 5),
        "has_groq_key": bool(settings.groq_api_key and len(settings.groq_api_key) > 5),
        "has_openai_key": bool(settings.openai_api_key and len(settings.openai_api_key) > 5),
    }

@app.post("/api/rag/config")
async def update_config(req: ConfigUpdateRequest):
    if req.sarvam_api_key is not None:
        settings.sarvam_api_key = req.sarvam_api_key
        rag_harness.stt_router.sarvam_client.api_key = req.sarvam_api_key
    if req.elevenlabs_api_key is not None:
        settings.elevenlabs_api_key = req.elevenlabs_api_key
        rag_harness.stt_router.elevenlabs_client.api_key = req.elevenlabs_api_key
    if req.gemini_api_key is not None:
        settings.gemini_api_key = req.gemini_api_key
    if req.groq_api_key is not None:
        settings.groq_api_key = req.groq_api_key
    if req.openai_api_key is not None:
        settings.openai_api_key = req.openai_api_key
    if req.stt_provider is not None:
        settings.stt_provider = req.stt_provider
    if req.llm_provider is not None:
        settings.llm_provider = req.llm_provider
    if req.default_chunking_strategy is not None:
        settings.default_chunking_strategy = req.default_chunking_strategy
        
    return {"status": "updated", "config": await get_config()}

@app.post("/api/rag/query", response_model=VoiceRAGResponse)
async def query_rag(req: VoiceRAGRequest):
    """Processes a text query or Base64 audio query through the Voice RAG harness."""
    try:
        response = await rag_harness.process_request(req)
        return response
    except Exception as e:
        logger.error(f"Error executing RAG request: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/rag/voice", response_model=VoiceRAGResponse)
async def voice_rag_upload(
    file: UploadFile = File(...),
    chunking_strategy: Optional[str] = Form(None),
    language_code: Optional[str] = Form("hi-IN"),
    stt_provider: Optional[str] = Form(None),
    llm_provider: Optional[str] = Form(None),
):
    """Direct audio file upload endpoint for microphone recordings or audio test files."""
    try:
        audio_bytes = await file.read()
        audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
        
        # Determine extension/format
        filename = file.filename or "audio.webm"
        ext = filename.split(".")[-1].lower() if "." in filename else "webm"

        req = VoiceRAGRequest(
            audio_base64=audio_b64,
            audio_format=ext,
            chunking_strategy=chunking_strategy,
            language_code=language_code,
            stt_provider=stt_provider,
            llm_provider=llm_provider
        )
        return await rag_harness.process_request(req)
    except Exception as e:
        logger.error(f"Error handling voice upload: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/rag/benchmark", response_model=BenchmarkSummary)
async def trigger_benchmark(strategy: Optional[str] = "hierarchical"):
    """Triggers the full P50/P70/P100 latency analytics benchmark."""
    suite = LatencyBenchmarkSuite()
    summary = await suite.run_benchmark(strategy=strategy)
    return summary

# Mount static web directory
static_dir = Path(__file__).resolve().parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

@app.get("/")
async def root():
    index_file = static_dir / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file))
    return HTMLResponse("<h1>Voice RAG Server Running</h1>")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("rag.server:app", host=settings.host, port=settings.port, reload=False)
