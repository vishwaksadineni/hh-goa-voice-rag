import httpx
import logging
import time
from typing import Dict, Any, Optional
from rag.stt.base import BaseSTTClient
from rag.config import settings

logger = logging.getLogger(__name__)

class ElevenLabsSTTClient(BaseSTTClient):
    """
    ElevenLabs Speech-to-Text (Scribe) Client.
    High-accuracy multilingual speech transcription.
    """
    ENDPOINT = "https://api.elevenlabs.io/v1/speech-to-text"

    def __init__(self, api_key: Optional[str] = None):
        super().__init__(name="elevenlabs")
        self.api_key = api_key or settings.elevenlabs_api_key

    async def transcribe(
        self, 
        audio_bytes: bytes, 
        audio_format: str = "webm", 
        language_code: Optional[str] = None
    ) -> Dict[str, Any]:
        start_time = time.perf_counter()

        if not self.api_key or self.api_key.strip() == "":
            raise ValueError("ElevenLabs API key is not configured. Please set ELEVENLABS_API_KEY.")

        headers = {
            "xi-api-key": self.api_key
        }

        filename = f"audio.{audio_format}"
        content_type = f"audio/{audio_format}" if audio_format != "wav" else "audio/wav"

        files = {
            "file": (filename, audio_bytes, content_type)
        }
        data = {
            "model_id": settings.elevenlabs_model or "scribe_v1"
        }
        if language_code:
            data["language_code"] = language_code

        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(
                self.ENDPOINT,
                headers=headers,
                data=data,
                files=files
            )

            if response.status_code != 200:
                logger.error(f"ElevenLabs STT error {response.status_code}: {response.text}")
                raise RuntimeError(f"ElevenLabs STT API returned status {response.status_code}: {response.text}")

            result = response.json()
            transcript = result.get("text", "").strip()
            detected_lang = result.get("language_code", language_code or "en")

            duration_s = time.perf_counter() - start_time
            return {
                "transcript": transcript,
                "language_detected": detected_lang,
                "confidence": 0.94,
                "provider": "elevenlabs",
                "duration_s": duration_s
            }
