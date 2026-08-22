import httpx
import logging
import time
from typing import Dict, Any, Optional
from rag.stt.base import BaseSTTClient
from rag.config import settings

logger = logging.getLogger(__name__)

class SarvamSTTClient(BaseSTTClient):
    """
    Sarvam AI Saaras Speech-to-Text Client.
    Specialized in Indian languages (Hindi, Bengali, Tamil, Telugu, Marathi, etc.)
    and Indian English accents.
    """
    ENDPOINT = "https://api.sarvam.ai/speech-to-text"

    def __init__(self, api_key: Optional[str] = None):
        super().__init__(name="sarvam")
        self.api_key = api_key or settings.sarvam_api_key

    async def transcribe(
        self, 
        audio_bytes: bytes, 
        audio_format: str = "webm", 
        language_code: Optional[str] = "hi-IN"
    ) -> Dict[str, Any]:
        start_time = time.perf_counter()
        
        if not self.api_key or self.api_key.strip() == "":
            raise ValueError("Sarvam AI API key is not configured. Please set SARVAM_API_KEY.")

        headers = {
            "api-subscription-key": self.api_key
        }

        # Sarvam speech-to-text multipart form payload
        filename = f"audio.{audio_format}"
        content_type = f"audio/{audio_format}" if audio_format != "wav" else "audio/wav"

        files = {
            "file": (filename, audio_bytes, content_type)
        }
        data = {
            "model": settings.sarvam_model or "saaras:v1",
            "language_code": language_code or settings.sarvam_language_code
        }

        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(
                self.ENDPOINT,
                headers=headers,
                data=data,
                files=files
            )
            
            if response.status_code != 200:
                logger.error(f"Sarvam STT API error {response.status_code}: {response.text}")
                raise RuntimeError(f"Sarvam STT API returned status {response.status_code}: {response.text}")

            result = response.json()
            transcript = result.get("transcript", "").strip()
            detected_lang = result.get("language_code", language_code)
            
            duration_s = time.perf_counter() - start_time
            return {
                "transcript": transcript,
                "language_detected": detected_lang,
                "confidence": 0.95,
                "provider": "sarvam",
                "duration_s": duration_s
            }
