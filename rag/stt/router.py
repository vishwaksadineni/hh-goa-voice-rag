import logging
from typing import Optional, Dict, Any
from rag.config import settings
from rag.stt.sarvam_stt import SarvamSTTClient
from rag.stt.elevenlabs_stt import ElevenLabsSTTClient
from rag.stt.local_stt import LocalAudioSTTClient

logger = logging.getLogger(__name__)

class STTRouter:
    """
    Intelligent Speech-to-Text Router.
    Routes incoming audio to Sarvam AI or ElevenLabs, with graceful fallback.
    """
    def __init__(self):
        self.sarvam_client = SarvamSTTClient()
        self.elevenlabs_client = ElevenLabsSTTClient()
        self.local_client = LocalAudioSTTClient()

    async def transcribe(
        self,
        audio_bytes: bytes,
        audio_format: str = "webm",
        provider_preference: Optional[str] = None,
        language_code: Optional[str] = "hi-IN"
    ) -> Dict[str, Any]:
        provider = (provider_preference or settings.stt_provider).lower()

        # 1. Sarvam AI Path
        if provider == "sarvam":
            try:
                if settings.sarvam_api_key and settings.sarvam_api_key.strip():
                    return await self.sarvam_client.transcribe(
                        audio_bytes=audio_bytes,
                        audio_format=audio_format,
                        language_code=language_code
                    )
                else:
                    logger.warning("Sarvam API key not set, using local high-speed speech recognizer.")
            except Exception as e:
                logger.error(f"Sarvam STT failed: {e}. Falling back to local STT.")

        # 2. ElevenLabs Path
        elif provider == "elevenlabs":
            try:
                if settings.elevenlabs_api_key and settings.elevenlabs_api_key.strip():
                    return await self.elevenlabs_client.transcribe(
                        audio_bytes=audio_bytes,
                        audio_format=audio_format,
                        language_code=language_code
                    )
                else:
                    logger.warning("ElevenLabs API key not set, using local high-speed speech recognizer.")
            except Exception as e:
                logger.error(f"ElevenLabs STT failed: {e}. Falling back to local STT.")

        # 3. Local Fallback Path (Zero-latency fallback)
        return await self.local_client.transcribe(
            audio_bytes=audio_bytes,
            audio_format=audio_format,
            language_code=language_code
        )

stt_router = STTRouter()
