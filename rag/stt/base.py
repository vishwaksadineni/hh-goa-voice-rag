from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

class BaseSTTClient(ABC):
    """Abstract Base Class for Speech-to-Text Clients."""
    
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    async def transcribe(
        self, 
        audio_bytes: bytes, 
        audio_format: str = "webm", 
        language_code: Optional[str] = "hi-IN"
    ) -> Dict[str, Any]:
        """
        Transcribes voice audio bytes to text.
        Returns dictionary containing:
        {
            "transcript": str,
            "language_detected": str,
            "confidence": float,
            "provider": str,
            "duration_s": float
        }
        """
        pass
