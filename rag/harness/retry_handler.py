import asyncio
import logging
import random
import time
from typing import Callable, Any, TypeVar

logger = logging.getLogger(__name__)
T = TypeVar("T")

class HarnessRetryHandler:
    """
    Exponential Backoff with Jitter and Error Recovery Handler.
    Orchestrates resilient API execution against rate-limits, network hiccups, and transient errors.
    """
    def __init__(
        self, 
        max_retries: int = 3, 
        initial_delay_s: float = 0.05, 
        backoff_multiplier: float = 2.0,
        max_delay_s: float = 0.5
    ):
        self.max_retries = max_retries
        self.initial_delay_s = initial_delay_s
        self.backoff_multiplier = backoff_multiplier
        self.max_delay_s = max_delay_s

    async def execute_with_retry(
        self, 
        operation_name: str, 
        func: Callable[[], Any],
        fallback_func: Callable[[Exception], Any] = None
    ) -> Any:
        delay = self.initial_delay_s
        last_exception = None

        for attempt in range(1, self.max_retries + 1):
            try:
                if asyncio.iscoroutinefunction(func):
                    return await func()
                else:
                    return func()
            except Exception as e:
                last_exception = e
                logger.warning(
                    f"[{operation_name}] Attempt {attempt}/{self.max_retries} failed: {e}. "
                    f"Retrying in {delay:.3f}s..."
                )
                if attempt == self.max_retries:
                    break
                
                # Exponential backoff + full jitter to prevent thundering herd
                jitter = random.uniform(0.8, 1.2)
                sleep_time = min(self.max_delay_s, delay * jitter)
                await asyncio.sleep(sleep_time)
                delay *= self.backoff_multiplier

        if fallback_func is not None:
            logger.info(f"[{operation_name}] Invoking fallback recovery after {self.max_retries} failed attempts.")
            return fallback_func(last_exception)

        raise last_exception
