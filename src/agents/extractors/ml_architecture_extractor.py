"""Pipeline 5. Извлечение описания архитектуры ML-решения из отчёта о разработке."""

import time
import logging

from openai import OpenAI

from ..config import call_llm_text
from ..prompts import ML_ARCHITECTURE_SYSTEM_PROMPT
from src.utils.llm_logging import log_llm_request, log_llm_error

logger = logging.getLogger(__name__)


class MlArchitectureExtractor:
    """Pipeline 5. Извлекает описание архитектуры ML-решения из отчёта о разработке."""

    def __init__(self, client: OpenAI, model: str, temperature: float = 0.0, max_tokens: int = 2048):
        self.client = client
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def extract(self, development_report: str) -> str:
        log_llm_request(logger, "MlArchitectureExtractor", development_report,
                        self.model, self.temperature, self.max_tokens)
        _start = time.time()
        try:
            result = call_llm_text(
                client=self.client, model=self.model,
                system_prompt=ML_ARCHITECTURE_SYSTEM_PROMPT,
                user_content=development_report,
                temperature=self.temperature, max_tokens=self.max_tokens,
            )
            logger.info(f"[MlArchitectureExtractor] Завершено за {time.time() - _start:.2f}с")
            return result.strip()
        except Exception as e:
            log_llm_error(logger, "MlArchitectureExtractor", e, time.time() - _start)
            return f"Ошибка при извлечении архитектуры: {e}"
