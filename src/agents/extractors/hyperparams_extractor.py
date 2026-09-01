"""Pipeline 3. Извлечение гиперпараметров генерации LLM из отчёта о разработке."""

import time
import logging
from typing import Dict, List

from openai import OpenAI
from pydantic import BaseModel, Field

from ..config import call_llm_structured
from ..prompts import HYPERPARAMS_SYSTEM_PROMPT
from src.utils.llm_logging import log_llm_request, log_llm_error

logger = logging.getLogger(__name__)


class HyperparamItem(BaseModel):
    name: str = Field(description="Название гиперпараметра, как указано в отчёте.")
    value: str = Field(description="Значение гиперпараметра в виде строки.")


class HyperparamsOutput(BaseModel):
    hyperparams: List[HyperparamItem] = Field(
        description="Список гиперпараметров. Пустой список, если не упомянуты."
    )


class HyperparamsExtractor:
    """Pipeline 3. Извлекает гиперпараметры генерации LLM из отчёта о разработке."""

    def __init__(self, client: OpenAI, model: str, temperature: float = 0.0, max_tokens: int = 2048):
        self.client = client
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def extract(self, development_report: str) -> Dict[str, str]:
        log_llm_request(logger, "HyperparamsExtractor", development_report,
                        self.model, self.temperature, self.max_tokens)
        _start = time.time()
        try:
            result: HyperparamsOutput = call_llm_structured(
                client=self.client, model=self.model,
                system_prompt=HYPERPARAMS_SYSTEM_PROMPT,
                user_content=development_report,
                schema=HyperparamsOutput,
                temperature=self.temperature, max_tokens=self.max_tokens,
            )
            logger.info(f"[HyperparamsExtractor] Завершено за {time.time() - _start:.2f}с")
            return {item.name: item.value for item in result.hyperparams}
        except Exception as e:
            log_llm_error(logger, "HyperparamsExtractor", e, time.time() - _start)
            return {'error': f"Ошибка при извлечении: {e}"}
