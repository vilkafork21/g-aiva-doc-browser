"""Pipeline 4. Извлечение описания выборок из отчёта о разработке."""

import time
import logging
from typing import List, Dict, Any, Optional

from openai import OpenAI
from pydantic import BaseModel, Field

from ..config import call_llm_structured
from ..prompts import SAMPLE_SYSTEM_PROMPT
from src.utils.llm_logging import log_llm_request, log_llm_error

logger = logging.getLogger(__name__)


class SampleRecord(BaseModel):
    split: str = Field(description="Часть выборки: train, val, oos или oot.")
    thema: str = Field(description="Тематика / канал / '-' если однородно.")
    volume: Optional[int] = Field(default=None, description="Количество наблюдений или null.")
    dates: str = Field(description="Период сбора данных. '-' если не указан.")


class SampleDescriptionOutput(BaseModel):
    samples: List[SampleRecord] = Field(
        description="Список записей. Пустой список, если информация отсутствует."
    )


class SampleExtractor:
    """Pipeline 4. Извлекает описание выборок из отчёта о разработке."""

    def __init__(self, client: OpenAI, model: str, temperature: float = 0.0, max_tokens: int = 2048):
        self.client = client
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    @staticmethod
    def _record_to_dict(s: SampleRecord) -> Dict[str, Any]:
        return {'split': s.split, 'thema': s.thema, 'volume': s.volume, 'dates': s.dates}

    def extract(self, development_report: str) -> List[Dict[str, Any]]:
        log_llm_request(logger, "SampleExtractor", development_report,
                        self.model, self.temperature, self.max_tokens)
        _start = time.time()
        try:
            result: SampleDescriptionOutput = call_llm_structured(
                client=self.client, model=self.model,
                system_prompt=SAMPLE_SYSTEM_PROMPT,
                user_content=development_report,
                schema=SampleDescriptionOutput,
                temperature=self.temperature, max_tokens=self.max_tokens,
            )
            logger.info(f"[SampleExtractor] Завершено за {time.time() - _start:.2f}с")
            return [self._record_to_dict(s) for s in result.samples]
        except Exception as e:
            log_llm_error(logger, "SampleExtractor", e, time.time() - _start)
            return [{'split': 'error', 'thema': '-', 'volume': None, 'dates': f"Ошибка: {e}"}]
