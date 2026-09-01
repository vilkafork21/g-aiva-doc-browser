"""Pipeline 2. Генерация суммаризации отчёта о разработке."""

import time
import logging
from typing import Dict

from openai import OpenAI

from ..config import call_llm_text
from ..prompts import SUMMARY_SYSTEM_PROMPT
from src.utils.llm_logging import log_llm_request, log_llm_error

logger = logging.getLogger(__name__)


class SummaryGenerator:
    """Pipeline 2. Генерирует суммаризацию отчёта о разработке."""

    def __init__(self, client: OpenAI, model: str, temperature: float = 0.0, max_tokens: int = 2048):
        self.client = client
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    @staticmethod
    def _key_points_have_errors(key_points: Dict[str, str]) -> bool:
        return any(
            isinstance(v, str) and v.startswith('Ошибка при извлечении:')
            for v in key_points.values()
        )

    def generate(self, development_report: str, key_points: Dict[str, str]) -> str:
        if self._key_points_have_errors(key_points):
            logger.warning("[SummaryGenerator] key_points содержат ошибки, пропускаем генерацию")
            return "Суммаризация недоступна: ключевые параметры не были извлечены."

        key_points_text = '\n'.join(f'  {k}: {v}' for k, v in key_points.items())
        user_content = 'Отчет о разработке \n' + development_report + '\n Ключевые параметры \n' + key_points_text
        log_llm_request(logger, "SummaryGenerator", user_content,
                        self.model, self.temperature, self.max_tokens)
        _start = time.time()
        try:
            result = call_llm_text(
                client=self.client, model=self.model,
                system_prompt=SUMMARY_SYSTEM_PROMPT,
                user_content=user_content,
                temperature=self.temperature, max_tokens=self.max_tokens,
            )
            logger.info(f"[SummaryGenerator] Завершено за {time.time() - _start:.2f}с")
            return result
        except Exception as e:
            log_llm_error(logger, "SummaryGenerator", e, time.time() - _start)
            return f"Ошибка при генерации суммаризации: {e}"
