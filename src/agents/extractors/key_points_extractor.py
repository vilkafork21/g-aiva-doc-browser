"""
Pipeline 1. Извлечение ключевых параметров бизнес-процесса из отчёта о разработке.

ml_task ограничен подтипами из src/metadata/task_types.json.
"""

import json
import time
import logging
from pathlib import Path
from typing import Optional, Dict

from openai import OpenAI
from pydantic import BaseModel, Field

from ..config import call_llm_structured
from ..prompts import build_key_points_prompt
from src.utils.llm_logging import log_llm_request, log_llm_error

logger = logging.getLogger(__name__)


class KeyPointsOutput(BaseModel):
    target_nature: str = Field(
        description="Природа целевой переменной. '-' если целевой переменной нет."
    )
    business_task: str = Field(description="Краткое описание бизнес-задачи (1–2 предложения).")
    ml_task: str = Field(description="Тип ML-задачи — одно из допустимых значений из списка в промпте.")
    model_inference_scenario: str = Field(
        description="Сценарий применения модели: когда, на каких данных, как используются результаты."
    )
    model_application_segment: str = Field(
        description="Сегмент данных, на которых будет применяться модель."
    )


def _load_allowed_ml_tasks() -> list:
    task_types_path = (
        Path(__file__).resolve().parent.parent.parent / "metadata" / "task_types.json"
    )
    with open(task_types_path, encoding="utf-8") as f:
        task_types = json.load(f)
    return [
        task["name"]
        for group_data in task_types.values()
        for task in group_data.get("tasks", [])
    ]


class KeyPointsExtractor:
    """
    Pipeline 1. Извлекает ключевые параметры бизнес-процесса из отчёта.
    ml_task ограничен подтипами из src/metadata/task_types.json.
    """

    def __init__(self, client: OpenAI, model: str, temperature: float = 0.0, max_tokens: int = 2048):
        self.client = client
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        allowed_tasks = _load_allowed_ml_tasks()
        self.system_prompt = build_key_points_prompt(allowed_tasks)

    def extract(self, development_report: str) -> Dict[str, str]:
        log_llm_request(logger, "KeyPointsExtractor", development_report,
                        self.model, self.temperature, self.max_tokens)
        _start = time.time()
        try:
            result: KeyPointsOutput = call_llm_structured(
                client=self.client, model=self.model,
                system_prompt=self.system_prompt,
                user_content=development_report,
                schema=KeyPointsOutput,
                temperature=self.temperature, max_tokens=self.max_tokens,
            )
            logger.info(f"[KeyPointsExtractor] Завершено за {time.time() - _start:.2f}с")
            return {
                'Природа целевой переменной': result.target_nature,
                'Бизнес задача': result.business_task,
                'ML-задача': result.ml_task,
                'Сценарий использования модели': result.model_inference_scenario,
                'Сегмент данных модели': result.model_application_segment,
            }
        except Exception as e:
            log_llm_error(logger, "KeyPointsExtractor", e, time.time() - _start)
            _err = f"Ошибка при извлечении: {e}"
            return {k: _err for k in (
                'Природа целевой переменной', 'Бизнес задача', 'ML-задача',
                'Сценарий использования модели', 'Сегмент данных модели',
            )}
