"""
Оркестратор извлечения данных из отчёта о разработке модели.

Использует тот же OpenAI-клиент (через gpt2giga прокси), что и основная логика doc-browser.
"""

import logging
import time
from typing import Any, Dict, List, Optional

from openai import OpenAI

from .extractors import (
    HyperparamsExtractor,
    KeyPointsExtractor,
    MlArchitectureExtractor,
    SampleExtractor,
    SummaryGenerator,
)

logger = logging.getLogger(__name__)

_MAX_RETRIES = 5
_RETRY_DELAY_SEC = 2.0


def _retry(fn, *args, max_retries: int = _MAX_RETRIES, delay: float = _RETRY_DELAY_SEC, **kwargs):
    last_exc: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            last_exc = exc
            logger.warning("[_retry] attempt %d/%d failed: %s", attempt, max_retries, exc)
            if attempt < max_retries:
                time.sleep(delay)
    raise last_exc


def extraction_agent(
    development_report: str,
    client: OpenAI,
    model: str,
    temperature: float = 0.0,
    max_tokens: int = 16384,
    title: str = "Описание бизнес-процесса",
) -> Dict[str, Any]:
    """
    Запускает 5 LLM-пайплайнов через gpt2giga-клиент и возвращает kwargs для
    BusinessProcessVisualizer.visualize() (передаются через порт extracted_fields).

    Порядок выполнения:
      1. KeyPointsExtractor  — ключевые параметры (ml_task ограничен task_types.json)
      2. SummaryGenerator    — суммаризация (зависит от п.1)
      3. HyperparamsExtractor — гиперпараметры генерации
      4. SampleExtractor     — описание выборок
      5. MlArchitectureExtractor — архитектура ML-решения

    Returns:
        {
            'title': str,
            'summary': str,
            'key_points': Dict[str, str],
            'generation_hyperparams': Dict[str, str],
            'sample_description': List[Dict[str, Any]],
            'ml_architecture': str,
        }
    """
    logger.info("[extraction_agent] start | report_len=%d", len(development_report))

    kp = KeyPointsExtractor(client=client, model=model, temperature=temperature, max_tokens=max_tokens)
    sg = SummaryGenerator(client=client, model=model, temperature=temperature, max_tokens=max_tokens)
    hp = HyperparamsExtractor(client=client, model=model, temperature=temperature, max_tokens=max_tokens)
    sa = SampleExtractor(client=client, model=model, temperature=temperature, max_tokens=max_tokens)
    ml = MlArchitectureExtractor(client=client, model=model, temperature=temperature, max_tokens=max_tokens)

    logger.info("[extraction_agent] pipeline 1: KeyPointsExtractor")
    key_points: Dict[str, str] = _retry(kp.extract, development_report)

    logger.info("[extraction_agent] pipeline 2: SummaryGenerator")
    summary: str = _retry(sg.generate, development_report, key_points)

    logger.info("[extraction_agent] pipeline 3: HyperparamsExtractor")
    generation_hyperparams: Dict[str, str] = _retry(hp.extract, development_report)

    logger.info("[extraction_agent] pipeline 4: SampleExtractor")
    sample_description: List[Dict[str, Any]] = _retry(sa.extract, development_report)

    logger.info("[extraction_agent] pipeline 5: MlArchitectureExtractor")
    ml_architecture: str = _retry(ml.extract, development_report)

    logger.info("[extraction_agent] done")

    return {
        'title': title,
        'summary': summary,
        'key_points': key_points,
        'generation_hyperparams': generation_hyperparams,
        'sample_description': sample_description,
        'ml_architecture': ml_architecture,
    }
