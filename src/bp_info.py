from __future__ import annotations
from src.prompts import (
    EXTRACT_METRIC_INFO_SYS_PROMPT,
    GROUP_EXTRACTION_SYS_PROMPT,
    TASK_EXTRACTION_SYS_PROMPT,
    DATASET_TYPE_CLASSIFICATION_SYS_PROMPT,
    SUMMARY_SYS_PROMPT,
)
import json
import re
import time
import logging
import pandas as pd
from pathlib import Path
from pydantic import BaseModel, Field, ValidationError
from typing import Dict, Any, List, Optional

from openai import OpenAI
from openai import RateLimitError, APIError

logger = logging.getLogger(__name__)


def _extract_metric_formula(
    metric_name: str,
    source_text: str,
    allow_labeled_formula: bool = True,
) -> Optional[str]:
    """Extract an explicit metric formula from LLM output or report text.

    The LLM occasionally returns the formula inside ``metric_desc`` while
    leaving ``metric_func`` empty. This fallback only copies an expression
    already present in an input artifact or structured response; it does not
    invent a formula.
    """
    if not metric_name or not source_text:
        return None

    # Allow Word-formula text to contain either normal spaces or no spaces in
    # the left-hand metric name (for example, "Custom Metric" or
    # "CustomMetric").
    metric_parts = re.split(r"\s+", metric_name.strip())
    metric_pattern = r"\s*".join(re.escape(part) for part in metric_parts)
    match = re.search(
        rf"(?<!\w){metric_pattern}\s*=\s*",
        source_text,
        flags=re.IGNORECASE,
    )

    if match:
        formula = source_text[match.start():].splitlines()[0].strip()
    else:
        if not allow_labeled_formula:
            return None
        # Keep compatibility with descriptions that explicitly label the
        # formula but omit its left-hand side.
        label_match = re.search(
            r"(?:формула(?:\s+расч[ёе]та)?|formula)\s*:\s*([^\r\n]+)",
            source_text,
            flags=re.IGNORECASE,
        )
        if not label_match:
            return None
        formula = label_match.group(1).strip()

    # Stop before a following prose sentence without treating decimal points
    # inside the formula as sentence boundaries.
    sentence_end = re.search(r"\.(?=\s+[A-ZА-ЯЁ])", formula)
    if sentence_end:
        formula = formula[:sentence_end.start()]

    return formula.rstrip(" .;,") or None


def _clean_metric_formula(value: Optional[str]) -> Optional[str]:
    """Normalize empty values returned by the LLM without changing a formula."""
    formula = value.strip() if value else ""
    if formula.casefold() in {"", "-", "—", "none", "null"}:
        return None
    return formula


def _resolve_metric_formula(
    metric_name: str,
    metric_desc: str,
    metric_func: Optional[str],
    report: str,
) -> Optional[str]:
    """Resolve a formula deterministically, preferring the input artifact."""
    return (
        _extract_metric_formula(
            metric_name=metric_name,
            source_text=report,
            allow_labeled_formula=False,
        )
        or _clean_metric_formula(metric_func)
        or _extract_metric_formula(
            metric_name=metric_name,
            source_text=metric_desc,
        )
    )


def _ensure_formula_in_metric_description(
    metric_name: str,
    metric_desc: str,
    metric_formula: Optional[str],
) -> str:
    """Append an explicit formula when the LLM omitted it from the description."""
    if not metric_formula:
        return metric_desc
    if _extract_metric_formula(metric_name, metric_desc):
        return metric_desc
    return f"{metric_desc.rstrip()}\nФормула расчёта: {metric_formula}"


def _call_llm_with_retry(
    client: OpenAI,
    model: str,
    messages: List[Dict[str, str]],
    response_format: Optional[Dict[str, Any]] = None,
    max_retries: int = 5,
    base_delay: float = 1.0,
) -> Any:
    """
    Вызывает LLM с экспоненциальной задержкой при ошибках.

    Parameters:
        client: Клиент OpenAI
        model: Название модели
        messages: Список сообщений
        response_format: Опционально - формат ответа (для structured output)
        max_retries: Максимальное количество повторов
        base_delay: Базовая задержка в секундах
    """
    last_error = None

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                response_format=response_format,
            )
            return response
        except (RateLimitError, APIError) as e:
            last_error = e
            if attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)
                logger.warning(
                    f"LLM error (attempt {attempt + 1}/{max_retries}): {e}. "
                    f"Retrying in {delay}s..."
                )
                time.sleep(delay)
            else:
                logger.error(f"Max retries reached. Last error: {e}")
                raise
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            raise

    raise last_error


def _convert_list_of_lists_to_dict(data: Any) -> dict:
    """
    Конвертирует [[key, value], ...] в {key: value}.
    Обрабатывает вложенные списки для списковых полей.
    """
    if isinstance(data, list) and all(
        isinstance(item, list) and len(item) == 2 for item in data
    ):
        result = {}
        for key, value in data:
            if isinstance(value, list) and value and isinstance(value[0], list):
                result[key] = [v[0] if len(v) == 1 else v for v in value]
            else:
                result[key] = value
        return result
    return data


def _call_llm_with_json(
    client: OpenAI,
    model: str,
    system_prompt: str,
    user_prompt: str,
    schema: type[BaseModel],
) -> BaseModel:
    """
    Вызывает LLM и парсит JSON ответ в указанную схему.
    Использует OpenAI SDK с response_format для structured output.
    С экспоненциальным retry при ошибках.
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": schema.__name__,
            "schema": schema.model_json_schema(),
        }
    }

    response = _call_llm_with_retry(
        client=client,
        model=model,
        messages=messages,
        response_format=response_format,
    )

    raw = response.choices[0].message.content

    if not raw:
        raise ValueError("LLM returned empty response")

    cleaned = raw.strip()
    if cleaned.startswith("```") and cleaned.endswith("```"):
        cleaned = cleaned.split("\n", 1)[-1][:-3].strip()

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Failed to parse JSON from LLM response: {raw}") from e

    data = _convert_list_of_lists_to_dict(data)

    try:
        parsed = schema.model_validate(data)
    except ValidationError as e:
        raise ValueError(
            f"JSON doesn't match schema {schema.__name__}: {data}") from e

    return parsed


class MetricInfo(BaseModel):
    """
    Информация о ключевой метрике оценки качества.
    """
    evaluation_metric: str = Field(...,
                                   description="Название метрики оценки качества")
    metric_desc: str = Field(
        ...,
        description="Описание ключевой метрики качества: обоснование выбора; целевые значения; формула вычисления",
    )
    threshold: Optional[float] = Field(
        ...,
        description="Порог ключевой метрики от Бизнеса (бизнес-заказчика), который необходимо достичь"
    )
    metric_func: Optional[str] = Field(
        default=None,
        description="Формула для подсчета метрики",
    )


class TaskInfo(BaseModel):
    """
    Получить информацию о решаемой задаче
    """
    selected_task: List[str] = Field(
        ...,
        description="Список с точным названием решаемых задач",
    )
    reasoning: str = Field(
        ...,
        description="Точное название задачи из списка",
    )


class TaskGroup(BaseModel):
    """
    Информация о группе решаемой задачи
    """
    reasoning: str = Field(..., description="Точное название задачи из списка")
    selected_group: str = Field(...,
                                description="Точное название группы задач")


class DatasetType(BaseModel):
    """
    Информация о предполагаемом типе датасета
    """
    reasoning: str = Field(...,
                           description="Описание процесса выбора типа датасета")
    dataset_type: str = Field(..., description="Тип датасета")


class BusinessProcessInfo:
    """
    Класс для агрегации информации о бизнес-процессе применения
    """

    def __init__(
        self,
        report: str,
        client: OpenAI,
        model: str,
        data: Optional[pd.DataFrame] = None
    ):
        """
        report: Текст отчета о разработке
        data: Тестовые данные
        client: Клиент OpenAI
        model: Название модели
        """
        self.report = report
        self.data = data
        base_dir = Path(__file__).resolve().parent
        with open(base_dir / "metadata" / "task_types.json") as f:
            task_types = json.load(f)
        self.task_types = task_types
        self.client = client
        self.model = model

    def get_metric(self) -> Dict[str, Any]:
        """
        Функция для выделения информации о ключевой метрике качества из отчета
        Return:
            Словарь в соответствии со схемой MetricInfo
        """
        user_prompt = f"Текст отчета о разработке:\n {self.report}"

        return _call_llm_with_json(
            client=self.client,
            model=self.model,
            system_prompt=EXTRACT_METRIC_INFO_SYS_PROMPT,
            user_prompt=user_prompt,
            schema=MetricInfo,
        )

    def get_task_group(self) -> Dict[str, Any]:
        """
        Функция для выделения группы задачи
        Return:
            Словарь в соответствии со схемой TaskGroup
        """
        tasks_groups = [
            "### Название группы {group}\n ### Краткое описание {desc}".format(
                group=group,
                desc=self.task_types[group]["description"],
            )
            for group in self.task_types
        ]

        system_prompt = GROUP_EXTRACTION_SYS_PROMPT.format(
            groups_desc="\n".join(tasks_groups)
        )
        user_prompt = f"Текст отчета о разработке:\n {self.report}"

        result = _call_llm_with_json(
            client=self.client,
            model=self.model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            schema=TaskGroup,
        )
        self.task_group = result.selected_group
        return result

    def get_task_type(self) -> Dict[str, Any]:
        """
        Функция для выделения типа задачи
        Return:
            Словарь в соответствии со схемой TaskType
        """
        if not hasattr(self, 'task_group') or not self.task_group:
            raise ValueError(
                "Must call get_task_group() before get_task_type()")

        tasks_list = self.task_types[self.task_group]["tasks"]
        tasks_desc = [
            "*Наименование задачи* \n {task_name} \n Описание задачи {task_desc}".format(
                task_name=task["name"],
                task_desc=task["description"],
            )
            for task in tasks_list
        ]

        system_prompt = TASK_EXTRACTION_SYS_PROMPT.format(
            task_types="\n".join(tasks_desc)
        )
        user_prompt = f"Текст отчета о разработке:\n {self.report}"

        result = _call_llm_with_json(
            client=self.client,
            model=self.model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            schema=TaskInfo,
        )

        self.task = result.selected_task
        return result

    def get_dataset_type(self):
        system_prompt = DATASET_TYPE_CLASSIFICATION_SYS_PROMPT
        user_prompt = (
            f"Текст отчета о разработке:\n {self.report} \n, "
            f"Примеры диалогов "
            f"{self.data.head().to_markdown()}"
        )

        return _call_llm_with_json(
            client=self.client,
            model=self.model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            schema=DatasetType,
        )

    def get_structured_fields(self) -> Dict[str, str]:
        """
        Возвращает структурированные поля первого этапа в текстовом формате.
        Должна вызываться после agg_info().
        Return:
            Словарь с текстовыми представлениями всех извлечённых полей первого этапа
        """
        if self.dataset_info is not None:
            dataset_type = self.dataset_info.dataset_type
        else:
            dataset_type = "-"

        metric_func = _resolve_metric_formula(
            metric_name=self.metric_info.evaluation_metric,
            metric_desc=self.metric_info.metric_desc,
            metric_func=self.metric_info.metric_func,
            report=self.report,
        )

        return {
            'task_group': self.group_info.selected_group,
            'task_type': ', '.join(self.task_type_info.selected_task),
            'evaluation_metric': self.metric_info.evaluation_metric,
            'metric_desc': self.metric_info.metric_desc,
            'threshold': str(self.metric_info.threshold) if self.metric_info.threshold is not None else '-',
            'metric_func': metric_func or '-',
            'dataset_type': dataset_type,
        }

    def agg_info(self) -> str:
        self.group_info: TaskGroup = self.get_task_group()
        self.task_type_info: TaskInfo = self.get_task_type()
        self.metric_info: MetricInfo = self.get_metric()
        self.dataset_info: Optional[DatasetType] = self.get_dataset_type(
        ) if self.data is not None else None
        messages = [
            {
                "content": SUMMARY_SYS_PROMPT,
                "role": "system",
            },
            {
                "content": f"Текст отчета о разработке:\n {self.report}",
                "role": "user",
            },
        ]
        summary = self.client.chat.completions.create(
            model=self.model, messages=messages)
        summary_text = summary.choices[0].message.content

        md_info = [summary_text]
        group_info = f"""
    ## Информация о решаемой задаче
    ### Группа
    ***{self.group_info.selected_group}***
    ### Обоснование
    {self.group_info.reasoning}
    """

        task_info = f"""
    ### Тип задачи
    ***{self.task_type_info.selected_task}***
    ### Обоснование
    {self.task_type_info.reasoning}
    """
        dataset_info = f"""
    ## Информация о данных
    ### Тип датасета
    {self.dataset_info.dataset_type}
    ### Обоснование
    {self.dataset_info.reasoning}
""" if self.data is not None else ""
        resolved_metric_formula = _resolve_metric_formula(
            metric_name=self.metric_info.evaluation_metric,
            metric_desc=self.metric_info.metric_desc,
            metric_func=self.metric_info.metric_func,
            report=self.report,
        )
        metric_description = _ensure_formula_in_metric_description(
            metric_name=self.metric_info.evaluation_metric,
            metric_desc=self.metric_info.metric_desc,
            metric_formula=resolved_metric_formula,
        )
        metric_info = f"""
    ## Информация о ключевой метрике
    ### Название
    {self.metric_info.evaluation_metric}
    ### Значение бизнес-метрики, которую необходимо достичь
    {self.metric_info.threshold if self.metric_info.threshold is not None else '-'}
    ### Описание
    {metric_description}
"""
        md_info = [
            summary_text,
            group_info,
            task_info,
            dataset_info,
            metric_info
        ]
        if self.metric_info.metric_func is not None:
            md_info.append(
                f"""
    ### Формула для вычисления метрики
    {self.metric_info.metric_func}
"""
            )
        return "\n".join(md_info)
