"""
Утилиты для вызова LLM через OpenAI SDK (используется gpt2giga прокси из main.py).
"""
import json
import time
import logging
from typing import Any, Dict, List, Optional, Type

from openai import OpenAI, RateLimitError, APIError
from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)


def _convert_list_of_lists_to_dict(data: Any) -> Any:
    """Конвертирует [[key, value], ...] в {key: value}. Обрабатывает вложенные списки."""
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


def call_llm_with_retry(
    client: OpenAI,
    model: str,
    messages: List[Dict[str, str]],
    temperature: float = 0.0,
    max_tokens: int = 2048,
    response_format: Optional[Dict[str, Any]] = None,
    max_retries: int = 5,
    base_delay: float = 1.0,
) -> Any:
    """Вызывает LLM с экспоненциальной задержкой при ошибках."""
    last_error = None
    for attempt in range(max_retries):
        try:
            call_kwargs: Dict[str, Any] = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            if response_format is not None:
                call_kwargs["response_format"] = response_format
            return client.chat.completions.create(**call_kwargs)
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


def call_llm_structured(
    client: OpenAI,
    model: str,
    system_prompt: str,
    user_content: str,
    schema: Type[BaseModel],
    temperature: float = 0.0,
    max_tokens: int = 2048,
) -> BaseModel:
    """Вызывает LLM и парсит JSON-ответ в указанную Pydantic-схему."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]
    response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": schema.__name__,
            "schema": schema.model_json_schema(),
        },
    }
    response = call_llm_with_retry(
        client=client,
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        response_format=response_format,
    )
    raw = response.choices[0].message.content
    if not raw:
        raise ValueError("LLM returned empty response")
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse JSON from LLM response: {raw}") from e
    data = _convert_list_of_lists_to_dict(data)
    try:
        return schema.model_validate(data)
    except ValidationError as e:
        raise ValueError(f"JSON doesn't match schema {schema.__name__}: {data}") from e


def call_llm_text(
    client: OpenAI,
    model: str,
    system_prompt: str,
    user_content: str,
    temperature: float = 0.0,
    max_tokens: int = 2048,
) -> str:
    """Вызывает LLM и возвращает сырой текстовый ответ."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]
    response = call_llm_with_retry(
        client=client,
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content or ""
