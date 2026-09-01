from __future__ import annotations

import importlib
import sys
from types import ModuleType, SimpleNamespace

from pydantic import BaseModel


class StructuredAnswer(BaseModel):
    answer: str


def test_structured_call_includes_schema_in_prompt(monkeypatch):
    openai = ModuleType("openai")
    openai.OpenAI = object
    openai.RateLimitError = type("RateLimitError", (Exception,), {})
    openai.APIError = type("APIError", (Exception,), {})
    monkeypatch.setitem(sys.modules, "openai", openai)
    monkeypatch.delitem(sys.modules, "src.agents.config", raising=False)
    config = importlib.import_module("src.agents.config")

    captured = {}

    def create(**kwargs):
        captured.update(kwargs)
        message = SimpleNamespace(content='{"answer":"ok"}')
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])

    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )

    result = config.call_llm_structured(
        client=client,
        model="test-model",
        system_prompt="Извлеки ответ.",
        user_content="Вход",
        schema=StructuredAnswer,
    )

    assert result.answer == "ok"
    system_prompt = captured["messages"][0]["content"]
    assert "только валидный JSON" in system_prompt
    assert '"answer"' in system_prompt


def test_business_process_parser_accepts_json_fence(monkeypatch):
    openai = ModuleType("openai")
    openai.OpenAI = object
    openai.RateLimitError = type("RateLimitError", (Exception,), {})
    openai.APIError = type("APIError", (Exception,), {})
    monkeypatch.setitem(sys.modules, "openai", openai)
    bp_info = importlib.import_module("src.bp_info")
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(
            content='```json\n[["answer", "ok"]]\n```'
        ))]
    )
    monkeypatch.setattr(bp_info, "_call_llm_with_retry", lambda **_: response)

    result = bp_info._call_llm_with_json(
        client=object(),
        model="test-model",
        system_prompt="system",
        user_prompt="user",
        schema=StructuredAnswer,
    )

    assert result.answer == "ok"
