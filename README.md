# G-AIVA

В ноду приходит отчёт о разработке и опциональная эталонная корзина
`reference_umr`. Нода собирает карточку бизнес-процесса и структурированные
поля `extracted_fields`; в wiring v4 этот выход передаётся в
`laim-kriteria-selector.doc_browser_result`.

---

## Поддерживаемые LLM
* GigaChat

Для OpenAI API используется локальный прокси `gpt2giga` на порту 8090.
Процесс создаётся на время вызова и завершается в `finally`.

* Opensource LLM из SberDS

Open-source LLM вызываются напрямую через OpenAI-совместимый AI Gateway.
`AI_GATEWAY_URL` обязателен; суффикс `/api/v1` можно передать сразу либо он
будет добавлен нодой. `AI_GATEWAY_API_KEY` используется, если задан, иначе для
внутреннего контура применяется `123`. Модель карточки задаёт `llm_selector`,
модель extraction-пайплайнов — `pr_extraction_model`.
