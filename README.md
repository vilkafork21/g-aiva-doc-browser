# g-aiva-doc-browser

Нода «Получение карточки бизнес-процесса» контура LAIM. Принимает **отчёт о
разработке** GenAI-агента (DOCX/PDF/XLSX в порту `report_dict`) и, опционально,
**эталонную корзину** `reference_umr` из `laim-baskets-adapter` (порт
`test_data`). Отдаёт **карточку бизнес-процесса** (`all_results.bp_card`) и
**структурированные поля** `extracted_fields`, из которых
`laim-kriteria-selector` берёт имя, описание, формулу и порог ключевой метрики (КМ).

## Зачем нода нужна

Отчёты о разработке пишутся в свободной форме: КМ и её порог стоят в тексте,
в таблице или в Word-формуле, тип задачи нигде не назван явно. Нода снимает эту
разницу: независимые LLM-вызовы читают весь текст отчёта и отвечают по
фиксированным JSON-схемам, а детерминированный код собирает карточку и поля;
корзина нужна только как пример строк для определения типа датасета.

Два решения. **Словари вместо свободного текста:** группа и тип задачи — из
`src/metadata/task_types.json` (8 групп, 35 подтипов), тип датасета — из семи
значений промпта (`Ping-Pong`, `RuBQ`, `NER`, `Structured NER`,
`Classification`, `LongAnswer`, `MMLU`). **Два режима отказа:** вызовы
карточки обязательны (ошибка после ретраев роняет ноду), пять экстракторов
`extracted_fields` деградируют в заглушки «Ошибка при извлечении: …».

## Место в контуре

По `monitoring/shared/port_wiring.json` (`laim-sberds-wiring.v7`):

```text
development_report (источник данных) ──► report_dict ─┐
laim-baskets-adapter.reference_umr ────► test_data   ─┤ g-aiva-doc-browser
                                                       ├─► extracted_fields ─► laim-kriteria-selector.doc_browser_result
                                                       └─► all_results      ─► в port_wiring не подключён
```

Селектор (`laim-kriteria-selector/main.py`, `_doc_browser_context`) читает из
`extracted_fields.bp_card_fields` только `evaluation_metric`, `metric_desc`,
`metric_func`, `threshold`; остальные поля `extracted_fields` никем не потребляются.

## Порты и настройки

### Входы

| Порт | Обязателен | Что приходит с платформы |
|---|---|---|
| `report_dict` | да | `dict` вида `{"bin": bytes, "ext": ".docx"}` — байты отчёта и расширение (`.docx`/`.doc`, `.pdf`, `.xlsx`/`.xls`). Словарь **без ключа `bin`** не ошибка: текст отчёта молча становится строкой `Unable to parse report`, и все LLM-вызовы идут по ней. Значение не-`dict` уходит в парсер как путь без расширения и роняет ноду |
| `test_data` | нет | `DataFrame` корзины `reference_umr` (для локального запуска принимаются также JSON-строка и `dict`/`list` записей). Без него тип датасета не классифицируется |

### Выходы

| Порт | Тип | Что отдаёт |
|---|---|---|
| `all_results` | default | `{"bp_card": str}` — карточка бизнес-процесса в Markdown |
| `extracted_fields` | default | `dict` с ключами `title`, `summary`, `key_points`, `generation_hyperparams`, `sample_description`, `ml_architecture`, `bp_card_fields` (см. «Форматы выхода») |

### Настройки

| Настройка | По умолчанию | Читается кодом | Зачем |
|---|---|---|---|
| `llm_selector` | `GigaChat-3-Ultra` | да | Модель вызовов карточки; имя с `giga` включает прокси `gpt2giga` |
| `pr_extraction_model` | `GigaChat-3-Ultra` | да | Модель экстракторов; в коде по умолчанию равна `llm_selector` |
| `pr_extraction_temperature` | `0.0` | да | Температура экстракторов |
| `pr_extraction_max_tokens` | `16384` | да | `max_tokens` экстракторов |
| `pr_temperature` | `0.1` | **нет** | Объявлена в `descriptor.json`, кодом не используется |
| `pr_top_p` | `0.05` | **нет** | То же |
| `pr_repetition_penalty` | `1.0` | **нет** | То же |

Вызовы карточки уходят без `temperature`/`max_tokens` — действуют значения
шлюза. `main` принимает также `current_card` и `user_answer` (правка карточки
по замечанию), но портов для них в `descriptor.json` нет.

## Как проходит прогон

```text
1. Парсинг       report_dict → дерево секций → сплошной текст отчёта
2. Клиент        AI_GATEWAY_URL (+ /api/v1); для *giga* — процесс gpt2giga на 127.0.0.1:8090
3. Карточка      4–5 LLM-вызовов bp_info.agg_info: группа → тип задачи → метрика → [тип датасета] → резюме
4. Экстракторы   5 LLM-вызовов extraction_agent: key_points → summary → hyperparams → sample → ml_architecture
5. Сборка        extracted_fields + bp_card_fields; прокси останавливается в finally
```

**Парсинг** (`src/parsing/`): DOCX через `python-docx` — абзацы стиля
`Heading N` открывают секцию, таблицы сериализуются через табуляцию, формулы
Office Math (OMML) переводятся в линейный текст (`F1 = 2*P*R/(P+R)`); PDF —
постранично (`PyPDF2`), XLSX — по листам (`openpyxl`). В каждый вызов LLM
уходит весь текст отчёта одной строкой (`get_all_plain_text()`).

**Карточка** (`src/bp_info.py`, `agg_info`): `get_task_group` → `get_task_type`
(подтипы выбранной группы) → `get_metric` → `get_dataset_type` только при
`test_data` (в промпт уходит `test_data.head().to_markdown()`) → резюме
текстовым вызовом. Структурные вызовы идут с `response_format: json_schema`,
ответ принимается и списком пар `[["ключ", "значение"], …]`, и в
Markdown-ограждении. Формула КМ — детерминированно: `<имя метрики> = …` в
тексте отчёта → поле `metric_func` ответа → формула в `metric_desc` → `-`.

**Экстракторы** (`src/agents/`): `KeyPointsExtractor` (`ml_task` ограничен
подтипами `task_types.json`), `SummaryGenerator`, `HyperparamsExtractor`,
`SampleExtractor`, `MlArchitectureExtractor`; каждый сам ловит исключения.

### Пример лога успешного прогона

Формат строк — из кода; значения условные. Нода пишет через `logging`,
карточка дополнительно печатается в stdout через `print`.

```text
INFO src.main: Запускаем извлечение структурированных полей из отчёта.
INFO src.agents.extraction_agent: [extraction_agent] start | report_len=48213
INFO src.agents.extraction_agent: [extraction_agent] pipeline 1: KeyPointsExtractor
INFO src.agents.extractors.key_points_extractor: [KeyPointsExtractor] LLM вызов | model=GigaChat-3-Ultra | temperature=0.0 | max_tokens=16384
INFO src.agents.extractors.key_points_extractor: [KeyPointsExtractor] Завершено за 14.21с
...
INFO src.agents.extraction_agent: [extraction_agent] done
INFO src.main: Извлечение завершено.
```

Транспортный ретрай и деградация экстрактора (ошибка экстрактора пишется
уровнем `INFO` — по уровню лога её не отфильтровать):

```text
WARNING src.bp_info: LLM error (attempt 1/5): Error code: 429 - rate limit. Retrying in 1.0s...
INFO src.agents.extractors.sample_extractor: [SampleExtractor] Ошибка при вызове LLM за 3.10с: JSON doesn't match schema SampleDescriptionOutput: {...}
WARNING src.agents.extractors.summary_generator: [SummaryGenerator] key_points содержат ошибки, пропускаем генерацию
```

## Форматы выхода и контракты

`all_results.bp_card` — Markdown: резюме отчёта, «Информация о решаемой задаче»
(группа, тип, обоснования), «Информация о данных» (только при `test_data`),
«Информация о ключевой метрике» (название, порог, описание с формулой).

`extracted_fields` — всегда все семь ключей:

| Ключ | Тип | Содержимое |
|---|---|---|
| `title` | str | Константа `Описание бизнес-процесса` |
| `summary` | str | 4–6 предложений: проблема «as is», модель, механизм применения |
| `key_points` | dict | Ключи `Природа целевой переменной`, `Бизнес задача`, `ML-задача`, `Сценарий использования модели`, `Сегмент данных модели` |
| `generation_hyperparams` | dict | `{имя: значение}`; пустой, если в отчёте нет |
| `sample_description` | list | Записи `{split, thema, volume, dates}`; `split` — `train`/`val`/`oos`/`oot` |
| `ml_architecture` | str | Описание архитектуры свободным текстом |
| `bp_card_fields` | dict | Все значения — строки: `task_group`, `task_type` (подтипы через `, `), `evaluation_metric`, `metric_desc`, `threshold` (строка числа, например `0.85`), `metric_func` (формула или `-`), `dataset_type` (значение или `-`). Единица наблюдения ноды — один отчёт |

## Падение против деградации

Собственных `reason_code` у ноды нет: исключение уходит платформе как есть.

| Причина падения | Исключение |
|---|---|
| Не задана `AI_GATEWAY_URL` | `ValueError` до первого LLM-вызова |
| `report_dict` не `dict` или расширение вне списка поддерживаемых | `AttributeError` / `ValueError` из `parse_file` |
| Вызов карточки: `RateLimitError`/`APIError` после 5 попыток (1, 2, 4, 8 с) | исходное исключение SDK |
| Вызов карточки: пустой ответ, невалидный JSON, ответ не по схеме | `ValueError` сразу, без повтора |
| Резюме карточки в `agg_info`: любая ошибка | исходное исключение — этот вызов идёт без ретраев |

| Событие | Реакция |
|---|---|
| `report_dict` без ключа `bin` | текст отчёта = `Unable to parse report`; ни warning, ни ошибки |
| `test_data` не подан | `dataset_type = "-"`, раздела «Информация о данных» в карточке нет |
| Экстрактор упал (транспорт после 5 ретраев, JSON, схема) | `key_points` — пять значений `Ошибка при извлечении: …`; `generation_hyperparams` — `{"error": …}`; `sample_description` — `[{"split": "error", …}]`; `ml_architecture` — `Ошибка при извлечении архитектуры: …` |
| `key_points` с ошибкой | `summary = "Суммаризация недоступна: ключевые параметры не были извлечены."`, LLM не вызывается |
| Порт прокси 8090 не отвечает через 1 с после старта | только `INFO Proxy is not running`; падение случится на первом вызове |

Внешний `_retry` в `extraction_agent.py` (5 попыток, пауза 2 с) не срабатывает:
экстракторы сами перехватывают все исключения.

## Внешние сервисы

- **Контурный шлюз** (`AI_GATEWAY_URL`, обязательна): OpenAI-совместимый
  `/chat/completions`; суффикс `/api/v1` добавляется, если его нет. Ключ —
  `AI_GATEWAY_API_KEY`, при отсутствии `123`. Модели без `giga` в имени — напрямую.
- **GigaChat через `gpt2giga`**: если `llm_selector` содержит `giga`, в дочернем
  процессе (`multiprocessing`, daemon) поднимается `gpt2giga.api_server` на
  `127.0.0.1:8090` с `GIGACHAT_BASE_URL` = адрес шлюза, `GIGACHAT_MODEL` =
  модель, `GIGACHAT_VERIFY_SSL_CERTS=False`, `GPT2GIGA_MODE=DEV`. Клиент ходит
  в прокси с ключом `123`; учётные данные GigaChat нода не задаёт — `gpt2giga`
  берёт их из окружения контейнера. Процесс завершается в `finally`
  (`terminate` + `join(5)`). Клиент один на карточку и экстракторы: при
  `llm_selector` с `giga` экстракторы тоже идут через прокси.
- **Ретраи**: 5 попыток с backoff 1/2/4/8 с только на `RateLimitError`/`APIError`;
  таймаут запроса — SDK по умолчанию. **Детерминированность**: экстракторы —
  `temperature` из настройки (`0.0`); вызовы карточки температуру не передают.

## Наблюдаемость

Порта журнала нет; след прогона — лог платформы и содержимое выходных портов.
В логе: статус прокси, маркеры `[extraction_agent] pipeline N`, для каждого
экстрактора модель, длина входа и время, warning ретраев. На сотне прогонов
триаж по портам: значения `extracted_fields`, начинающиеся с `Ошибка`,
`split == "error"`, `summary` с `Суммаризация недоступна` — деградировавшие
экстракторы; `metric_func == "-"` — формула не найдена; `dataset_type == "-"` —
корзины не было; `Unable to parse report` в `bp_card` — битый вход, не LLM.

## Карта кода

```text
src/main.py                     main(): клиент, карточка, экстракторы, finally для прокси
src/config_gpt2giga.py          start_proxy(): окружение и запуск gpt2giga.api_server
src/report_browser.py           ReportBrowser: первичная карточка, поля первого этапа, правка по ответу
src/bp_info.py, prompts.py      BusinessProcessInfo: 4 структурных вызова + резюме, формула КМ; промпты карточки
src/parsing/                    parser.py (диспетчер по расширению), parse_docx/pdf/xlsx_with_sections.py
src/agents/extraction_agent.py  порядок пяти экстракторов, сборка extracted_fields
src/agents/config.py            call_llm_structured / call_llm_text, ретраи, список пар → dict
src/agents/extractors/, prompts/  пять экстракторов и их системные промпты
src/utils/llm_logging.py        единый формат строк лога LLM-вызовов
src/metadata/task_types.json    группы и подтипы задач для карточки и key_points
tests/                          test_main.py (клиент, настройки, прокси, descriptor, requirements), test_structured_config.py
```

Остальные файлы `src/metadata/` и `src/agents/configs/` кодом не читаются и в `sourceFiles` не входят.

## Что делать, если

- **Нода упала с `AI_GATEWAY_URL не задана`** — переменная не проброшена в
  контейнер; без неё нода не делает ни одного вызова.
- **В `bp_card` и `summary` текст `Unable to parse report`** — платформа отдала
  `report_dict` без `bin`; проверять источник `development_report`, не LLM.
- **Нода упала на `JSON doesn't match schema …` / `Failed to parse JSON`** —
  модель карточки не удержала схему, повторов нет: сменить `llm_selector`.
- **В `extracted_fields` заглушки `Ошибка при извлечении`** при живой карточке —
  искать `Ошибка при вызове LLM` в логе (уровень `INFO`); чаще всего дело в
  `pr_extraction_model` или лимите `pr_extraction_max_tokens`.
- **`Proxy is not running` и ошибка соединения с `127.0.0.1:8090`** — пакет
  `gpt2giga` не установился или порт занят.

## Деплой

База — `py312-simple`; синтаксис и stdlib новее Python 3.12 не используются.
Точка входа — `main` в `src/main.py`. `script.runConfiguration.sourceFiles`
перечисляет 28 файлов: все `*.py` из `src/` (27) и `src/metadata/task_types.json`;
тест `test_descriptor_contains_runtime_sources_and_wiring_shapes` требует, чтобы
каждый `*.py` из `src/` был в списке и каждый элемент списка лежал на диске.

Зависимости `requirements.txt`: `python-docx==1.1.2`, `pydantic`,
`pandas==3.0.3`, `openpyxl==3.1.5`, `PyPDF2==3.0.1`, `tabulate==0.9.0`
(`DataFrame.to_markdown` для типа датасета), `loguru` (кодом не импортируется),
`openai`, `gpt2giga`. Проверка перед сборкой из корня ноды: `ruff check .` и
`python -m pytest -q` (7 тестов); то же делает GitHub Actions
(`.github/workflows/ci.yml`, Python 3.12). ZIP для платформы —
`descriptor.json`, `requirements.txt` и `src/` из ветки `dev` без `.git`,
`tests/`, кэшей и `.## Глоссарий

- **Отчёт о разработке** — документ команды агента с описанием задачи, данных,
  архитектуры и критериев приёмки; единственный источник для карточки.
- **Карточка бизнес-процесса** — Markdown-описание задачи агента (`all_results.bp_card`).
- **КМ** — ключевая метрика качества агента: имя, описание, формула, порог.
- **Structured output** — вызов с `response_format: json_schema`; ответ
  проверяется Pydantic-схемой.
- **`gpt2giga`** — локальный прокси из OpenAI-совместимых запросов в API GigaChat.
