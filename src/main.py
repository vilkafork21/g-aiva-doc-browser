import io
from docx import Document
import os
import socket
import pickle
import logging
import requests
import pandas as pd
import multiprocessing
from openai import OpenAI
from typing import Dict, Optional, Any, Tuple

from src.parsing.parser import parse_file
from src.config_gpt2giga import start_proxy
from src.report_browser import ReportBrowser
from src.agents.extraction_agent import extraction_agent

logger = logging.getLogger(__name__)


def configure_client(
    model: str
) -> Tuple[OpenAI, multiprocessing.Process]:
    """
    Функция для конфигурации клиента OpenAI
    Parameters:
        model: Строка с названием модели 
    Return:
        Клиент OpenAI и процесс с запущенной прокси gpt2giga
    """
    if "giga" in model.lower():
        base_url = f"{os.environ.get("AI_GATEWAY_URL")}/api/v1"
        host = "127.0.0.1"
        port = 8090
        process = multiprocessing.Process(
            target=start_proxy,
            daemon=True,
            kwargs={
                "model": model,
                "base_url": base_url,
                "host": host,
                "port": str(port)
            }
        )
        process.start()

        try:
            with socket.create_connection((host, port), timeout=1):
                logger.info("Proxy is running")
        except Exception as e:
            logger.info("Proxy is not running")

        client = OpenAI(
            base_url=f"http://127.0.0.1:8090",
            api_key="123"
        )
        return client, process
    else:
        AI_GATEWAY_URL = os.environ.get("AI_GATEWAY_URL")
        base_url = f"{AI_GATEWAY_URL}/api/v1"
        client = OpenAI(
            base_url="http://sds-ai-gateway:8097/api/v1",
            api_key="123"
        )
        return client, None


def get_report(report_source):
    """Извлекает и парсит отчёт из различных форматов входных данных."""
    if isinstance(report_source, dict):
        if "bin" not in report_source:
            return None
        binary_report = io.BytesIO(report_source["bin"])
        return parse_file(binary_report, report_source.get("ext", "").lower())
    else:
        return parse_file(report_source)


def _convert_test_data(data: Any) -> pd.DataFrame:
    """
    Converts test_data to pandas DataFrame.
    Accepts DataFrame or dict (parsed JSON).
    """
    if isinstance(data, pd.DataFrame):
        return data
    elif isinstance(data, dict) or isinstance(data, list):
        # Dict can be orient='records' (list of row dicts) or dict of lists
        # pd.DataFrame handles both cases
        return pd.DataFrame(data)
    else:
        raise ValueError(f"Unsupported test_data type: {type(data)}")


def main(
    report_dict: Dict[str, Any],
    test_data: Optional[Any] = None,
    current_card: Optional[str] = None,
    user_answer: Optional[str] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Основная функция doc-browser.

    Выполняет два независимых действия над отчётом о разработке:
      1. Формирует карточку бизнес-процесса (суммаризация, тип задачи, метрика, датасет).
      2. Запускает 5 LLM-пайплайнов извлечения и возвращает структурированные поля
         через порт extracted_fields для последующей передачи в g-aiva-business-process.
    """
    model = kwargs.get("llm_selector", "GigaChat-3-Ultra")

    _parsed = get_report(report_dict)
    report_text = _parsed.get_all_plain_text(
    ) if _parsed is not None else "Unable to parse report"

    # Convert test_data to DataFrame if needed
    test_data_df = _convert_test_data(
        test_data) if test_data is not None else None

    client, proc = configure_client(model)

    # --- 1. Карточка бизнес-процесса (существующая логика) ---
    first_stage_fields = {}
    if not current_card and not user_answer:
        browser = ReportBrowser(
            report_text,
            test_data=test_data_df,
            client=client,
            model=model
        )
        bp_card = browser.handle_initial_creation()
        first_stage_fields = browser.get_first_stage_fields()
    elif current_card and not user_answer:
        bp_card = current_card
    else:
        browser = ReportBrowser(report_text, test_data_df,
                                client=client, model=model)
        bp_card = browser.handle_answer_and_update(user_answer)

    # --- 2. Извлечение структурированных полей (gpt2giga, тот же клиент) ---
    logger.info('Запускаем извлечение структурированных полей из отчёта.')
    extracted_fields = extraction_agent(
        development_report=report_text,
        client=client,
        model=model,
        temperature=kwargs.get('pr_extraction_temperature', 0.0),
        max_tokens=kwargs.get('pr_extraction_max_tokens', 16384),
    )
    logger.info('Извлечение завершено.')

    extracted_fields['bp_card_fields'] = first_stage_fields

    return {
        "all_results": {
            "bp_card": bp_card,
        },
        "extracted_fields": extracted_fields,
    }
