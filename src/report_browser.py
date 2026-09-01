from __future__ import annotations
import time
import logging
from typing import Optional, Dict, List, Any

import pandas as pd
from openai import OpenAI
from openai import RateLimitError, APIError

from src.bp_info import BusinessProcessInfo
from src.prompts import (
    UPDATE_BP_CARD_SYS_PROMPT,
    UPDATE_BP_CARD_USER_PROMPT,
)

logger = logging.getLogger(__name__)


class ReportBrowser:
    """
    Класс для агрегации информации из отчета о разработке
    """

    def __init__(
        self,
        report: str,
        client: OpenAI,
        model: str,
        bp_card: Optional[str] = None,
        test_data: pd.DataFrame = None
    ):
        """
        report: Текст отчета о разработке
        data: Тестовые данные
        client: Клиент OpenAI 
        model: Название модели
        bp_card: Опицонально - существующая версия карточки бизнес-процесса
        """
        self.report = report
        self.data = test_data
        self.client = client
        self.model = model
        self.bp_card = bp_card

    def _call_llm_with_retry(
        self,
        model: str,
        messages: List[Dict[str, str]],
        max_retries: int = 5,
        base_delay: float = 1.0,
    ) -> Any:
        """
        Вызывает LLM с экспоненциальной задержкой при ошибках.

        Parameters:
            model: Название модели
            messages: Список сообщений
            max_retries: Максимальное количество повторов
            base_delay: Базовая задержка в секундах
        """
        last_error = None

        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    stream=False
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

    def handle_initial_creation(self) -> str:
        """
        Создание первичной карточки бизнес-процесса модели
        Return:
            Строка с карточкой бизнес-процесса
        """
        if self.bp_card:
            raise ValueError(
                "You can't provide card or user feedback on initial creation")
        self._aggregator = BusinessProcessInfo(
            self.report, self.client, self.model, self.data)
        self.bp_card = self._aggregator.agg_info()
        print(self.bp_card)
        return self.bp_card

    def get_first_stage_fields(self) -> Dict[str, str]:
        """
        Возвращает структурированные поля первого этапа в текстовом формате.
        Должна вызываться после handle_initial_creation().
        Return:
            Словарь с текстовыми представлениями всех извлечённых полей первого этапа
        """
        if not hasattr(self, '_aggregator'):
            return {}
        return self._aggregator.get_structured_fields()

    def handle_answer_and_update(self, user_answer: str) -> str:
        """
        Обработка ответа и обновление карточки бизнес-процесса на основе ответа пользователя
        Parameters:
            user_answer: Опционально - комментарии пользотваля о том, что необходимо изменить в текущей карточке
        Return:
            Строка с обновленной карточкой
        """
        if not self.bp_card:
            raise ValueError("Card must not be empty")
        sys_prompt = UPDATE_BP_CARD_SYS_PROMPT
        user_prompt = UPDATE_BP_CARD_USER_PROMPT.format(
            self.bp_card,
            user_answer
        )
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt}
        ]
        response = self._call_llm_with_retry(
            model=self.model,
            messages=messages,
        )
        self.bp_card = response.choices[0].message.content
        return self.bp_card
