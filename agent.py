import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

# В этом модуле я полностью изолирую логику работы с LLM:
# - системный промпт с бизнес-правилами (перевёрнутая воронка, бюджет, опыт и т.д.);
# - интеграция с Chroma через retriever (RAG);
# - нормализация ответов в строгую JSON-схему, удобную для Telegram-слоя.

from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage, SystemMessage, AIMessage

from database import get_vectorstore


load_dotenv()


@dataclass
class LeadData:
    """
    Структурированное представление лида.
    Я так разделяю ответственность: Telegram-боту не нужно понимать
    внутреннюю кухню LLM, он получает уже нормализованную сущность.
    """

    qualified: bool
    reason: str
    reply_to_user: str
    name: Optional[str] = None
    experience: Optional[str] = None
    budget: Optional[str] = None
    goals: Optional[str] = None
    raw_llm_output: Optional[str] = None


class LeadQualificationAgent:
    """
    Сердце воронки — агент-квалификатор.

    Задачи:
    - объединить Lead Magnet (пользы из книг), Profiling (вопросы) и Scoring (решение);
    - обеспечить перевёрнутую воронку: не продавать в лоб, а отсеивать слабых;
    - выдать Telegram-слою понятный результат: текст ответа и флаг qualified.
    """

    def __init__(self) -> None:
        """
        В конструкторе я переключаюсь на Gemini через langchain-google-genai.

        Это даёт нам:
        - доступ к актуальным моделям Gemini (в том числе бесплатным/дешёвым),
        - простой конфиг через GEMINI_API_KEY и имя модели.
        """
        chat_model_name = os.getenv("GEMINI_CHAT_MODEL", "gemini-1.5-flash")
        google_api_key = os.getenv("GEMINI_API_KEY")

        if not google_api_key:
            # Я предпочитаю упасть сразу и явно, чем тихо работать "непонятно как".
            # В противном случае ты получишь непредсказуемые ошибки уже в момент диалога с лидом.
            raise RuntimeError("GEMINI_API_KEY не задан. Заполни .env и перезапусти бота.")

        # Я беру gemini-1.5-flash как дефолт — это быстрая и относительно дешевая модель,
        # которой достаточно для фильтрации лидов и работы с маркетинговыми текстами.
        # Температуру оставляю 0.4, как и раньше.
        # Важно явно прокинуть google_api_key, чтобы не зависеть от глобальной
        # конфигурации среды. Так ты контролируешь, какой ключ и проект используются.
        self.llm = ChatGoogleGenerativeAI(
            model=chat_model_name,
            temperature=0.4,
            google_api_key=google_api_key,
        )

        # Подключаем Chroma, если база знаний уже есть.
        self.vectorstore = get_vectorstore()
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 5}) if self.vectorstore else None

        # Простейшая in-memory история диалогов по user_id.
        # На проде я бы вынес это в Redis с TTL.
        self.conversations: Dict[int, List[Any]] = {}

        self.training_channel_url = os.getenv("TRAINING_CHANNEL_URL", "").strip()
        self.training_guide_url = os.getenv("TRAINING_GUIDE_URL", "").strip()

        self.system_prompt = self._build_system_prompt()

    def _build_system_prompt(self) -> str:
        """
        Здесь я максимально жёстко фиксирую бизнес-логику квалификации.
        Модель получает не "общий чат", а роль конкретного маркетолога
        с чёткими критериями.
        """
        base_prompt = (
            "Ты — опытный маркетолог-аналитик и руководитель отдела лидогенерации "
            "в сетевом маркетинге (MLM). Ты работаешь по принципу перевёрнутой воронки: "
            "твоё время и время основателя — самый дорогой ресурс, поэтому твоя задача "
            "не продать любой ценой, а отсеять слабых и выявить амбициозных лидеров.\n\n"
            "Ты используешь методики из книги [Название] для проверки мотивации: "
            "оцениваешь адекватность ожиданий, готовность учиться и внедрять, способность "
            "держать удар реальности, а не жить в иллюзиях \"кнопки бабло\".\n\n"
            "Твоя работа делится на три блока:\n"
            "1) Lead Magnet — ты даёшь пользу из книг по маркетингу/MLM, отвечаешь по делу, "
            "но не делаешь бесплатный полноценный консалт.\n"
            "2) Profiling — задаёшь 3–4 ключевых вопроса: про опыт в MLM/маркетинге/бизнесе, "
            "про бюджет на трафик, про готовность выделять время и про цели.\n"
            "3) Scoring — на основании ответов принимаешь решение: рентабелен ли человек.\n\n"
            "Критерии ПОЛОЖИТЕЛЬНОЙ квалификации (лид рентабелен):\n"
            "- есть реальный бюджет на трафик (не последние $50, а рабочий диапазон);\n"
            "- человек готов выделять время на обучение и внедрение (не \"хочу на пассиве\");\n"
            "- цели достаточно конкретны и реалистичны (нет магического мышления);\n"
            "- психоэмоционально человек ровный: без агрессии, тотальной жертвенности и т.п.\n\n"
            "Жёсткое правило отсечения:\n"
            "- если по ходу диалога выясняется, что бюджет < $100 И при этом нет опыта в MLM/маркетинге/бизнесе,\n"
            "  и не видно внятной готовности интенсивно учиться — ты считаешь такого лида нерентабельным.\n\n"
            "Если лид нерентабелен:\n"
            "- даёшь ему короткий, но конкретный совет по маркетингу/MLM на 1–3 шага (на основе книг);\n"
            "- рекомендуешь обучающий канал и/или PDF-гайд для самостоятельного прогрева;\n"
            "- вежливо прощаешься, без попыток дожать.\n\n"
            "Если лид рентабелен:\n"
            "- аккуратно собираешь его данные (имя, опыт, бюджет, цели);\n"
            "- подчёркиваешь, что глубинный разбор и стратегия доступны после короткого созвона со специалистом/основателем;\n"
            "- мягко просишь удобный контакт для связи (телефон, Telegram @username или другой надёжный способ).\n\n"
            "СТИЛЬ общения:\n"
            "- профессиональный, уверенный, человеческий; без инфоцыганщины и без ИИ-канцелярита;\n"
            "- не используй фразы про \"я всего лишь ИИ\" и т.п.;\n"
            "- отвечай кратко и по делу, но с уважением к человеку.\n\n"
            "Ты всегда отвечаешь СТРОГО в JSON-формате БЕЗ пояснений и без лишнего текста. "
            "Никаких комментариев, markdown и кода — только чистый JSON.\n\n"
            "Структура JSON, которую ты ДОЛЖЕН вернуть в КАЖДОМ ответе:\n"
            "{\n"
            '  \"qualified\": true/false,\n'
            '  \"reason\": \"строка\",           // почему ты решил, что лид (не)рентабелен на этом шаге\n'
            '  \"reply_to_user\": \"строка\",    // текст ответа, который сразу уходит пользователю в Telegram\n'
            '  \"name\": \"строка или null\",      // имя лида, если уже известно\n'
            '  \"experience\": \"строка или null\", // краткое описание опыта\n'
            '  \"budget\": \"строка или null\",     // формулировка бюджета (например, \"$300–500 в месяц\")\n'
            '  \"goals\": \"строка или null\"       // формулировка целей своими словами\n'
            "}\n\n"
            "Если данных пока недостаточно для окончательного решения, ты можешь временно "
            "ставить qualified=false, но в reply_to_user задавать уточняющие вопросы и двигать диалог вперёд.\n\n"
        )

        # Я дополнительно подсказываю модели, как ей использовать обучающий канал и гайд,
        # если они настроены в окружении. Это позволяет гибко менять ссылки без правки кода.
        if self.training_channel_url or self.training_guide_url:
            extra = "Дополнительные указания по обучающим материалам:\n"
            if self.training_channel_url:
                extra += (
                    f"- если человек нерентабелен, рекомендуй обучающий Telegram-канал: {self.training_channel_url}\n"
                )
            if self.training_guide_url:
                extra += (
                    f"- можешь упоминать PDF-гайд/методичку для старта: {self.training_guide_url}\n"
                )
            extra += "Упоминать эти ссылки нужно естественно и уместно, без навязчивости.\n\n"
            base_prompt += extra

        base_prompt += (
            "Иногда к тебе будут приходить люди только за быстрым советом. "
            "В таком случае ты всё равно мягко переводишь разговор к профилированию "
            "(опыт, бюджет, время, цели), но без давления.\n"
        )

        return base_prompt

    def _get_conversation_history(self, user_id: int, limit: int = 10) -> List[Any]:
        """
        Достаём последние N сообщений для сохранения контекста.
        Этого достаточно для связного диалога без тяжёлой памяти.
        """
        history = self.conversations.get(user_id, [])
        return history[-limit:]

    def _append_to_history(self, user_id: int, message: Any) -> None:
        """
        Добавляем сообщение (HumanMessage или AIMessage) в историю пользователя.
        """
        if user_id not in self.conversations:
            self.conversations[user_id] = []
        self.conversations[user_id].append(message)

    @staticmethod
    def _build_context_from_docs(docs: List) -> str:
        """
        Собираем релевантные чанки из Chroma в один текстовый блок.
        Я намеренно ограничиваю длину, чтобы не раздувать промпт и не убивать latency.
        """
        if not docs:
            return ""

        contents: List[str] = []
        for d in docs:
            contents.append(d.page_content)

        joined = "\n\n---\n\n".join(contents)
        # Ограничение по длине оставляю "на глаз" — дальше можно
        # отрегулировать под конкретную модель.
        return joined[:6000]

    @staticmethod
    def _safe_parse_json(raw: str) -> Dict[str, Any]:
        """
        Аккуратно парсим JSON из ответа модели.
        На практике модель иногда оборачивает JSON в ```json ... ```,
        поэтому я заранее чищу форматирование.
        """
        text = raw.strip()

        if text.startswith("```"):
            lines = text.splitlines()
            if len(lines) >= 2:
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1].startswith("```"):
                    lines = lines[:-1]
                text = "\n".join(lines).strip()

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Здесь я предпочитаю не падать, а дать пользователю аккуратный ответ
            # и залогировать проблему.
            print("[agent] Ошибка JSONDecodeError, сырой ответ:", raw)
            return {
                "qualified": False,
                "reason": "LLM вернул некорректный JSON, поэтому лид помечен как не прошедший фильтр.",
                "reply_to_user": (
                    "Похоже, обработка ответа дала сбой. "
                    "Напишите, пожалуйста, чуть более кратко и по сути, я попробую ещё раз."
                ),
                "name": None,
                "experience": None,
                "budget": None,
                "goals": None,
            }

    @staticmethod
    def _norm_optional_str(value: Any) -> Optional[str]:
        """
        Приводим произвольное значение к опциональной строке.
        Это защищает нас от странных типов в JSON и пустых строчек.
        """
        if value is None:
            return None
        s = str(value).strip()
        return s if s else None

    def _json_to_lead_data(self, data: Dict[str, Any], raw: str) -> LeadData:
        """
        Маппинг "сырого" JSON в наш dataclass LeadData.
        Так мы фиксируем контракт между LLM и слоем Telegram-бота.
        """
        qualified = bool(data.get("qualified", False))
        reason = str(data.get("reason", "") or "")
        reply_to_user = str(data.get("reply_to_user", "") or "")

        name = self._norm_optional_str(data.get("name"))
        experience = self._norm_optional_str(data.get("experience"))
        budget = self._norm_optional_str(data.get("budget"))
        goals = self._norm_optional_str(data.get("goals"))

        return LeadData(
            qualified=qualified,
            reason=reason,
            reply_to_user=reply_to_user,
            name=name,
            experience=experience,
            budget=budget,
            goals=goals,
            raw_llm_output=raw,
        )

    async def _aget_relevant_docs(self, query: str) -> List:
        """
        Асинхронная обёртка для поиска релевантных документов в Chroma.
        Chroma — синхронная библиотека, поэтому я уношу её вызов в executor,
        чтобы не блокировать event loop aiogram.
        """
        if self.retriever is None:
            return []

        import asyncio

        loop = asyncio.get_running_loop()
        docs = await loop.run_in_executor(None, self.retriever.get_relevant_documents, query)
        return docs

    async def process_message(self, user_id: int, user_text: str) -> LeadData:
        """
        Основной публичный метод агента.

        На входе: raw-текст от пользователя и его Telegram user_id.
        На выходе: LeadData с флагом qualified и готовым текстом-ответом.
        """
        from langchain.schema import HumanMessage  # локальный импорт, чтобы избежать циклов

        user_msg = HumanMessage(content=user_text)
        self._append_to_history(user_id, user_msg)

        history = self._get_conversation_history(user_id)

        # Подтягиваем контекст из книг (если база знаний уже есть).
        rag_context = ""
        if self.retriever is not None:
            relevant_docs = await self._aget_relevant_docs(user_text)
            rag_context = self._build_context_from_docs(relevant_docs)

        messages: List[Any] = []

        full_system_prompt = self.system_prompt
        if rag_context:
            full_system_prompt += (
                "\n\nНиже дан дополнительный контекст из книг по маркетингу и MLM. "
                "Используй его для формулировки советов и вопросов, но не цитируй огромные куски дословно:\n\n"
                f"{rag_context}"
            )

        messages.append(SystemMessage(content=full_system_prompt))
        messages.extend(history)

        # Вызов модели — самое "хрупкое" место пайплайна:
        # здесь возможны лимиты, сетевые сбои и т.д.
        # Для Gemini на бесплатном тарифе типичная боль — 429 (rate limit).
        # Я добавляю мягкий ретрай с backoff, чтобы бот не умирал и не бесил лидов.
        max_attempts = 3
        backoff_seconds = 2.0
        last_err_text = ""
        raw_content: Optional[str] = None

        for attempt in range(1, max_attempts + 1):
            try:
                ai_response = await self.llm.ainvoke(messages)
                raw_content = ai_response.content
                break
            except Exception as e:
                last_err_text = str(e)
                print(f"[agent] Ошибка при вызове LLM (attempt {attempt}/{max_attempts}):", last_err_text)

                is_rate_limited = (
                    "429" in last_err_text
                    or "rate" in last_err_text.lower()
                    or "resource_exhausted" in last_err_text.lower()
                    or "quota" in last_err_text.lower()
                )

                if attempt < max_attempts and is_rate_limited:
                    import asyncio

                    await asyncio.sleep(backoff_seconds)
                    backoff_seconds *= 2
                    continue

                break

        if raw_content is None:
            err_text = last_err_text
            print("[agent] Фатальная ошибка при вызове LLM:", err_text)

            if (
                "429" in err_text
                or "rate" in err_text.lower()
                or "resource_exhausted" in err_text.lower()
                or "quota" in err_text.lower()
            ):
                reply = (
                    "Сейчас сервис перегружен/упёрся в лимиты (free tier). "
                    "Попробуйте через 1–2 минуты."
                )
            else:
                reply = (
                    "Сейчас есть техническая проблема при обработке сообщения.\n"
                    "Попробуйте повторить запрос через пару минут."
                )

            return LeadData(
                qualified=False,
                reason=f"Ошибка LLM: {err_text[:300]}",
                reply_to_user=reply,
                name=None,
                experience=None,
                budget=None,
                goals=None,
                raw_llm_output=None,
            )

        self._append_to_history(user_id, AIMessage(content=raw_content))

        data = self._safe_parse_json(raw_content)
        lead = self._json_to_lead_data(data, raw=raw_content)
        return lead

