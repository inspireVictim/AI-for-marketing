import os
from typing import Any

# Здесь я держу все Telegram-хэндлеры отдельно от LLM-логики.
# Задача этого слоя — чистый UX: принять текст, отдать в агента,
# отправить ответ и, при необходимости, уведомить админа.

from aiogram import Router, F
from aiogram.filters import CommandStart
from aiogram.types import Message
from aiogram.fsm.context import FSMContext
from aiogram.fsm.storage.memory import MemoryStorage

from agent import LeadQualificationAgent, LeadData


# Отдельный роутер под воронку квалификации.
router = Router(name="lead_qualification_router")

# В этом примере достаточно памяти процесса; если проект вырастет,
# можно заменить на Redis-хранилище для FSM.
storage = MemoryStorage()

# Единственный экземпляр агента на процесс.
agent = LeadQualificationAgent()

ADMIN_CHAT_ID = int(os.getenv("ADMIN_CHAT_ID", "0"))


async def _maybe_notify_admin(message: Message, lead: LeadData) -> None:
    """
    Если лид квалифицирован, собираем аккуратное сообщение для админа
    и отправляем его в ADMIN_CHAT_ID.

    Здесь я намеренно не тяну LLM — формат делаю жёстким и лаконичным.
    """
    if not lead.qualified:
        return

    if ADMIN_CHAT_ID == 0:
        # В продовой системе сюда стоит повесить логгер/алерт.
        print("[handlers] ADMIN_CHAT_ID не задан, но лид квалифицирован:")
        print(lead)
        return

    parts = [
        "🔥 Новый квалифицированный лид из Telegram",
        "",
        f"Telegram: @{message.from_user.username or 'без username'} (id: {message.from_user.id})",
    ]

    if lead.name:
        parts.append(f"Имя: {lead.name}")
    if lead.experience:
        parts.append(f"Опыт: {lead.experience}")
    if lead.budget:
        parts.append(f"Бюджет: {lead.budget}")
    if lead.goals:
        parts.append(f"Цели: {lead.goals}")

    parts.append("")
    parts.append(f"Комментарий фильтра: {lead.reason}")

    text = "\n".join(parts)

    await message.bot.send_message(chat_id=ADMIN_CHAT_ID, text=text)


@router.message(CommandStart())
async def cmd_start(message: Message, state: FSMContext) -> Any:
    """
    Стартовый сценарий.

    Здесь я не лезу в LLM, а просто проговариваю человеку правила игры:
    - сначала он получает короткий совет;
    - затем — несколько вопросов про опыт, бюджет и время;
    - дальше — либо прогрев, либо созвон.
    """
    await state.clear()

    intro = (
        "Привет. Я помогаю отфильтровать заявки на работу с нашим маркетингом в MLM.\n\n"
        "Как это устроено:\n"
        "1) Сначала я дам вам короткий совет по вашей ситуации (на основе маркетинговых книг).\n"
        "2) Затем задам несколько вопросов про опыт, бюджет на трафик и готовность уделять время.\n"
        "3) Если мы подходим друг другу — передам вас основателю/специалисту на короткий созвон.\n\n"
        "Напишите, пожалуйста, в свободной форме:\n"
        "- что у вас сейчас за ситуация (MLM, бизнес, найм и т.п.);\n"
        "- чего вы хотите достичь;\n"
        "- как вы оцениваете свою готовность по времени и деньгам."
    )
    await message.answer(intro)


@router.message(F.text)
async def handle_text(message: Message, state: FSMContext) -> Any:
    """
    Основной хэндлер текстовых сообщений.

    Логика проста:
    - любую реплику пользователя прокидываем в агента;
    - отправляем обратно сгенерированный ответ;
    - при положительной квалификации — уведомляем админа.
    """
    user_text = message.text.strip()
    if not user_text:
        return

    # Я оборачиваю вызов агента в try/except.
    # Даже если провайдер LLM "упадёт" по балансу/лимитам/сети,
    # Telegram-бот обязан оставаться живым и отвечать пользователю.
    try:
        lead: LeadData = await agent.process_message(
            user_id=message.from_user.id,
            user_text=user_text,
        )
    except Exception as e:
        err_text = str(e)
        print("[handlers] Ошибка при обработке сообщения:", err_text)
        await message.answer(
            "Сейчас есть техническая проблема при обработке сообщения. Попробуйте повторить позже."
        )
        return

    await message.answer(lead.reply_to_user)
    await _maybe_notify_admin(message, lead)

