import asyncio
import logging
import os

# main.py — это точка входа.
# Здесь я запускаю aiogram, подключаю роутеры и не трогаю бизнес-логику.

import uvloop
from aiogram import Bot, Dispatcher
from aiogram.enums import ParseMode
from aiogram.client.default import DefaultBotProperties
from dotenv import load_dotenv

from handlers import router as lead_router


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)


async def main() -> None:
    """
    Базовая инициализация приложения:
    - читаем .env;
    - поднимаем бота и диспетчер;
    - подключаем роутеры;
    - стартуем long polling.
    """
    load_dotenv()

    bot_token = os.getenv("BOT_TOKEN")
    if not bot_token:
        raise RuntimeError("BOT_TOKEN не задан. Заполните .env на основе .env.example")

    # Начиная с aiogram 3.7 parse_mode нужно указывать через DefaultBotProperties,
    # поэтому я передаю его в аргумент default.
    bot = Bot(
        token=bot_token,
        default=DefaultBotProperties(parse_mode=ParseMode.HTML),
    )
    dp = Dispatcher()

    # Подключаем основной роутер с логикой квалификации.
    dp.include_router(lead_router)

    logger.info("Запуск Telegram-бота квалификации лидов...")
    await dp.start_polling(bot)


if __name__ == "__main__":
    """
    Использую uvloop как более быстрый event loop под Linux.
    Для Windows его можно не ставить.
    """
    uvloop.install()
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        logger.info("Бот остановлен.")

