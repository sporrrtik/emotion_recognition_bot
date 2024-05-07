# Импортируем функции из нейромоделей
from models.model_cnn import predict_from_audio
from models.model_lstm import predict_from_text

import os
import logging

from aiogram import Bot, Dispatcher, executor, types
from aiogram.types import ContentType
from config import API_TOKEN

# Configure logging
logging.basicConfig(level=logging.INFO)

# Initialize bot and dispatcher
bot = Bot(token=API_TOKEN)
dp = Dispatcher(bot)

emotion_dict = {
    "disgust": "отвращение",
    "happiness": "счастье",
    "anger": "злость",
    "fear": "страх",
    "enthusiasm": "энтузиазм",
    "neutral": "нейтральная",
    "sadness": "грусть",
}

@dp.message_handler(content_types=ContentType.VOICE)
async def echo(message: types.Message):
    # old style:
    # await bot.send_message(message.chat.id, message.text)
    voice = await message.voice.get_file()
    file_path = f"audio/{voice['file_unique_id']}"
    await bot.download_file(file_path=voice['file_path'], destination=file_path)
    msg = await message.answer("Извлечение эмоций из аудио...")
    new_text = f"В вашем аудио наиболее выраженная эмоция - {emotion_dict[predict_from_audio(file_path)[0][0]]}"
    await bot.edit_message_text(chat_id=message.from_user.id, message_id=msg.message_id, text=new_text)
    os.remove(file_path)


@dp.message_handler(commands=['start', 'help'])
async def send_welcome(message: types.Message):
    """
    This handler will be called when user sends `/start` or `/help` command
    """
    await message.reply("Привет! Я могу извлекать эмоции из текста или аудио. Просто перешли мне текстовое или голосовое сообщение, а я подскажу тебе какую эмоцию испытывает твой собеседник)")


@dp.message_handler()
async def text_recognition(message: types.Message):
    # old style:
    # await bot.send_message(message.chat.id, message.text)
    msg = await message.answer("Извлечение эмоций из текста...")
    new_text = f"В вашем тексте наиболее выраженная эмоция - {emotion_dict[predict_from_text(message.text)[0][0]]}"
    await bot.edit_message_text(chat_id=message.from_user.id, message_id=msg.message_id, text=new_text)


if __name__ == '__main__':
    executor.start_polling(dp, skip_updates=True)
