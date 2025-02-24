from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes
import logging
import asyncio
import random
import os
from hashids import Hashids
from telegram.error import TelegramError
from dotenv import load_dotenv

load_dotenv()

# تنظیمات لاگ‌گیری
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# اطلاعات بات و کانال‌ها
BOT_TOKEN = os.getenv("BOT1_TOKEN")  
STORAGE_CHANNEL = int(os.getenv("STORAGE_CHANNEL", "0").strip('"'))
REQUIRED_CHANNELS = os.getenv("REQUIRED_CHANNELS", "").split(",")
salt = os.getenv("salt")

# Use the same salt and configuration as Bot 1
hashids = Hashids(salt=salt, min_length=6)

# تنظیمات محدودیت نرخ
RATE_LIMIT = 20  # حداکثر ۲۰ پیام در دقیقه (مطابق محدودیت تلگرام)
semaphore = asyncio.Semaphore(RATE_LIMIT // 2)  # کنترل همزمانی

def decode_movie_token(token: str) -> list:
    """Decode token into list of message IDs"""
    decoded = hashids.decode(token)
    return list(decoded) if decoded else []

# تابع بررسی عضویت کاربر
async def get_unjoined_channels(user_id: int, context: ContextTypes.DEFAULT_TYPE):
    unjoined_channels = []
    for channel in REQUIRED_CHANNELS:
        try:
            member = await context.bot.get_chat_member(chat_id=channel, user_id=user_id)
            if member.status not in ["member", "administrator", "creator"]:
                unjoined_channels.append(channel)
        except TelegramError as e:
            logger.error(f"خطا در بررسی عضویت کاربر {user_id} در {channel}: {e}")
            unjoined_channels.append(channel)
    return unjoined_channels

# تابع ایجاد دکمه‌های عضویت و تأیید
def get_verification_menu(unjoined_channels):
    keyboard = [[InlineKeyboardButton(f"✅ عضویت در {ch}", url=f"https://t.me/{ch[1:]}")] for ch in unjoined_channels]
    keyboard.append([InlineKeyboardButton("🔄 بررسی مجدد عضویت", callback_data="verify")])
    return InlineKeyboardMarkup(keyboard)

# ارسال محتوا با تلاش مجدد و مدیریت محدودیت نرخ
async def send_with_retry(context, content_code, user_id):
    backoff = 1
    max_retries = 5
    for _ in range(max_retries):
        try:
            async with semaphore:
                await asyncio.sleep(random.uniform(0.1, 0.5))
                return await context.bot.forward_message(
                    chat_id=user_id,
                    from_chat_id=STORAGE_CHANNEL,
                    message_id=int(content_code)
                )
        except TelegramError as e:
            if "Too Many Requests" in str(e):
                wait = int(str(e).split()[-2])
                await asyncio.sleep(wait + backoff)
                backoff *= 2
            else:
                logger.error(f"ارسال ناموفق {content_code}: {e}")
                break
    return None

# ارسال محتوا به کاربر با تأخیر و کنترل محدودیت نرخ
async def send_timed_messages(user_id: int, context: ContextTypes.DEFAULT_TYPE, content_codes: list):
    try:
        tasks = [send_with_retry(context, code, user_id) for code in content_codes]
        sent_messages = await asyncio.gather(*tasks)
        sent_messages = [msg for msg in sent_messages if msg]

        if not sent_messages:
            await context.bot.send_message(user_id, "⚠️ خطا در ارسال محتوا")
            return

        countdown = await context.bot.send_message(user_id, "⏳ این محتوا پس از 5 دقیقه حذف خواهد شد!")
        await asyncio.sleep(300)

        # حذف پیام‌ها
        for msg in sent_messages:
            try: await context.bot.delete_message(user_id, msg.message_id)
            except: pass
        await countdown.delete()

    except Exception as e:
        logger.error(f"خطای کلی: {e}")
        await context.bot.send_message(user_id, "⚠️ خطا در پردازش")

# دستور /start
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    args = context.args if context.args else []
    content_codes = []
    if args:
        tokens = args[0].split('_')
        for token in tokens:
            decoded_ids = decode_movie_token(token)
            content_codes.extend(str(id) for id in decoded_ids)
    
    try:
        unjoined_channels = await get_unjoined_channels(user.id, context)
        if not unjoined_channels:
            if content_codes:
                await update.message.reply_text("📩 محتوا در حال ارسال...")
                await send_timed_messages(user.id, context, content_codes)
            else:
                await update.message.reply_text("✅ خوش آمدید! برای دریافت رسانه، از لینک‌های محتوایی استفاده کنید.")
        else:
            await update.message.reply_text(
                "⚠️ برای دسترسی به محتوا، ابتدا در کانال‌های زیر عضو شوید:",
                reply_markup=get_verification_menu(unjoined_channels)
            )
    except Exception as e:
        logger.error(f"خطا در پردازش دستور /start برای کاربر {user.id}: {e}")
        await update.message.reply_text("⚠️ مشکلی رخ داده است. لطفاً دوباره تلاش کنید.")

# بررسی عضویت از طریق دکمه تأیید
async def verify_membership(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()
    try:
        unjoined_channels = await get_unjoined_channels(query.from_user.id, context)
        if not unjoined_channels:
            await query.edit_message_text("✅ عضویت شما تأیید شد! اکنون می‌توانید به محتوای رسانه‌ای دسترسی داشته باشید.")
        else:
            await query.edit_message_text(
                "⚠️ شما هنوز در تمام کانال‌ها عضو نشده‌اید. لطفاً ابتدا عضو شوید:",
                reply_markup=get_verification_menu(unjoined_channels)
            )
    except Exception as e:
        logger.error(f"خطا در بررسی عضویت کاربر {query.from_user.id}: {e}")
        await query.edit_message_text("⚠️ مشکلی در بررسی عضویت رخ داده است. لطفاً مجدداً تلاش کنید.")

# راه‌اندازی ربات
def main():
    application = (
        Application.builder()
        .token(BOT_TOKEN)
        .concurrent_updates(True)  # فعال‌سازی پردازش غیرهمزمان
        .pool_timeout(100)
        .get_updates_http_version("1.1")
        .http_version("1.1")
        .build()
    )
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CallbackQueryHandler(verify_membership, pattern="verify"))
    application.run_polling()

if __name__ == '__main__':
    main()