import asyncio
import random
import logging
import redis
from aiogram import Bot, Dispatcher, types
from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup
from aiogram.exceptions import TelegramRetryAfter, TelegramForbiddenError
from aiogram.filters import Command
from hashids import Hashids
import json
from redis.asyncio import Redis
from contextlib import asynccontextmanager
import time
from dataclasses import dataclass
from typing import List, Optional, Union, Dict, Any, Set
import psutil
import os
from aiogram.fsm.state import State, StatesGroup
from aiogram.fsm.context import FSMContext
from aiogram.fsm.storage.memory import MemoryStorage

# Advanced Logging Configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("bot.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Bot Configuration
BOT_TOKEN = "7679219636:AAGF0XhfcYPdGCinA2TbX7rscrO74x6d9cg"
STORAGE_CHANNEL = -1002463367628
REQUIRED_CHANNELS = ["@zapas_kcrang", "@kcrang"]

REDIS_QUEUE_KEY = "video_queue"
REDIS_USER_CACHE_PREFIX = "user_subscriptions:"
REDIS_USER_CACHE_TTL = 3600  # Cache subscription status for 1 hour

# Rate Limiting Config
MAX_MESSAGES_PER_SECOND = 20
MAX_MESSAGES_PER_MINUTE = 300  # Adjust based on Telegram's limits
SUBSCRIPTION_CHECK_COOLDOWN = 1000  # Seconds between subscription checks

# Redis Configuration for Railway
REDIS_URL = "redis://default:ZEWvatsColwbVZEOYlrWpFFDIMhfAyFW@switchback.proxy.rlwy.net:25159"

# Sync Redis Client
redis_client = redis.Redis.from_url(REDIS_URL, decode_responses=True)

# Async Redis Client
async_redis_pool = redis.asyncio.ConnectionPool.from_url(REDIS_URL, decode_responses=True)

@asynccontextmanager
async def get_async_redis():
    """Context manager for getting an async Redis connection from Railway"""
    client = Redis(connection_pool=async_redis_pool)
    try:
        yield client
    finally:
        await client.aclose()



# Persian (Farsi) Message Templates
MESSAGES = {
    "welcome": "✅ خوش آمدید! برای دریافت مدیا، از لینک‌های محتوا استفاده کنید.",
    "media_preparing": "📩 {count} فایل مدیای شما در حال آماده‌سازی است و به زودی ارسال می‌شود...\n محتوا پس از 5 دقیقه حذف خواهد شد.",
    "join_channels": "⚠️ برای دسترسی به محتوا، لطفاً ابتدا در کانال‌های زیر عضو شوید:",
    "pending_media": "شما {count} مدیا در انتظار دارید که پس از عضویت در کانال‌ها ارسال خواهد شد.",
    "membership_verified": "✅ عضویت شما تأیید شد! اکنون می‌توانید به مدیا دسترسی داشته باشید.",
    "still_not_member": "⚠️ شما هنوز عضو همه کانال‌های مورد نیاز نیستید. لطفاً ابتدا عضو شوید:",
    "error_occurred": "⚠️ خطایی رخ داد. لطفاً دوباره تلاش کنید یا با پشتیبانی تماس بگیرید.",
    "wait_before_check": "لطفاً {seconds} ثانیه صبر کنید تا دوباره بررسی کنید",
    "pending_media_sending": "📩 {count} فایل مدیای در انتظار شما به زودی ارسال خواهد شد.",
    "help_message": "🔍 **نحوه استفاده از این ربات**\n\n"
                   "۱. برای دسترسی به محتوا، در کانال‌های مورد نیاز عضو شوید\n"
                   "۲. از لینک‌های محتوا موجود در کانال اصلی  استفاده کنید\n"
                   "۳. ربات مدیای درخواستی را برای شما ارسال می‌کند\n\n"
                   "اگر به کمک نیاز دارید، با @admin تماس بگیرید",
    "status_message": "📊 اندازه صف فعلی: {queue_size} مورد",
    "status_error": "در حال حاضر امکان بررسی وضعیت وجود ندارد.",
    "join_button": "✅ عضویت در {channel}",
    "check_again_button": "🔄 بررسی مجدد",
    "admin_menu": "🛠 **پنل مدیریت**\n\nلطفا عملیات مورد نظر را انتخاب کنید:",
    "admin_stats": """📊 **آمار لحظه‌ای ربات**
    
📥 تعداد در صف: {queue_size}
👥 کاربران فعال: {active_users}
🚫 کاربران مسدود شده: {banned_users}
🧠 مصرف حافظه: {memory_usage} MB
🔄 ورشکرها: {worker_count}""",
    "confirm_clear": "⚠️ آیا مطمئنید می‌خواهید صف را پاک کنید؟",
    "queue_cleared": "✅ صف با موفقیت پاک شد!",
    "broadcast_sent": "✅ پیام به {count} کاربر ارسال شد!",
    "user_banned": "✅ کاربر [{user_id}](tg://user?id={user_id}) مسدود شد!",
    "user_unbanned": "✅ کاربر [{user_id}](tg://user?id={user_id}) از حالت مسدود خارج شد!",
    "channel_updated": "✅ کانال‌ها به‌روزرسانی شدند:\n{channels}",
    "rate_limit_updated": "✅ محدودیت ارسال به {limit} پیام در دقیقه تنظیم شد!",
    "access_denied": "⛔️ دسترسی محدود به ادمین‌ها!",
}




# Bot initialization
bot = Bot(token=BOT_TOKEN)
dp = Dispatcher(storage=MemoryStorage())
hashids = Hashids(salt="Admiral23", min_length=6)

# Optimized Rate Limiter with burst capability and semaphore
class OptimizedRateLimiter:
    def __init__(self):
        self.semaphore = asyncio.Semaphore(MAX_MESSAGES_PER_SECOND * 2)  # Allow burst
        self.minute_limit = MAX_MESSAGES_PER_MINUTE
        self.minute_count = 0
        self.lock = asyncio.Lock()
        
    async def can_send(self) -> bool:
        async with self.lock:
            if self.minute_count >= self.minute_limit:
                return False
            self.minute_count += 1
            return True
            
    async def wait_for_slot(self):
        async with self.semaphore:
            while not await self.can_send():
                await asyncio.sleep(random.uniform(0.05, 0.1))  # Reduced sleep

rate_limiter = OptimizedRateLimiter()

# Data models for better type safety and serialization
@dataclass
class QueueItem:
    user_id: int
    message_id: int
    timestamp: float = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()
    
    def to_json(self) -> str:
        return json.dumps(self.__dict__)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'QueueItem':
        data = json.loads(json_str)
        return cls(**data)

# ========================= Utility Functions =========================

def decode_movie_token(token: str) -> list:
    """Decode token into a list of message IDs."""
    decoded = hashids.decode(token)
    return list(decoded) if decoded else []

async def get_subscription_cache_key(user_id: int) -> str:
    """Generate Redis key for user subscription cache."""
    return f"{REDIS_USER_CACHE_PREFIX}{user_id}"

async def get_cached_subscription_status(user_id: int) -> Optional[List[str]]:
    """Get cached subscription status for user."""
    async with get_async_redis() as redis:
        cached = await redis.get(await get_subscription_cache_key(user_id))
        if cached:
            return json.loads(cached)
    return None

async def set_cached_subscription_status(user_id: int, unjoined_channels: List[str]):
    """Cache user's subscription status."""
    async with get_async_redis() as redis:
        key = await get_subscription_cache_key(user_id)
        await redis.setex(key, REDIS_USER_CACHE_TTL, json.dumps(unjoined_channels))

async def get_unjoined_channels(user_id: int, force_check: bool = False) -> list:
    """Check if user is subscribed to required channels, with caching."""
    if not force_check:
        cached = await get_cached_subscription_status(user_id)
        if cached is not None:
            return cached
    
    unjoined_channels = []
    for channel in REQUIRED_CHANNELS:
        try:
            member = await bot.get_chat_member(chat_id=channel, user_id=user_id)
            if member.status not in ["member", "administrator", "creator"]:
                unjoined_channels.append(channel)
        except TelegramForbiddenError:
            logger.warning(f"Bot is not an admin in {channel}, skipping check.")
        except Exception as e:
            logger.error(f"Error checking {channel} for user {user_id}: {e}")
            unjoined_channels.append(channel)
    
    await set_cached_subscription_status(user_id, unjoined_channels)
    return unjoined_channels

def get_verification_menu(unjoined_channels: list) -> InlineKeyboardMarkup:
    """Generate a verification menu with channel join buttons in Persian."""
    keyboard = [[InlineKeyboardButton(text=MESSAGES["join_button"].format(channel=ch), url=f"https://t.me/{ch[1:]}")] for ch in unjoined_channels]
    keyboard.append([InlineKeyboardButton(text=MESSAGES["check_again_button"], callback_data="verify")])
    return InlineKeyboardMarkup(inline_keyboard=keyboard)

async def handle_media_requests(user_id: int, content_codes: List[str]) -> int:
    """Add media requests to queue and return number of queued items."""
    async with get_async_redis() as redis:
        pipeline = redis.pipeline()
        count = 0
        
        for content_id in content_codes:
            item = QueueItem(user_id=user_id, message_id=int(content_id))
            await pipeline.lpush(REDIS_QUEUE_KEY, item.to_json())
            count += 1
        
        await pipeline.execute()
        return count

async def rate_limited_forward(user_id: int, message_id: int):
    """Forward a message with intelligent rate limiting and schedule deletion."""
    retry_attempts = 5

    for attempt in range(retry_attempts):
        try:
            await rate_limiter.wait_for_slot()
            forwarded_msg = await bot.forward_message(chat_id=user_id, from_chat_id=STORAGE_CHANNEL, message_id=message_id)
            logger.info(f"Successfully forwarded message {message_id} to user {user_id}")
            
            # Schedule message deletion after 5 minutes
            asyncio.create_task(delete_message_later(user_id, forwarded_msg.message_id))
            return True
            
        except TelegramRetryAfter as e:
            wait_time = e.retry_after + random.uniform(0.1, 1.0)
            logger.warning(f"Flood control: waiting {wait_time}s before retrying message {message_id} to {user_id}")
            await asyncio.sleep(wait_time)
            
        except Exception as e:
            if "blocked" in str(e).lower() or "deactivated" in str(e).lower():
                logger.info(f"User {user_id} blocked the bot or deactivated: {e}")
                return False
                
            logger.error(f"Error forwarding message {message_id} to {user_id}: {e}")
            if attempt < retry_attempts - 1:
                await asyncio.sleep(random.uniform(1, 3) * (attempt + 1))
            else:
                return False
                
    return False

async def delete_message_later(chat_id: int, message_id: int):
    """Delete a message after 5 minutes."""
    try:
        await asyncio.sleep(300)  # 5 minutes = 300 seconds
        await bot.delete_message(chat_id=chat_id, message_id=message_id)
        logger.info(f"Deleted message {message_id} from chat {chat_id}")
    except Exception as e:
        logger.error(f"Failed to delete message {message_id} from chat {chat_id}: {e}")

# ========================= Optimized Queue Worker =========================

WORKER_COUNT = 6  # Optimal for 2 vCPU

async def process_queue_worker():
    """Optimized queue processing with batch operations."""
    worker_id = random.randint(1000, 9999)
    logger.info(f"Starting optimized queue worker {worker_id}")
    
    while True:
        try:
            async with get_async_redis() as redis:
                items = []
                for _ in range(10):  # Process in batches of 10
                    raw_item = await redis.rpop(REDIS_QUEUE_KEY)
                    if raw_item:
                        items.append(raw_item)
                        
                if not items:
                    await asyncio.sleep(0.1)  # Shorter sleep
                    continue
                
                tasks = []
                for raw_item in items:
                    try:
                        item = QueueItem.from_json(raw_item)
                        tasks.append(
                            asyncio.create_task(
                                rate_limited_forward(item.user_id, item.message_id)
                            )
                        )
                    except Exception as e:
                        logger.error(f"Worker {worker_id}: Invalid item: {e}")
                
                done, pending = await asyncio.wait(
                    tasks,
                    timeout=10.0,
                    return_when=asyncio.ALL_COMPLETED
                )
                
                for task in pending:
                    task.cancel()
                    
                for task in done:
                    if task.exception():
                        logger.error(f"Task failed: {task.exception()}")
                        
        except redis.RedisError as e:
            logger.error(f"Worker {worker_id}: Redis error: {e}")
            await asyncio.sleep(0.5)
            
        except Exception as e:
            logger.error(f"Worker {worker_id}: Error: {e}")
            await asyncio.sleep(1)

# ========================= Bot Handlers =========================

@dp.message(Command("start"))
async def start(message: types.Message):
    """Handle /start command with ban check"""
    if message.from_user.id in BANNED_USERS:
        await message.answer("⛔️ دسترسی شما مسدود شده است!")
        return
        
    user_id = message.from_user.id
    args = message.text.split()[1:] if len(message.text.split()) > 1 else []
    content_codes = []

    logger.info(f"User {user_id} sent /start command with args: {args}")

    try:
        if args:
            tokens = args[0].split('_')
            for token in tokens:
                content_codes.extend(str(id) for id in decode_movie_token(token))

        unjoined_channels = await get_unjoined_channels(user_id)
        
        if not unjoined_channels:
            if content_codes:
                request_count = await handle_media_requests(user_id, content_codes)
                await message.answer(MESSAGES["media_preparing"].format(count=request_count))
            else:
                await message.answer(MESSAGES["welcome"])
        else:
            keyboard = get_verification_menu(unjoined_channels)
            await message.answer(
                MESSAGES["join_channels"],
                reply_markup=keyboard
            )
            
            if content_codes:
                await message.answer(MESSAGES["pending_media"].format(count=len(content_codes)))
    except Exception as e:
        logger.error(f"Error in /start for user {user_id}: {e}", exc_info=True)
        await message.answer(MESSAGES["error_occurred"])

@dp.callback_query(lambda query: query.data == "verify")
async def verify_membership(query: types.CallbackQuery):
    """Verify membership when the user clicks the check button."""
    user_id = query.from_user.id
    
    try:
        async with get_async_redis() as redis:
            last_check_key = f"last_verify:{user_id}"
            last_check = await redis.get(last_check_key)
            
            if last_check and (time.time() - float(last_check)) < SUBSCRIPTION_CHECK_COOLDOWN:
                remain_time = int(SUBSCRIPTION_CHECK_COOLDOWN - (time.time() - float(last_check)))
                await query.answer(MESSAGES["wait_before_check"].format(seconds=remain_time))
                return
                
            await redis.set(last_check_key, str(time.time()), ex=SUBSCRIPTION_CHECK_COOLDOWN)
        
        unjoined_channels = await get_unjoined_channels(user_id, force_check=True)
        
        if not unjoined_channels:
            await query.message.edit_text(MESSAGES["membership_verified"])
            
            pending_media = await check_pending_media_for_user(user_id)
            if pending_media > 0:
                await query.message.answer(MESSAGES["pending_media_sending"].format(count=pending_media))
        else:
            await query.message.edit_text(
                MESSAGES["still_not_member"],
                reply_markup=get_verification_menu(unjoined_channels)
            )
    except Exception as e:
        logger.error(f"Error verifying membership for user {user_id}: {e}", exc_info=True)
        await query.answer(MESSAGES["error_occurred"])

async def check_pending_media_for_user(user_id: int) -> int:
    """Check and activate any pending media for newly verified users."""
    try:
        count = 0
        async with get_async_redis() as redis:
            # Get all pending items from queue
            queue_items = await redis.lrange(REDIS_QUEUE_KEY, 0, -1)
            
            # Filter items for this user and requeue them
            for item in queue_items:
                try:
                    queue_item = QueueItem.from_json(item)
                    if queue_item.user_id == user_id:
                        count += 1
                        # Re-add to queue to ensure processing
                        await redis.lpush(REDIS_QUEUE_KEY, item.to_json())
                except Exception as e:
                    logger.error(f"Error processing queue item: {e}")
                    
        return count
    except Exception as e:
        logger.error(f"Error checking pending media: {e}")
        return 0

@dp.message(Command("help"))
async def send_help(message: types.Message):
    """Send help information to the user."""
    await message.answer(MESSAGES["help_message"])

@dp.message(Command("status"))
async def status(message: types.Message):
    """Check queue status (admin only)."""
    user_id = message.from_user.id
    
    try:
        async with get_async_redis() as redis:
            queue_size = await redis.llen(REDIS_QUEUE_KEY)
            await message.answer(MESSAGES["status_message"].format(queue_size=queue_size))
    except Exception as e:
        logger.error(f"Error in /status command: {e}")
        await message.answer(MESSAGES["status_error"])

@dp.message(Command("admintest"))
async def admin_test(message: types.Message):
    """Test admin functionality"""
    user_id = message.from_user.id
    logger.info(f"Admin test from user {user_id}")
    await message.answer(
        f"Your ID: {user_id}\n"
        f"Is Admin: {user_id in ADMIN_IDS}\n"
        f"Admin IDs: {ADMIN_IDS}"
    )

# ========================= Start Bot With Optimized Workers =========================

async def main():
    """Start the bot with optimized worker configuration."""
    workers = []
    
    for _ in range(WORKER_COUNT):
        workers.append(asyncio.create_task(process_queue_worker()))
    
    logger.info(f"Started {WORKER_COUNT} optimized workers")
    
    try:
        await dp.start_polling(bot, skip_updates=True)  # Skip old updates
    finally:
        for worker in workers:
            worker.cancel()
        await asyncio.gather(*workers, return_exceptions=True)
        await async_redis_pool.disconnect()

# Add after the existing configuration section
ADMIN_IDS = {374057584, 485488475}  # Replace with actual admin IDs
BANNED_USERS: Set[int] = set()
logger.info(f"Configured admin IDs: {ADMIN_IDS}")

# Add admin keyboard markups
ADMIN_KEYBOARD = InlineKeyboardMarkup(inline_keyboard=[
    [InlineKeyboardButton(text="📊 آمار ربات", callback_data="admin_stats")],
    [InlineKeyboardButton(text="🧹 پاکسازی صف", callback_data="admin_clear_queue"),
     InlineKeyboardButton(text="📣 ارسال پیام همگانی", callback_data="admin_broadcast")],
    [InlineKeyboardButton(text="⚙️ تنظیمات پیشرفته", callback_data="admin_advanced")],
])

ADVANCED_KEYBOARD = InlineKeyboardMarkup(inline_keyboard=[
    [InlineKeyboardButton(text="🔧 تغییر کانال‌ها", callback_data="admin_set_channels"),
     InlineKeyboardButton(text="🎚 تنظیم محدودیت ارسال", callback_data="admin_set_limits")],
    [InlineKeyboardButton(text="👤 مدیریت کاربران", callback_data="admin_manage_users")],
    [InlineKeyboardButton(text="🔙 بازگشت", callback_data="admin_back")],
])

USER_MANAGEMENT_KEYBOARD = InlineKeyboardMarkup(inline_keyboard=[
    [InlineKeyboardButton(text="🚫 مسدود کردن کاربر", callback_data="admin_ban_user"),
     InlineKeyboardButton(text="✅ آزاد کردن کاربر", callback_data="admin_unban_user")],
    [InlineKeyboardButton(text="🔙 بازگشت", callback_data="admin_back")],
])

CONFIRM_CLEAR_KEYBOARD = InlineKeyboardMarkup(inline_keyboard=[
    [InlineKeyboardButton(text="✅ تایید پاکسازی", callback_data="confirm_clear")],
    [InlineKeyboardButton(text="❌ لغو", callback_data="admin_back")],
])

# Update the admin decorator
def admin_required(func):
    async def wrapper(message: types.Message, **kwargs):
        if message.from_user.id not in ADMIN_IDS:
            await message.answer(MESSAGES["access_denied"])
            return
        return await func(message)
    return wrapper

# Update the admin panel handler
@dp.message(Command("admin"))
async def admin_panel(message: types.Message):
    """Show admin panel with interactive keyboard"""
    if message.from_user.id not in ADMIN_IDS:
        await message.answer(MESSAGES["access_denied"])
        return
        
    logger.info(f"Admin command received from user {message.from_user.id}")
    try:
        await message.answer(
            text=MESSAGES["admin_menu"],
            reply_markup=ADMIN_KEYBOARD,
            parse_mode="Markdown"
        )
    except Exception as e:
        logger.error(f"Error in admin panel: {e}")
        await message.answer("Error showing admin panel. Check logs.")

# Add this class with your other classes
class AdminStates(StatesGroup):
    waiting_for_broadcast = State()

# Add these new handlers for broadcast functionality
@dp.callback_query(lambda query: query.data == "admin_broadcast")
async def start_broadcast(query: types.CallbackQuery, state: FSMContext):
    """Start broadcast process"""
    if query.from_user.id not in ADMIN_IDS:
        await query.answer(MESSAGES["access_denied"])
        return
        
    await state.set_state(AdminStates.waiting_for_broadcast)
    await query.message.edit_text(
        "📣 لطفا پیام خود را برای ارسال همگانی وارد کنید:\n\n"
        "❗️ هر پیامی که الان ارسال کنید به همه کاربران ارسال خواهد شد.",
        reply_markup=InlineKeyboardMarkup(inline_keyboard=[[
            InlineKeyboardButton(text="❌ لغو", callback_data="admin_cancel_broadcast")
        ]])
    )
    await query.answer()

@dp.callback_query(lambda query: query.data == "admin_cancel_broadcast")
async def cancel_broadcast(query: types.CallbackQuery, state: FSMContext):
    """Cancel broadcast process"""
    current_state = await state.get_state()
    if current_state is not None:
        await state.clear()
    
    await query.message.edit_text(
        MESSAGES["admin_menu"],
        reply_markup=ADMIN_KEYBOARD,
        parse_mode="Markdown"
    )
    await query.answer("Broadcast cancelled")

@dp.message(AdminStates.waiting_for_broadcast)
async def handle_broadcast(message: types.Message, state: FSMContext):
    """Handle the broadcast message"""
    if message.from_user.id not in ADMIN_IDS:
        return
        
    await state.clear()
    
    try:
        async with get_async_redis() as redis:
            # Get all unique users from Redis
            user_keys = await redis.keys(f"{REDIS_USER_CACHE_PREFIX}*")
            user_ids = [int(key.split(':')[1]) for key in user_keys]
            
            sent_count = 0
            failed_count = 0
            
            # Send the message to all users
            for user_id in user_ids:
                try:
                    await message.copy_to(user_id)
                    sent_count += 1
                    await asyncio.sleep(0.05)  # Small delay to prevent flooding
                except Exception as e:
                    logger.error(f"Failed to send broadcast to {user_id}: {e}")
                    failed_count += 1
            
            # Send summary to admin
            summary = (
                f"📣 نتیجه ارسال پیام همگانی:\n\n"
                f"✅ ارسال موفق: {sent_count}\n"
                f"❌ ارسال ناموفق: {failed_count}\n"
                f"📊 مجموع: {sent_count + failed_count}"
            )
            
            await message.answer(
                summary,
                reply_markup=ADMIN_KEYBOARD
            )
            
    except Exception as e:
        logger.error(f"Broadcast error: {e}")
        await message.answer(
            "⚠️ خطا در ارسال پیام همگانی",
            reply_markup=ADMIN_KEYBOARD
        )

@dp.callback_query(lambda query: query.data.startswith("admin_") or query.data == "confirm_clear")
async def handle_admin_actions(query: types.CallbackQuery):
    """Handle all admin panel interactions"""
    user_id = query.from_user.id
    if user_id not in ADMIN_IDS:
        await query.answer(MESSAGES["access_denied"])
        return

    try:
        if query.data == "admin_stats":
            async with get_async_redis() as redis:
                queue_size = await redis.llen(REDIS_QUEUE_KEY)
                memory_usage = psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2
                stats_msg = MESSAGES["admin_stats"].format(
                    queue_size=queue_size,
                    active_users=len(await redis.keys(f"{REDIS_USER_CACHE_PREFIX}*")),
                    banned_users=len(BANNED_USERS),
                    memory_usage=f"{memory_usage:.1f}",
                    worker_count=WORKER_COUNT
                )
                await query.message.edit_text(stats_msg, reply_markup=ADMIN_KEYBOARD)
                
        elif query.data == "admin_clear_queue":
            await query.message.edit_text(MESSAGES["confirm_clear"], reply_markup=CONFIRM_CLEAR_KEYBOARD)
            
        elif query.data == "confirm_clear":
            async with get_async_redis() as redis:
                await redis.delete(REDIS_QUEUE_KEY)
                await query.message.edit_text(MESSAGES["queue_cleared"], reply_markup=ADMIN_KEYBOARD)
            
        elif query.data == "admin_broadcast":
            await start_broadcast(query)  # Call the new broadcast handler
            
        elif query.data == "admin_advanced":
            await query.message.edit_text(
                "⚙️ **تنظیمات پیشرفته**",
                reply_markup=ADVANCED_KEYBOARD,
                parse_mode="Markdown"
            )
            
        elif query.data == "admin_manage_users":
            await query.message.edit_text(
                "👤 **مدیریت کاربران**",
                reply_markup=USER_MANAGEMENT_KEYBOARD,
                parse_mode="Markdown"
            )
            
        elif query.data == "admin_back":
            await query.message.edit_text(
                MESSAGES["admin_menu"],
                reply_markup=ADMIN_KEYBOARD,
                parse_mode="Markdown"
            )
            
        elif query.data == "admin_set_channels":
            await query.message.edit_text(
                "🔧 برای تغییر کانال‌ها از دستور زیر استفاده کنید:\n"
                "/channels @channel1 @channel2",
                reply_markup=InlineKeyboardMarkup(inline_keyboard=[[
                    InlineKeyboardButton(text="🔙 بازگشت", callback_data="admin_advanced")
                ]])
            )
            
        elif query.data == "admin_set_limits":
            await query.message.edit_text(
                "🎚 برای تنظیم محدودیت ارسال از دستور زیر استفاده کنید:\n"
                "/limit <تعداد پیام در دقیقه>",
                reply_markup=InlineKeyboardMarkup(inline_keyboard=[[
                    InlineKeyboardButton(text="🔙 بازگشت", callback_data="admin_advanced")
                ]])
            )
            
        elif query.data == "admin_ban_user":
            await query.message.edit_text(
                "🚫 برای مسدود کردن کاربر از دستور زیر استفاده کنید:\n"
                "/ban <user_id>",
                reply_markup=InlineKeyboardMarkup(inline_keyboard=[[
                    InlineKeyboardButton(text="🔙 بازگشت", callback_data="admin_manage_users")
                ]])
            )
            
        elif query.data == "admin_unban_user":
            await query.message.edit_text(
                "✅ برای خارج کردن کاربر از حالت مسدود از دستور زیر استفاده کنید:\n"
                "/unban <user_id>",
                reply_markup=InlineKeyboardMarkup(inline_keyboard=[[
                    InlineKeyboardButton(text="🔙 بازگشت", callback_data="admin_manage_users")
                ]])
            )

        await query.answer()
        
    except Exception as e:
        logger.error(f"Admin action error: {e}")
        await query.answer("⚠️ خطا در انجام عملیات!")

@dp.message(Command("ban"))
async def ban_user(message: types.Message):
    """Ban a user"""
    if message.from_user.id not in ADMIN_IDS:
        await message.answer(MESSAGES["access_denied"])
        return
        
    try:
        user_id = int(message.text.split()[1])
        BANNED_USERS.add(user_id)
        await message.answer(MESSAGES["user_banned"].format(user_id=user_id))
    except:
        await message.answer("⚠️ فرمت دستور نادرست!\nاستفاده: /ban <user_id>")

@dp.message(Command("unban"))
async def unban_user(message: types.Message):
    """Unban a user"""
    if message.from_user.id not in ADMIN_IDS:
        await message.answer(MESSAGES["access_denied"])
        return
        
    try:
        user_id = int(message.text.split()[1])
        BANNED_USERS.discard(user_id)
        await message.answer(MESSAGES["user_unbanned"].format(user_id=user_id))
    except:
        await message.answer("⚠️ فرمت دستور نادرست!\nاستفاده: /unban <user_id>")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        logger.info("Bot stopped by user")
    except Exception as e:
        logger.critical(f"Unexpected fatal error: {e}", exc_info=True)