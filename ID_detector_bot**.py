import logging
import time
from telegram import Update, MessageEntity
from telegram.ext import Application, MessageHandler, filters, ContextTypes, JobQueue
from hashids import Hashids

# Define the same secret salt and configuration in both bots
hashids = Hashids(salt="Admiral23", min_length=6)

# Store grouped messages with timestamps
related_messages = {}
message_timestamps = {}

# Add new global variables for timer management
active_timer_messages = {}
last_timer_message = None

# Modify global variables
active_window_messages = {}  # Store messages for each time window
active_window_timers = {}    # Store timer messages for each window
message_window_map = {}      # Map message IDs to their time window
ADMIN_IDS = [485488475, 374057584]

# Enable logging
logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
TOKEN = "7733604493:AAEuzdRdSv0l0xnAb1GDyaSVnzFWXbXN1c4"
CHANNEL_ID = -1002463367628
BOT1_LINK = "https://telegram.me/toop_toop_bot?start={}"
BOT2_LINK = "https://telegram.me/toop_toop_2_bot?start={}"
TIME_WINDOW = 30  # seconds

def encode_multiple_ids(message_ids: list) -> str:
    """Encodes multiple message IDs into a single token."""
    return hashids.encode(*message_ids)

async def update_timer(context: ContextTypes.DEFAULT_TYPE) -> None:
    """Update timer messages for active windows."""
    global message_window_map
    current_time = time.time()
    windows_to_remove = []

    for window_start, timer_message in active_window_timers.items():
        remaining_time = int(window_start + TIME_WINDOW - current_time)
        
        if remaining_time <= 0:
            windows_to_remove.append(window_start)
            window_messages = [mid for mid, w in message_window_map.items() if w == window_start]
            try:
                await timer_message.edit_text("⏰ Time window closed!")
            except Exception as e:
                logger.error(f"Error updating final timer message: {e}")
        else:
            window_messages = [mid for mid, w in message_window_map.items() if w == window_start]
            try:
                if len(window_messages) >= 5:
                    group_notice = (
                        f"🔄 Contents grouped by time window!\n"
                        f"⏳ Time remaining: {remaining_time} seconds"
                    )
                else:
                    group_notice = (
                        f"📝 {len(window_messages)}/5 contents in group\n"
                        f"⏳ Time remaining: {remaining_time} seconds"
                    )
                await timer_message.edit_text(group_notice)
            except Exception as e:
                logger.error(f"Error updating timer message: {e}")

    # Clean up expired windows
    for window in windows_to_remove:
        active_window_timers.pop(window, None)
        active_window_messages.pop(window, None)
        message_window_map = {mid: w for mid, w in message_window_map.items() if w != window}

async def append_content_id(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handles new and edited messages, detects groups, and appends encoded content ID."""
    global message_timestamps, message_window_map, active_window_messages, active_window_timers
    message = update.channel_post or update.edited_channel_post
    if not message:
        return

    current_time = time.time()
    message_id = message.message_id

    # Determine which time window this message belongs to
    window_start = int(current_time / TIME_WINDOW) * TIME_WINDOW
    
    # Clean up old windows and messages
    current_windows = [w for w in active_window_messages.keys() if w + TIME_WINDOW > current_time]
    active_window_messages = {w: m for w, m in active_window_messages.items() if w in current_windows}
    active_window_timers = {w: m for w, m in active_window_timers.items() if w in current_windows}
    message_window_map = {mid: w for mid, w in message_window_map.items() if w in current_windows}

    # Add current message to its window
    message_window_map[message_id] = window_start
    message_timestamps[message_id] = current_time

    # Get all messages in the current window
    window_messages = [mid for mid, w in message_window_map.items() if w == window_start]

    # Generate encoded token and links for current window messages
    encoded_token = encode_multiple_ids(sorted(window_messages))
    content_links = (
        f"Bot 1 : [پک]({BOT1_LINK.format(encoded_token)})\n"
        f"Bot 2 : [پک]({BOT2_LINK.format(encoded_token)})\n"
        f"@KcRang مرجع تخصصی نود @KcRang"
    )

    try:
        # Send or update links message
        if window_start in active_window_messages:
            try:
                await active_window_messages[window_start].edit_text(
                    content_links,
                    parse_mode="Markdown"
                )
            except Exception as e:
                logger.error(f"Error editing window message: {e}")
        else:
            # Send new messages for this window
            active_window_messages[window_start] = await message.reply_text(
                content_links,
                parse_mode="Markdown"
            )
            
            # Initial timer message
            if len(window_messages) >= 5:
                group_notice = (
                    f"🔄 Contents grouped by time window!\n"
                    f"⏳ Time remaining: {int(window_start + TIME_WINDOW - current_time)} seconds"
                )
            else:
                group_notice = (
                    f"📝 {len(window_messages)}/5 contents in group\n"
                    f"⏳ Time remaining: {int(window_start + TIME_WINDOW - current_time)} seconds"
                )
            
            active_window_timers[window_start] = await message.reply_text(
                group_notice
            )

    except Exception as e:
        logger.error(f"Error sending content ID message: {e}")

async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handles unexpected errors gracefully."""
    logger.error("Exception while handling an update:", exc_info=context.error)

if __name__ == "__main__":
    # Build application with job_queue enabled
    app = Application.builder().token(TOKEN).build()
    
    # Add handlers
    app.add_handler(MessageHandler(filters.Chat(CHANNEL_ID) & filters.ALL, append_content_id))
    app.add_error_handler(error_handler)

    # Set up timer update job
    if app.job_queue:
        app.job_queue.run_repeating(update_timer, interval=1.0, first=1.0)

    logger.info("Bot is running...")
    app.run_polling()
