"""
Reddit Comment Listener Bot
Monitors a subreddit for comments containing "!postinfo" trigger
and replies with the parent post's details.
"""

import os
import praw
from dotenv import load_dotenv
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Configuration
TARGET_SUBREDDIT = os.getenv("TARGET_SUBREDDIT", "PokemonPocketTradeCo")
TRIGGER_KEYWORD = "!postinfo"


def authenticate_reddit():
    """
    Authenticate with Reddit using credentials from .env file.
    Returns a Reddit instance.
    """
    # Debug: Print loaded credentials (masked)
    client_id = os.getenv("REDDIT_CLIENT_ID")
    client_secret = os.getenv("REDDIT_CLIENT_SECRET")
    username = os.getenv("REDDIT_USERNAME")
    password = os.getenv("REDDIT_PASSWORD")
    user_agent = os.getenv("REDDIT_USER_AGENT", "PostInfoBot v1.0")
    
    logger.info("🔍 Checking credentials...")
    logger.info(f"   Client ID: {client_id[:5]}...{client_id[-3:] if client_id and len(client_id) > 8 else 'NOT SET'}")
    logger.info(f"   Client Secret: {'*' * 10 if client_secret else 'NOT SET'}")
    logger.info(f"   Username: {username if username else 'NOT SET'}")
    logger.info(f"   Password: {'*' * len(password) if password else 'NOT SET'}")
    logger.info(f"   User Agent: {user_agent}")
    
    if not all([client_id, client_secret, username, password]):
        logger.error("❌ Missing credentials! Check your .env file.")
        raise ValueError("Missing Reddit credentials")
    
    try:
        reddit = praw.Reddit(
            client_id=client_id,
            client_secret=client_secret,
            username=username,
            password=password,
            user_agent=user_agent
        )
        # Verify authentication
        logger.info(f"✅ Authenticated as: u/{reddit.user.me()}")
        return reddit
    except Exception as e:
        logger.error(f"❌ Authentication failed: {e}")
        logger.error("")
        logger.error("🔧 TROUBLESHOOTING TIPS:")
        logger.error("   1. Is 2FA (Two-Factor Auth) DISABLED on your Reddit account?")
        logger.error("   2. Is your Reddit app type set to 'script' (not 'web app')?")
        logger.error("   3. Can you login to Reddit with this username/password in browser?")
        logger.error("   4. Is your Reddit account email verified?")
        logger.error("   5. Try logging into Reddit in browser first, then run bot again.")
        logger.error("")
        raise


def format_post_info(submission):
    """
    Format the post information into a clean Markdown reply.
    
    Args:
        submission: PRAW Submission object
        
    Returns:
        Formatted string with post details
    """
    # Get author name safely (deleted users return None)
    author_name = submission.author.name if submission.author else "[deleted]"
    
    # Get media URL - check for different media types
    media_url = submission.url
    if hasattr(submission, 'is_gallery') and submission.is_gallery:
        media_url = f"https://reddit.com{submission.permalink} (Gallery Post)"
    elif submission.is_self:
        media_url = "No media (Text Post)"
    
    reply_message = f"""
📌 **Post Info**

📝 **Title:** {submission.title}

👤 **Posted by:** u/{author_name}

🔗 **Media:** {media_url}

💬 **Subreddit:** r/{submission.subreddit.display_name}

👍 **Score:** {submission.score}

---
^(I am a bot. Reply with !postinfo on any post to get its details.)
"""
    return reply_message


def has_bot_already_replied(comment, bot_username):
    """
    Check if the bot has already replied to this comment.
    
    Args:
        comment: PRAW Comment object
        bot_username: The bot's username
        
    Returns:
        Boolean indicating if bot already replied
    """
    try:
        comment.refresh()
        for reply in comment.replies:
            if reply.author and reply.author.name.lower() == bot_username.lower():
                return True
    except Exception:
        # If we can't check replies, assume not replied
        pass
    return False


def handle_comment(comment, reddit):
    """
    Handle a comment that contains the trigger keyword.
    
    Args:
        comment: PRAW Comment object
        reddit: Authenticated Reddit instance
    """
    try:
        bot_username = reddit.user.me().name
        
        # Ignore own comments to prevent loops
        if comment.author and comment.author.name.lower() == bot_username.lower():
            return
        
        # Check if trigger keyword is in comment (case-insensitive)
        if TRIGGER_KEYWORD.lower() not in comment.body.lower():
            return
        
        # Check if bot already replied
        if has_bot_already_replied(comment, bot_username):
            logger.info(f"⏭️ Already replied to comment {comment.id}, skipping...")
            return
        
        # Get the parent post (submission)
        submission = comment.submission
        
        # Format and send reply
        reply_message = format_post_info(submission)
        comment.reply(reply_message)
        
        logger.info(f"✅ Replied to comment by u/{comment.author.name} on post: {submission.title[:50]}...")
        
    except praw.exceptions.RedditAPIException as e:
        logger.error(f"❌ Reddit API Error: {e}")
    except Exception as e:
        logger.error(f"❌ Error handling comment {comment.id}: {e}")


def run_bot():
    """
    Main function to run the bot.
    Streams comments from the target subreddit and processes them.
    """
    # Authenticate
    reddit = authenticate_reddit()
    
    # Get subreddit
    subreddit = reddit.subreddit(TARGET_SUBREDDIT)
    logger.info(f"🤖 Bot started! Monitoring r/{TARGET_SUBREDDIT} for '{TRIGGER_KEYWORD}' trigger...")
    logger.info("Press Ctrl+C to stop the bot.")
    
    try:
        # Stream comments continuously, skip existing ones
        for comment in subreddit.stream.comments(skip_existing=True):
            handle_comment(comment, reddit)
            
    except KeyboardInterrupt:
        logger.info("🛑 Bot stopped by user.")
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        # Continue running after errors
        logger.info("🔄 Restarting bot in 10 seconds...")
        import time
        time.sleep(10)
        run_bot()


if __name__ == "__main__":
    run_bot()
