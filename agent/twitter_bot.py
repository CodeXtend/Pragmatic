"""
Twitter Bot Agent - Polls mentions and replies with fact-checks

This agent continuously monitors Twitter/X for mentions, analyzes the tagged tweets
using the TwitterWatcher agent, and posts fact-check replies.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import time
from typing import Optional, Dict, Any, List, Set
from dataclasses import dataclass

import requests
from requests_oauthlib import OAuth1
from dotenv import load_dotenv

from twitter_watcher import TwitterWatcher


@dataclass
class BotConfig:
    """Configuration for the Twitter bot."""
    poll_interval_seconds: int = 15
    auto_reply: bool = True
    max_retries: int = 3


class TwitterBot:
    """
    Twitter Bot Agent for fact-checking mentions.
    
    This bot monitors mentions, uses TwitterWatcher to analyze tagged tweets,
    and posts fact-check replies.
    """
    
    def __init__(self, config: Optional[BotConfig] = None):
        """
        Initialize the Twitter Bot.
        
        Args:
            config: Bot configuration. Uses defaults if None.
        """
        load_dotenv()
        
        self.config = config or BotConfig()
        
        # Twitter OAuth credentials
        self.api_key = os.getenv("TWITTER_API_KEY", "").strip()
        self.api_secret = os.getenv("TWITTER_API_SECRET", "").strip()
        self.access_token = os.getenv("TWITTER_ACCESS_TOKEN", "").strip()
        self.access_secret = os.getenv("TWITTER_ACCESS_SECRET", "").strip()
        
        # Setup OAuth1
        if self.api_key and self.api_secret and self.access_token and self.access_secret:
            self.oauth = OAuth1(
                self.api_key,
                client_secret=self.api_secret,
                resource_owner_key=self.access_token,
                resource_owner_secret=self.access_secret
            )
        else:
            self.oauth = None
            raise ValueError("Twitter OAuth credentials not configured")
        
        # Initialize watcher agent
        self.watcher = TwitterWatcher()
        
        # Bot state
        self.bot_user_id: Optional[str] = None
        self.bot_username: Optional[str] = None
        self.processed_mentions: Set[str] = set()
        self.last_mention_id: Optional[str] = None
        self.is_running: bool = False
    
    def _print_header(self, text: str):
        """Print formatted header."""
        print(f"\n{'═' * 60}")
        print(f"  {text}")
        print(f"{'═' * 60}")
    
    def _print_section(self, text: str):
        """Print section divider."""
        print(f"\n{'─' * 50}")
        print(f"  {text}")
        print(f"{'─' * 50}")
    
    def get_me(self, retries: int = 3) -> Optional[Dict[str, Any]]:
        """
        Get authenticated user info.
        
        Args:
            retries: Number of retry attempts
            
        Returns:
            User data dictionary or None
        """
        url = "https://api.twitter.com/2/users/me?user.fields=id,name,username,profile_image_url,public_metrics"
        
        print("\n📡 API CALL: GET /users/me")
        
        for attempt in range(1, retries + 1):
            try:
                response = requests.get(url, auth=self.oauth)
                print(f"   Status: {response.status_code}")
                
                if response.status_code == 429:
                    wait_time = attempt * 60
                    print(f"   ⏳ Rate limited! Waiting {wait_time}s (attempt {attempt}/{retries})...")
                    time.sleep(wait_time)
                    continue
                
                return response.json()
            except Exception as e:
                print(f"   ❌ ERROR: {e}")
                if attempt == retries:
                    raise
                time.sleep(5)
        
        return None
    
    def get_mentions(self, user_id: str, since_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get mentions for the bot user.
        
        Args:
            user_id: Bot's user ID
            since_id: Only get mentions after this ID
            
        Returns:
            Mentions data dictionary
        """
        url = (
            f"https://api.twitter.com/2/users/{user_id}/mentions"
            f"?tweet.fields=created_at,author_id,conversation_id,referenced_tweets"
            f"&expansions=author_id,referenced_tweets.id"
            f"&user.fields=id,name,username"
            f"&max_results=5"
        )
        
        if since_id:
            url += f"&since_id={since_id}"
        
        try:
            response = requests.get(url, auth=self.oauth)
            return response.json()
        except Exception as e:
            print(f"   ❌ FETCH ERROR: {e}")
            raise
    
    def post_reply(self, text: str, reply_to_tweet_id: str) -> Dict[str, Any]:
        """
        Post a reply tweet.
        
        Args:
            text: Reply text content
            reply_to_tweet_id: Tweet ID to reply to
            
        Returns:
            Response data dictionary
        """
        url = "https://api.twitter.com/2/tweets"
        
        print(f"\n📡 API CALL: POST /tweets (reply)")
        print(f"   Reply to: {reply_to_tweet_id}")
        
        body = {
            "text": text,
            "reply": {
                "in_reply_to_tweet_id": reply_to_tweet_id
            }
        }
        
        try:
            response = requests.post(url, auth=self.oauth, json=body)
            print(f"   Status: {response.status_code}")
            return response.json()
        except Exception as e:
            print(f"   ❌ POST ERROR: {e}")
            raise
    
    def process_mention(self, mention: Dict[str, Any], includes_users: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """
        Process a single mention - analyze and reply.
        
        Args:
            mention: Mention tweet data
            includes_users: User data from includes
            
        Returns:
            Result dictionary or None
        """
        mention_id = mention.get("id")
        
        # Skip if already processed
        if mention_id in self.processed_mentions:
            return None
        
        self.processed_mentions.add(mention_id)
        
        # Get mention author
        mention_author = next(
            (u for u in includes_users if u["id"] == mention.get("author_id")),
            None
        )
        username = mention_author.get("username", "there") if mention_author else "there"
        
        self._print_section(f"📩 NEW MENTION from @{username}")
        print(f"   Text: {mention.get('text')}")
        
        # Find the tweet they're replying to
        target_tweet_id = None
        referenced_tweets = mention.get("referenced_tweets", [])
        
        replied_to = next(
            (ref for ref in referenced_tweets if ref.get("type") == "replied_to"),
            None
        )
        if replied_to:
            target_tweet_id = replied_to.get("id")
            print(f"   🎯 Found replied-to tweet: {target_tweet_id}")
        
        if not target_tweet_id:
            conversation_id = mention.get("conversation_id")
            if conversation_id and conversation_id != mention_id:
                target_tweet_id = conversation_id
                print(f"   🎯 Using conversation root: {target_tweet_id}")
        
        try:
            if target_tweet_id:
                print(f"   🔍 Analyzing tweet {target_tweet_id}...")
                analysis = self.watcher.watch(target_tweet_id)
                
                if analysis and "error" not in analysis:
                    print("   ✅ Analysis complete!")
                    print(f"   📊 Misleading: {analysis['details'].get('is_potentially_misleading')}")
                    print(f"   🎯 Confidence: {analysis['details'].get('confidence_score')}%")
                    reply_message = self.watcher.generate_reply(analysis, username)
                else:
                    reply_message = (
                        f"Hello @{username}! 👋\n\n"
                        f"I couldn't analyze that tweet. It may not contain media or there was an error.\n\n"
                        f"🤖 Pragmatic"
                    )
            else:
                reply_message = (
                    f"Hello @{username}! 👋\n\n"
                    f"To fact-check a post, please reply to a tweet with an image or video and tag me.\n\n"
                    f"Example: Reply to a suspicious post and mention @{self.bot_username}\n\n"
                    f"🤖 Pragmatic"
                )
            
            if self.config.auto_reply:
                print("   💬 Sending reply...")
                reply_result = self.post_reply(reply_message, mention_id)
                
                if "data" in reply_result:
                    print(f"   ✅ Reply sent! ID: {reply_result['data'].get('id')}")
                else:
                    print(f"   ⚠️ Reply failed: {json.dumps(reply_result)}")
            else:
                print(f"   📝 Reply (not sent - auto_reply disabled):\n{reply_message}")
            
            return {"success": True, "mention_id": mention_id}
        
        except Exception as e:
            print(f"   ❌ Error: {e}")
            return None
    
    def poll_mentions(self):
        """Poll for new mentions and process them."""
        try:
            response = self.get_mentions(self.bot_user_id, self.last_mention_id)
            
            if "errors" in response:
                error_msg = response["errors"][0].get("message", json.dumps(response["errors"]))
                print(f"\n⚠️ API Error: {error_msg}")
                return
            
            mentions = response.get("data", [])
            includes_users = response.get("includes", {}).get("users", [])
            
            if not mentions:
                print(".", end="", flush=True)
                return
            
            print(f"\n\n📬 Found {len(mentions)} new mention(s)!")
            
            # Update pagination
            if mentions:
                self.last_mention_id = mentions[0].get("id")
            
            # Process each mention
            for mention in mentions:
                self.process_mention(mention, includes_users)
        
        except Exception as e:
            print(f"\n❌ Poll error: {e}")
    
    def start(self):
        """Initialize and start the bot."""
        self._print_header("🤖 PRAGMATIC - TWITTER FACT-CHECK BOT")
        
        # Check Gemini
        if not self.watcher.gemini_model:
            print("\n⚠️  GEMINI_API_KEY not found!")
            print("   The bot will run but cannot analyze media.")
        
        # Verify credentials
        print("\n🔐 Verifying Twitter credentials...")
        
        try:
            me = self.get_me()
            
            if not me or "data" not in me:
                print("\n❌ Failed to authenticate!")
                print(f"   Error: {json.dumps(me)}")
                return
            
            self.bot_user_id = me["data"]["id"]
            self.bot_username = me["data"]["username"]
            
            print(f"\n✅ Authenticated as @{self.bot_username}")
            print(f"   User ID: {self.bot_user_id}")
            print(f"   Followers: {me['data'].get('public_metrics', {}).get('followers_count', 0)}")
        
        except Exception as e:
            print(f"\n❌ Auth error: {e}")
            return
        
        # Start polling
        self._print_section("🚀 Bot is now running!")
        print(f"   Polling every {self.config.poll_interval_seconds} seconds")
        print(f"   Auto-reply: {'✓ Enabled' if self.config.auto_reply else '✗ Disabled'}")
        print(f"   Gemini: {'✓ Ready' if self.watcher.gemini_model else '✗ Not configured'}")
        
        print("\n📖 HOW TO USE:")
        print("   1. Find a tweet with video/image to fact-check")
        print(f"   2. Reply to that tweet and tag @{self.bot_username}")
        print("   3. Wait for the bot to analyze and reply")
        print(f"\n📡 Listening for mentions... (Ctrl+C to stop)\n")
        
        self.is_running = True
        
        try:
            while self.is_running:
                self.poll_mentions()
                time.sleep(self.config.poll_interval_seconds)
        except KeyboardInterrupt:
            self._print_header("🛑 Shutting down...")
            print(f"   Processed {len(self.processed_mentions)} mentions")
            self.is_running = False
    
    def stop(self):
        """Stop the bot."""
        self.is_running = False


# Example usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Pragmatic Twitter Fact-Check Bot")
    parser.add_argument("--interval", type=int, default=15, help="Poll interval in seconds")
    parser.add_argument("--no-reply", action="store_true", help="Disable auto-reply")
    
    args = parser.parse_args()
    
    config = BotConfig(
        poll_interval_seconds=args.interval,
        auto_reply=not args.no_reply
    )
    
    bot = TwitterBot(config)
    bot.start()
