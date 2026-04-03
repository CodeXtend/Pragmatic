import instaloader
import os
import time
import random

def get_reel_data(reel_url, num_comments=10, username=None, password=None):
    """
    Extract Instagram reel data using instaloader.
    Supports URLs like:
    - https://www.instagram.com/reels/DRjH6HKCEoX/
    - https://www.instagram.com/reel/DRjH6HKCEoX/
    - https://www.instagram.com/p/DRjH6HKCEoX/
    
    Args:
        reel_url: Instagram reel URL
        num_comments: Number of comments to fetch (default: 10)
        username: Instagram username for login (needed for comments)
        password: Instagram password for login (needed for comments)
    """
    # Extract shortcode from URL
    shortcode = None
    for part in reel_url.rstrip('/').split('/'):
        if part and part not in ['https:', 'http:', '', 'www.instagram.com', 'instagram.com', 'reels', 'reel', 'p']:
            shortcode = part
    
    if not shortcode:
        return None
        
    try:
        # Add delay to avoid rate limiting
        L = instaloader.Instaloader(
            download_comments=True,
            max_connection_attempts=3,
            request_timeout=300,
            sleep=True,  # Enable sleep between requests
        )
        
        # Try to login if credentials provided (required for fetching comments)
        logged_in = False
        if username and password:
            try:
                L.login(username, password)
                logged_in = True
                print(f"✅ Logged in as {username}")
            except Exception as e:
                print(f"⚠️ Login failed: {e}")
        else:
            # Try to load session from file
            try:
                L.load_session_from_file(username or "session")
                logged_in = True
                print("✅ Loaded existing session")
            except Exception:
                print("⚠️ No login session. Comments may not be available.")
        
        # Add random delay to appear more human-like
        time.sleep(random.uniform(2, 5))
        
        # Get the post by shortcode
        post = instaloader.Post.from_shortcode(L.context, shortcode)
        
        # Fetch comments (requires login)
        comments_list = []
        if logged_in:
            try:
                # Add delay before fetching comments
                print("⏳ Fetching comments (this may take a moment)...")
                time.sleep(random.uniform(3, 6))
                
                comments_iterator = post.get_comments()
                for i, comment in enumerate(comments_iterator):
                    if i >= num_comments:
                        break
                    comment_data = {
                        "username": comment.owner.username,
                        "text": comment.text,
                        "created_at": str(comment.created_at_utc),
                        "likes": comment.likes_count,
                        "answers_count": comment.answers_count,
                    }
                    # Fetch replies to this comment (up to 3 replies)
                    replies = []
                    try:
                        for j, reply in enumerate(comment.answers):
                            if j >= 3:
                                break
                            replies.append({
                                "username": reply.owner.username,
                                "text": reply.text,
                                "created_at": str(reply.created_at_utc),
                                "likes": reply.likes_count,
                            })
                    except Exception:
                        pass
                    comment_data["replies"] = replies
                    comments_list.append(comment_data)
                    # Small delay between comments to avoid detection
                    time.sleep(random.uniform(0.5, 1.5))
            except Exception as e:
                print(f"⚠️ Could not fetch comments: {e}")
                print("   Instagram may be rate-limiting. Try again in a few minutes.")
                comments_list = []
        else:
            print("⚠️ Login required to fetch comments. Pass username and password.")
        
        # Extract relevant data
        data = {
            "caption": post.caption,
            "owner_username": post.owner_username,
            "owner_id": post.owner_id,
            "total_comments": post.comments,
            "thumbnail_url": post.url,
            "is_video": post.is_video,
            "date": str(post.date),
            "typename": post.typename,
            "video_url": post.video_url if post.is_video else None,
            "comments": comments_list,
        }
        return data
        
    except instaloader.exceptions.InstaloaderException as e:
        print(f"Error: {e}")
        return None
    except Exception as e:
        print(f"Error: {e}")
        return None


url = "https://www.instagram.com/reels/DRkYNeTjAST/"

# ⚠️ Instagram requires login to fetch comments!
# Run save_session.py first (as admin) to save your session

import instaloader

INSTAGRAM_USERNAME = "192.168.0.29"

L = instaloader.Instaloader()
logged_in = False

# Try to load existing saved session
try:
    L.load_session_from_file(INSTAGRAM_USERNAME)
    print("✅ Loaded saved session!")
    logged_in = True
except FileNotFoundError:
    print("❌ No saved session found!")
    print("\n📋 To fix this, run save_session.py first:")
    print("   1. Open PowerShell as Administrator")
    print("   2. Run: python instagram_bot/save_session.py")
    print("   3. Enter your password when prompted")
    print("   4. Then run this script again")

if logged_in:
    data = get_reel_data(url, num_comments=10, username=INSTAGRAM_USERNAME)
else:
    print("\n⚠️ Running without login - comments won't be available")
    data = get_reel_data(url, num_comments=10)

if data:
    print("=" * 50)
    print(f"Reel by: @{data['owner_username']}")
    print(f"Date: {data['date']}")
    print(f"Caption: {data['caption'][:100] if data['caption'] else 'No caption'}...")
    print(f"Total Comments: {data['total_comments']}")
    print("=" * 50)
    print("\nFetched Comments:")
    for i, comment in enumerate(data['comments'], 1):
        print(f"\n{i}. @{comment['username']}: {comment['text']}")
        print(f"   ❤️ {comment['likes']} likes | 💬 {comment['answers_count']} replies")
        for reply in comment['replies']:
            print(f"   └─ @{reply['username']}: {reply['text']}")
    print("\nFull data dict:")
    print(data)