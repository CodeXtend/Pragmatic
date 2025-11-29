import instaloader

def get_reel_data(reel_url):
    """
    Extract Instagram reel data using instaloader.
    Supports URLs like:
    - https://www.instagram.com/reels/DRjH6HKCEoX/
    - https://www.instagram.com/reel/DRjH6HKCEoX/
    - https://www.instagram.com/p/DRjH6HKCEoX/
    
    Args:
        reel_url: Instagram reel URL
        num_comments: Number of comments to fetch (default: 5)
    """
    # Extract shortcode from URL
    shortcode = None
    for part in reel_url.rstrip('/').split('/'):
        if part and part not in ['https:', 'http:', '', 'www.instagram.com', 'instagram.com', 'reels', 'reel', 'p']:
            shortcode = part
    
    if not shortcode:
        return None
        
    try:
        L = instaloader.Instaloader()
        
        # Get the post by shortcode
        post = instaloader.Post.from_shortcode(L.context, shortcode)
        
        # Extract relevant data
        data = {
            "caption": post.caption,
            "owner_username": post.owner_username,
            "owner_id": post.owner_id,
            # "total_comments": post.comments,
            "thumbnail_url": post.url,
            "is_video": post.is_video,
            "date": str(post.date),
            "typename": post.typename,
            "video_url": post.video_url if post.is_video else None,
        }
        return data
        
    except instaloader.exceptions.InstaloaderException as e:
        return None
    except Exception as e:
        return None


url = "https://www.instagram.com/reels/DRkYNeTjAST/"
data = get_reel_data(url, num_comments=5)

if data:
    print(data)