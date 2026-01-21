"""
Twitter Watcher Agent - Analyzes Twitter/X media content using Gemini Vision

This agent watches tweets, downloads media (video/image), and uses Gemini 2.0 Flash
for visual analysis and speech-to-text transcription.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import re
import json
import base64
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from datetime import datetime

import requests
from requests_oauthlib import OAuth1
from dotenv import load_dotenv
import google.generativeai as genai


@dataclass
class VisualSummary:
    """Data class for visual analysis summary."""
    objects_detected: List[str] = field(default_factory=list)
    people_in_video: str = "Not analyzed"
    speaker_actions: str = "Not analyzed"
    scene_description: str = "Not analyzed"
    text_overlays: str = "None detected"
    key_moments: str = "Not analyzed"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "objects_detected": self.objects_detected,
            "people_in_video": self.people_in_video,
            "speaker_actions": self.speaker_actions,
            "scene_description": self.scene_description,
            "text_overlays": self.text_overlays,
            "key_moments": self.key_moments
        }


@dataclass
class WatcherResult:
    """Data class for watcher agent results."""
    user: str = ""
    post_caption: str = ""
    few_comments: List[str] = field(default_factory=list)
    gemini_scan_details: str = ""
    person_talking_about: str = "No speech detected"
    visual_summary: VisualSummary = field(default_factory=VisualSummary)
    detail_analysis: str = ""
    is_potentially_misleading: Optional[bool] = None
    confidence_score: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "details": {
                "User": self.user,
                "post_caption": self.post_caption,
                "few_comments": self.few_comments,
                "Gemini_Scan_Details": self.gemini_scan_details,
                "Person_talking_about": self.person_talking_about,
                "visual_summary": self.visual_summary.to_dict() if isinstance(self.visual_summary, VisualSummary) else self.visual_summary,
                "detail_analysis": self.detail_analysis,
                "is_potentially_misleading": self.is_potentially_misleading,
                "confidence_score": self.confidence_score
            }
        }
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)


class TwitterWatcher:
    """
    Twitter Watcher Agent for media analysis.
    
    This agent monitors Twitter/X, downloads media content (videos/images),
    and uses Gemini 2.0 Flash for visual analysis including:
    - Object detection
    - Person identification
    - Speech-to-text transcription
    - Scene analysis
    - Potential misinformation detection
    """
    
    def __init__(self, gemini_api_key: Optional[str] = None):
        """
        Initialize the Twitter Watcher agent.
        
        Args:
            gemini_api_key: Gemini API key. If None, reads from GEMINI_API_KEY env variable.
        """
        load_dotenv()
        
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
            print("⚠️ Twitter OAuth credentials not configured")
        
        # Gemini setup
        self.gemini_api_key = gemini_api_key or os.getenv("GEMINI_API_KEY", "").strip()
        if self.gemini_api_key:
            genai.configure(api_key=self.gemini_api_key)
            self.gemini_model = genai.GenerativeModel("gemini-2.5-flash")
        else:
            self.gemini_model = None
            print("⚠️ Gemini API key not configured")
    
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
    
    def get_tweet_details(self, tweet_id: str) -> Dict[str, Any]:
        """
        Fetch tweet details with media expansions.
        
        Args:
            tweet_id: The Twitter/X tweet ID
            
        Returns:
            Dictionary containing tweet data
        """
        url = (
            f"https://api.twitter.com/2/tweets/{tweet_id}"
            f"?tweet.fields=created_at,author_id,public_metrics,entities,conversation_id,lang,source,referenced_tweets,attachments"
            f"&expansions=author_id,attachments.media_keys,referenced_tweets.id"
            f"&user.fields=id,name,username,profile_image_url,verified,public_metrics,description"
            f"&media.fields=media_key,type,url,preview_image_url,width,height,alt_text,variants,duration_ms"
        )
        
        print("🔍 Fetching tweet details...")
        
        response = requests.get(url, auth=self.oauth)
        return response.json()
    
    def get_user_by_id(self, user_id: str) -> Optional[Dict[str, Any]]:
        """
        Get user details by user ID.
        
        Args:
            user_id: Twitter user ID
            
        Returns:
            User data dictionary or None
        """
        url = (
            f"https://api.twitter.com/2/users/{user_id}"
            f"?user.fields=id,name,username,profile_image_url,verified,public_metrics,description,created_at,location"
        )
        
        print(f"🔍 Fetching user details for ID: {user_id}...")
        
        response = requests.get(url, auth=self.oauth)
        data = response.json()
        
        if "data" in data:
            print(f"   ✅ Found user: @{data['data']['username']} ({data['data']['name']})")
            return data["data"]
        
        print(f"   ❌ User not found or error")
        return None
    
    def get_video_mp4_url(self, media: Dict[str, Any]) -> Optional[str]:
        """
        Extract best quality MP4 URL from video media.
        
        Args:
            media: Media object from Twitter API
            
        Returns:
            MP4 video URL or None
        """
        if not media or media.get("type") != "video":
            return None
        
        variants = media.get("variants", [])
        mp4_variants = [v for v in variants if v.get("content_type") == "video/mp4"]
        mp4_variants.sort(key=lambda x: x.get("bit_rate", 0), reverse=True)
        
        return mp4_variants[0]["url"] if mp4_variants else None
    
    def download_video(self, video_url: str, filename: Optional[str] = None) -> bytes:
        """
        Download video to buffer.
        
        Args:
            video_url: URL of the video
            filename: Optional filename to save locally
            
        Returns:
            Video bytes buffer
        """
        print("📥 Downloading video...")
        
        response = requests.get(video_url)
        buffer = response.content
        
        if filename:
            with open(filename, "wb") as f:
                f.write(buffer)
            print(f"   ✅ Saved to: {filename}")
        
        print(f"   📦 Size: {len(buffer) / 1024 / 1024:.2f} MB")
        return buffer
    
    def download_image(self, image_url: str) -> bytes:
        """
        Download image to buffer.
        
        Args:
            image_url: URL of the image
            
        Returns:
            Image bytes buffer
        """
        print("📥 Downloading image...")
        response = requests.get(image_url)
        return response.content
    
    def analyze_video(self, video_buffer: bytes, tweet_context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Analyze video using Gemini 2.0 Flash.
        
        Args:
            video_buffer: Video bytes
            tweet_context: Context about the tweet
            
        Returns:
            Analysis result dictionary
        """
        if not self.gemini_model:
            print("❌ Gemini model not configured")
            return None
        
        print("\n🤖 Analyzing video with Gemini 2.0 Flash...")
        
        video_base64 = base64.b64encode(video_buffer).decode("utf-8")
        
        prompt = f"""You are a fact-checking assistant analyzing a video from Twitter/X.

Analyze this video carefully and extract the following information in JSON format:

1. **Gemini_Scan_Details**: Describe what you see in the video - people, objects, scenes, any flags or symbols, text overlays, etc.

2. **Person_talking_about (Speech to Text)**: If someone is speaking in the video, transcribe what they are saying. Include the language if it's not English.

3. **visual_summary**: Provide a detailed summary of the visual content including:
   - Objects detected in the video (list all visible objects like phone, flag, building, etc.)
   - People identified (describe their appearance, clothing, expressions)
   - Who is speaking and their visual actions (gestures, body language, facial expressions while speaking)
   - Scene description (indoor/outdoor, location type, lighting)
   - Any text overlays or captions visible on screen
   - Key visual moments or actions happening in the video

4. **detail_analysis**: Provide your analysis of what this video/reel appears to be about. Is it making any claims? Does it appear to be edited or manipulated? Any signs of misinformation?

5. **is_potentially_misleading**: true/false - Does this content appear to be potentially misleading or fake?

6. **confidence_score**: 0-100 - How confident are you in your analysis?

Tweet context:
- Author: {tweet_context.get('author', 'Unknown')}
- Tweet text: {tweet_context.get('text', 'N/A')}
- Posted: {tweet_context.get('created_at', 'Unknown')}

Return ONLY valid JSON in this exact format:
{{
  "User": "The question or context from the tweet",
  "post_caption": "The caption/text of the post",
  "few_comments": ["comment1", "comment2"],
  "Gemini_Scan_Details": "Detailed description of visual content in the video",
  "Person_talking_about": "Speech to text transcription of what is said in the video",
  "visual_summary": {{
    "objects_detected": ["object1", "object2", "object3"],
    "people_in_video": "Description of people visible - their appearance, clothing, expressions",
    "speaker_actions": "Who is speaking and their gestures, body language, facial expressions",
    "scene_description": "Indoor/outdoor, location type, environment details",
    "text_overlays": "Any text visible on screen",
    "key_moments": "Important visual moments or actions in the video"
  }},
  "detail_analysis": "Analysis of what this content is about and if it seems real or fake",
  "is_potentially_misleading": false,
  "confidence_score": 85
}}"""

        try:
            result = self.gemini_model.generate_content([
                {"mime_type": "video/mp4", "data": video_base64},
                prompt
            ])
            
            response_text = result.text
            print("   ✅ Gemini analysis complete!")
            
            return self._parse_json_response(response_text)
        
        except Exception as e:
            print(f"   ❌ Gemini error: {e}")
            return {"error": str(e)}
    
    def analyze_image(self, image_buffer: bytes, tweet_context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Analyze image using Gemini 2.0 Flash.
        
        Args:
            image_buffer: Image bytes
            tweet_context: Context about the tweet
            
        Returns:
            Analysis result dictionary
        """
        if not self.gemini_model:
            print("❌ Gemini model not configured")
            return None
        
        print("\n🤖 Analyzing image with Gemini 2.0 Flash...")
        
        image_base64 = base64.b64encode(image_buffer).decode("utf-8")
        
        prompt = f"""You are a fact-checking assistant analyzing an image from Twitter/X.

Analyze this image carefully and extract the following information in JSON format:

1. **Gemini_Scan_Details**: Describe what you see - people, objects, text, any signs of editing or manipulation

2. **text_in_image**: Any text visible in the image

3. **visual_summary**: Provide a detailed summary of the visual content including:
   - Objects detected in the image (list all visible objects)
   - People identified (describe their appearance, clothing, expressions, actions)
   - Scene description (indoor/outdoor, location type, lighting)
   - Any text overlays or captions visible
   - Signs of editing or manipulation

4. **detail_analysis**: Your analysis - does this appear real, edited, or potentially misleading?

5. **is_potentially_misleading**: true/false

6. **confidence_score**: 0-100

Tweet context:
- Author: {tweet_context.get('author', 'Unknown')}
- Tweet text: {tweet_context.get('text', 'N/A')}

Return ONLY valid JSON:
{{
  "User": "Context from tweet",
  "post_caption": "Caption of the post",
  "Gemini_Scan_Details": "Description of what is in the image",
  "text_in_image": "Any text visible in the image",
  "visual_summary": {{
    "objects_detected": ["object1", "object2"],
    "people_in_image": "Description of people visible",
    "scene_description": "Environment and setting details",
    "text_overlays": "Any text visible on the image",
    "editing_signs": "Any signs of manipulation or editing"
  }},
  "detail_analysis": "Analysis of the image authenticity",
  "is_potentially_misleading": false,
  "confidence_score": 85
}}"""

        try:
            result = self.gemini_model.generate_content([
                {"mime_type": "image/jpeg", "data": image_base64},
                prompt
            ])
            
            response_text = result.text
            print("   ✅ Gemini analysis complete!")
            
            return self._parse_json_response(response_text)
        
        except Exception as e:
            print(f"   ❌ Gemini error: {e}")
            return {"error": str(e)}
    
    def _parse_json_response(self, response_text: str) -> Dict[str, Any]:
        """
        Parse JSON from Gemini response.
        
        Args:
            response_text: Raw response text from Gemini
            
        Returns:
            Parsed dictionary
        """
        try:
            # Try to extract JSON from code block
            json_match = re.search(r'```json\n?([\s\S]*?)\n?```', response_text)
            if json_match:
                json_str = json_match.group(1)
            else:
                # Try to find raw JSON object
                obj_match = re.search(r'\{[\s\S]*\}', response_text)
                json_str = obj_match.group(0) if obj_match else response_text
            
            return json.loads(json_str)
        except json.JSONDecodeError:
            print("   ⚠️ Could not parse JSON, returning raw response")
            return {"raw_response": response_text}
    
    def watch(self, tweet_id: str) -> Dict[str, Any]:
        """
        Main method to watch and analyze a tweet.
        
        Args:
            tweet_id: The Twitter/X tweet ID to analyze
            
        Returns:
            Complete analysis result dictionary
        """
        self._print_header("🔍 TWITTER WATCHER AGENT")
        print("   (Gemini Vision, SpeechToText)")
        
        # Step 1: Get tweet details
        tweet_data = self.get_tweet_details(tweet_id)
        
        # Error handling
        if not tweet_data:
            print("❌ No response from Twitter API")
            return {"error": "No response from Twitter API"}
        
        if "errors" in tweet_data:
            print(f"❌ API Errors: {json.dumps(tweet_data['errors'], indent=2)}")
            return {"error": tweet_data["errors"]}
        
        if tweet_data.get("status") == 401:
            print("❌ Unauthorized - Check your API credentials")
            return {"error": "Unauthorized"}
        
        if tweet_data.get("status") == 429:
            print("❌ Rate limited - Too many requests")
            return {"error": "Rate limited"}
        
        if "data" not in tweet_data:
            print(f"❌ No tweet data returned")
            return {"error": "No tweet data found"}
        
        tweet = tweet_data["data"]
        includes = tweet_data.get("includes", {})
        users = includes.get("users", [])
        media = includes.get("media", [])
        
        # Find author
        author = next((u for u in users if u["id"] == tweet.get("author_id")), None)
        if not author and tweet.get("author_id"):
            print("   ℹ️ Author not in includes, fetching by ID...")
            author = self.get_user_by_id(tweet["author_id"])
        
        author_username = author.get("username", "unknown") if author else "unknown"
        
        print(f'\n📝 Tweet: "{tweet.get("text", "")}"')
        print(f'👤 Author: @{author_username} (ID: {tweet.get("author_id")})')
        print(f'📅 Posted: {tweet.get("created_at")}')
        
        tweet_context = {
            "author": author_username,
            "text": tweet.get("text"),
            "created_at": tweet.get("created_at"),
        }
        
        analysis_result = None
        
        # Step 2: Process media
        for media_item in media:
            print(f"\n📎 Media found: {media_item.get('type')}")
            
            if media_item.get("type") == "video":
                video_url = self.get_video_mp4_url(media_item)
                if not video_url:
                    print("   ❌ Could not extract video URL")
                    continue
                
                print(f"   🎬 Video URL: {video_url[:60]}...")
                buffer = self.download_video(video_url, f"tweet_{tweet_id}.mp4")
                analysis_result = self.analyze_video(buffer, tweet_context)
            
            elif media_item.get("type") == "photo":
                image_url = media_item.get("url")
                print(f"   📷 Image URL: {image_url}")
                buffer = self.download_image(image_url)
                analysis_result = self.analyze_image(buffer, tweet_context)
        
        # If no media, create text-only result
        if not media:
            print("\n📝 No media found, text-only tweet")
            analysis_result = {
                "User": "Text analysis request",
                "post_caption": tweet.get("text"),
                "Gemini_Scan_Details": "No visual content - text only tweet",
                "detail_analysis": "Cannot perform visual analysis on text-only content",
                "is_potentially_misleading": None,
                "confidence_score": 0,
            }
        
        # Step 3: Build final output
        self._print_header("📊 WATCHER AGENT OUTPUT")
        
        # Create default visual summary if not present
        default_visual_summary = {
            "objects_detected": [],
            "people_in_video": "Not analyzed",
            "speaker_actions": "Not analyzed",
            "scene_description": "Not analyzed",
            "text_overlays": "None detected",
            "key_moments": "Not analyzed"
        }
        
        final_output = {
            "details": {
                "User": analysis_result.get("User", tweet.get("text")) if analysis_result else tweet.get("text"),
                "post_caption": analysis_result.get("post_caption", tweet.get("text")) if analysis_result else tweet.get("text"),
                "few_comments": analysis_result.get("few_comments", []) if analysis_result else [],
                "Gemini_Scan_Details": analysis_result.get("Gemini_Scan_Details", "Analysis pending") if analysis_result else "Analysis pending",
                "Person_talking_about": analysis_result.get("Person_talking_about", "No speech detected") if analysis_result else "No speech detected",
                "visual_summary": analysis_result.get("visual_summary", default_visual_summary) if analysis_result else default_visual_summary,
                "detail_analysis": analysis_result.get("detail_analysis", "Analysis pending") if analysis_result else "Analysis pending",
                "is_potentially_misleading": analysis_result.get("is_potentially_misleading", False) if analysis_result else False,
                "confidence_score": analysis_result.get("confidence_score", 0) if analysis_result else 0,
            },
            "tweet_info": {
                "id": tweet.get("id"),
                "author": author_username,
                "author_id": tweet.get("author_id"),
                "url": f"https://x.com/{author_username}/status/{tweet.get('id')}",
                "created_at": tweet.get("created_at"),
                "public_metrics": tweet.get("public_metrics"),
            },
            "media_info": [
                {
                    "type": m.get("type"),
                    "url": self.get_video_mp4_url(m) if m.get("type") == "video" else m.get("url"),
                    "preview": m.get("preview_image_url"),
                }
                for m in media
            ],
            "analyzed_at": datetime.now().isoformat(),
        }
        
        print(json.dumps(final_output, indent=2))
        return final_output
    
    def generate_reply(self, analysis: Dict[str, Any], mention_author: str) -> str:
        """
        Generate a fact-check reply message from analysis.
        
        Args:
            analysis: The watcher analysis result
            mention_author: Username of the person who tagged the bot
            
        Returns:
            Reply message string (max 280 chars)
        """
        if not analysis or "details" not in analysis:
            return f"Hello @{mention_author}! 👋\n\nI couldn't analyze that content. Please tag me on a tweet with a video or image.\n\n🤖 Pragmatic"
        
        result = analysis["details"]
        is_misleading = result.get("is_potentially_misleading")
        confidence = result.get("confidence_score", 0)
        
        if is_misleading is True:
            emoji = "⚠️"
            verdict = "Potentially Misleading"
        elif is_misleading is False:
            emoji = "✅"
            verdict = "No Red Flags Detected"
        else:
            emoji = "ℹ️"
            verdict = "Analysis Complete"
        
        reply = f"{emoji} Fact-Check for @{mention_author}\n\n"
        reply += f"📊 Verdict: {verdict}\n"
        reply += f"🎯 Confidence: {confidence}%\n\n"
        
        brief_analysis = result.get("detail_analysis", "Analysis complete")[:150]
        reply += f"📝 {brief_analysis}{'...' if len(result.get('detail_analysis', '')) >= 150 else ''}\n\n"
        reply += "🤖 #FactCheck #Pragmatic"
        
        # Twitter 280 char limit
        if len(reply) > 280:
            reply = reply[:277] + "..."
        
        return reply


# Example usage
if __name__ == "__main__":
    import sys
    
    # Initialize the watcher
    watcher = TwitterWatcher()
    
    # Get tweet ID from command line or use default
    tweet_id = sys.argv[1] if len(sys.argv) > 1 else "1994477335979200513"
    
    print(f"\n🎯 Analyzing tweet: {tweet_id}")
    
    # Run analysis
    result = watcher.watch(tweet_id)
    
    if "error" not in result:
        print("\n✅ Analysis complete!")
        
        # Generate sample reply
        reply = watcher.generate_reply(result, "testuser")
        print(f"\n📝 Sample Reply:\n{reply}")
