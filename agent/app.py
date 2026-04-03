import json
from flask import Flask, request, jsonify
from flask_cors import CORS
from fact_extracter import FactExtracter
from decision_maker import DecisionMaker
from twitter_watcher import TwitterWatcher
from twitter_bot import TwitterBot, BotConfig
from content_classifier import ContentClassifier, UserDisclosure, quick_analyze
from tools.google_factcheck_tool import GoogleFactCheckTool, verify_video_content
import threading
import requests as re

app = Flask(__name__)
CORS(app)

# Initialize agents
extracter = FactExtracter()
decision_maker = DecisionMaker()
twitter_watcher = TwitterWatcher()
content_classifier = ContentClassifier()
google_fact_checker = GoogleFactCheckTool()

# Bot instance (created on demand)
twitter_bot = None
bot_thread = None


@app.route("/", methods=["GET"])
def health_check():
    """Health check endpoint."""
    return jsonify({"status": "ok", "message": "Pragmatic API is running"})


def extract_message_from_json(data):
    """
    Extract the message/query from various JSON formats.
    Handles unpredictable JSON structures from Agent 1.
    
    Args:
        data: The JSON data (dict, list, or string)
        
    Returns:
        A string representation of the message/query
    """
    if data is None:
        return None
    
    # If it's already a string, return it
    if isinstance(data, str):
        return data.strip() if data.strip() else None
    
    # If it's a list, try to extract content or convert to string
    if isinstance(data, list):
        # Try to find a meaningful item in the list
        for item in data:
            extracted = extract_message_from_json(item)
            if extracted:
                return extracted
        # If nothing found, convert the whole list to string
        return json.dumps(data, indent=2)
    
    # If it's a dict, try common keys first
    if isinstance(data, dict):
        # List of common keys that might contain the message (in priority order)
        common_keys = [
            "message", "msg", "query", "question", "text", "content",
            "input", "prompt", "claim", "statement", "body",
            "request", "user_input", "user_message", "payload"
        ]
        
        # First, try to find an exact match (case-insensitive)
        lower_keys = {k.lower(): k for k in data.keys()}
        for key in common_keys:
            if key in lower_keys:
                value = data[lower_keys[key]]
                if isinstance(value, str) and value.strip():
                    return value.strip()
                elif isinstance(value, (dict, list)):
                    extracted = extract_message_from_json(value)
                    if extracted:
                        return extracted
        
        # If this looks like structured analysis data (has keys like 'details', 'tweet_info', 
        # 'visual_summary', 'Gemini_Scan_Details', etc.), convert the entire thing to a 
        # formatted string for the agent to process
        analysis_indicators = [
            "details", "tweet_info", "visual_summary", "Gemini_Scan_Details",
            "detail_analysis", "is_potentially_misleading", "confidence_score",
            "post_caption", "scene_description", "objects_detected", "analyzed_at"
        ]
        
        lower_data_keys = [k.lower() for k in data.keys()]
        if any(indicator.lower() in lower_data_keys for indicator in analysis_indicators):
            # This is structured analysis data - format it nicely for the agent
            return format_analysis_data(data)
        
        # If no common key found, try the first substantial string value
        for key, value in data.items():
            if isinstance(value, str) and len(value.strip()) > 10:  # Require meaningful content
                return value.strip()
        
        # If still nothing, try nested extraction on dict/list values
        for value in data.values():
            if isinstance(value, (dict, list)):
                extracted = extract_message_from_json(value)
                if extracted and len(extracted) > 10:
                    return extracted
        
        # Last resort: convert the entire dict to a readable string
        return json.dumps(data, indent=2)
    
    # For any other type, convert to string
    return str(data)


def format_analysis_data(data):
    """
    Format structured analysis data into a readable string for the agent.
    
    Args:
        data: Dictionary containing analysis data from Agent 1
        
    Returns:
        A formatted string representation of the analysis
    """
    parts = []
    
    # Extract details section
    details = data.get("details", data)
    
    if isinstance(details, dict):
        if "User" in details or "author" in details:
            parts.append(f"Author: {details.get('User') or details.get('author', 'Unknown')}")
        
        if "post_caption" in details:
            parts.append(f"Post Caption: {details['post_caption']}")
        
        if "Gemini_Scan_Details" in details:
            parts.append(f"Image Analysis: {details['Gemini_Scan_Details']}")
        
        if "Person_talking_about" in details:
            parts.append(f"Audio/Speech: {details['Person_talking_about']}")
        
        # Visual summary
        visual = details.get("visual_summary", {})
        if isinstance(visual, dict):
            if visual.get("objects_detected"):
                parts.append(f"Objects Detected: {', '.join(visual['objects_detected']) if isinstance(visual['objects_detected'], list) else visual['objects_detected']}")
            if visual.get("scene_description"):
                parts.append(f"Scene: {visual['scene_description']}")
            if visual.get("text_overlays") and visual.get("text_overlays") != "None":
                parts.append(f"Text Overlays: {visual['text_overlays']}")
            if visual.get("editing_signs") and visual.get("editing_signs") != "None":
                parts.append(f"Editing Signs: {visual['editing_signs']}")
        
        if "detail_analysis" in details:
            parts.append(f"Analysis: {details['detail_analysis']}")
        
        if "is_potentially_misleading" in details:
            parts.append(f"Potentially Misleading: {details['is_potentially_misleading']}")
        
        if "confidence_score" in details:
            parts.append(f"Confidence Score: {details['confidence_score']}")
    
    # Tweet/post info
    tweet_info = data.get("tweet_info", {})
    if isinstance(tweet_info, dict):
        if tweet_info.get("url"):
            parts.append(f"Source URL: {tweet_info['url']}")
        if tweet_info.get("created_at"):
            parts.append(f"Posted At: {tweet_info['created_at']}")
    
    if data.get("analyzed_at"):
        parts.append(f"Analyzed At: {data['analyzed_at']}")
    
    # If we extracted meaningful parts, join them
    if parts:
        return "Social Media Post Analysis:\n" + "\n".join(parts)
    
    # Fallback to JSON dump
    return json.dumps(data, indent=2)


@app.route("/api/query", methods=["POST"])
def query():
    """
    Process a user query with full content moderation pipeline.
    
    Request body:
        Accepts flexible JSON formats including:
        - {"message": "Your question here"}
        - {"query": "Your question here"}
        - {"text": "Your question here"}
        - {"content": "Your question here"}
        - Or any nested JSON structure
    
    Response:
        Complete analysis with content classification, risk assessment, and fact-check results.
    """
    data = request.get_json()
    
    if not data:
        return jsonify({"error": "Missing request body", "success": False}), 400
    
    user_input = extract_message_from_json(data)
    
    if not user_input or (isinstance(user_input, str) and len(user_input.strip()) == 0):
        return jsonify({"error": "Could not extract message from request body", "success": False}), 400
    
    try:
        # Agent 1: Extract facts and gather evidence
        fact_data = extracter.run(user_input)
        
        # Agent 2: Make final decision based on the extracted facts
        decision = decision_maker.make_decision(fact_data, user_input)
        
        return jsonify({"response": decision, "success": True})
    except Exception as e:
        return jsonify({"error": str(e), "success": False}), 500


@app.route("/api/memory", methods=["GET"])
def get_memory():
    """Get the current conversation memory."""
    return jsonify({"memory": extracter.get_memory(), "success": True})


@app.route("/api/memory", methods=["DELETE"])
def clear_memory():
    """Clear the conversation memory."""
    extracter.clear_memory()
    return jsonify({"message": "Memory cleared", "success": True})


# ============== VIDEO CONTENT ANALYSIS ENDPOINTS ==============

@app.route("/api/video/analyze", methods=["POST"])
def analyze_video_content():
    """
    Analyze video content using multi-layer moderation system.
    
    Request body:
        {
            "transcript": "Video transcript text",
            "is_self_made": false,        // Optional: User declaration
            "is_ai_generated": false,     // Optional: AI generation declaration
            "content_type_declared": "",  // Optional: "vlog", "tutorial", "news"
            "video_id": ""                // Optional: Video identifier
        }
    
    Response:
        Full moderation analysis with risk score, labels, and recommended action.
    """
    data = request.get_json()
    
    if not data or "transcript" not in data:
        return jsonify({
            "error": "Missing 'transcript' field in request body",
            "success": False
        }), 400
    
    transcript = data["transcript"]
    
    if len(transcript.strip()) < 20:
        return jsonify({
            "error": "Transcript too short for analysis (minimum 20 characters)",
            "success": False
        }), 400
    
    try:
        # Create user disclosure from request
        disclosure = UserDisclosure(
            is_self_made=data.get("is_self_made", False),
            is_ai_generated=data.get("is_ai_generated", False),
            content_type_declared=data.get("content_type_declared", ""),
            disclosure_provided=data.get("is_self_made", False) or data.get("is_ai_generated", False)
        )
        
        # Run full analysis
        record = content_classifier.analyze_content(
            transcript=transcript,
            user_disclosure=disclosure,
            video_id=data.get("video_id")
        )
        
        # Determine if fact-checking is needed
        should_check, reason = content_classifier.should_fact_check(record)
        
        # Get claims for fact-checking if needed
        claims_to_verify = []
        if should_check:
            claims_to_verify = content_classifier.get_claims_for_fact_check(record)
        
        return jsonify({
            "success": True,
            "content_type": record.content_classification["content_type"],
            "content_classification": record.content_classification,
            "requires_fact_check": should_check,
            "fact_check_reason": reason,
            "risk_assessment": {
                "total_score": record.risk_score["total_score"],
                "level": "low" if record.risk_score["total_score"] < 0.3 else 
                         "medium" if record.risk_score["total_score"] < 0.6 else "high",
                "components": record.risk_score
            },
            "self_made_analysis": record.self_made_signals,
            "ai_detection": record.ai_generation_signals,
            "deceptive_framing": record.deceptive_framing,
            "harm_assessment": record.harm_assessment,
            "claims_to_verify": claims_to_verify,
            "final_label": record.final_label,
            "final_action": record.final_action,
            "reasoning": record.reasoning,
            "full_record": content_classifier.to_dict(record)
        })
    except Exception as e:
        return jsonify({"error": str(e), "success": False}), 500


@app.route("/api/video/classify", methods=["POST"])
def classify_content():
    """
    Quick content classification (lightweight).
    
    Request body:
        {
            "transcript": "Video transcript text"
        }
    
    Response:
        Content type classification with confidence score.
    """
    data = request.get_json()
    
    if not data or "transcript" not in data:
        return jsonify({
            "error": "Missing 'transcript' field",
            "success": False
        }), 400
    
    try:
        classification = content_classifier.classify_content(data["transcript"])
        self_made = content_classifier.detect_self_made_signals(data["transcript"])
        
        return jsonify({
            "success": True,
            "content_type": classification.content_type,
            "confidence": classification.confidence,
            "reason": classification.reason,
            "requires_fact_check": classification.requires_fact_check,
            "ownership_confidence": self_made.ownership_confidence,
            "self_made_indicators": self_made.detected_phrases
        })
    except Exception as e:
        return jsonify({"error": str(e), "success": False}), 500


@app.route("/api/video/extract-claims", methods=["POST"])
def extract_claims():
    """
    Extract factual claims from content.
    
    Request body:
        {
            "transcript": "Video transcript text"
        }
    
    Response:
        List of extracted factual claims (ignores opinions).
    """
    data = request.get_json()
    
    if not data or "transcript" not in data:
        return jsonify({
            "error": "Missing 'transcript' field",
            "success": False
        }), 400
    
    try:
        claims = content_classifier.extract_claims(data["transcript"])
        
        return jsonify({
            "success": True,
            "total_claims": claims.total_claims,
            "verifiable_claims": claims.verifiable_claims,
            "opinion_statements": claims.opinion_statements,
            "claims": claims.claims
        })
    except Exception as e:
        return jsonify({"error": str(e), "success": False}), 500


@app.route("/api/video/fact-check", methods=["POST"])
def full_video_fact_check():
    """
    Full video fact-check pipeline:
    1. Classify content
    2. Determine if fact-checking is needed
    3. Extract claims (if needed)
    4. Fact-check claims via agents
    5. Return comprehensive result
    
    Request body:
        {
            "transcript": "Video transcript text",
            "is_self_made": false,
            "is_ai_generated": false
        }
    
    Response:
        Complete fact-check result with verdicts.
    """
    data = request.get_json()
    
    if not data or "transcript" not in data:
        return jsonify({
            "error": "Missing 'transcript' field",
            "success": False
        }), 400
    
    transcript = data["transcript"]
    
    try:
        # Step 1: Analyze content
        disclosure = UserDisclosure(
            is_self_made=data.get("is_self_made", False),
            is_ai_generated=data.get("is_ai_generated", False),
            disclosure_provided=data.get("is_self_made", False) or data.get("is_ai_generated", False)
        )
        
        record = content_classifier.analyze_content(transcript, disclosure)
        should_check, reason = content_classifier.should_fact_check(record)
        
        # Step 2: If no fact-check needed, return early
        if not should_check:
            return jsonify({
                "success": True,
                "fact_check_performed": False,
                "content_type": record.content_classification["content_type"],
                "reason": reason,
                "final_label": record.final_label,
                "final_action": record.final_action,
                "risk_score": record.risk_score["total_score"],
                "message": "Content classified as self-made/opinion - fact-checking not applicable"
            })
        
        # Step 3: Extract and verify claims
        claims_to_verify = content_classifier.get_claims_for_fact_check(record)
        
        if not claims_to_verify:
            return jsonify({
                "success": True,
                "fact_check_performed": False,
                "content_type": record.content_classification["content_type"],
                "reason": "No verifiable factual claims found",
                "final_label": record.final_label,
                "final_action": record.final_action,
                "risk_score": record.risk_score["total_score"]
            })
        
        # Step 3.5: Comprehensive Fake Check for non-personal video content
        is_personal_video = record.self_made_signals.get("ownership_confidence", 0) > 0.6
        google_verification = None
        
        if not is_personal_video and not data.get("is_self_made", False):
            print("\n" + "="*70)
            print("🎬 VIDEO FACT-CHECK: Running Comprehensive Fake Detection")
            print("="*70)
            
            # Use full transcript for comprehensive analysis
            google_verdict = google_fact_checker.comprehensive_fake_check(
                transcript,
                verbose=True  # Print all steps to terminal
            )
            
            google_verification = {
                "checked": True,
                "is_fake": google_verdict.is_fake,
                "confidence": google_verdict.confidence,
                "verdict": google_verdict.verdict_summary,
                "action": google_verdict.action,
                "sources": google_verdict.sources,
                "detection_method": google_verdict.detection_method
            }
            
            # If comprehensive check verifies as fake - BLOCK immediately
            if google_verdict.is_fake and google_verdict.action == "block":
                print("\n🚫 VIDEO BLOCKED - Verified as FAKE")
                return jsonify({
                    "success": True,
                        "fact_check_performed": True,
                        "blocked": True,
                        "content_type": record.content_classification["content_type"],
                        "google_fact_check": google_verification,
                        "risk_score": 1.0,
                        "final_label": "❌ BLOCKED: Verified as FAKE VIDEO/NEWS",
                        "final_action": "block",
                        "reasoning": f"Google Fact Check verified this video as fake news. {google_verdict.verdict_summary}",
                        "response": f"🚫 NOT ALLOWED: This video has been verified as FAKE by Google Fact Check."
                    })
        
        # Step 4: Fact-check each claim using existing agents
        verified_claims = []
        for claim in claims_to_verify:
            claim_text = claim.get("claim", "")
            if claim_text:
                try:
                    # Use existing fact-check pipeline
                    fact_data = extracter.run(f"Verify this claim: {claim_text}")
                    decision = decision_maker.make_decision(fact_data, claim_text)
                    
                    verified_claims.append({
                        "claim": claim_text,
                        "type": claim.get("type", "unknown"),
                        "severity": claim.get("severity", "medium"),
                        "verification_result": decision
                    })
                except Exception as e:
                    verified_claims.append({
                        "claim": claim_text,
                        "type": claim.get("type", "unknown"),
                        "verification_result": f"Error verifying: {str(e)}"
                    })
        
        # Step 5: Determine overall verdict
        return jsonify({
            "success": True,
            "fact_check_performed": True,
            "blocked": False,
            "content_type": record.content_classification["content_type"],
            "google_fact_check": google_verification if google_verification else {"checked": False, "reason": "Personal content - skipped"},
            "risk_score": record.risk_score["total_score"],
            "deceptive_framing_detected": record.deceptive_framing["is_deceptive"],
            "ai_generated_probability": record.ai_generation_signals["ai_generated_probability"],
            "total_claims_found": len(claims_to_verify),
            "verified_claims": verified_claims,
            "final_label": record.final_label,
            "final_action": record.final_action,
            "reasoning": record.reasoning,
            "moderation_record": content_classifier.to_dict(record)
        })
        
    except Exception as e:
        return jsonify({"error": str(e), "success": False}), 500


@app.route("/api/video/quick-check", methods=["POST"])
def quick_video_check():
    """
    Quick check endpoint for simple integration.
    
    Request body:
        {
            "transcript": "Video transcript text",
            "is_self_made": false,
            "is_ai_generated": false
        }
    
    Response:
        Quick analysis result with action recommendation.
    """
    data = request.get_json()
    
    if not data or "transcript" not in data:
        return jsonify({
            "error": "Missing 'transcript' field",
            "success": False
        }), 400
    
    try:
        result = quick_analyze(
            transcript=data["transcript"],
            is_self_made=data.get("is_self_made", False),
            is_ai_generated=data.get("is_ai_generated", False)
        )
        
        return jsonify({
            "success": True,
            **result
        })
    except Exception as e:
        return jsonify({"error": str(e), "success": False}), 500


# ============== TWITTER ENDPOINTS ==============

@app.route("/api/analyze/<tweet_id>", methods=["GET"])
def analyze_tweet_get(tweet_id: str):
    """Analyze a tweet by ID."""
    try:
        if not tweet_id:
            return jsonify({"error": "Tweet ID is required", "success": False}), 400
        
        print(f"\n🔍 Analyzing tweet: {tweet_id}")
        result = twitter_watcher.watch(tweet_id)
        
        if "error" in result:
            return jsonify({"error": result["error"], "success": False}), 400
        
        return jsonify({"data": result, "success": True})
    except Exception as e:
        return jsonify({"error": str(e), "success": False}), 500


@app.route("/api/analyze", methods=["POST"])
def analyze_tweet_post():
    """Analyze a tweet by ID or URL (POST)."""
    try:
        data = request.get_json()
        tweet_id = data.get("tweetId")
        tweet_url = data.get("tweetUrl")
        
        # Extract ID from URL if provided
        if not tweet_id and tweet_url:
            match = re.search(r'status/(\d+)', tweet_url)
            if match:
                tweet_id = match.group(1)
        
        if not tweet_id:
            return jsonify({
                "error": "Tweet ID or URL is required",
                "example": {"tweetId": "1234567890", "tweetUrl": "https://x.com/user/status/1234567890"},
                "success": False
            }), 400
        
        print(f"\n🔍 Analyzing tweet: {tweet_id}")
        result = twitter_watcher.watch(tweet_id)
        
        if "error" in result:
            return jsonify({"error": result["error"], "success": False}), 400
        
        return jsonify({"data": result, "success": True})
    except Exception as e:
        return jsonify({"error": str(e), "success": False}), 500


@app.route("/api/tweet/<tweet_id>", methods=["GET"])
def get_tweet(tweet_id: str):
    """Get tweet details only (no analysis)."""
    try:
        if not tweet_id:
            return jsonify({"error": "Tweet ID is required", "success": False}), 400
        
        tweet_data = twitter_watcher.get_tweet_details(tweet_id)
        
        if "errors" in tweet_data:
            return jsonify({"error": "Twitter API Error", "details": tweet_data["errors"], "success": False}), 400
        
        return jsonify({"data": tweet_data, "success": True})
    except Exception as e:
        return jsonify({"error": str(e), "success": False}), 500


@app.route("/api/user/<user_id>", methods=["GET"])
def get_user(user_id: str):
    """Get Twitter user details by ID."""
    try:
        if not user_id:
            return jsonify({"error": "User ID is required", "success": False}), 400
        
        user_data = twitter_watcher.get_user_by_id(user_id)
        
        if not user_data:
            return jsonify({"error": "User not found", "success": False}), 404
        
        return jsonify({"data": user_data, "success": True})
    except Exception as e:
        return jsonify({"error": str(e), "success": False}), 500


# ============== BOT CONTROL ENDPOINTS ==============

@app.route("/api/bot/start", methods=["POST"])
def start_bot():
    """Start the Twitter bot in a background thread."""
    global twitter_bot, bot_thread
    
    if twitter_bot and twitter_bot.is_running:
        return jsonify({"message": "Bot is already running", "success": True})
    
    try:
        data = request.get_json() or {}
        interval = data.get("interval", 15)
        auto_reply = data.get("autoReply", True)
        
        config = BotConfig(
            poll_interval_seconds=interval,
            auto_reply=auto_reply
        )
        
        twitter_bot = TwitterBot(config)
        
        # Start in background thread
        bot_thread = threading.Thread(target=twitter_bot.start, daemon=True)
        bot_thread.start()
        
        return jsonify({
            "message": "Bot started",
            "config": {
                "interval": interval,
                "autoReply": auto_reply
            },
            "success": True
        })
    except Exception as e:
        return jsonify({"error": str(e), "success": False}), 500


@app.route("/api/bot/stop", methods=["POST"])
def stop_bot():
    """Stop the Twitter bot."""
    global twitter_bot
    
    if not twitter_bot or not twitter_bot.is_running:
        return jsonify({"message": "Bot is not running", "success": True})
    
    twitter_bot.stop()
    return jsonify({"message": "Bot stopped", "success": True})


@app.route("/api/bot/status", methods=["GET"])
def bot_status():
    """Get current bot status."""
    global twitter_bot
    
    if not twitter_bot:
        return jsonify({
            "running": False,
            "username": None,
            "processed_mentions": 0,
            "success": True
        })
    
    return jsonify({
        "running": twitter_bot.is_running,
        "username": twitter_bot.bot_username,
        "user_id": twitter_bot.bot_user_id,
        "processed_mentions": len(twitter_bot.processed_mentions),
        "config": {
            "interval": twitter_bot.config.poll_interval_seconds,
            "autoReply": twitter_bot.config.auto_reply
        },
        "success": True
    })


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000, use_reloader=False)