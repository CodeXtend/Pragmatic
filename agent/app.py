import re
import sys
import os
import threading
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, request, jsonify
from flask_cors import CORS
from fact_extracter import FactExtracter
from decision_maker import DecisionMaker
from twitter_watcher import TwitterWatcher
from twitter_bot import TwitterBot, BotConfig
from content_classifier import ContentClassifier, UserDisclosure, quick_analyze
from tools.google_factcheck_tool import GoogleFactCheckTool, verify_video_content

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
    return jsonify({
        "status": "ok",
        "message": "Pragmatic API is running",
        "version": "2.0.0",
        "agents": {
            "fact_extracter": "ready",
            "decision_maker": "ready",
            "twitter_watcher": "ready" if twitter_watcher.gemini_model else "no gemini",
            "twitter_bot": "running" if twitter_bot and twitter_bot.is_running else "stopped",
            "content_classifier": "ready"
        },
        "endpoints": {
            "GET /": "Health check",
            "GET /api/health": "Detailed health",
            "POST /api/query": "Fact-check a claim",
            "GET /api/analyze/<tweet_id>": "Analyze a tweet",
            "POST /api/analyze": "Analyze tweet (JSON)",
            "GET /api/tweet/<tweet_id>": "Get tweet details",
            "GET /api/user/<user_id>": "Get user details",
            "POST /api/bot/start": "Start the bot",
            "POST /api/bot/stop": "Stop the bot",
            "GET /api/bot/status": "Bot status",
            "POST /api/video/analyze": "Analyze video content",
            "POST /api/video/classify": "Classify content type",
            "POST /api/video/extract-claims": "Extract factual claims",
            "POST /api/video/fact-check": "Full fact-check pipeline"
        }
    })


@app.route("/api/query", methods=["POST"])
def query():
    """
    Process a user query with full content moderation pipeline.
    
    Request body:
        {
            "message": "Your text/transcript here",
            "is_self_made": false,        // Optional: User declaration
            "is_ai_generated": false      // Optional: AI generation declaration
        }
    
    Response:
        Complete analysis with content classification, risk assessment, and fact-check results.
    """
    data = request.get_json()
    
    if not data or "message" not in data:
        return jsonify({"error": "Missing 'message' field in request body", "success": False}), 400
    
    user_input = data["message"]
    
    if len(user_input.strip()) < 10:
        return jsonify({"error": "Message too short for analysis", "success": False}), 400
    
    try:
        # ============== STEP 1: Content Classification ==============
        disclosure = UserDisclosure(
            is_self_made=data.get("is_self_made", False),
            is_ai_generated=data.get("is_ai_generated", False),
            content_type_declared=data.get("content_type", ""),
            disclosure_provided=data.get("is_self_made", False) or data.get("is_ai_generated", False)
        )
        
        # Run full content analysis
        record = content_classifier.analyze_content(
            transcript=user_input,
            user_disclosure=disclosure
        )
        
        # ============== STEP 2: Determine if Fact-Check Needed ==============
        should_check, reason = content_classifier.should_fact_check(record)
        
        # ============== STEP 3: Handle Self-Made / Non-Factual Content ==============
        if not should_check:
            # Skip fact-checking for self-made/opinion content
            return jsonify({
                "success": True,
                "fact_check_performed": False,
                
                # Content Analysis
                "content_analysis": {
                    "content_type": record.content_classification["content_type"],
                    "confidence": record.content_classification["confidence"],
                    "reason": record.content_classification["reason"]
                },
                
                # Self-Made Detection
                "self_made_analysis": {
                    "ownership_confidence": record.self_made_signals["ownership_confidence"],
                    "detected_indicators": record.self_made_signals["detected_phrases"],
                    "first_person_score": record.self_made_signals["first_person_score"],
                    "demonstration_score": record.self_made_signals["demonstration_score"]
                },
                
                # AI Detection
                "ai_detection": {
                    "probability": record.ai_generation_signals["ai_generated_probability"],
                    "indicators": record.ai_generation_signals["synthetic_voice_indicators"],
                    "note": "Risk signal only, not definitive proof"
                },
                
                # Risk Assessment
                "risk_assessment": {
                    "total_score": record.risk_score["total_score"],
                    "level": "low" if record.risk_score["total_score"] < 0.3 else 
                             "medium" if record.risk_score["total_score"] < 0.6 else "high",
                    "deceptive_framing": record.deceptive_framing["is_deceptive"],
                    "deceptive_score": record.deceptive_framing["deceptive_score"]
                },
                
                # Final Result
                "result": {
                    "label": record.final_label,
                    "action": record.final_action,
                    "reasoning": record.reasoning,
                    "skip_reason": reason
                },
                
                "message": "✅ Content classified as self-made/opinion - fact-checking not applicable"
            })
        
        # ============== STEP 4: Extract Claims for Fact-Checking ==============
        claims_to_verify = content_classifier.get_claims_for_fact_check(record)
        
        if not claims_to_verify:
            return jsonify({
                "success": True,
                "fact_check_performed": False,
                
                "content_analysis": {
                    "content_type": record.content_classification["content_type"],
                    "confidence": record.content_classification["confidence"]
                },
                
                "risk_assessment": {
                    "total_score": record.risk_score["total_score"],
                    "level": "low" if record.risk_score["total_score"] < 0.3 else 
                             "medium" if record.risk_score["total_score"] < 0.6 else "high",
                    "deceptive_framing": record.deceptive_framing["is_deceptive"]
                },
                
                "result": {
                    "label": record.final_label,
                    "action": record.final_action,
                    "reasoning": "No verifiable factual claims found in content"
                },
                
                "message": "ℹ️ No verifiable factual claims found to check"
            })
        
        # ============== STEP 5: Fact-Check Each Claim ==============
        # Optimization: Priority based limiting to avoid API quota exhaustion
        # specific to Gemini Free Tier (15 RPM)
        
        # Sort by severity (high > medium > low)
        severity_map = {"high": 3, "medium": 2, "low": 1}
        claims_to_verify.sort(key=lambda x: severity_map.get(x.get("severity", "medium"), 1), reverse=True)
        
        # Limit to top 1 claim to prevent rate limit exhaustion
        claims_to_verify = claims_to_verify[:1]
        
        # ============== STEP 5.5: Comprehensive Fake Check for Non-Personal Content ==============
        # If content is NOT personal/self-made, verify using Google Fact Check API + Gemini AI
        is_personal_content = record.self_made_signals.get("ownership_confidence", 0) > 0.6
        google_verification = None
        
        if not is_personal_content and not data.get("is_self_made", False):
            print("\n" + "="*70)
            print("🔍 NON-PERSONAL CONTENT DETECTED - Running Comprehensive Fake Check")
            print("="*70)
            
            # Use the full content for comprehensive analysis
            content_to_check = user_input
            
            # Run comprehensive check (Google API + Gemini AI)
            google_verdict = google_fact_checker.comprehensive_fake_check(
                content_to_check,
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
            
            # If comprehensive check says it's fake, take action based on severity
            if google_verdict.is_fake and google_verdict.action == "block":
                print("\n🚫 FINAL DECISION: CONTENT BLOCKED - Verified as FAKE")
                return jsonify({
                    "success": True,
                    "fact_check_performed": True,
                    "blocked": True,
                    
                    # Content Analysis
                    "content_analysis": {
                        "content_type": record.content_classification["content_type"],
                        "confidence": record.content_classification["confidence"],
                        "reason": record.content_classification["reason"]
                    },
                    
                    # Comprehensive Fact Check Result
                    "google_fact_check": google_verification,
                    
                    # Risk Assessment
                    "risk_assessment": {
                        "total_score": 1.0,  # Maximum risk
                        "level": "critical",
                        "deceptive_framing": record.deceptive_framing["is_deceptive"]
                    },
                    
                    # Final Result
                    "result": {
                        "label": "❌ BLOCKED: Verified as FAKE NEWS/VIDEO",
                        "action": "block",
                        "reasoning": google_verdict.verdict_summary
                    },
                    
                    "response": f"🚫 NOT ALLOWED: This content has been verified as FAKE. {google_verdict.verdict_summary}"
                    })
        else:
            google_verification = {
                "checked": False,
                "reason": "Personal/self-made content - Google verification skipped"
            }
        
        verified_claims = []
        overall_verdict = "verified"
        false_claims_count = 0
        
        for claim in claims_to_verify:
            claim_text = claim.get("claim", "")
            if claim_text:
                try:
                    # Optimized: Use fast_check instead of agent loop
                    # This saves 5+ API calls per claim
                    result = extracter.fast_check(claim_text)
                    
                    verdict_status = result.get("verdict", "Unverified")
                    analysis = result.get("analysis", "No analysis provided")
                    
                    # Parse decision to determine verdict
                    is_false = "false" in verdict_status.lower() or "fake" in verdict_status.lower()
                    if is_false:
                        false_claims_count += 1
                    
                    # Store result
                    decision_json = {
                        "details": {
                            "fact": f"This is {verdict_status}",
                            "analysis": analysis
                        }
                    }
                    
                    verified_claims.append({
                        "claim": claim_text,
                        "type": claim.get("type", "unknown"),
                        "severity": claim.get("severity", "medium"),
                        "verdict": "False" if is_false else verdict_status,
                        "verification_result": decision_json
                    })
                except Exception as e:
                    verified_claims.append({
                        "claim": claim_text,
                        "type": claim.get("type", "unknown"),
                        "verdict": "Error",
                        "verification_result": f"Error verifying: {str(e)}"
                    })
        
        # Determine overall verdict
        # Also consider Google Fact Check results
        google_says_fake = google_verification and google_verification.get("checked") and google_verification.get("is_fake")
        
        if google_says_fake or false_claims_count > 0:
            if google_says_fake and google_verification.get("action") == "reduce_reach":
                overall_verdict = "restricted_fake"
                final_label = f"⚠️ RESTRICTED: {google_verification.get('verdict', 'Flagged as potentially fake')}"
                final_action = "reduce_reach"
            elif false_claims_count == len(verified_claims) and verified_claims:
                overall_verdict = "all_false"
                final_label = "❌ Contains false claims"
                final_action = "block" if google_says_fake else "label"
            else:
                overall_verdict = "partially_false"
                final_label = f"⚠️ Contains {false_claims_count} false claim(s)"
                final_action = "label"
        else:
            overall_verdict = "verified"
            final_label = "✅ Claims verified"
            final_action = "allow"
        
        # ============== STEP 6: Return Complete Result ==============
        return jsonify({
            "success": True,
            "fact_check_performed": True,
            "blocked": final_action == "block",
            
            # Content Analysis
            "content_analysis": {
                "content_type": record.content_classification["content_type"],
                "confidence": record.content_classification["confidence"],
                "reason": record.content_classification["reason"]
            },
            
            # Self-Made Detection
            "self_made_analysis": {
                "ownership_confidence": record.self_made_signals["ownership_confidence"],
                "detected_indicators": record.self_made_signals["detected_phrases"]
            },
            
            # AI Detection
            "ai_detection": {
                "probability": record.ai_generation_signals["ai_generated_probability"],
                "indicators": record.ai_generation_signals["synthetic_voice_indicators"]
            },
            
            # Google Fact Check Results (NEW)
            "google_fact_check": google_verification,
            
            # Deceptive Framing
            "deceptive_framing": {
                "is_deceptive": record.deceptive_framing["is_deceptive"],
                "score": record.deceptive_framing["deceptive_score"],
                "red_flags": record.deceptive_framing["red_flags"],
                "authority_claims": record.deceptive_framing["authority_claims"],
                "urgency_indicators": record.deceptive_framing["urgency_indicators"]
            },
            
            # Risk Assessment
            "risk_assessment": {
                "total_score": 1.0 if google_says_fake else record.risk_score["total_score"],
                "level": "critical" if google_says_fake else (
                         "low" if record.risk_score["total_score"] < 0.3 else 
                         "medium" if record.risk_score["total_score"] < 0.6 else "high"),
                "components": {
                    "ai_generated": record.risk_score["ai_generated_component"],
                    "deceptive_framing": record.risk_score["deceptive_framing_component"],
                    "claim_severity": record.risk_score["factual_claim_severity"],
                    "disclosure_penalty": record.risk_score["disclosure_penalty"],
                    "google_fake_flag": 1.0 if google_says_fake else 0.0
                }
            },
            
            # Harm Assessment
            "harm_assessment": {
                "category": record.harm_assessment["harm_category"],
                "severity": record.harm_assessment["severity"],
                "reach_impact": record.harm_assessment["potential_reach_impact"],
                "requires_action": record.harm_assessment["requires_immediate_action"]
            },
            
            # Fact-Check Results
            "fact_check_results": {
                "total_claims": len(claims_to_verify),
                "false_claims": false_claims_count,
                "verified_claims": len(verified_claims) - false_claims_count,
                "overall_verdict": overall_verdict,
                "claims": verified_claims
            },
            
            # Final Result
            "result": {
                "label": final_label,
                "action": final_action,
                "reasoning": record.reasoning + (
                    f" Google Fact Check: {google_verification.get('verdict', 'N/A')}" 
                    if google_verification and google_verification.get("checked") else ""
                )
            },
            
            "response": (
                f"🚫 NOT ALLOWED: {final_label}" if final_action == "block" else
                f"⚠️ RESTRICTED: {final_label}" if final_action == "reduce_reach" else
                f"{final_label} - Checked {len(verified_claims)} claim(s), {false_claims_count} found false."
            )
        })
        
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
    app.run(debug=True, host="0.0.0.0", port=5000)