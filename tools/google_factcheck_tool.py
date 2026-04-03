import requests
from smolagents import Tool
from typing import Optional, Dict, Any, List
import os
import re
import json
from dataclasses import dataclass
from dotenv import load_dotenv

# Try to import Google Generative AI
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    genai = None
    GEMINI_AVAILABLE = False

# Import News Verification Tool
try:
    from tools.news_verification_tool import NewsVerificationTool, verify_content_with_news
    NEWS_VERIFICATION_AVAILABLE = True
except ImportError:
    try:
        from news_verification_tool import NewsVerificationTool, verify_content_with_news
        NEWS_VERIFICATION_AVAILABLE = True
    except ImportError:
        NEWS_VERIFICATION_AVAILABLE = False
        NewsVerificationTool = None


@dataclass
class FactCheckVerdict:
    """Result of a fact-check verification."""
    is_fake: bool
    confidence: float  # 0.0 to 1.0
    sources: List[str]
    verdict_summary: str
    action: str  # "allow", "label", "reduce_reach", "block"
    raw_results: Dict[str, Any]
    detection_method: str = "unknown"  # "google_api", "gemini_ai", "combined"


class GoogleFactCheckTool(Tool):
    name = "google_fact_check"
    description = "Search for fact-checks on a given claim using Google Fact Check Tools API."
    inputs = {
        "query": {
            "type": "string",
            "description": "The claim or statement to fact-check"
        }
    }
    output_type = "string"

    # Keywords indicating fake/false content
    FAKE_INDICATORS = [
        "false", "fake", "hoax", "misleading", "debunked", 
        "fabricated", "not true", "inaccurate", "satire",
        "manipulated", "doctored", "altered", "scam", "rumor",
        "unverified", "baseless", "conspiracy", "disinformation",
        "misinformation", "pants on fire", "mostly false"
    ]
    
    # Keywords indicating true content
    TRUE_INDICATORS = [
        "true", "verified", "accurate", "confirmed", "factual",
        "correct", "authentic", "legitimate", "real"
    ]
    
    # Dangerous/impossible scenario indicators - patterns that indicate staged/fake content
    DANGEROUS_SCENARIO_INDICATORS = [
        "skateboarding in front of bus", "skateboarding on their stomach",
        "lying on road", "laying on road", "dangerous stunt",
        "in front of traffic", "middle of road", "blocking traffic",
        "stomach on road", "laying in street", "viral stunt",
        "directly in front of", "center of a busy", "in front of a large",
        "daring atmosphere", "chaotic yet daring", "in front of a moving"
    ]
    
    # High-risk scenario patterns (regex-like patterns)
    FAKE_CONTENT_PATTERNS = [
        # Person doing dangerous things in traffic
        r"person.*skateboard.*front.*bus",
        r"person.*skateboard.*middle.*road",
        r"person.*stomach.*road",
        r"person.*lying.*traffic",
        r"skateboard.*directly.*front.*vehicle",
        r"skateboard.*busy.*road",
        r"skateboard.*stomach.*center",
        # Impossible survival scenarios
        r"walking.*middle.*busy.*road",
        r"lying.*front.*truck",
        r"person.*directly.*front.*moving",
        # Viral stunt indicators
        r"daring.*amidst.*traffic",
        r"chaotic.*daring",
        r"stunt.*traffic"
    ]

    def __init__(self):
        super().__init__()
        load_dotenv()
        self.api_key = os.getenv("GOOGLE_FACT_CHECK_API_KEY")
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        
        # Initialize Gemini
        self.gemini_model = None
        if GEMINI_AVAILABLE and self.gemini_api_key:
            try:
                genai.configure(api_key=self.gemini_api_key)
                self.gemini_model = genai.GenerativeModel("gemini-2.5-flash")
                print("✅ Gemini AI initialized for fake content detection")
            except Exception as e:
                print(f"⚠️ Could not initialize Gemini: {e}")
        
        # Initialize News Verification Tool
        self.news_verifier = None
        if NEWS_VERIFICATION_AVAILABLE:
            try:
                self.news_verifier = NewsVerificationTool()
                print("✅ News Verification Tool initialized")
            except Exception as e:
                print(f"⚠️ Could not initialize News Verification: {e}")
    
    def rule_based_fake_detection(self, content: str, verbose: bool = True) -> FactCheckVerdict:
        """
        Rule-based fake content detection as fallback when Gemini is unavailable.
        Uses pattern matching to detect dangerous/fake scenarios.
        
        Args:
            content: Content description to analyze
            verbose: Print steps to terminal
            
        Returns:
            FactCheckVerdict with rule-based analysis
        """
        if verbose:
            print("\n" + "-"*50)
            print("📋 FALLBACK: Rule-Based Fake Detection")
            print("-"*50)
        
        content_lower = content.lower()
        detected_indicators = []
        detected_patterns = []
        
        # Check for dangerous scenario indicators
        for indicator in self.DANGEROUS_SCENARIO_INDICATORS:
            if indicator in content_lower:
                detected_indicators.append(indicator)
        
        # Check for fake content patterns (regex)
        for pattern in self.FAKE_CONTENT_PATTERNS:
            if re.search(pattern, content_lower):
                detected_patterns.append(pattern)
        
        total_matches = len(detected_indicators) + len(detected_patterns)
        
        if verbose:
            print(f"   🔍 Dangerous indicators found: {len(detected_indicators)}")
            for ind in detected_indicators[:5]:
                print(f"      - '{ind}'")
            print(f"   🔍 Fake patterns matched: {len(detected_patterns)}")
            for pat in detected_patterns[:5]:
                print(f"      - '{pat}'")
        
        # Determine if fake based on matches
        if total_matches >= 3:
            is_fake = True
            confidence = min(0.95, 0.6 + (total_matches * 0.1))
            action = "block"
            verdict = f"❌ BLOCKED: Rule-based detection found {total_matches} dangerous/fake indicators"
        elif total_matches >= 2:
            is_fake = True
            confidence = 0.7
            action = "reduce_reach"
            verdict = f"⚠️ RESTRICTED: Detected {total_matches} suspicious patterns"
        elif total_matches >= 1:
            is_fake = True
            confidence = 0.5
            action = "label"
            verdict = f"ℹ️ LABELED: Detected suspicious content pattern"
        else:
            is_fake = False
            confidence = 0.3
            action = "allow"
            verdict = "✅ No dangerous patterns detected"
        
        if verbose:
            print(f"\n   🎯 Rule-Based Verdict: {'FAKE' if is_fake else 'ALLOWED'}")
            print(f"   📊 Confidence: {confidence*100:.0f}%")
            print(f"   📋 Action: {action.upper()}")
        
        return FactCheckVerdict(
            is_fake=is_fake,
            confidence=confidence,
            sources=["Rule-based pattern detection"],
            verdict_summary=verdict,
            action=action,
            raw_results={
                "detected_indicators": detected_indicators,
                "detected_patterns": detected_patterns,
                "total_matches": total_matches
            },
            detection_method="rule_based"
        )
       
    def forward(self, query: str) -> str:
        """
        Search for fact-checks on the given query.

        Args:
            query: The claim to fact-check

        Returns:
            A string containing the fact-check results
        """
        url = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
        params = {
            "query": query,
            "key": self.api_key
        }

        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()

            return data
        except requests.exceptions.RequestException as e:
            return f"Error querying Google Fact Check API: {str(e)}"
        except Exception as e:
            return f"Error processing fact-check results: {str(e)}"

    def verify_content(self, query: str) -> FactCheckVerdict:
        """
        Verify content and determine if it's fake based on Google Fact Check API.
        
        Args:
            query: The claim or content description to verify
            
        Returns:
            FactCheckVerdict with determination and action
        """
        url = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
        params = {
            "query": query,
            "key": self.api_key
        }

        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            return self._analyze_fact_check_results(data, query)
            
        except requests.exceptions.RequestException as e:
            return FactCheckVerdict(
                is_fake=False,
                confidence=0.0,
                sources=[],
                verdict_summary=f"Could not verify: {str(e)}",
                action="allow",
                raw_results={"error": str(e)}
            )
        except Exception as e:
            return FactCheckVerdict(
                is_fake=False,
                confidence=0.0,
                sources=[],
                verdict_summary=f"Error processing results: {str(e)}",
                action="allow",
                raw_results={"error": str(e)}
            )

    def _analyze_fact_check_results(self, data: Dict[str, Any], original_query: str) -> FactCheckVerdict:
        """
        Analyze Google Fact Check API results to determine if content is fake.
        
        Args:
            data: Raw API response
            original_query: Original search query
            
        Returns:
            FactCheckVerdict with analysis
        """
        claims = data.get("claims", [])
        
        if not claims:
            # No fact-checks found - can't verify
            return FactCheckVerdict(
                is_fake=False,
                confidence=0.0,
                sources=[],
                verdict_summary="No fact-check results found for this content",
                action="allow",
                raw_results=data
            )
        
        fake_score = 0
        true_score = 0
        sources = []
        verdicts = []
        
        for claim in claims:
            claim_text = claim.get("text", "")
            claim_reviews = claim.get("claimReview", [])
            
            for review in claim_reviews:
                publisher = review.get("publisher", {}).get("name", "Unknown")
                rating = review.get("textualRating", "").lower()
                url = review.get("url", "")
                
                if url:
                    sources.append(f"{publisher}: {url}")
                
                verdicts.append({
                    "publisher": publisher,
                    "rating": rating,
                    "claim": claim_text
                })
                
                # Score based on rating keywords
                rating_lower = rating.lower()
                
                for indicator in self.FAKE_INDICATORS:
                    if indicator in rating_lower:
                        fake_score += 1
                        break
                
                for indicator in self.TRUE_INDICATORS:
                    if indicator in rating_lower:
                        true_score += 1
                        break
        
        total_reviews = len(verdicts)
        
        if total_reviews == 0:
            return FactCheckVerdict(
                is_fake=False,
                confidence=0.0,
                sources=sources,
                verdict_summary="No reviews found",
                action="allow",
                raw_results=data
            )
        
        # Calculate fake probability
        fake_ratio = fake_score / total_reviews if total_reviews > 0 else 0
        confidence = min(1.0, total_reviews / 3)  # More reviews = higher confidence
        
        # Determine if fake
        is_fake = fake_ratio >= 0.5 and fake_score > 0
        
        # Determine action based on results
        if is_fake and fake_ratio >= 0.8 and confidence >= 0.6:
            action = "block"
            verdict_summary = f"❌ BLOCKED: Content verified as FAKE by {fake_score}/{total_reviews} fact-checkers"
        elif is_fake and fake_ratio >= 0.5:
            action = "reduce_reach"
            verdict_summary = f"⚠️ RESTRICTED: Content flagged as potentially false by {fake_score}/{total_reviews} fact-checkers"
        elif fake_score > 0:
            action = "label"
            verdict_summary = f"ℹ️ LABELED: Content disputed by some fact-checkers ({fake_score}/{total_reviews})"
        else:
            action = "allow"
            verdict_summary = f"✅ ALLOWED: No fake indicators found ({true_score}/{total_reviews} verified true)"
        
        return FactCheckVerdict(
            is_fake=is_fake,
            confidence=confidence,
            sources=sources[:5],  # Top 5 sources
            verdict_summary=verdict_summary,
            action=action,
            raw_results={
                "total_reviews": total_reviews,
                "fake_score": fake_score,
                "true_score": true_score,
                "fake_ratio": fake_ratio,
                "verdicts": verdicts
            }
        )

    def search_for_fake_news(self, content_description: str, additional_keywords: List[str] = None) -> FactCheckVerdict:
        """
        Search specifically for fake news indicators about the content.
        Combines the content with "fake" and "fact check" keywords for better results.
        
        Args:
            content_description: Description of the video/content
            additional_keywords: Optional additional search terms
            
        Returns:
            FactCheckVerdict with determination
        """
        # Build search queries
        queries = [
            f"{content_description} fact check",
            f"{content_description} fake",
            f"{content_description} hoax debunked"
        ]
        
        if additional_keywords:
            for kw in additional_keywords[:3]:
                queries.append(f"{kw} fact check")
        
        # Aggregate results from multiple queries
        all_verdicts = []
        all_sources = []
        
        for query in queries[:3]:  # Limit to 3 API calls
            verdict = self.verify_content(query)
            if verdict.confidence > 0:
                all_verdicts.append(verdict)
                all_sources.extend(verdict.sources)
        
        if not all_verdicts:
            return FactCheckVerdict(
                is_fake=False,
                confidence=0.0,
                sources=[],
                verdict_summary="No fact-check information found in Google search",
                action="allow",
                raw_results={"queries": queries}
            )
        
        # Aggregate: If any verdict says fake with high confidence, mark as fake
        fake_verdicts = [v for v in all_verdicts if v.is_fake]
        
        if fake_verdicts:
            # Take the most severe verdict
            most_severe = max(fake_verdicts, key=lambda v: v.confidence)
            return FactCheckVerdict(
                is_fake=True,
                confidence=most_severe.confidence,
                sources=list(set(all_sources))[:5],
                verdict_summary=most_severe.verdict_summary,
                action=most_severe.action,
                raw_results={
                    "total_queries": len(queries),
                    "fake_verdicts": len(fake_verdicts),
                    "total_verdicts": len(all_verdicts)
                },
                detection_method="google_api"
            )
        
        # No fake verdicts found from Google API
        best_verdict = max(all_verdicts, key=lambda v: v.confidence) if all_verdicts else None
        return FactCheckVerdict(
            is_fake=False,
            confidence=best_verdict.confidence if best_verdict else 0.0,
            sources=list(set(all_sources))[:5],
            verdict_summary=f"ℹ️ No fact-check results found in Google database",
            action="allow",
            raw_results={
                "total_queries": len(queries),
                "total_verdicts": len(all_verdicts)
            },
            detection_method="google_api"
        )

    def analyze_with_gemini(self, content_description: str, verbose: bool = True) -> FactCheckVerdict:
        """
        Use Gemini AI to analyze if content is fake/manipulated.
        This is used when Google Fact Check API doesn't have results.
        
        Args:
            content_description: Description of the video/image content
            verbose: Print steps to terminal
            
        Returns:
            FactCheckVerdict with AI analysis
        """
        if verbose:
            print("\n" + "="*60)
            print("🤖 STEP: Gemini AI Fake Content Analysis")
            print("="*60)
            print(f"📝 Content: {content_description[:200]}...")
        
        if not self.gemini_model:
            if verbose:
                print("❌ Gemini not available - skipping AI analysis")
            return FactCheckVerdict(
                is_fake=False,
                confidence=0.0,
                sources=[],
                verdict_summary="Gemini AI not available for analysis",
                action="allow",
                raw_results={"error": "Gemini not initialized"},
                detection_method="none"
            )
        
        prompt = f"""You are a fact-checking AI specialized in detecting fake, manipulated, or staged content.

Analyze the following video/image description and determine if it's likely FAKE or REAL.

CONTENT DESCRIPTION:
{content_description}

ANALYSIS CRITERIA:
1. **Physical Impossibility**: Does the scene describe something physically impossible or extremely dangerous that would result in death/injury?
2. **Staged/Viral Stunt**: Does it appear to be a staged viral stunt designed to get views?
3. **Logical Inconsistency**: Are there logical inconsistencies (e.g., person calmly skateboarding in front of a moving bus)?
4. **Clickbait Pattern**: Does it follow clickbait/viral fake content patterns?
5. **Safety Reality**: Would a real person survive this scenario? Would authorities allow this?
6. **Camera Perspective**: Does the camera angle suggest it's staged (perfect framing of dangerous act)?

RED FLAGS FOR FAKE CONTENT:
- Person doing dangerous stunts in front of vehicles
- "Miracle" survival scenarios
- Perfect camera timing for dangerous events
- No visible concern from bystanders in life-threatening situations
- Content designed to shock/go viral

Return your analysis as JSON:
{{
    "is_fake": true/false,
    "confidence": 0.0-1.0,
    "verdict": "FAKE" or "REAL" or "LIKELY_FAKE" or "UNCERTAIN",
    "reasoning": "Detailed explanation",
    "red_flags": ["list", "of", "red", "flags", "detected"],
    "danger_level": "none/low/medium/high/extreme",
    "recommendation": "block/reduce_reach/label/allow"
}}"""

        try:
            if verbose:
                print("🔄 Sending to Gemini for analysis...")
            
            response = self.gemini_model.generate_content(prompt)
            response_text = response.text
            
            if verbose:
                print(f"📥 Gemini response received")
            
            # Parse JSON from response
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group(0))
            else:
                result = {"is_fake": False, "confidence": 0.5, "verdict": "UNCERTAIN", "reasoning": response_text}
            
            is_fake = result.get("is_fake", False) or result.get("verdict", "").upper() in ["FAKE", "LIKELY_FAKE"]
            confidence = float(result.get("confidence", 0.7))
            reasoning = result.get("reasoning", "No reasoning provided")
            red_flags = result.get("red_flags", [])
            recommendation = result.get("recommendation", "label")
            
            if verbose:
                print("\n" + "-"*50)
                print(f"🎯 GEMINI VERDICT: {'❌ FAKE' if is_fake else '✅ REAL'}")
                print(f"📊 Confidence: {confidence*100:.1f}%")
                print(f"🚨 Red Flags: {', '.join(red_flags) if red_flags else 'None'}")
                print(f"💬 Reasoning: {reasoning[:200]}...")
                print(f"📋 Recommendation: {recommendation.upper()}")
                print("-"*50)
            
            # Determine action based on analysis
            if is_fake and confidence >= 0.8:
                action = "block"
                verdict_summary = f"❌ BLOCKED: Gemini AI detected FAKE content with {confidence*100:.0f}% confidence. {reasoning[:100]}"
            elif is_fake and confidence >= 0.6:
                action = "reduce_reach"
                verdict_summary = f"⚠️ RESTRICTED: Likely fake content detected. {reasoning[:100]}"
            elif is_fake:
                action = "label"
                verdict_summary = f"ℹ️ LABELED: Possibly fake content. {reasoning[:100]}"
            else:
                action = "allow"
                verdict_summary = f"✅ Content appears genuine. {reasoning[:100]}"
            
            return FactCheckVerdict(
                is_fake=is_fake,
                confidence=confidence,
                sources=["Gemini AI Analysis"],
                verdict_summary=verdict_summary,
                action=action,
                raw_results={
                    "gemini_response": result,
                    "red_flags": red_flags,
                    "danger_level": result.get("danger_level", "unknown")
                },
                detection_method="gemini_ai"
            )
            
        except Exception as e:
            if verbose:
                print(f"❌ Gemini analysis error: {e}")
            return FactCheckVerdict(
                is_fake=False,
                confidence=0.0,
                sources=[],
                verdict_summary=f"Gemini analysis failed: {str(e)}",
                action="allow",
                raw_results={"error": str(e)},
                detection_method="error"
            )

    def comprehensive_fake_check(self, content_description: str, verbose: bool = True) -> FactCheckVerdict:
        """
        Comprehensive fake content check using News Verification + Gemini AI.
        
        NEW APPROACH - News-Based Verification:
        1. Extract key incident/event from content
        2. Search Google News for the incident
        3. Verify against news sources (NDTV, Times of India, BBC, etc.)
        4. If found in news - verify if real or fake according to news
        5. If NOT found in news - analyze if staged/fake content
        
        Args:
            content_description: Description of the video/content
            verbose: Print all steps to terminal
            
        Returns:
            FactCheckVerdict with news-based verification
        """
        if verbose:
            print("\n" + "="*70)
            print("🔍 NEWS-BASED FAKE CONTENT VERIFICATION")
            print("="*70)
            print(f"📝 Content to verify: {content_description[:150]}...")
            print("="*70)
        
        # ============== STEP 1: NEWS-BASED VERIFICATION (PRIMARY) ==============
        if self.news_verifier:
            if verbose:
                print("\n📋 STEP 1: News-Based Verification (Primary Method)")
                print("   Checking if incident is covered in news sources...")
            
            try:
                news_result = self.news_verifier.comprehensive_verify(content_description, verbose=verbose)
                
                # If news verification gave a clear verdict
                if news_result.verdict in ["FAKE", "REAL", "LIKELY_FAKE"]:
                    if verbose:
                        print(f"\n✅ News Verification Complete: {news_result.verdict}")
                    
                    return FactCheckVerdict(
                        is_fake=news_result.is_fake,
                        confidence=news_result.confidence,
                        sources=[s.source_name for s in news_result.news_sources[:5]],
                        verdict_summary=news_result.verdict_summary,
                        action=news_result.action,
                        raw_results={
                            "news_verification": {
                                "verdict": news_result.verdict,
                                "detection_method": news_result.detection_method,
                                "news_sources_count": len(news_result.news_sources),
                                "incident_extracted": news_result.extracted_incident.main_event if news_result.extracted_incident else ""
                            }
                        },
                        detection_method=f"news_{news_result.detection_method}"
                    )
                elif news_result.verdict == "LIKELY_REAL":
                    # Verified by news - return as real
                    return FactCheckVerdict(
                        is_fake=False,
                        confidence=news_result.confidence,
                        sources=[s.source_name for s in news_result.news_sources[:5]],
                        verdict_summary=news_result.verdict_summary,
                        action="allow",
                        raw_results={"news_verification": news_result.raw_results},
                        detection_method="news_verified"
                    )
                else:
                    if verbose:
                        print(f"   ⚠️ News verification inconclusive: {news_result.verdict}")
                        print("   Proceeding with additional checks...")
                    
            except Exception as e:
                if verbose:
                    print(f"   ❌ News verification error: {e}")
                    print("   Proceeding with fallback methods...")
        else:
            if verbose:
                print("\n⚠️ News Verification Tool not available")
                print("   Using fallback verification methods...")
        
        # ============== STEP 2: GOOGLE FACT CHECK API (FALLBACK) ==============
        if verbose:
            print("\n📋 STEP 2: Google Fact Check API...")
        
        google_verdict = self.search_for_fake_news(content_description[:200])
        
        if verbose:
            print(f"   📊 Google API Result: {google_verdict.verdict_summary}")
        
        if google_verdict.is_fake and google_verdict.confidence >= 0.5:
            if verbose:
                print("\n✅ Google Fact Check found existing fact-check - FAKE confirmed")
            google_verdict.detection_method = "google_factcheck_api"
            return google_verdict
        
        # ============== STEP 3: GEMINI AI ANALYSIS (FALLBACK) ==============
        if verbose:
            print("\n📋 STEP 3: Gemini AI Content Analysis...")
        
        gemini_verdict = self.analyze_with_gemini(content_description, verbose=verbose)
        
        # If Gemini failed, use rule-based
        if gemini_verdict.detection_method == "error" or gemini_verdict.confidence == 0.0:
            if verbose:
                print("\n📋 STEP 3.5: Rule-based detection fallback...")
            gemini_verdict = self.rule_based_fake_detection(content_description, verbose=verbose)
        
        # ============== STEP 4: COMBINE RESULTS ==============
        if verbose:
            print("\n📋 STEP 4: Combining analysis results...")
        
        # Check dangerous scenario patterns
        content_lower = content_description.lower()
        dangerous_found = []
        for indicator in self.DANGEROUS_SCENARIO_INDICATORS:
            if indicator in content_lower:
                dangerous_found.append(indicator)
        
        # Final decision
        if gemini_verdict.is_fake or len(dangerous_found) >= 2:
            final_confidence = max(gemini_verdict.confidence, 0.8 if len(dangerous_found) >= 2 else 0.5)
            final_action = "block" if final_confidence >= 0.8 else "reduce_reach"
            
            verdict_summary = gemini_verdict.verdict_summary
            if dangerous_found:
                verdict_summary = f"⚠️ No news coverage + dangerous content patterns detected: {', '.join(dangerous_found[:3])}"
            
            final_verdict = FactCheckVerdict(
                is_fake=True,
                confidence=final_confidence,
                sources=gemini_verdict.sources + (["Pattern Detection"] if dangerous_found else []),
                verdict_summary=verdict_summary,
                action=final_action,
                raw_results={
                    "google_result": google_verdict.raw_results,
                    "gemini_result": gemini_verdict.raw_results,
                    "dangerous_indicators": dangerous_found
                },
                detection_method="combined_fallback"
            )
        else:
            final_verdict = FactCheckVerdict(
                is_fake=False,
                confidence=gemini_verdict.confidence,
                sources=gemini_verdict.sources,
                verdict_summary=gemini_verdict.verdict_summary,
                action="allow",
                raw_results={
                    "google_result": google_verdict.raw_results,
                    "gemini_result": gemini_verdict.raw_results
                },
                detection_method="combined_fallback"
            )
        
        if verbose:
            print("\n" + "="*70)
            print(f"🎯 FINAL VERDICT: {'❌ FAKE - NOT ALLOWED' if final_verdict.is_fake else '✅ ALLOWED'}")
            print(f"📊 Confidence: {final_verdict.confidence*100:.1f}%")
            print(f"📋 Action: {final_verdict.action.upper()}")
            print(f"💬 Summary: {final_verdict.verdict_summary}")
            print("="*70 + "\n")
        
        return final_verdict


# Utility function for quick verification
def verify_video_content(content_description: str, is_personal: bool = False, verbose: bool = True) -> Dict[str, Any]:
    """
    Quick utility to verify if video content is fake.
    Uses comprehensive check with both Google API and Gemini AI.
    
    Args:
        content_description: Description or transcript of the video
        is_personal: If True, skips verification (personal content)
        verbose: Print steps to terminal
        
    Returns:
        Dictionary with verification result
    """
    if is_personal:
        if verbose:
            print("ℹ️ Personal/self-made content - verification skipped")
        return {
            "verified": True,
            "is_fake": False,
            "action": "allow",
            "reason": "Personal/self-made content - verification skipped"
        }
    
    tool = GoogleFactCheckTool()
    verdict = tool.comprehensive_fake_check(content_description, verbose=verbose)
    
    return {
        "verified": True,
        "is_fake": verdict.is_fake,
        "confidence": verdict.confidence,
        "action": verdict.action,
        "verdict": verdict.verdict_summary,
        "sources": verdict.sources,
        "detection_method": verdict.detection_method,
        "raw_results": verdict.raw_results
    }