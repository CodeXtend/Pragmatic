"""
News Verification Tool - Verify content against real news sources

This tool:
1. Extracts key events/incidents from text/video content using Gemini
2. Searches Google News/Custom Search for the incident
3. Cross-references with news sources to verify if real or fake
4. Returns verdict based on news coverage
"""
import requests
import os
import re
import json
import time
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Try to import Google Generative AI
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    genai = None
    GEMINI_AVAILABLE = False


@dataclass
class NewsSource:
    """A news source that covered the story."""
    title: str
    source_name: str
    url: str
    published_date: str = ""
    snippet: str = ""
    is_credible: bool = True


@dataclass
class ExtractedIncident:
    """Extracted incident/event from content."""
    main_event: str
    location: str = ""
    date_mentioned: str = ""
    people_involved: List[str] = field(default_factory=list)
    key_claims: List[str] = field(default_factory=list)
    search_queries: List[str] = field(default_factory=list)


@dataclass
class NewsVerificationResult:
    """Result of news-based verification."""
    is_verified: bool  # Found in legitimate news
    is_fake: bool  # Determined to be fake
    confidence: float  # 0.0 to 1.0
    verdict: str  # "REAL", "FAKE", "UNVERIFIED", "LIKELY_FAKE", "LIKELY_REAL"
    verdict_summary: str
    news_sources: List[NewsSource]
    extracted_incident: Optional[ExtractedIncident]
    action: str  # "allow", "label", "reduce_reach", "block"
    detection_method: str  # "news_verified", "no_news_found", "contradicted_by_news"
    raw_results: Dict[str, Any] = field(default_factory=dict)


class NewsVerificationTool:
    """
    News-based verification tool that checks if content/incidents
    are covered by legitimate news sources.
    """
    
    # Credible news sources (Indian and International)
    CREDIBLE_SOURCES = [
        # Indian News
        "ndtv", "times of india", "hindustan times", "indian express",
        "the hindu", "india today", "news18", "zee news", "aaj tak",
        "republic", "scroll.in", "the wire", "the print", "firstpost",
        "deccan herald", "telegraph india", "economic times", "mint",
        "business standard", "moneycontrol", "livemint",
        # International
        "bbc", "cnn", "reuters", "ap news", "associated press",
        "al jazeera", "guardian", "new york times", "washington post",
        "afp", "france 24", "dw", "sky news", "abc news", "nbc news",
        # Fact-checkers
        "factcheck", "snopes", "politifact", "alt news", "boomlive",
        "vishvas news", "quint", "india today fact check", "afp fact check"
    ]
    
    # Known fake/satire sources
    UNRELIABLE_SOURCES = [
        "fauxy", "the fauxy", "unreal times", "fakingnews",
        "satire", "parody", "theonion", "babylon bee"
    ]
    
    def __init__(self, google_api_key: Optional[str] = None, 
                 google_cse_id: Optional[str] = None,
                 gemini_api_key: Optional[str] = None):
        """
        Initialize the News Verification Tool.
        
        Args:
            google_api_key: Google Custom Search API key
            google_cse_id: Google Custom Search Engine ID
            gemini_api_key: Gemini API key for content analysis
        """
        load_dotenv()
        
        # Google Custom Search API (for searching news)
        self.google_api_key = google_api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GOOGLE_FACT_CHECK_API_KEY")
        self.google_cse_id = google_cse_id or os.getenv("GOOGLE_CSE_ID", "")
        
        # SerpAPI for Google Search (better than CSE)
        self.serp_api_key = os.getenv("SERP_API_KEY", "")
        
        # Gemini for content analysis
        self.gemini_api_key = gemini_api_key or os.getenv("GEMINI_API_KEY")
        self.gemini_model = None
        self._last_gemini_call = 0  # For rate limiting
        
        if GEMINI_AVAILABLE and self.gemini_api_key:
            try:
                genai.configure(api_key=self.gemini_api_key)
                self.gemini_model = genai.GenerativeModel("gemini-2.5-flash")
                print("✅ News Verification Tool initialized with Gemini AI")
            except Exception as e:
                print(f"⚠️ Gemini initialization failed: {e}")
        else:
            print("⚠️ Gemini not available for news verification")
    
    def _call_gemini_with_retry(self, prompt: str, max_retries: int = 3, verbose: bool = True) -> Optional[str]:
        """Call Gemini API with retry logic for rate limiting."""
        if not self.gemini_model:
            return None
        
        for attempt in range(max_retries):
            # Rate limiting - wait between calls
            elapsed = time.time() - self._last_gemini_call
            if elapsed < 2:  # Minimum 2 seconds between calls
                time.sleep(2 - elapsed)
            
            try:
                self._last_gemini_call = time.time()
                response = self.gemini_model.generate_content(prompt)
                return response.text
            except Exception as e:
                error_str = str(e)
                if "429" in error_str or "quota" in error_str.lower():
                    # Rate limited - extract retry delay
                    wait_time = 15 * (attempt + 1)  # Exponential backoff
                    if verbose:
                        print(f"   ⚠️ Rate limited, waiting {wait_time}s... (attempt {attempt + 1}/{max_retries})")
                    time.sleep(wait_time)
                else:
                    if verbose:
                        print(f"   ❌ Gemini error: {e}")
                    return None
        
        return None
    
    def extract_incident_from_content(self, content: str, verbose: bool = True) -> ExtractedIncident:
        """
        Use Gemini to extract the main incident/event from content.
        
        Args:
            content: Text content or video transcript
            verbose: Print progress
            
        Returns:
            ExtractedIncident with key details
        """
        if verbose:
            print("\n" + "="*60)
            print("📰 STEP 1: Extracting incident from content")
            print("="*60)
        
        if not self.gemini_model:
            # Fallback: Simple extraction
            if verbose:
                print("⚠️ Gemini not available, using simple extraction")
            return self._simple_incident_extraction(content)
        
        prompt = f"""You are an expert at extracting news-worthy incidents from text/video descriptions.

Analyze this content and extract the main incident/event that could be verified through news sources:

CONTENT:
{content}

Extract and return as JSON:
{{
    "main_event": "One-line description of the main incident (what happened)",
    "location": "Where did it happen (city, state, country if mentioned)",
    "date_mentioned": "Any date or time reference",
    "people_involved": ["List of people/groups involved"],
    "key_claims": ["List of specific claims made that can be fact-checked"],
    "search_queries": [
        "Query 1 for Google News search",
        "Query 2 - more specific",
        "Query 3 - alternate keywords"
    ],
    "is_news_worthy": true/false,
    "incident_type": "accident/political/crime/health/entertainment/sports/other"
}}

RULES:
1. Focus on the main newsworthy event, not descriptions
2. Search queries should be what someone would type in Google to find news about this
3. Include location in search queries if available
4. If content describes something that would make news, is_news_worthy = true
5. For video descriptions, extract what HAPPENED, not what is being shown

Example:
- Input: "Person skateboarding in front of bus on busy Mumbai road"
- main_event: "Person skateboarding in front of bus Mumbai"
- search_queries: ["skateboard stunt bus Mumbai", "viral skateboard video traffic India", "dangerous stunt Mumbai road"]
"""

        try:
            if verbose:
                print("🔄 Analyzing content with Gemini...")
            
            response = self.gemini_model.generate_content(prompt)
            response_text = response.text
            
            # Parse JSON
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group(0))
            else:
                return self._simple_incident_extraction(content)
            
            incident = ExtractedIncident(
                main_event=result.get("main_event", ""),
                location=result.get("location", ""),
                date_mentioned=result.get("date_mentioned", ""),
                people_involved=result.get("people_involved", []),
                key_claims=result.get("key_claims", []),
                search_queries=result.get("search_queries", [])
            )
            
            if verbose:
                print(f"✅ Main Event: {incident.main_event}")
                print(f"📍 Location: {incident.location or 'Not specified'}")
                print(f"🔍 Search Queries: {incident.search_queries}")
            
            return incident
            
        except Exception as e:
            if verbose:
                print(f"❌ Extraction error: {e}")
            return self._simple_incident_extraction(content)
    
    def _simple_incident_extraction(self, content: str) -> ExtractedIncident:
        """Simple keyword-based incident extraction as fallback."""
        # Extract key phrases
        content_lower = content.lower()
        
        # Common news patterns
        key_terms = []
        
        # Location patterns
        location = ""
        location_keywords = ["mumbai", "delhi", "bangalore", "chennai", "kolkata", 
                           "hyderabad", "pune", "india", "road", "highway"]
        for loc in location_keywords:
            if loc in content_lower:
                location = loc.title()
                break
        
        # Event patterns
        event_keywords = ["accident", "crash", "stunt", "viral", "video", 
                        "incident", "breaking", "news", "alert"]
        for event in event_keywords:
            if event in content_lower:
                key_terms.append(event)
        
        # Build search queries
        words = content.split()[:20]  # First 20 words
        main_event = " ".join(words[:10])
        
        search_queries = [
            " ".join(key_terms[:3]) + " " + location if key_terms else main_event,
            main_event[:50],
            content[:100]
        ]
        
        return ExtractedIncident(
            main_event=main_event,
            location=location,
            search_queries=search_queries
        )
    
    def search_google_news(self, query: str, verbose: bool = True) -> List[NewsSource]:
        """
        Search Google for news articles about the query.
        Tries multiple methods:
        1. SerpAPI (best for real Google News results)
        2. Google Custom Search API
        3. Gemini AI knowledge base
        
        Args:
            query: Search query
            verbose: Print progress
            
        Returns:
            List of NewsSource objects
        """
        if verbose:
            print(f"\n🔍 Searching news for: {query[:60]}...")
        
        news_sources = []
        
        # Method 1: SerpAPI (Google News)
        if self.serp_api_key:
            try:
                news_sources = self._search_with_serpapi(query, verbose)
                if news_sources:
                    return news_sources
            except Exception as e:
                if verbose:
                    print(f"   ⚠️ SerpAPI error: {e}")
        
        # Method 2: Google Custom Search API
        if self.google_api_key and self.google_cse_id:
            try:
                news_sources = self._search_with_google_cse(query, verbose)
                if news_sources:
                    return news_sources
            except Exception as e:
                if verbose:
                    print(f"   ⚠️ Google CSE error: {e}")
        
        # Method 3: Use Gemini to search and verify
        if self.gemini_model:
            news_sources = self._search_with_gemini(query, verbose)
        
        return news_sources
    
    def _search_with_serpapi(self, query: str, verbose: bool = True) -> List[NewsSource]:
        """Search using SerpAPI for Google News results."""
        url = "https://serpapi.com/search"
        
        params = {
            "api_key": self.serp_api_key,
            "engine": "google_news",
            "q": query,
            "gl": "in",  # India
            "hl": "en"
        }
        
        try:
            response = requests.get(url, params=params, timeout=15)
            data = response.json()
            
            news_sources = []
            
            # Parse news results
            news_results = data.get("news_results", [])
            for item in news_results:
                source_info = item.get("source", {})
                source_name = source_info.get("name", "").lower()
                
                is_credible = any(cred in source_name for cred in self.CREDIBLE_SOURCES)
                is_unreliable = any(fake in source_name for fake in self.UNRELIABLE_SOURCES)
                
                if is_unreliable:
                    is_credible = False
                
                news_sources.append(NewsSource(
                    title=item.get("title", ""),
                    source_name=source_info.get("name", ""),
                    url=item.get("link", ""),
                    published_date=item.get("date", ""),
                    snippet=item.get("snippet", ""),
                    is_credible=is_credible
                ))
            
            if verbose and news_sources:
                print(f"   ✅ Found {len(news_sources)} results from Google News (SerpAPI)")
            
            return news_sources
            
        except Exception as e:
            if verbose:
                print(f"   ❌ SerpAPI error: {e}")
            return []
        
        # Method 2: Use Gemini to search and verify (if CSE not available)
        if self.gemini_model:
            news_sources = self._search_with_gemini(query, verbose)
        
        return news_sources
    
    def _search_with_google_cse(self, query: str, verbose: bool = True) -> List[NewsSource]:
        """Search using Google Custom Search Engine API."""
        url = "https://www.googleapis.com/customsearch/v1"
        
        # Add "news" to query to prioritize news results
        search_query = f"{query} news"
        
        params = {
            "key": self.google_api_key,
            "cx": self.google_cse_id,
            "q": search_query,
            "num": 10,
            "dateRestrict": "m1",  # Last month
            "sort": "date"
        }
        
        try:
            response = requests.get(url, params=params, timeout=10)
            data = response.json()
            
            news_sources = []
            items = data.get("items", [])
            
            for item in items:
                source_name = item.get("displayLink", "").lower()
                is_credible = any(cred in source_name for cred in self.CREDIBLE_SOURCES)
                is_unreliable = any(fake in source_name for fake in self.UNRELIABLE_SOURCES)
                
                if is_unreliable:
                    is_credible = False
                
                news_sources.append(NewsSource(
                    title=item.get("title", ""),
                    source_name=item.get("displayLink", ""),
                    url=item.get("link", ""),
                    snippet=item.get("snippet", ""),
                    is_credible=is_credible
                ))
            
            if verbose:
                print(f"   ✅ Found {len(news_sources)} results from Google CSE")
            
            return news_sources
            
        except Exception as e:
            if verbose:
                print(f"   ❌ Google CSE error: {e}")
            return []
    
    def _search_with_gemini(self, query: str, verbose: bool = True) -> List[NewsSource]:
        """Use Gemini to verify if the incident was covered in news."""
        if not self.gemini_model:
            return []
        
        prompt = f"""You are a news verification assistant. I need to verify if this incident/event was covered in legitimate news sources.

QUERY/INCIDENT:
{query}

Please search your knowledge and tell me:
1. Is this a REAL incident that was covered by news media?
2. If yes, which news sources covered it?
3. What was the actual story according to news reports?
4. Is this a known FAKE/viral hoax that was debunked?

Return as JSON:
{{
    "found_in_news": true/false,
    "is_real_incident": true/false,
    "is_known_fake": true/false,
    "news_sources": [
        {{"source": "News Source Name", "headline": "What they reported", "url": "URL if known"}}
    ],
    "actual_story": "What actually happened according to news",
    "debunked_by": ["List of fact-checkers if fake"],
    "explanation": "Detailed explanation"
}}

Be accurate. If you don't have information about this specific incident, say so.
Only report news sources you are confident about."""

        try:
            response = self.gemini_model.generate_content(prompt)
            response_text = response.text
            
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if not json_match:
                return []
            
            result = json.loads(json_match.group(0))
            
            news_sources = []
            for source in result.get("news_sources", []):
                news_sources.append(NewsSource(
                    title=source.get("headline", ""),
                    source_name=source.get("source", ""),
                    url=source.get("url", ""),
                    snippet=result.get("actual_story", "")[:200],
                    is_credible=True
                ))
            
            # Store Gemini's analysis for later use
            self._last_gemini_analysis = result
            
            if verbose:
                found = result.get("found_in_news", False)
                is_fake = result.get("is_known_fake", False)
                print(f"   📰 Gemini found in news: {found}")
                print(f"   🚨 Known fake/hoax: {is_fake}")
            
            return news_sources
            
        except Exception as e:
            if verbose:
                print(f"   ❌ Gemini search error: {e}")
            return []
    
    def verify_against_news(self, content: str, verbose: bool = True) -> NewsVerificationResult:
        """
        Main verification method - checks if content matches real news.
        
        Args:
            content: Text content or video transcript to verify
            verbose: Print progress
            
        Returns:
            NewsVerificationResult with verdict
        """
        if verbose:
            print("\n" + "="*70)
            print("📰 NEWS-BASED VERIFICATION")
            print("="*70)
            print(f"📝 Content to verify: {content[:150]}...")
            print("="*70)
        
        # Step 1: Extract incident from content
        incident = self.extract_incident_from_content(content, verbose)
        
        if not incident.main_event:
            if verbose:
                print("\n⚠️ Could not extract clear incident from content")
            return NewsVerificationResult(
                is_verified=False,
                is_fake=False,
                confidence=0.3,
                verdict="UNVERIFIED",
                verdict_summary="Could not extract clear incident to verify",
                news_sources=[],
                extracted_incident=incident,
                action="label",
                detection_method="extraction_failed"
            )
        
        # Step 2: Search for news about the incident
        if verbose:
            print("\n" + "="*60)
            print("📰 STEP 2: Searching news sources")
            print("="*60)
        
        all_news_sources = []
        for query in incident.search_queries[:3]:  # Try top 3 queries
            sources = self.search_google_news(query, verbose)
            all_news_sources.extend(sources)
        
        # Remove duplicates
        seen_urls = set()
        unique_sources = []
        for source in all_news_sources:
            if source.url not in seen_urls:
                seen_urls.add(source.url)
                unique_sources.append(source)
        
        # Step 3: Analyze findings
        if verbose:
            print("\n" + "="*60)
            print("📰 STEP 3: Analyzing news coverage")
            print("="*60)
        
        return self._analyze_news_results(content, incident, unique_sources, verbose)
    
    def _analyze_news_results(self, content: str, incident: ExtractedIncident, 
                             news_sources: List[NewsSource], 
                             verbose: bool = True) -> NewsVerificationResult:
        """Analyze news results and determine verdict."""
        
        credible_sources = [s for s in news_sources if s.is_credible]
        total_sources = len(news_sources)
        credible_count = len(credible_sources)
        
        if verbose:
            print(f"📊 Total sources found: {total_sources}")
            print(f"✅ Credible sources: {credible_count}")
        
        # Check if Gemini found it as known fake
        gemini_analysis = getattr(self, '_last_gemini_analysis', {})
        is_known_fake = gemini_analysis.get("is_known_fake", False)
        found_in_news = gemini_analysis.get("found_in_news", False)
        debunked_by = gemini_analysis.get("debunked_by", [])
        
        if verbose and is_known_fake:
            print(f"🚨 KNOWN FAKE detected by Gemini!")
            print(f"📋 Debunked by: {debunked_by}")
        
        # Decision logic
        if is_known_fake and debunked_by:
            # Content is a known fake/hoax
            verdict = "FAKE"
            is_fake = True
            confidence = 0.95
            action = "block"
            verdict_summary = f"❌ FAKE: This incident was debunked by fact-checkers: {', '.join(debunked_by[:3])}"
            detection_method = "debunked_by_factcheckers"
            
        elif credible_count >= 3:
            # Multiple credible sources confirm it's real news
            verdict = "REAL"
            is_fake = False
            confidence = 0.9
            action = "allow"
            sources_names = [s.source_name for s in credible_sources[:3]]
            verdict_summary = f"✅ VERIFIED: Confirmed by news sources: {', '.join(sources_names)}"
            detection_method = "news_verified"
            
        elif credible_count >= 1 and found_in_news:
            # At least one credible source + Gemini confirms
            verdict = "LIKELY_REAL"
            is_fake = False
            confidence = 0.75
            action = "allow"
            verdict_summary = f"✅ LIKELY REAL: Found in news source: {credible_sources[0].source_name}"
            detection_method = "news_verified"
            
        elif total_sources == 0 and not found_in_news:
            # No news coverage at all - suspicious for viral content
            # Use Gemini to analyze if it's staged/fake
            if verbose:
                print("\n⚠️ No news coverage found - checking if staged/fake content...")
            
            fake_check_result = self._check_if_staged_content(content, incident, verbose)
            
            if fake_check_result["is_likely_fake"]:
                verdict = "LIKELY_FAKE"
                is_fake = True
                confidence = fake_check_result["confidence"]
                action = "reduce_reach" if confidence < 0.8 else "block"
                verdict_summary = f"⚠️ SUSPICIOUS: No news coverage found. {fake_check_result['reason']}"
                detection_method = "no_news_found_likely_staged"
            else:
                verdict = "UNVERIFIED"
                is_fake = False
                confidence = 0.4
                action = "label"
                verdict_summary = "ℹ️ UNVERIFIED: No news coverage found. Cannot confirm or deny."
                detection_method = "no_news_found"
        
        else:
            # Some sources but not credible ones
            verdict = "UNVERIFIED"
            is_fake = False
            confidence = 0.5
            action = "label"
            verdict_summary = "ℹ️ UNVERIFIED: Found in sources but not major news outlets."
            detection_method = "insufficient_coverage"
        
        if verbose:
            print("\n" + "="*70)
            print(f"🎯 VERDICT: {verdict}")
            print(f"📊 Confidence: {confidence*100:.1f}%")
            print(f"📋 Action: {action.upper()}")
            print(f"💬 {verdict_summary}")
            print("="*70 + "\n")
        
        return NewsVerificationResult(
            is_verified=found_in_news or credible_count > 0,
            is_fake=is_fake,
            confidence=confidence,
            verdict=verdict,
            verdict_summary=verdict_summary,
            news_sources=news_sources,
            extracted_incident=incident,
            action=action,
            detection_method=detection_method,
            raw_results={
                "gemini_analysis": gemini_analysis,
                "credible_count": credible_count,
                "total_sources": total_sources
            }
        )
    
    def _check_if_staged_content(self, content: str, incident: ExtractedIncident, 
                                  verbose: bool = True) -> Dict[str, Any]:
        """Check if content appears to be staged/fake when no news coverage found."""
        
        if not self.gemini_model:
            return {"is_likely_fake": False, "confidence": 0.5, "reason": "Cannot analyze"}
        
        prompt = f"""Analyze this content that has NO NEWS COVERAGE. Determine if it's likely STAGED/FAKE content.

CONTENT:
{content}

EXTRACTED INCIDENT:
{incident.main_event}
Location: {incident.location}

ANALYSIS CRITERIA:
1. If this incident really happened, would news media cover it?
2. Does it describe something dangerous/sensational that would definitely make news?
3. Does it appear to be a viral stunt designed for views?
4. Is it physically impossible or extremely unlikely?
5. Would there be official records (police, hospital) if real?

IMPORTANT: Viral videos of dangerous stunts in traffic that have NO news coverage are usually STAGED/FAKE because:
- Real dangerous incidents get reported to police
- Real accidents make local news
- People don't casually film life-threatening situations

Return JSON:
{{
    "is_likely_fake": true/false,
    "confidence": 0.0-1.0,
    "reason": "Why you think it's fake/real",
    "red_flags": ["List of red flags"],
    "would_make_news_if_real": true/false
}}"""

        try:
            response = self.gemini_model.generate_content(prompt)
            response_text = response.text
            
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group(0))
                return {
                    "is_likely_fake": result.get("is_likely_fake", False),
                    "confidence": result.get("confidence", 0.5),
                    "reason": result.get("reason", ""),
                    "red_flags": result.get("red_flags", [])
                }
        except Exception as e:
            if verbose:
                print(f"❌ Analysis error: {e}")
        
        return {"is_likely_fake": False, "confidence": 0.5, "reason": "Analysis failed"}
    
    def comprehensive_verify(self, content: str, verbose: bool = True) -> NewsVerificationResult:
        """
        Comprehensive verification using news search.
        This is the main method to call.
        
        Args:
            content: Content to verify
            verbose: Print progress
            
        Returns:
            NewsVerificationResult
        """
        return self.verify_against_news(content, verbose)


# Utility function for quick verification
def verify_content_with_news(content: str, verbose: bool = True) -> Dict[str, Any]:
    """
    Quick utility to verify content against news sources.
    
    Args:
        content: Text content or video transcript
        verbose: Print progress
        
    Returns:
        Dictionary with verification result
    """
    tool = NewsVerificationTool()
    result = tool.comprehensive_verify(content, verbose)
    
    return {
        "verified": result.is_verified,
        "is_fake": result.is_fake,
        "confidence": result.confidence,
        "verdict": result.verdict,
        "verdict_summary": result.verdict_summary,
        "action": result.action,
        "detection_method": result.detection_method,
        "news_sources": [
            {
                "title": s.title,
                "source": s.source_name,
                "url": s.url,
                "is_credible": s.is_credible
            } for s in result.news_sources
        ],
        "incident_extracted": {
            "main_event": result.extracted_incident.main_event if result.extracted_incident else "",
            "location": result.extracted_incident.location if result.extracted_incident else "",
            "search_queries": result.extracted_incident.search_queries if result.extracted_incident else []
        } if result.extracted_incident else None
    }


if __name__ == "__main__":
    # Test with the skateboard video content
    test_content = """
    This frame captures a bustling urban street scene during daytime, with a man walking across the road in the foreground while various vehicles populate the lanes. A yellow auto-rickshaw, a white and red bus, and a colorful yellow-orange truck are prominent, alongside cars and a motorcycle, all framed by an elevated concrete structure on the left and lush green trees and buildings in the background. The bright sunlight illuminates the active flow of traffic and pedestrians, creating a dynamic city atmosphere. A person is skateboarding on their stomach down the center of a busy urban road, directly in front of a large white and red bus, while various other vehicles like an SUV, a colorful truck, and a motorcycle navigate the surrounding lanes. The scene is set on a multi-lane street lined with trees and an overhead concrete flyover, conveying a chaotic yet daring atmosphere amidst regular city traffic.
    """
    
    print("="*70)
    print("🧪 NEWS VERIFICATION TOOL TEST")
    print("="*70)
    
    result = verify_content_with_news(test_content)
    
    print("\n📊 FINAL RESULT:")
    print(f"Verdict: {result['verdict']}")
    print(f"Is Fake: {result['is_fake']}")
    print(f"Confidence: {result['confidence']*100:.1f}%")
    print(f"Summary: {result['verdict_summary']}")
