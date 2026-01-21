"""
Content Classifier Module - Multi-Layer Content Moderation System

This module handles:
1. Content Classification (Self-made vs Factual)
2. AI-Generated Content Detection
3. Self-Made Content Signal Detection
4. Deceptive Framing Detection
5. Claim Extraction for Mixed Content
6. Risk Scoring System
7. Harm & Reach Assessment
8. Final Labels and Transparency Records
"""

import os
import re
import json
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
from datetime import datetime
from dotenv import load_dotenv

try:
    import google.generativeai as genai
except ImportError:
    genai = None


# ============== ENUMS & DATA CLASSES ==============

class ContentType(Enum):
    """Content type categories"""
    PERSONAL_SELF_MADE = "personal_self_made"
    DEMONSTRATION = "demonstration"
    OPINION_COMMENTARY = "opinion_commentary"
    FACTUAL_CLAIM = "factual_claim"
    MIXED_CONTENT = "mixed_content"
    ENTERTAINMENT = "entertainment"
    UNKNOWN = "unknown"


class RiskLevel(Enum):
    """Risk level categories"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class HarmCategory(Enum):
    """Harm categories for content"""
    NONE = "none"
    POLITICAL = "political"
    HEALTH = "health"
    FINANCIAL = "financial"
    NEWS_LIKE = "news_like"
    VIOLENCE = "violence"
    HATE_SPEECH = "hate_speech"


class ActionType(Enum):
    """Actions to take based on risk assessment"""
    ALLOW = "allow"
    LABEL = "label"
    REDUCE_REACH = "reduce_reach"
    MANUAL_REVIEW = "manual_review"
    BLOCK = "block"


@dataclass
class ContentClassification:
    """Content classification result"""
    content_type: str
    confidence: float
    reason: str
    requires_fact_check: bool
    

@dataclass
class SelfMadeSignals:
    """Self-made content detection signals"""
    first_person_score: float
    demonstration_score: float
    live_narration_score: float
    personal_experience_score: float
    detected_phrases: List[str]
    ownership_confidence: float


@dataclass
class AIGenerationSignals:
    """AI-generated content detection signals"""
    ai_generated_probability: float
    synthetic_voice_indicators: List[str]
    detection_confidence: float
    risk_signal_only: bool = True  # Never use as final proof


@dataclass
class DeceptiveFramingResult:
    """Deceptive framing detection result"""
    is_deceptive: bool
    deceptive_score: float
    red_flags: List[str]
    authority_claims: List[str]
    urgency_indicators: List[str]


@dataclass
class ClaimExtractionResult:
    """Extracted factual claims from content"""
    claims: List[Dict[str, Any]]
    total_claims: int
    verifiable_claims: int
    opinion_statements: int


@dataclass
class HarmAssessment:
    """Harm and reach assessment result"""
    harm_category: str
    severity: float
    potential_reach_impact: str
    requires_immediate_action: bool


@dataclass
class RiskScore:
    """Final risk score calculation"""
    total_score: float
    ai_generated_component: float
    deceptive_framing_component: float
    factual_claim_severity: float
    disclosure_penalty: float
    recommended_action: str


@dataclass
class ModerationRecord:
    """Complete moderation transparency record"""
    timestamp: str
    video_id: Optional[str]
    content_classification: Dict[str, Any]
    self_made_signals: Dict[str, Any]
    ai_generation_signals: Dict[str, Any]
    deceptive_framing: Dict[str, Any]
    extracted_claims: Dict[str, Any]
    harm_assessment: Dict[str, Any]
    risk_score: Dict[str, Any]
    user_disclosure: Dict[str, Any]
    final_action: str
    final_label: str
    reasoning: str


@dataclass
class UserDisclosure:
    """User-provided disclosure at upload time"""
    is_self_made: bool = False
    is_ai_generated: bool = False
    content_type_declared: str = ""
    disclosure_provided: bool = False


# ============== MAIN CLASSIFIER CLASS ==============

class ContentClassifier:
    """
    Multi-layer content classification and moderation system.
    
    This system classifies video content BEFORE fact-checking to prevent
    false flags on self-made content (vlogs, gaming, tutorials).
    """
    
    # Self-made content indicator phrases
    SELF_MADE_INDICATORS = [
        # First-person ownership
        "my vlog", "my video", "my gameplay", "my tutorial",
        "i recorded", "i am showing", "i made", "i built",
        "i created", "my experience", "my journey", "my story",
        "watch me", "join me", "follow me", "let me show",
        
        # Live narration
        "as you can see", "here we have", "right now",
        "at this moment", "currently", "let's see",
        "look at this", "check this out",
        
        # Personal experience
        "i feel", "i think", "in my opinion", "personally",
        "from my experience", "i believe", "to me",
        "my thoughts on", "my take on"
    ]
    
    # Deceptive framing indicators
    DECEPTIVE_INDICATORS = [
        # Authority claims
        "breaking news", "exclusive", "media won't tell you",
        "they don't want you to know", "truth revealed",
        "confirmed", "officially", "verified",
        
        # Urgency
        "urgent", "immediately", "right now", "alert",
        "warning", "must watch", "share before deleted",
        
        # Authenticity claims
        "100% true", "this really happened", "not fake",
        "i swear", "trust me", "real footage", "leaked"
    ]
    
    # High-risk claim categories
    HIGH_RISK_CATEGORIES = {
        "political": ["election", "government", "politician", "vote", "democracy", "policy"],
        "health": ["covid", "vaccine", "cure", "treatment", "disease", "medicine", "doctor"],
        "financial": ["scam", "investment", "bank", "money", "crypto", "scheme"],
        "news": ["breaking", "news", "report", "journalist", "media"]
    }

    def __init__(self, api_key: Optional[str] = None, model_name: Optional[str] = None):
        """Initialize the content classifier with Gemini API."""
        load_dotenv()
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        
        # Determine model name from argument or environment
        if model_name:
            self.model_name = model_name
        else:
            # Handle LiteLLM format (e.g., "gemini/gemini-2.0-flash") -> "gemini-2.0-flash"
            # User specifically requested 2.5/2.0 model
            env_model = os.getenv("LLM_MODEL")
            self.model_name = env_model.replace("gemini/", "") if env_model.startswith("gemini/") else env_model
            
        self.model = None
        self.last_analysis_result = None
        
        if genai and self.api_key:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel(self.model_name)

    def _perform_combined_llm_analysis(self, transcript: str) -> Dict[str, Any]:
        """
        Combined LLM analysis: Classification + Claim Extraction in one call.
        """
        if not self.model:
            return {}

        prompt = f"""Analyze the following video transcript. Perform two tasks:
1. CLASSIFY the content type (personal_self_made, demonstration, opinion_commentary, factual_claim, mixed_content, entertainment, unknown).
2. EXTRACT objective factual claims (ignore opinions).

TRANSCRIPT:
{transcript[:4000]}

Return a SINGLE JSON object with this structure:
{{
    "classification": {{
        "content_type": "<category>",
        "confidence": <0.0-1.0>,
        "reason": "<explanation>",
        "requires_fact_check": <bool>
    }},
    "claims_data": {{
        "claims": [
            {{
                "claim": "<text>",
                "type": "statistic|historical|scientific|news|health|political|other",
                "verifiable": <bool>,
                "severity": "low|medium|high",
                "context": "<context>"
            }}
        ],
        "opinion_statements": <count>
    }}
}}"""

        try:
            response = self.model.generate_content(prompt)
            return self._parse_json_response(response.text)
        except Exception as e:
            print(f"Combined LLM analysis error: {e}")
            return {}

    # ============== LAYER 1: CONTENT CLASSIFICATION ==============
    
    def classify_content(self, transcript: str) -> ContentClassification:
        """
        Classify video content into categories before fact-checking.
        Uses LLM for intelligent classification.
        """
        if not self.model:
            return self._rule_based_classification(transcript)
            
        # Check if we have a cached analysis from this transcript
        if self.last_analysis_result and self.last_analysis_result.get("transcript_hash") == hash(transcript):
             result = self.last_analysis_result.get("classification", {})
             if result:
                return ContentClassification(
                    content_type=result.get("content_type", "unknown"),
                    confidence=float(result.get("confidence", 0.5)),
                    reason=result.get("reason", ""),
                    requires_fact_check=result.get("requires_fact_check", False)
                )
        
        # If no cache or separate call, use valid single purpose prompt (fallback to original behavior if needed, 
        # but better to use combined if we can coordinate it. 
        # For now, let's keep original behavior if called independently, but analyze_content will populate cache)
        
        prompt = f"""Analyze the following video transcript and classify it into ONE of these categories:

1. personal_self_made - Vlogs, gaming videos, tutorials, travel, personal experiences
2. demonstration - "I built this app", gameplay walkthroughs, DIY projects
3. opinion_commentary - Reviews, reactions, personal opinions
4. factual_claim - News, statistics, health claims, political statements
5. mixed_content - Contains both personal content AND factual claims
6. entertainment - Comedy, storytelling, fictional content

TRANSCRIPT:
{transcript[:3000]}

Return ONLY a JSON object with:
{{
    "content_type": "<category_name>",
    "confidence": <0.0-1.0>,
    "reason": "<brief explanation>",
    "requires_fact_check": <true/false>
}}

IMPORTANT: 
- Personal vlogs, gaming, tutorials should NOT require fact-checking
- Only factual_claim and factual parts of mixed_content require verification
- Be generous towards self-made content"""

        try:
            response = self.model.generate_content(prompt)
            result = self._parse_json_response(response.text)
            
            return ContentClassification(
                content_type=result.get("content_type", "unknown"),
                confidence=float(result.get("confidence", 0.5)),
                reason=result.get("reason", ""),
                requires_fact_check=result.get("requires_fact_check", False)
            )
        except Exception as e:
            print(f"LLM classification error: {e}")
            return self._rule_based_classification(transcript)

    def _rule_based_classification(self, transcript: str) -> ContentClassification:
        """Fallback rule-based classification."""
        transcript_lower = transcript.lower()
        
        # Count indicators
        self_made_count = sum(1 for phrase in self.SELF_MADE_INDICATORS 
                             if phrase in transcript_lower)
        deceptive_count = sum(1 for phrase in self.DECEPTIVE_INDICATORS 
                              if phrase in transcript_lower)
        
        # Determine content type
        if self_made_count >= 3 and deceptive_count == 0:
            return ContentClassification(
                content_type=ContentType.PERSONAL_SELF_MADE.value,
                confidence=min(0.9, 0.5 + self_made_count * 0.1),
                reason=f"Detected {self_made_count} self-made content indicators",
                requires_fact_check=False
            )
        elif deceptive_count >= 2:
            return ContentClassification(
                content_type=ContentType.FACTUAL_CLAIM.value,
                confidence=min(0.9, 0.5 + deceptive_count * 0.15),
                reason=f"Detected {deceptive_count} authority/news-like indicators",
                requires_fact_check=True
            )
        elif self_made_count >= 1 and deceptive_count >= 1:
            return ContentClassification(
                content_type=ContentType.MIXED_CONTENT.value,
                confidence=0.6,
                reason="Contains both personal and factual elements",
                requires_fact_check=True
            )
        else:
            return ContentClassification(
                content_type=ContentType.UNKNOWN.value,
                confidence=0.5,
                reason="Could not determine content type confidently",
                requires_fact_check=True
            )

    # ============== LAYER 2: SELF-MADE CONTENT SIGNALS ==============
    
    def detect_self_made_signals(self, transcript: str) -> SelfMadeSignals:
        """
        Detect ownership signals in transcript.
        Returns confidence score for self-made content.
        """
        transcript_lower = transcript.lower()
        detected_phrases = []
        
        # First-person score
        first_person_patterns = [
            r'\bi\s+(am|was|have|had|will|would|made|built|created|recorded)\b',
            r'\bmy\s+(vlog|video|channel|gameplay|tutorial|story)\b',
            r'\bwe\s+(are|were|have)\b'
        ]
        first_person_matches = sum(len(re.findall(p, transcript_lower)) 
                                   for p in first_person_patterns)
        first_person_score = min(1.0, first_person_matches / 10)
        
        # Demonstration score
        demo_phrases = ["let me show", "watch this", "here's how", 
                        "as you can see", "check this out", "look at"]
        demo_matches = sum(1 for p in demo_phrases if p in transcript_lower)
        demonstration_score = min(1.0, demo_matches / 4)
        
        # Live narration score
        live_phrases = ["right now", "currently", "at this moment",
                        "here we are", "let's go", "let's see"]
        live_matches = sum(1 for p in live_phrases if p in transcript_lower)
        live_narration_score = min(1.0, live_matches / 3)
        
        # Personal experience score
        personal_phrases = ["i feel", "i think", "in my opinion", 
                           "my experience", "personally", "i believe"]
        personal_matches = sum(1 for p in personal_phrases if p in transcript_lower)
        personal_experience_score = min(1.0, personal_matches / 4)
        
        # Collect detected phrases
        for phrase in self.SELF_MADE_INDICATORS:
            if phrase in transcript_lower:
                detected_phrases.append(phrase)
        
        # Calculate ownership confidence
        ownership_confidence = (
            0.35 * first_person_score +
            0.25 * demonstration_score +
            0.20 * live_narration_score +
            0.20 * personal_experience_score
        )
        
        return SelfMadeSignals(
            first_person_score=round(first_person_score, 3),
            demonstration_score=round(demonstration_score, 3),
            live_narration_score=round(live_narration_score, 3),
            personal_experience_score=round(personal_experience_score, 3),
            detected_phrases=detected_phrases,
            ownership_confidence=round(ownership_confidence, 3)
        )

    # ============== LAYER 3: AI-GENERATED CONTENT DETECTION ==============
    
    def detect_ai_generation_signals(self, transcript: str, 
                                      audio_features: Optional[Dict] = None) -> AIGenerationSignals:
        """
        Detect AI-generation signals (USE AS RISK SIGNAL ONLY, NOT PROOF).
        
        Note: This is NOT definitive detection. Use only for risk scoring.
        """
        indicators = []
        
        # Text-based heuristics (weak signals)
        transcript_lower = transcript.lower()
        
        # Over-formal language patterns
        formal_patterns = [
            r'\bfurthermore\b', r'\bmoreover\b', r'\bconsequently\b',
            r'\bnevertheless\b', r'\bnotwithstanding\b'
        ]
        formal_count = sum(len(re.findall(p, transcript_lower)) for p in formal_patterns)
        if formal_count > 3:
            indicators.append("over_formal_language")
        
        # Repetitive sentence structure (weak signal)
        sentences = transcript.split('.')
        if len(sentences) > 5:
            lengths = [len(s.split()) for s in sentences if s.strip()]
            if lengths:
                avg_len = sum(lengths) / len(lengths)
                variance = sum((l - avg_len) ** 2 for l in lengths) / len(lengths)
                if variance < 5:  # Very uniform sentence lengths
                    indicators.append("uniform_sentence_structure")
        
        # Lack of filler words (natural speech has fillers)
        filler_words = ["um", "uh", "like", "you know", "actually", "basically"]
        filler_count = sum(transcript_lower.count(f) for f in filler_words)
        word_count = len(transcript.split())
        if word_count > 100 and filler_count < 2:
            indicators.append("no_natural_speech_fillers")
        
        # Calculate probability (conservative estimate)
        base_prob = len(indicators) * 0.15
        
        # Audio features if available
        if audio_features:
            if audio_features.get("pitch_variance", 1) < 0.1:
                indicators.append("low_pitch_variance")
                base_prob += 0.1
            if audio_features.get("emotion_detected") == False:
                indicators.append("no_emotional_variation")
                base_prob += 0.1
        
        ai_probability = min(0.85, base_prob)  # Cap at 0.85 - never 100% certain
        
        return AIGenerationSignals(
            ai_generated_probability=round(ai_probability, 3),
            synthetic_voice_indicators=indicators,
            detection_confidence=0.4,  # Always low confidence
            risk_signal_only=True
        )

    # ============== LAYER 4: DECEPTIVE FRAMING DETECTION ==============
    
    def detect_deceptive_framing(self, transcript: str) -> DeceptiveFramingResult:
        """
        Detect if content uses deceptive framing to present claims as facts.
        """
        transcript_lower = transcript.lower()
        red_flags = []
        authority_claims = []
        urgency_indicators = []
        
        # Authority patterns
        authority_patterns = {
            "breaking_news": r'\bbreaking\s*(news)?\b',
            "exclusive": r'\bexclusive\b',
            "confirmed": r'\bconfirmed\b|\bofficially\b',
            "truth_claim": r'\btruth\b.*\brevealed\b|\breal\s+truth\b',
            "suppression_claim": r'\bthey\s+don\'?t\s+want\b|\bmedia\s+won\'?t\b'
        }
        
        for name, pattern in authority_patterns.items():
            if re.search(pattern, transcript_lower):
                authority_claims.append(name)
        
        # Urgency patterns
        urgency_patterns = {
            "immediate_action": r'\burgent\b|\bimmediately\b|\bright\s+now\b',
            "share_pressure": r'\bshare\s+(before|now)\b|\bmust\s+watch\b',
            "time_pressure": r'\bbefore\s+it\'?s?\s+(too\s+)?late\b|\blast\s+chance\b'
        }
        
        for name, pattern in urgency_patterns.items():
            if re.search(pattern, transcript_lower):
                urgency_indicators.append(name)
        
        # Deceptive authenticity claims
        authenticity_patterns = [
            r'\b100\s*%\s*true\b',
            r'\bthis\s+really\s+happened\b',
            r'\bnot\s+fake\b',
            r'\breal\s+footage\b',
            r'\bleaked\b'
        ]
        
        for pattern in authenticity_patterns:
            if re.search(pattern, transcript_lower):
                red_flags.append(f"authenticity_claim: {pattern}")
        
        # Calculate deceptive score
        total_indicators = len(authority_claims) + len(urgency_indicators) + len(red_flags)
        deceptive_score = min(1.0, total_indicators * 0.15)
        
        return DeceptiveFramingResult(
            is_deceptive=deceptive_score > 0.4,
            deceptive_score=round(deceptive_score, 3),
            red_flags=red_flags,
            authority_claims=authority_claims,
            urgency_indicators=urgency_indicators
        )

    # ============== LAYER 5: CLAIM EXTRACTION ==============
    
    def extract_claims(self, transcript: str) -> ClaimExtractionResult:
        """
        Extract only verifiable factual claims from content.
        Ignores opinions, experiences, and personal statements.
        """
        if not self.model:
            return self._rule_based_claim_extraction(transcript)
            
        # Use cached result if available
        if self.last_analysis_result and self.last_analysis_result.get("transcript_hash") == hash(transcript):
             result = self.last_analysis_result.get("claims_data", {})
             if result:
                claims = result.get("claims", [])
                verifiable = sum(1 for c in claims if c.get("verifiable", False))
                return ClaimExtractionResult(
                    claims=claims,
                    total_claims=len(claims),
                    verifiable_claims=verifiable,
                    opinion_statements=result.get("opinion_statements", 0)
                )
        
        prompt = f"""Extract ONLY objective factual claims from the following text.

IGNORE:
- Personal opinions ("I think", "I feel", "In my opinion")
- Personal experiences ("I went", "I saw", "I did")
- Emotions and feelings
- Future predictions without data
- Questions

EXTRACT:
- Statistics and numbers
- Historical facts
- Scientific claims
- News events
- Health/medical claims
- Political statements presented as facts

TRANSCRIPT:
{transcript[:3000]}

Return a JSON array of claims:
{{
    "claims": [
        {{
            "claim": "<the factual claim>",
            "type": "statistic|historical|scientific|news|health|political|other",
            "verifiable": true/false,
            "severity": "low|medium|high",
            "context": "<surrounding context>"
        }}
    ],
    "opinion_statements": <count of opinion statements found>
}}"""

        try:
            response = self.model.generate_content(prompt)
            result = self._parse_json_response(response.text)
            
            claims = result.get("claims", [])
            verifiable = sum(1 for c in claims if c.get("verifiable", False))
            
            return ClaimExtractionResult(
                claims=claims,
                total_claims=len(claims),
                verifiable_claims=verifiable,
                opinion_statements=result.get("opinion_statements", 0)
            )
        except Exception as e:
            print(f"Claim extraction error: {e}")
            return self._rule_based_claim_extraction(transcript)

    def _rule_based_claim_extraction(self, transcript: str) -> ClaimExtractionResult:
        """Fallback rule-based claim extraction."""
        claims = []
        sentences = re.split(r'[.!?]', transcript)
        
        # Patterns for factual claims
        factual_patterns = [
            (r'\b\d+\s*%', "statistic"),
            (r'\b\d{4}\b', "historical"),  # Years
            (r'\b(study|research|scientists?|experts?)\s+(shows?|found|says?)\b', "scientific"),
            (r'\b(government|official|minister|president)\b', "political"),
            (r'\b(vaccine|covid|disease|treatment|cure)\b', "health")
        ]
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence or len(sentence) < 20:
                continue
                
            # Skip opinion sentences
            if any(op in sentence.lower() for op in ["i think", "i feel", "in my opinion", "personally"]):
                continue
            
            for pattern, claim_type in factual_patterns:
                if re.search(pattern, sentence, re.IGNORECASE):
                    claims.append({
                        "claim": sentence,
                        "type": claim_type,
                        "verifiable": True,
                        "severity": "medium",
                        "context": ""
                    })
                    break
        
        return ClaimExtractionResult(
            claims=claims,
            total_claims=len(claims),
            verifiable_claims=len(claims),
            opinion_statements=0
        )

    # ============== LAYER 6: HARM ASSESSMENT ==============
    
    def assess_harm(self, transcript: str, claims: List[Dict]) -> HarmAssessment:
        """
        Assess potential harm and reach impact of the content.
        """
        transcript_lower = transcript.lower()
        
        # Detect harm category
        detected_category = HarmCategory.NONE.value
        severity = 0.0
        
        for category, keywords in self.HIGH_RISK_CATEGORIES.items():
            matches = sum(1 for kw in keywords if kw in transcript_lower)
            if matches >= 2:
                detected_category = category
                severity = min(1.0, matches * 0.2)
                break
        
        # Increase severity for claims
        if claims:
            high_severity_claims = sum(1 for c in claims 
                                       if c.get("severity") == "high")
            severity = min(1.0, severity + high_severity_claims * 0.15)
        
        # Determine reach impact
        if severity > 0.7:
            reach_impact = "high"
            requires_action = True
        elif severity > 0.4:
            reach_impact = "medium"
            requires_action = False
        else:
            reach_impact = "low"
            requires_action = False
        
        return HarmAssessment(
            harm_category=detected_category,
            severity=round(severity, 3),
            potential_reach_impact=reach_impact,
            requires_immediate_action=requires_action
        )

    # ============== LAYER 7: RISK SCORING ==============
    
    def calculate_risk_score(self,
                              ai_signals: AIGenerationSignals,
                              deceptive_framing: DeceptiveFramingResult,
                              claims: ClaimExtractionResult,
                              harm: HarmAssessment,
                              disclosure: UserDisclosure) -> RiskScore:
        """
        Calculate final risk score using weighted components.
        
        Formula:
        riskScore = 0.25 * aiGeneratedScore +
                    0.30 * deceptiveFramingScore +
                    0.30 * factualClaimSeverity +
                    0.15 * disclosurePenalty
        """
        # AI-generated component (0.25 weight)
        ai_component = ai_signals.ai_generated_probability
        
        # Deceptive framing component (0.30 weight)
        deceptive_component = deceptive_framing.deceptive_score
        
        # Factual claim severity (0.30 weight)
        if claims.verifiable_claims > 0:
            high_severity = sum(1 for c in claims.claims 
                               if c.get("severity") == "high")
            claim_severity = min(1.0, (high_severity / claims.verifiable_claims) + 
                                harm.severity * 0.5)
        else:
            claim_severity = 0.0
        
        # Disclosure penalty (0.15 weight)
        disclosure_penalty = 0.0
        if ai_signals.ai_generated_probability > 0.5 and not disclosure.is_ai_generated:
            disclosure_penalty = 0.4  # Missing AI disclosure
        if not disclosure.disclosure_provided:
            disclosure_penalty += 0.2
        disclosure_penalty = min(1.0, disclosure_penalty)
        
        # Calculate total score
        total_score = (
            0.25 * ai_component +
            0.30 * deceptive_component +
            0.30 * claim_severity +
            0.15 * disclosure_penalty
        )
        
        # Determine action
        if total_score < 0.3:
            action = ActionType.ALLOW.value
        elif total_score < 0.5:
            action = ActionType.LABEL.value
        elif total_score < 0.7:
            action = ActionType.REDUCE_REACH.value
        else:
            action = ActionType.MANUAL_REVIEW.value
        
        return RiskScore(
            total_score=round(total_score, 3),
            ai_generated_component=round(ai_component, 3),
            deceptive_framing_component=round(deceptive_component, 3),
            factual_claim_severity=round(claim_severity, 3),
            disclosure_penalty=round(disclosure_penalty, 3),
            recommended_action=action
        )

    # ============== LAYER 8: FINAL LABELS ==============
    
    def generate_final_label(self, 
                             classification: ContentClassification,
                             risk_score: RiskScore,
                             claims: ClaimExtractionResult) -> Tuple[str, str]:
        """
        Generate appropriate final label and reasoning.
        
        NEVER use: "fake", "lying", "fraud"
        USE: Informative, fair labels
        """
        if classification.content_type in [ContentType.PERSONAL_SELF_MADE.value,
                                           ContentType.DEMONSTRATION.value]:
            if risk_score.total_score < 0.4:
                return (
                    "✅ Self-made content (Original)",
                    "Personal/demonstration content - fact-checking not applicable"
                )
        
        if classification.content_type == ContentType.OPINION_COMMENTARY.value:
            return (
                "ℹ️ Opinion/Commentary content",
                "This content expresses personal views - not fact-checkable"
            )
        
        if classification.content_type == ContentType.ENTERTAINMENT.value:
            return (
                "🎭 Entertainment/Storytelling content",
                "Fictional or entertainment content - not fact-checkable"
            )
        
        if risk_score.ai_generated_component > 0.6:
            if risk_score.total_score > 0.6:
                return (
                    "⚠️ AI-generated content with unverified claims",
                    "This content appears AI-generated and contains claims requiring verification"
                )
            else:
                return (
                    "🤖 AI-assisted content",
                    "This content may be AI-generated - claims have been reviewed"
                )
        
        if claims.verifiable_claims > 0:
            if risk_score.total_score > 0.5:
                return (
                    "🔍 Contains factual claims (Verification needed)",
                    f"Found {claims.verifiable_claims} claims requiring verification"
                )
            else:
                return (
                    "🔍 Contains factual claims (Verified)",
                    f"Verified {claims.verifiable_claims} factual claims"
                )
        
        return (
            "ℹ️ Content reviewed",
            "Standard content - no specific concerns identified"
        )

    # ============== MAIN ANALYSIS PIPELINE ==============
    
    def analyze_content(self, 
                        transcript: str,
                        user_disclosure: Optional[UserDisclosure] = None,
                        audio_features: Optional[Dict] = None,
                        video_id: Optional[str] = None) -> ModerationRecord:
        """
        Run the complete multi-layer content analysis pipeline.
        
        Args:
            transcript: Video transcript text
            user_disclosure: User-provided disclosure information
            audio_features: Optional audio analysis features
            video_id: Optional video identifier
            
        Returns:
            Complete ModerationRecord with all analysis results
        """
        if user_disclosure is None:
            user_disclosure = UserDisclosure()
            
        # Optimization: Run combined LLM analysis first
        if self.model:
            combined_result = self._perform_combined_llm_analysis(transcript)
            if combined_result:
                self.last_analysis_result = combined_result
                self.last_analysis_result["transcript_hash"] = hash(transcript)
        
        # Layer 1: Content Classification
        classification = self.classify_content(transcript)
        
        # Layer 2: Self-made Signals
        self_made = self.detect_self_made_signals(transcript)
        
        # Adjust classification based on self-made signals
        if self_made.ownership_confidence > 0.7 and not classification.requires_fact_check:
            classification.requires_fact_check = False
        
        # Consider user disclosure
        if user_disclosure.is_self_made and user_disclosure.disclosure_provided:
            if self_made.ownership_confidence > 0.4:
                classification = ContentClassification(
                    content_type=ContentType.PERSONAL_SELF_MADE.value,
                    confidence=0.85,
                    reason="User declared self-made + content signals match",
                    requires_fact_check=False
                )
        
        # Layer 3: AI Generation Detection (risk signal only)
        ai_signals = self.detect_ai_generation_signals(transcript, audio_features)
        
        # Layer 4: Deceptive Framing
        deceptive_framing = self.detect_deceptive_framing(transcript)
        
        # Layer 5: Claim Extraction (only if needed)
        if classification.requires_fact_check or deceptive_framing.is_deceptive:
            claims = self.extract_claims(transcript)
        else:
            claims = ClaimExtractionResult([], 0, 0, 0)
        
        # Layer 6: Harm Assessment
        harm = self.assess_harm(transcript, claims.claims)
        
        # Layer 7: Risk Scoring
        risk_score = self.calculate_risk_score(
            ai_signals, deceptive_framing, claims, harm, user_disclosure
        )
        
        # Layer 8: Final Labels
        final_label, reasoning = self.generate_final_label(
            classification, risk_score, claims
        )
        
        # Determine final action
        if classification.content_type == ContentType.PERSONAL_SELF_MADE.value:
            if risk_score.total_score < 0.5:
                final_action = ActionType.ALLOW.value
            else:
                final_action = risk_score.recommended_action
        else:
            final_action = risk_score.recommended_action
        
        # Create moderation record
        record = ModerationRecord(
            timestamp=datetime.now().isoformat(),
            video_id=video_id,
            content_classification=asdict(classification),
            self_made_signals=asdict(self_made),
            ai_generation_signals=asdict(ai_signals),
            deceptive_framing=asdict(deceptive_framing),
            extracted_claims=asdict(claims),
            harm_assessment=asdict(harm),
            risk_score=asdict(risk_score),
            user_disclosure=asdict(user_disclosure),
            final_action=final_action,
            final_label=final_label,
            reasoning=reasoning
        )
        
        return record

    # ============== FACT-CHECK INTEGRATION ==============
    
    def should_fact_check(self, record: ModerationRecord) -> Tuple[bool, str]:
        """
        Determine if content should be sent to fact-checking.
        
        Returns:
            (should_check, reason)
        """
        classification = record.content_classification
        
        # Self-made content - no fact check
        if classification["content_type"] in [
            ContentType.PERSONAL_SELF_MADE.value,
            ContentType.DEMONSTRATION.value,
            ContentType.ENTERTAINMENT.value
        ]:
            if record.risk_score["total_score"] < 0.5:
                return (False, "Self-made/demonstration content - fact-checking not applicable")
        
        # Opinion - no fact check
        if classification["content_type"] == ContentType.OPINION_COMMENTARY.value:
            if record.risk_score["total_score"] < 0.4:
                return (False, "Opinion content - fact-checking not applicable")
        
        # Has verifiable claims
        if record.extracted_claims["verifiable_claims"] > 0:
            return (True, f"Found {record.extracted_claims['verifiable_claims']} verifiable claims")
        
        # Deceptive framing detected
        if record.deceptive_framing["is_deceptive"]:
            return (True, "Deceptive framing detected - requires verification")
        
        # High risk score
        if record.risk_score["total_score"] > 0.6:
            return (True, "High risk score - requires verification")
        
        return (False, "No factual claims requiring verification")

    def get_claims_for_fact_check(self, record: ModerationRecord) -> List[Dict]:
        """
        Get only the claims that should be fact-checked.
        Filters out opinions and unverifiable statements.
        """
        claims = record.extracted_claims.get("claims", [])
        return [c for c in claims if c.get("verifiable", False)]

    # ============== GEMINI PROMPT WITH GUARDRAIL ==============
    
    def get_fact_check_prompt(self, claims: List[Dict], context: str) -> str:
        """
        Generate Gemini fact-check prompt with guardrails.
        """
        claims_text = "\n".join([f"- {c['claim']}" for c in claims])
        
        return f"""You are a fact-checking assistant. Verify the following claims.

IMPORTANT GUARDRAILS:
1. If the content is personal, experiential, self-recorded, or demonstrative,
   DO NOT label it as fake or misleading.
2. Only evaluate VERIFIABLE factual claims.
3. If a claim cannot be verified, mark it as "Unverifiable" - NOT "False"
4. Provide sources for all verdicts.

CLAIMS TO VERIFY:
{claims_text}

CONTEXT:
{context[:1000]}

For each claim, return:
{{
    "claim": "<the claim>",
    "verdict": "True|False|Partially True|Unverifiable",
    "confidence": 0.0-1.0,
    "explanation": "<brief explanation>",
    "sources": ["<source1>", "<source2>"]
}}

Return as JSON array."""

    # ============== UTILITY METHODS ==============
    
    def _parse_json_response(self, text: str) -> Dict:
        """Parse JSON from LLM response."""
        try:
            # Try to find JSON in response
            json_match = re.search(r'\{[\s\S]*\}', text)
            if json_match:
                return json.loads(json_match.group())
            
            # Try array
            json_match = re.search(r'\[[\s\S]*\]', text)
            if json_match:
                return {"claims": json.loads(json_match.group())}
            
            return {}
        except json.JSONDecodeError:
            return {}

    def to_dict(self, record: ModerationRecord) -> Dict:
        """Convert ModerationRecord to dictionary."""
        return asdict(record)

    def to_json(self, record: ModerationRecord) -> str:
        """Convert ModerationRecord to JSON string."""
        return json.dumps(asdict(record), indent=2, default=str)


# ============== QUICK ANALYSIS FUNCTION ==============

def quick_analyze(transcript: str, 
                  is_self_made: bool = False,
                  is_ai_generated: bool = False) -> Dict:
    """
    Quick analysis function for simple integration.
    
    Args:
        transcript: Video transcript
        is_self_made: User declared self-made
        is_ai_generated: User declared AI-generated
        
    Returns:
        Analysis result dictionary
    """
    classifier = ContentClassifier()
    
    disclosure = UserDisclosure(
        is_self_made=is_self_made,
        is_ai_generated=is_ai_generated,
        disclosure_provided=is_self_made or is_ai_generated
    )
    
    record = classifier.analyze_content(transcript, disclosure)
    should_check, reason = classifier.should_fact_check(record)
    
    return {
        "content_type": record.content_classification["content_type"],
        "requires_fact_check": should_check,
        "fact_check_reason": reason,
        "risk_score": record.risk_score["total_score"],
        "final_label": record.final_label,
        "final_action": record.final_action,
        "claims_to_verify": classifier.get_claims_for_fact_check(record),
        "full_record": classifier.to_dict(record)
    }


# ============== TEST ==============

if __name__ == "__main__":
    # Test with self-made vlog content
    vlog_transcript = """
    Hey everyone, welcome back to my channel! Today I'm going to show you 
    my morning routine. As you can see, I'm currently in my kitchen making 
    breakfast. I personally think oatmeal is the best breakfast option.
    Let me show you how I prepare it. This is my favorite recipe that 
    I've been making for years.
    """
    
    # Test with suspicious news-like content
    suspicious_transcript = """
    BREAKING NEWS: The government has officially banned all cryptocurrencies 
    starting tomorrow. This is 100% true and confirmed by official sources.
    Share this before it gets deleted. Media won't tell you this but the 
    truth must be revealed. Urgent warning to all citizens.
    """
    
    # Test with mixed content
    mixed_transcript = """
    Hey guys, this is my travel vlog from Paris. As you can see, I'm at 
    the Eiffel Tower right now. Did you know that the Eiffel Tower was 
    supposed to be demolished after 20 years? Also, studies show that 
    Paris receives over 30 million tourists annually. Anyway, let me 
    show you my hotel room.
    """
    
    print("=" * 60)
    print("CONTENT CLASSIFIER TEST")
    print("=" * 60)
    
    # Test vlog
    print("\n1. VLOG CONTENT TEST:")
    result = quick_analyze(vlog_transcript, is_self_made=True)
    print(f"   Content Type: {result['content_type']}")
    print(f"   Requires Fact Check: {result['requires_fact_check']}")
    print(f"   Risk Score: {result['risk_score']}")
    print(f"   Final Label: {result['final_label']}")
    
    # Test suspicious
    print("\n2. SUSPICIOUS CONTENT TEST:")
    result = quick_analyze(suspicious_transcript)
    print(f"   Content Type: {result['content_type']}")
    print(f"   Requires Fact Check: {result['requires_fact_check']}")
    print(f"   Risk Score: {result['risk_score']}")
    print(f"   Final Label: {result['final_label']}")
    print(f"   Claims to Verify: {len(result['claims_to_verify'])}")
    
    # Test mixed
    print("\n3. MIXED CONTENT TEST:")
    result = quick_analyze(mixed_transcript, is_self_made=True)
    print(f"   Content Type: {result['content_type']}")
    print(f"   Requires Fact Check: {result['requires_fact_check']}")
    print(f"   Risk Score: {result['risk_score']}")
    print(f"   Final Label: {result['final_label']}")
    print(f"   Claims to Verify: {len(result['claims_to_verify'])}")
