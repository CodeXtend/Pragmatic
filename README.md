
# 🛡️ Pragmatic

## Information

Project: Pragmatic

Pragmatic is an AI-powered fact-checking platform that detects and analyzes potentially misleading content across social platforms. It uses a multi-agent approach: an extraction agent gathers claims and evidence from web search and fact-check databases, and a decision agent evaluates credibility and outputs a structured verdict. Integrations include a Python Flask API, a Node.js Twitter/X bot for monitoring and responding to mentions, and an Instagram reel extractor for media metadata.

Key components:
- `agent/fact_extracter.py` — gathers evidence using search and fact-check tools.
- `agent/decision_maker.py` — analyzes evidence and formats a final verdict.
- `agent/app.py` — Flask API exposing endpoints for queries and memory management.
- `api/bot.js`, `api/server.js` — Node.js services for Twitter integration and media analysis.
- `instagram_bot/reel_scrapper.py` — extracts reel metadata using Instaloader.
- `tools/google_factcheck_tool.py` — wrapper for Google Fact Check API.

## Problem Statement

Social platforms spread claims rapidly; many are unverified or false. Manual fact-checking is slow and cannot scale to the volume of posts, especially posts containing media (videos/reels) that may be manipulated or miscontextualized. Teams need an automated, auditable pipeline that:

- Accepts a user-submitted claim or a social post (text or media).
- Gathers supporting and contradicting evidence from authoritative sources and web search.
- Produces a transparent, source-cited decision (e.g., "This is Fake" with analysis).
- Integrates with social platforms to monitor mentions and provide timely responses.

![Image](https://github.com/user-attachments/assets/2bfef727-e990-4ac2-91dc-2654d92257a6)

## Solution

TruthShield provides a pragmatic, modular solution:

1. Ingest: Accept a claim or social post via the Flask API or by monitoring Twitter/X mentions with the Node.js bot.
2. Extract Evidence: `FactExtracter` runs search tools (DuckDuckGo) and the Google Fact Check API, collects timestamps and source links, and builds structured evidence.
3. Decide: `DecisionMaker` analyzes the collected evidence using a lightweight LLM agent and returns a JSON-formatted verdict with an explanation and cited sources.
4. Respond & Store: The system returns the decision to the caller (API response) and optionally replies on the originating social post (bot reply). Conversation memory is stored in the agent to provide context-aware follow-ups and can be cleared or retrieved via API endpoints.

![Image](https://github.com/user-attachments/assets/428a740d-20dc-47ba-8e9b-fddb00633ac5)
![Image](https://github.com/user-attachments/assets/d4fc3f94-8559-45bb-b012-2e48c1b06d32)


Design benefits:
- Automated multi-source verification reduces manual workload and improves response time.
- Structured JSON decisions make results auditable and easy to integrate.
- Modular agents allow swapping or upgrading search and LLM components independently.

Quick example (conceptual):

Request: POST `/api/query` { "message": "Do COVID-19 vaccines contain microchips?" }

Response: {
  "details": {
    "fact": "This is Fake",
    "analysis": "Multiple authoritative sources (WHO, CDC, Reuters Fact Check) find no evidence; Google Fact Check entries debunk this claim."
  }
}

If you'd like, I can now:
- Run a quick verification example locally (needs API keys)
- Trim or expand any section or change wording/tone
- Add minimal Docker or deploy instructions back into the README

