# Pragmatic 🔍

**AI-Powered Fact-Checking Bot for Social Media**

Pragmatic is an intelligent fact-checking bot that can be tagged on social media posts (Reddit, Instagram, X/Twitter) to verify claims and provide evidence-based verdicts. It uses a multi-agent AI architecture powered by Google Gemini and smolagents to analyze content, extract facts, and deliver structured decisions.

---

## 🎯 How It Works

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              USER INTERACTION                                │
│   User tags @PragmaticBot on a post with a query like "Is this true?"       │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           SOCIAL MEDIA BOTS                                  │
│   • Reddit Bot (PRAW)                                                        │
│   • Instagram Bot (Instaloader)                                              │
│   • X/Twitter Bot (API)                                                      │
│                                                                              │
│   Collects: User query, Post content, Caption, Comments, Media              │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         AGENT 1: WATCHER AGENT                               │
│   📸 Analyzes post deeply using Gemini Vision                                │
│   📝 Extracts text, context, and claims from images/videos                   │
│   🔗 Processes captions and comments for context                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      AGENT 2: FACT EXTRACTOR AGENT                           │
│   🌐 Web search via DuckDuckGo                                               │
│   ✅ Google Fact Check API verification                                      │
│   📚 Gathers evidence from multiple sources                                  │
│   📊 Returns structured fact data with references                            │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      AGENT 3: DECISION MAKER AGENT                           │
│   ⚖️ Analyzes all collected evidence                                         │
│   🎯 Makes final TRUE/FAKE verdict                                           │
│   📋 Provides detailed analysis with sources                                 │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              BOT RESPONSE                                    │
│   {                                                                          │
│     "details": {                                                             │
│       "fact": "This is Fake",                                                │
│       "analysis": "Drinking hot water doesn't prevent COVID-19.             │
│                    Sources: WHO, CDC, Reuters Fact Check..."                │
│     }                                                                        │
│   }                                                                          │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
Pragmatic/
├── agent/                      # AI Agents (smolagents)
│   ├── app.py                  # Flask API server
│   ├── watcher.py              # Agent 1: Vision analysis
│   ├── fact_extracter.py       # Agent 2: Fact extraction
│   └── decision_maker.py       # Agent 3: Final decision
│
├── tools/                      # Custom tools for agents
│   └── google_factcheck_tool.py
│
├── reddit_bot/                 # Reddit integration
│   ├── bot.py                  # Reddit listener bot
│   ├── requirements.txt
│   └── README.md
│
├── instagram_bot/              # Instagram integration
│   └── reel_scrapper.py        # Extract reel/post data
│
├── api/                        # External API integrations
│   ├── server.js               # Node.js server
│   └── Xapi.js                 # X/Twitter API
│
├── .env                        # Environment variables
├── requirements.txt            # Python dependencies
└── README.md
```

---

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/CodeXtend/Pragmatic.git
cd Pragmatic
```

### 2. Install Dependencies

```bash
# Python dependencies
pip install -r requirements.txt

# For Reddit bot
cd reddit_bot && pip install -r requirements.txt && cd ..

# For Node.js API (optional)
cd api && npm install && cd ..
```

### 3. Configure Environment Variables

Create a `.env` file in the root directory:

```env
# LLM Configuration
LLM_MODEL=gemini/gemini-2.0-flash
GEMINI_API_KEY=your_gemini_api_key

# Google Fact Check API
GOOGLE_FACT_CHECK_API_KEY=your_google_factcheck_api_key

# Reddit Bot (optional)
REDDIT_CLIENT_ID=your_reddit_client_id
REDDIT_CLIENT_SECRET=your_reddit_client_secret
REDDIT_USERNAME=your_bot_username
REDDIT_PASSWORD=your_bot_password
REDDIT_USER_AGENT=PragmaticBot v1.0
TARGET_SUBREDDIT=your_target_subreddit
```

### 4. Run the API Server

```bash
cd agent
python app.py
```

The server will start at `http://localhost:5000`

---

## 📡 API Endpoints

### Health Check
```http
GET /
```
**Response:**
```json
{
  "status": "ok",
  "message": "Pragmatic API is running"
}
```

### Fact Check Query
```http
POST /api/query
Content-Type: application/json

{
  "message": "Is it true that drinking hot water prevents COVID-19?"
}
```

**Response:**
```json
{
  "response": {
    "details": {
      "fact": "This is Fake",
      "analysis": "Drinking hot water does not prevent COVID-19. Sources: WHO states there is no evidence that hot water prevents coronavirus infection. CDC confirms that only vaccines and proper hygiene measures are effective preventive measures."
    }
  },
  "success": true
}
```

### Memory Management
```http
GET /api/memory          # Get conversation history
DELETE /api/memory       # Clear conversation history
```

---

## 🤖 Agents Overview

### Agent 1: Watcher Agent
- **Purpose:** Deep analysis of social media posts
- **Capabilities:**
  - Gemini Vision for image/video analysis
  - Text extraction from media
  - Context understanding from captions and comments

### Agent 2: Fact Extractor Agent
- **Purpose:** Gather evidence and verify claims
- **Tools:**
  - `DuckDuckGoSearchTool` - Web search for related information
  - `GoogleFactCheckTool` - Official fact-check database queries
- **Output:** Structured fact data with sources and references

### Agent 3: Decision Maker Agent
- **Purpose:** Final verdict and analysis
- **Tools:**
  - `format_decision` - Structures the final output
- **Output:** 
  ```json
  {
    "details": {
      "fact": "This is True/Fake",
      "analysis": "Detailed explanation with sources..."
    }
  }
  ```

---

## 🔧 Supported Platforms

| Platform | Status | Trigger |
|----------|--------|---------|
| Reddit | ✅ Active | `!postinfo` or tag bot |
| Instagram | 🔄 In Progress | Tag @PragmaticBot |
| X/Twitter | 🔄 In Progress | Tag @PragmaticBot |

---

## 📦 Dependencies

### Python
```
flask
flask-cors
smolagents[litellm]
python-dotenv
diskcache
requests
praw (for Reddit)
instaloader (for Instagram)
```

### Node.js (for X/Twitter API)
```
See api/package.json
```

---

## 🔑 API Keys Required

1. **Google Gemini API Key** - For LLM and Vision capabilities
   - Get it from: [Google AI Studio](https://aistudio.google.com/)

2. **Google Fact Check API Key** - For official fact-check database
   - Get it from: [Google Cloud Console](https://console.cloud.google.com/)

3. **Reddit API Credentials** - For Reddit bot
   - Get it from: [Reddit Apps](https://www.reddit.com/prefs/apps)

---

## 🛠️ Development

### Running Tests
```bash
python -m pytest tests/
```

### Running Individual Bots

**Reddit Bot:**
```bash
cd reddit_bot
python bot.py
```

**Flask API Server:**
```bash
cd agent
python app.py
```

---

## 📄 License

MIT License - See [LICENSE](LICENSE) for details.

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 👥 Team

**CodeXtend** - [GitHub](https://github.com/CodeXtend)

---

## 📞 Support

For issues and questions, please open a [GitHub Issue](https://github.com/CodeXtend/Pragmatic/issues).

---

<p align="center">
  <b>Fighting Misinformation, One Fact at a Time 🎯</b>
</p>
