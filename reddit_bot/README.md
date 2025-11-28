# Reddit Comment Listener Bot 🤖

A Python bot that monitors a subreddit for comments containing the `!postinfo` trigger keyword and automatically replies with the parent post's details.

## Features

- 🔍 Monitors a configurable subreddit for new comments
- 📝 Responds to `!postinfo` trigger (case-insensitive)
- 📌 Replies with formatted post information including:
  - Post title
  - Author username
  - Media/image URL
  - Subreddit name
  - Score/upvotes
- 🔄 Prevents duplicate replies
- 🛡️ Ignores own messages to avoid loops
- 📊 Clean logging output
- ⚡ Continuous streaming with error recovery

## Prerequisites

- Python 3.7+
- A Reddit account for the bot
- Reddit API credentials (client_id and client_secret)

## Getting Reddit API Credentials

1. Go to [Reddit Apps](https://www.reddit.com/prefs/apps)
2. Click "Create App" or "Create Another App"
3. Fill in the details:
   - **Name:** Your bot name
   - **App type:** Select "script"
   - **Description:** Optional
   - **About URL:** Optional
   - **Redirect URI:** `http://localhost:8080`
4. Click "Create app"
5. Note down:
   - **client_id:** The string under "personal use script"
   - **client_secret:** The "secret" field

## Installation

1. Navigate to the reddit_bot folder:
   ```bash
   cd reddit_bot
   ```

2. Install dependencies:
   ```bash
   pip install praw python-dotenv
   ```
   
   Or using requirements.txt:
   ```bash
   pip install -r requirements.txt
   ```

3. Create your `.env` file:
   ```bash
   cp .env.example .env
   ```

4. Edit `.env` with your credentials:
   ```env
   REDDIT_CLIENT_ID=your_client_id_here
   REDDIT_CLIENT_SECRET=your_client_secret_here
   REDDIT_USERNAME=your_bot_username
   REDDIT_PASSWORD=your_bot_password
   REDDIT_USER_AGENT=PostInfoBot v1.0 by u/your_username
   TARGET_SUBREDDIT=PokemonPocketTradeCo
   ```

## Usage

Run the bot:
```bash
python bot.py
```

The bot will:
1. Authenticate with Reddit
2. Start monitoring the configured subreddit
3. Reply to any comment containing `!postinfo`

To stop the bot, press `Ctrl+C`.

## Reply Format Example

When someone comments `!postinfo`, the bot replies with:

```
📌 **Post Info**

📝 **Title:** Check out this awesome Pokemon card!

👤 **Posted by:** u/trainer123

🔗 **Media:** https://i.redd.it/example.jpg

💬 **Subreddit:** r/PokemonPocketTradeCo

👍 **Score:** 42

---
^(I am a bot. Reply with !postinfo on any post to get its details.)
```

## Configuration

You can modify these settings in your `.env` file:

| Variable | Description | Default |
|----------|-------------|---------|
| `TARGET_SUBREDDIT` | Subreddit to monitor (without r/) | `PokemonPocketTradeCo` |
| `REDDIT_USER_AGENT` | User agent string | `PostInfoBot v1.0` |

## File Structure

```
reddit_bot/
├── bot.py              # Main bot code
├── .env.example        # Example environment variables
├── .env                # Your actual credentials (git-ignored)
├── requirements.txt    # Python dependencies
└── README.md           # This file
```

## Troubleshooting

### Authentication Errors
- Double-check your client_id and client_secret
- Ensure your Reddit account has 2FA disabled or use an app password
- Verify your user_agent is unique

### Rate Limiting
- Reddit has rate limits; the bot handles this automatically
- If you get rate-limited, the bot will wait and retry

### Bot Not Responding
- Check if the subreddit name is correct
- Ensure the bot account has permission to comment in the subreddit
- Check the logs for any error messages

## License

MIT License - Feel free to modify and use as needed!
