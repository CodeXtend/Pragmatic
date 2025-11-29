/**
 * 🤖 PRAGMATIC - TWITTER FACT-CHECK BOT
 * 
 * When someone tags this bot on a tweet with media (video/image):
 * 1. Downloads the media
 * 2. Analyzes it with Gemini 2.0 Flash
 * 3. Replies with fact-check analysis
 * 
 * RUN: node bot.js
 */

import fetch from "node-fetch";
import dotenv from "dotenv";
import crypto from "crypto";
import OAuth from "oauth-1.0a";
import fs from "fs";
import path from "path";
import { GoogleGenerativeAI } from "@google/generative-ai";

dotenv.config();

// ============== CONFIGURATION ==============

const CONFIG = {
  BOT_NAME: "Pragmatic",
  POLL_INTERVAL_MS: 15000, // 15 seconds
  AUTO_REPLY: true,
};

// Environment variables
const BEARER_TOKEN = process.env.Bearer_Token?.trim();
const API_KEY = process.env.TWITTER_API_KEY?.trim();
const API_SECRET = process.env.TWITTER_API_SECRET?.trim();
const ACCESS_TOKEN = process.env.TWITTER_ACCESS_TOKEN?.trim();
const ACCESS_SECRET = process.env.TWITTER_ACCESS_SECRET?.trim();
const GEMINI_API_KEY = process.env.GEMINI_API_KEY?.trim();

console.log("\n🔧 DEBUG: Checking environment variables...");
console.log(`   BEARER_TOKEN: ${BEARER_TOKEN ? "✓ Set" : "✗ Missing"}`);
console.log(`   API_KEY: ${API_KEY ? "✓ Set" : "✗ Missing"}`);
console.log(`   API_SECRET: ${API_SECRET ? "✓ Set" : "✗ Missing"}`);
console.log(`   ACCESS_TOKEN: ${ACCESS_TOKEN ? "✓ Set" : "✗ Missing"}`);
console.log(`   ACCESS_SECRET: ${ACCESS_SECRET ? "✓ Set" : "✗ Missing"}`);
console.log(`   GEMINI_API_KEY: ${GEMINI_API_KEY ? "✓ Set" : "✗ Missing"}`);

// OAuth 1.0a for user context endpoints
const oauth = new OAuth({
  consumer: {
    key: API_KEY,
    secret: API_SECRET,
  },
  signature_method: "HMAC-SHA1",
  hash_function(base_string, key) {
    return crypto.createHmac("sha1", key).update(base_string).digest("base64");
  },
});

const token = {
  key: ACCESS_TOKEN,
  secret: ACCESS_SECRET,
};

// Gemini AI setup
const genAI = GEMINI_API_KEY ? new GoogleGenerativeAI(GEMINI_API_KEY) : null;

// Bot state
let processedMentions = new Set();
let lastMentionId = null;
let botUserId = null;
let botUsername = null;
let isRunning = false;

// ============== TWITTER API FUNCTIONS ==============

/**
 * Sleep helper
 */
function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

/**
 * Get authenticated user info - Using OAuth 1.0a
 */
async function getMe(retries = 3) {
  const url = "https://api.twitter.com/2/users/me?user.fields=id,name,username,profile_image_url,public_metrics";

  console.log("\n📡 API CALL: GET /users/me");
  
  const authHeader = oauth.toHeader(
    oauth.authorize({ url, method: "GET" }, token)
  );

  for (let attempt = 1; attempt <= retries; attempt++) {
    try {
      const response = await fetch(url, {
        method: "GET",
        headers: { ...authHeader },
      });

      console.log(`   Status: ${response.status}`);
      
      if (response.status === 429) {
        const waitTime = attempt * 60000; // Wait 1 min, 2 min, 3 min...
        console.log(`   ⏳ Rate limited! Waiting ${waitTime/1000} seconds before retry ${attempt}/${retries}...`);
        await sleep(waitTime);
        continue;
      }

      return await response.json();
    } catch (error) {
      console.error(`   ❌ ERROR: ${error.message}`);
      if (attempt === retries) throw error;
      await sleep(5000);
    }
  }
}

/**
 * Get mentions - Using OAuth 1.0a (required for user context)
 */
async function getMentions(userId, sinceId = null) {
  let url = `https://api.twitter.com/2/users/${userId}/mentions?tweet.fields=created_at,author_id,conversation_id,referenced_tweets&expansions=author_id,referenced_tweets.id&user.fields=id,name,username&max_results=5`;

  if (sinceId) {
    url += `&since_id=${sinceId}`;
  }

  // Use OAuth 1.0a for mentions (user context required)
  const authHeader = oauth.toHeader(
    oauth.authorize({ url, method: "GET" }, token)
  );

  try {
    const response = await fetch(url, {
      method: "GET",
      headers: { ...authHeader },
    });

    const data = await response.json();
    return data;
  } catch (error) {
    console.error(`   ❌ FETCH ERROR: ${error.message}`);
    throw error;
  }
}

/**
 * Get tweet details with media
 */
async function getTweetDetails(tweetId) {
  const url = `https://api.twitter.com/2/tweets/${tweetId}?tweet.fields=created_at,author_id,public_metrics,entities,conversation_id,lang,source,referenced_tweets,attachments&expansions=author_id,attachments.media_keys,referenced_tweets.id&user.fields=id,name,username,profile_image_url,verified,public_metrics,description&media.fields=media_key,type,url,preview_image_url,width,height,alt_text,variants,duration_ms`;

  console.log("🔍 Fetching tweet details...");

  const authHeader = oauth.toHeader(
    oauth.authorize({ url, method: "GET" }, token)
  );

  const response = await fetch(url, {
    method: "GET",
    headers: { ...authHeader },
  });

  return await response.json();
}

/**
 * Get user details by user ID
 */
async function getUserById(userId) {
  const url = `https://api.twitter.com/2/users/${userId}?user.fields=id,name,username,profile_image_url,verified,public_metrics,description,created_at,location`;

  console.log(`🔍 Fetching user details for ID: ${userId}...`);

  const authHeader = oauth.toHeader(
    oauth.authorize({ url, method: "GET" }, token)
  );

  const response = await fetch(url, {
    method: "GET",
    headers: { ...authHeader },
  });

  const data = await response.json();
  
  if (data.data) {
    console.log(`   ✅ Found user: @${data.data.username} (${data.data.name})`);
    return data.data;
  }
  
  console.log(`   ❌ User not found or error:`, JSON.stringify(data, null, 2));
  return null;
}

/**
 * Post a reply tweet
 */
async function postReply(text, replyToTweetId) {
  const url = "https://api.twitter.com/2/tweets";

  console.log(`\n📡 API CALL: POST /tweets (reply)`);
  console.log(`   Reply to: ${replyToTweetId}`);

  const body = {
    text: text,
    reply: {
      in_reply_to_tweet_id: replyToTweetId,
    },
  };

  const authHeader = oauth.toHeader(
    oauth.authorize({ url, method: "POST" }, token)
  );

  try {
    const response = await fetch(url, {
      method: "POST",
      headers: {
        ...authHeader,
        "Content-Type": "application/json",
      },
      body: JSON.stringify(body),
    });

    const data = await response.json();
    console.log(`   Status: ${response.status}`);
    return data;
  } catch (error) {
    console.error(`   ❌ POST ERROR: ${error.message}`);
    throw error;
  }
}

// ============== MEDIA FUNCTIONS ==============

/**
 * Extract best video MP4 URL from media
 */
function getVideoMp4Url(media) {
  if (!media || media.type !== "video") return null;

  const variants = media.variants || [];
  const mp4Variants = variants
    .filter(v => v.content_type === "video/mp4")
    .sort((a, b) => (b.bit_rate || 0) - (a.bit_rate || 0));

  return mp4Variants.length > 0 ? mp4Variants[0].url : null;
}

/**
 * Download video to local file
 */
async function downloadVideo(videoUrl, filename) {
  console.log("📥 Downloading video...");
  
  const response = await fetch(videoUrl);
  const buffer = await response.buffer();
  
  const filepath = path.join(process.cwd(), filename);
  fs.writeFileSync(filepath, buffer);
  
  console.log(`   ✅ Saved to: ${filepath}`);
  console.log(`   📦 Size: ${(buffer.length / 1024 / 1024).toFixed(2)} MB`);
  
  return { filepath, buffer };
}

/**
 * Download image to buffer
 */
async function downloadImage(imageUrl) {
  console.log("📥 Downloading image...");
  const response = await fetch(imageUrl);
  return await response.buffer();
}

// ============== GEMINI ANALYSIS ==============

/**
 * Analyze video with Gemini 2.0 Flash
 */
async function analyzeVideoWithGemini(videoBuffer, tweetData) {
  if (!genAI) {
    console.log("❌ GEMINI_API_KEY not found in .env");
    return null;
  }

  console.log("\n🤖 Analyzing video with Gemini 2.0 Flash...");

  const model = genAI.getGenerativeModel({ model: "gemini-2.0-flash" });
  const videoBase64 = videoBuffer.toString("base64");

  const prompt = `You are a fact-checking assistant analyzing a video from Twitter/X.

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
- Author: ${tweetData.author || "Unknown"}
- Tweet text: ${tweetData.text || "N/A"}
- Posted: ${tweetData.created_at || "Unknown"}

Return ONLY valid JSON in this exact format:
{
  "User": "The question or context from the tweet",
  "post_caption": "The caption/text of the post",
  "few_comments": ["comment1", "comment2"],
  "Gemini_Scan_Details": "Detailed description of visual content in the video",
  "Person_talking_about": "Speech to text transcription of what is said in the video",
  "visual_summary": {
    "objects_detected": ["object1", "object2", "object3"],
    "people_in_video": "Description of people visible - their appearance, clothing, expressions",
    "speaker_actions": "Who is speaking and their gestures, body language, facial expressions",
    "scene_description": "Indoor/outdoor, location type, environment details",
    "text_overlays": "Any text visible on screen",
    "key_moments": "Important visual moments or actions in the video"
  },
  "detail_analysis": "Analysis of what this content is about and if it seems real or fake",
  "is_potentially_misleading": false,
  "confidence_score": 85
}`;

  try {
    const result = await model.generateContent([
      {
        inlineData: {
          mimeType: "video/mp4",
          data: videoBase64,
        },
      },
      prompt,
    ]);

    const responseText = result.response.text();
    console.log("   ✅ Gemini analysis complete!");

    try {
      let jsonStr = responseText;
      const jsonMatch = responseText.match(/```json\n?([\s\S]*?)\n?```/);
      if (jsonMatch) {
        jsonStr = jsonMatch[1];
      } else {
        const objMatch = responseText.match(/\{[\s\S]*\}/);
        if (objMatch) {
          jsonStr = objMatch[0];
        }
      }
      
      return JSON.parse(jsonStr);
    } catch (parseError) {
      console.log("   ⚠️ Could not parse JSON, returning raw response");
      return { raw_response: responseText };
    }
  } catch (error) {
    console.error("   ❌ Gemini error:", error.message);
    return { error: error.message };
  }
}

/**
 * Analyze image with Gemini 2.0 Flash
 */
async function analyzeImageWithGemini(imageBuffer, tweetData) {
  if (!genAI) {
    console.log("❌ GEMINI_API_KEY not found in .env");
    return null;
  }

  console.log("\n🤖 Analyzing image with Gemini 2.0 Flash...");

  const model = genAI.getGenerativeModel({ model: "gemini-2.0-flash" });
  const imageBase64 = imageBuffer.toString("base64");

  const prompt = `You are a fact-checking assistant analyzing an image from Twitter/X.

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
- Author: ${tweetData.author || "Unknown"}
- Tweet text: ${tweetData.text || "N/A"}

Return ONLY valid JSON:
{
  "User": "Context from tweet",
  "post_caption": "Caption of the post",
  "Gemini_Scan_Details": "Description of what is in the image",
  "text_in_image": "Any text visible in the image",
  "visual_summary": {
    "objects_detected": ["object1", "object2"],
    "people_in_image": "Description of people visible",
    "scene_description": "Environment and setting details",
    "text_overlays": "Any text visible on the image",
    "editing_signs": "Any signs of manipulation or editing"
  },
  "detail_analysis": "Analysis of the image authenticity",
  "is_potentially_misleading": false,
  "confidence_score": 85
}`;

  try {
    const result = await model.generateContent([
      {
        inlineData: {
          mimeType: "image/jpeg",
          data: imageBase64,
        },
      },
      prompt,
    ]);

    const responseText = result.response.text();
    console.log("   ✅ Gemini analysis complete!");

    try {
      let jsonStr = responseText;
      const jsonMatch = responseText.match(/```json\n?([\s\S]*?)\n?```/);
      if (jsonMatch) {
        jsonStr = jsonMatch[1];
      } else {
        const objMatch = responseText.match(/\{[\s\S]*\}/);
        if (objMatch) jsonStr = objMatch[0];
      }
      return JSON.parse(jsonStr);
    } catch {
      return { raw_response: responseText };
    }
  } catch (error) {
    console.error("   ❌ Gemini error:", error.message);
    return { error: error.message };
  }
}

// ============== WATCHER AGENT ==============

/**
 * Agent 1: Watcher Agent
 * Analyzes Twitter content using Gemini Vision and Speech-to-Text
 */
async function watcherAgent(tweetId) {
  console.log("\n" + "═".repeat(60));
  console.log("🔍 AGENT 1: WATCHER AGENT");
  console.log("   (Gemini Vision, SpeechToText)");
  console.log("═".repeat(60));

  // Step 1: Get tweet details
  const tweetData = await getTweetDetails(tweetId);

  // Better error handling
  if (!tweetData) {
    console.log("❌ No response from Twitter API");
    return null;
  }

  if (tweetData.errors) {
    console.log("❌ API Errors:", JSON.stringify(tweetData.errors, null, 2));
    return null;
  }

  if (tweetData.status === 401) {
    console.log("❌ Unauthorized - Check your API credentials");
    return null;
  }

  if (tweetData.status === 429) {
    console.log("❌ Rate limited - Too many requests. Wait a few minutes and try again.");
    return null;
  }

  if (!tweetData.data) {
    console.log("❌ No tweet data returned. Response:", JSON.stringify(tweetData, null, 2));
    return null;
  }

  const tweet = tweetData.data;
  let author = tweetData.includes?.users?.find(u => u.id === tweet.author_id);
  
  // If author not found in includes, fetch by author_id
  if (!author && tweet.author_id) {
    console.log("   ℹ️ Author not in includes, fetching by ID...");
    author = await getUserById(tweet.author_id);
  }
  
  const media = tweetData.includes?.media || [];

  console.log(`\n📝 Tweet: "${tweet.text}"`);
  console.log(`👤 Author: @${author?.username || "unknown"} (ID: ${tweet.author_id})`);
  console.log(`📅 Posted: ${tweet.created_at}`);

  const tweetContext = {
    author: author?.username,
    text: tweet.text,
    created_at: tweet.created_at,
  };

  let analysisResult = null;

  // Step 2: Process media
  for (const mediaItem of media) {
    console.log(`\n📎 Media found: ${mediaItem.type}`);

    if (mediaItem.type === "video") {
      // Get video URL
      const videoUrl = getVideoMp4Url(mediaItem);
      if (!videoUrl) {
        console.log("   ❌ Could not extract video URL");
        continue;
      }

      console.log(`   🎬 Video URL: ${videoUrl.substring(0, 60)}...`);

      // Download video
      const { buffer } = await downloadVideo(videoUrl, `tweet_${tweetId}.mp4`);

      // Analyze with Gemini
      analysisResult = await analyzeVideoWithGemini(buffer, tweetContext);

    } else if (mediaItem.type === "photo") {
      // Analyze image
      const imageUrl = mediaItem.url;
      console.log(`   📷 Image URL: ${imageUrl}`);

      const buffer = await downloadImage(imageUrl);
      analysisResult = await analyzeImageWithGemini(buffer, tweetContext);
    }
  }

  // If no media, analyze text only
  if (media.length === 0) {
    console.log("\n📝 No media found, analyzing text only...");
    analysisResult = {
      User: "Text analysis request",
      post_caption: tweet.text,
      Gemini_Scan_Details: "No visual content - text only tweet",
      detail_analysis: "Cannot perform visual analysis on text-only content",
      is_potentially_misleading: null,
      confidence_score: 0,
    };
  }

  // Step 3: Output final JSON
  console.log("\n" + "═".repeat(60));
  console.log("📊 WATCHER AGENT OUTPUT");
  console.log("═".repeat(60));

  const finalOutput = {
    details: {
      User: analysisResult?.User || tweet.text,
      post_caption: analysisResult?.post_caption || tweet.text,
      few_comments: analysisResult?.few_comments || ["Analyzing...", "Checking facts..."],
      Gemini_Scan_Details: analysisResult?.Gemini_Scan_Details || "Analysis pending",
      Person_talking_about: analysisResult?.Person_talking_about || "No speech detected",
      visual_summary: analysisResult?.visual_summary || {
        objects_detected: [],
        people_in_video: "Not analyzed",
        speaker_actions: "Not analyzed",
        scene_description: "Not analyzed",
        text_overlays: "None detected",
        key_moments: "Not analyzed"
      },
      detail_analysis: analysisResult?.detail_analysis || "Analysis pending",
      is_potentially_misleading: analysisResult?.is_potentially_misleading || false,
      confidence_score: analysisResult?.confidence_score || 0,
    },
    tweet_info: {
      id: tweet.id,
      author: author?.username,
      url: `https://x.com/${author?.username}/status/${tweet.id}`,
      created_at: tweet.created_at,
    },
    analyzed_at: new Date().toISOString(),
  };

  console.log(JSON.stringify(finalOutput, null, 2));

  return finalOutput;
}

// ============== GENERATE REPLY MESSAGE ==============

/**
 * Generate a fact-check reply message from analysis
 */
function generateFactCheckReply(analysis, mentionAuthor) {
  if (!analysis?.details) {
    return `Hello @${mentionAuthor}! 👋\n\nI couldn't analyze that content. Please make sure to tag me on a tweet that contains a video or image.\n\n🤖 Pragmatic`;
  }

  const result = analysis.details;
  const isMisleading = result.is_potentially_misleading;
  const confidence = result.confidence_score || 0;

  let emoji = "ℹ️";
  let verdict = "Analysis Complete";
  
  if (isMisleading === true) {
    emoji = "⚠️";
    verdict = "Potentially Misleading";
  } else if (isMisleading === false) {
    emoji = "✅";
    verdict = "No Red Flags Detected";
  }

  let reply = `${emoji} Fact-Check for @${mentionAuthor}\n\n`;
  reply += `📊 Verdict: ${verdict}\n`;
  reply += `🎯 Confidence: ${confidence}%\n\n`;
  
  // Add brief analysis (keep it short for Twitter)
  const briefAnalysis = result.detail_analysis?.substring(0, 150) || "Analysis complete";
  reply += `📝 ${briefAnalysis}${briefAnalysis.length >= 150 ? "..." : ""}\n\n`;
  
  reply += `🤖 #FactCheck #Pragmatic`;

  // Twitter 280 char limit
  if (reply.length > 280) {
    reply = reply.substring(0, 277) + "...";
  }

  return reply;
}

// ============== BOT CORE LOGIC ==============

/**
 * Process a single mention - Analyze the tagged tweet and reply
 */
async function processMention(mention, includesUsers, includesTweets) {
  const mentionId = mention.id;
  
  // Skip if already processed
  if (processedMentions.has(mentionId)) {
    return null;
  }
  
  processedMentions.add(mentionId);
  
  // Get mention author
  const mentionAuthor = includesUsers?.find((u) => u.id === mention.author_id);
  const username = mentionAuthor?.username || "there";
  
  console.log(`\n${"─".repeat(50)}`);
  console.log(`📩 NEW MENTION from @${username}`);
  console.log(`   Text: ${mention.text}`);
  console.log(`${"─".repeat(50)}`);
  
  // Find the tweet they're replying to (the one to analyze)
  let targetTweetId = null;
  
  // Check if this mention is a reply to another tweet
  const repliedTo = mention.referenced_tweets?.find(ref => ref.type === "replied_to");
  if (repliedTo) {
    targetTweetId = repliedTo.id;
    console.log(`   🎯 Found replied-to tweet: ${targetTweetId}`);
  }
  
  // If no reply, check conversation
  if (!targetTweetId && mention.conversation_id && mention.conversation_id !== mentionId) {
    targetTweetId = mention.conversation_id;
    console.log(`   🎯 Using conversation root: ${targetTweetId}`);
  }

  try {
    let replyMessage;
    
    if (targetTweetId) {
      // Analyze the target tweet with Watcher Agent
      console.log(`   🔍 Analyzing tweet ${targetTweetId}...`);
      const analysis = await watcherAgent(targetTweetId);
      
      if (analysis) {
        console.log(`   ✅ Analysis complete!`);
        console.log(`   📊 Misleading: ${analysis.details?.is_potentially_misleading}`);
        console.log(`   🎯 Confidence: ${analysis.details?.confidence_score}%`);
        
        replyMessage = generateFactCheckReply(analysis, username);
      } else {
        replyMessage = `Hello @${username}! 👋\n\nI couldn't analyze that tweet. It may not contain media or there was an error.\n\n🤖 Pragmatic`;
      }
    } else {
      // No target tweet found - direct mention without reply
      replyMessage = `Hello @${username}! 👋\n\nTo fact-check a post, please reply to a tweet containing an image or video and tag me.\n\nExample: Reply to a suspicious post and mention @${botUsername}\n\n🤖 Pragmatic`;
    }
    
    console.log(`   💬 Sending reply...`);
    const replyResult = await postReply(replyMessage, mentionId);
    
    if (replyResult.data) {
      console.log(`   ✅ Reply sent! ID: ${replyResult.data.id}`);
    } else {
      console.log(`   ⚠️ Reply failed:`, JSON.stringify(replyResult));
    }
    
    return { success: true, mention_id: mentionId };
    
  } catch (error) {
    console.log(`   ❌ Error: ${error.message}`);
    return null;
  }
}

/**
 * Poll for new mentions
 */
async function pollMentions() {
  try {
    // Fetch mentions
    const response = await getMentions(botUserId, lastMentionId);
    
    if (response.errors) {
      console.log(`\n⚠️ API Error: ${response.errors[0]?.message || JSON.stringify(response.errors)}`);
      return;
    }
    
    const mentions = response.data || [];
    const includesUsers = response.includes?.users || [];
    const includesTweets = response.includes?.tweets || [];
    
    if (mentions.length === 0) {
      process.stdout.write(".");
      return;
    }
    
    console.log(`\n\n📬 Found ${mentions.length} new mention(s)!`);
    
    // Update pagination
    if (mentions.length > 0) {
      lastMentionId = mentions[0].id;
    }
    
    // Process each mention
    for (const mention of mentions) {
      await processMention(mention, includesUsers, includesTweets);
    }
    
  } catch (error) {
    console.log(`\n❌ Poll error: ${error.message}`);
  }
}

/**
 * Initialize and start the bot
 */
async function startBot() {
  console.log(`\n${"═".repeat(60)}`);
  console.log(`🤖 PRAGMATIC - TWITTER FACT-CHECK BOT`);
  console.log(`${"═".repeat(60)}`);
  
  // Check Gemini
  if (!genAI) {
    console.log(`\n⚠️  GEMINI_API_KEY not found!`);
    console.log(`   The bot will run but cannot analyze media.`);
    console.log(`   Add GEMINI_API_KEY to your .env file.`);
  }
  
  // Verify credentials
  console.log(`\n🔐 Verifying Twitter credentials...`);
  
  try {
    const me = await getMe();
    
    if (!me.data) {
      console.log(`\n❌ Failed to authenticate!`);
      console.log(`   Error: ${JSON.stringify(me)}`);
      process.exit(1);
    }
    
    botUserId = me.data.id;
    botUsername = me.data.username;
    
    console.log(`\n✅ Authenticated as @${botUsername}`);
    console.log(`   User ID: ${botUserId}`);
    console.log(`   Followers: ${me.data.public_metrics?.followers_count || 0}`);
    
  } catch (error) {
    console.log(`\n❌ Auth error: ${error.message}`);
    process.exit(1);
  }
  
  // Start polling
  console.log(`\n${"─".repeat(50)}`);
  console.log(`🚀 Bot is now running!`);
  console.log(`   Polling every ${CONFIG.POLL_INTERVAL_MS / 1000} seconds`);
  console.log(`   Gemini: ${genAI ? "✓ Ready" : "✗ Not configured"}`);
  console.log(`${"─".repeat(50)}`);
  console.log(`\n📖 HOW TO USE:`);
  console.log(`   1. Find a tweet with video/image to fact-check`);
  console.log(`   2. Reply to that tweet and tag @${botUsername}`);
  console.log(`   3. Wait for the bot to analyze and reply`);
  console.log(`${"─".repeat(50)}`);
  console.log(`\n📡 Listening for mentions... (Ctrl+C to stop)\n`);
  
  isRunning = true;
  
  // Initial poll
  await pollMentions();
  
  // Set interval for continuous polling
  const intervalId = setInterval(async () => {
    if (isRunning) {
      await pollMentions();
    }
  }, CONFIG.POLL_INTERVAL_MS);
  
  // Handle graceful shutdown
  process.on("SIGINT", () => {
    console.log(`\n\n${"═".repeat(60)}`);
    console.log(`🛑 Shutting down...`);
    console.log(`   Processed ${processedMentions.size} mentions`);
    console.log(`${"═".repeat(60)}\n`);
    
    isRunning = false;
    clearInterval(intervalId);
    process.exit(0);
  });
}

// ============== RUN BOT ==============

startBot();
