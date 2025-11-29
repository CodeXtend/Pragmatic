/**
 * 🤖 PRAGMATIC - EXPRESS API SERVER
 * 
 * REST API endpoints for the Pragmatic Fact-Check Bot
 * 
 * RUN: node server.js
 */

import express from "express";
import cors from "cors";
import fetch from "node-fetch";
import dotenv from "dotenv";
import crypto from "crypto";
import OAuth from "oauth-1.0a";
import fs from "fs";
import path from "path";
import { GoogleGenerativeAI } from "@google/generative-ai";

dotenv.config();

const app = express();
const PORT = process.env.PORT || 3000;

// Middleware
app.use(cors());
app.use(express.json());

// ============== CONFIGURATION ==============

const API_KEY = process.env.TWITTER_API_KEY?.trim();
const API_SECRET = process.env.TWITTER_API_SECRET?.trim();
const ACCESS_TOKEN = process.env.TWITTER_ACCESS_TOKEN?.trim();
const ACCESS_SECRET = process.env.TWITTER_ACCESS_SECRET?.trim();
const GEMINI_API_KEY = process.env.GEMINI_API_KEY?.trim();

// OAuth 1.0a setup for Twitter
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

// ============== TWITTER FUNCTIONS ==============

/**
 * Get tweet details with media
 */
async function getTweetDetails(tweetId) {
  const url = `https://api.twitter.com/2/tweets/${tweetId}?tweet.fields=created_at,author_id,public_metrics,entities,conversation_id,lang,source,referenced_tweets,attachments&expansions=author_id,attachments.media_keys,referenced_tweets.id&user.fields=id,name,username,profile_image_url,verified,public_metrics,description&media.fields=media_key,type,url,preview_image_url,width,height,alt_text,variants,duration_ms`;

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

  const authHeader = oauth.toHeader(
    oauth.authorize({ url, method: "GET" }, token)
  );

  const response = await fetch(url, {
    method: "GET",
    headers: { ...authHeader },
  });

  const data = await response.json();
  return data.data || null;
}

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
 * Download video to buffer
 */
async function downloadVideo(videoUrl) {
  const response = await fetch(videoUrl);
  const buffer = await response.buffer();
  return buffer;
}

/**
 * Download image to buffer
 */
async function downloadImage(imageUrl) {
  const response = await fetch(imageUrl);
  return await response.buffer();
}

// ============== GEMINI ANALYSIS ==============

/**
 * Analyze video with Gemini 2.0 Flash
 */
async function analyzeVideoWithGemini(videoBuffer, tweetData) {
  if (!genAI) {
    return { error: "GEMINI_API_KEY not configured" };
  }

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
      return { raw_response: responseText };
    }
  } catch (error) {
    return { error: error.message };
  }
}

/**
 * Analyze image with Gemini 2.0 Flash
 */
async function analyzeImageWithGemini(imageBuffer, tweetData) {
  if (!genAI) {
    return { error: "GEMINI_API_KEY not configured" };
  }

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
    return { error: error.message };
  }
}

// ============== WATCHER AGENT ==============

/**
 * Agent 1: Watcher Agent
 * Analyzes Twitter content using Gemini Vision and Speech-to-Text
 */
async function watcherAgent(tweetId) {
  // Step 1: Get tweet details
  const tweetData = await getTweetDetails(tweetId);

  if (!tweetData) {
    return { error: "No response from Twitter API" };
  }

  if (tweetData.errors) {
    return { error: "API Error", details: tweetData.errors };
  }

  if (tweetData.status === 401) {
    return { error: "Unauthorized - Check API credentials" };
  }

  if (tweetData.status === 429) {
    return { error: "Rate limited - Too many requests" };
  }

  if (!tweetData.data) {
    return { error: "No tweet data returned", response: tweetData };
  }

  const tweet = tweetData.data;
  let author = tweetData.includes?.users?.find(u => u.id === tweet.author_id);
  
  if (!author && tweet.author_id) {
    author = await getUserById(tweet.author_id);
  }
  
  const media = tweetData.includes?.media || [];

  const tweetContext = {
    author: author?.username,
    text: tweet.text,
    created_at: tweet.created_at,
  };

  let analysisResult = null;

  // Step 2: Process media
  for (const mediaItem of media) {
    if (mediaItem.type === "video") {
      const videoUrl = getVideoMp4Url(mediaItem);
      if (!videoUrl) continue;

      const buffer = await downloadVideo(videoUrl);
      analysisResult = await analyzeVideoWithGemini(buffer, tweetContext);

    } else if (mediaItem.type === "photo") {
      const imageUrl = mediaItem.url;
      const buffer = await downloadImage(imageUrl);
      analysisResult = await analyzeImageWithGemini(buffer, tweetContext);
    }
  }

  // If no media, analyze text only
  if (media.length === 0) {
    analysisResult = {
      User: "Text analysis request",
      post_caption: tweet.text,
      Gemini_Scan_Details: "No visual content - text only tweet",
      detail_analysis: "Cannot perform visual analysis on text-only content",
      is_potentially_misleading: null,
      confidence_score: 0,
    };
  }

  // Step 3: Return final JSON
  const finalOutput = {
    details: {
      User: analysisResult?.User || tweet.text,
      post_caption: analysisResult?.post_caption || tweet.text,
      few_comments: analysisResult?.few_comments || [],
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
      author_id: tweet.author_id,
      url: `https://x.com/${author?.username}/status/${tweet.id}`,
      created_at: tweet.created_at,
      public_metrics: tweet.public_metrics,
    },
    media_info: media.map(m => ({
      type: m.type,
      url: m.type === "video" ? getVideoMp4Url(m) : m.url,
      preview: m.preview_image_url,
    })),
    analyzed_at: new Date().toISOString(),
  };

  return finalOutput;
}

// ============== API ROUTES ==============

/**
 * Health check
 */
app.get("/", (req, res) => {
  res.json({
    name: "Pragmatic Fact-Check API",
    version: "1.0.0",
    status: "running",
    endpoints: {
      "GET /": "Health check",
      "GET /api/health": "API health status",
      "GET /api/analyze/:tweetId": "Analyze a tweet by ID",
      "POST /api/analyze": "Analyze a tweet (body: { tweetId })",
      "GET /api/tweet/:tweetId": "Get tweet details only",
      "GET /api/user/:userId": "Get user details by ID",
    },
  });
});

/**
 * API Health check
 */
app.get("/api/health", (req, res) => {
  res.json({
    status: "ok",
    twitter: API_KEY ? "configured" : "not configured",
    gemini: GEMINI_API_KEY ? "configured" : "not configured",
    timestamp: new Date().toISOString(),
  });
});

/**
 * Analyze tweet by ID (GET)
 */
app.get("/api/analyze/:tweetId", async (req, res) => {
  try {
    const { tweetId } = req.params;
    
    if (!tweetId) {
      return res.status(400).json({ error: "Tweet ID is required" });
    }

    console.log(`\n🔍 Analyzing tweet: ${tweetId}`);
    const result = await watcherAgent(tweetId);
    
    if (result.error) {
      return res.status(400).json(result);
    }

    res.json(result);
  } catch (error) {
    console.error("Error:", error.message);
    res.status(500).json({ error: error.message });
  }
});

/**
 * Analyze tweet by ID (POST)
 */
app.post("/api/analyze", async (req, res) => {
  try {
    const { tweetId, tweetUrl } = req.body;
    
    let id = tweetId;
    
    // Extract ID from URL if provided
    if (!id && tweetUrl) {
      const match = tweetUrl.match(/status\/(\d+)/);
      if (match) {
        id = match[1];
      }
    }
    
    if (!id) {
      return res.status(400).json({ 
        error: "Tweet ID or URL is required",
        example: {
          tweetId: "1234567890",
          tweetUrl: "https://x.com/user/status/1234567890"
        }
      });
    }

    console.log(`\n🔍 Analyzing tweet: ${id}`);
    const result = await watcherAgent(id);
    
    if (result.error) {
      return res.status(400).json(result);
    }

    res.json(result);
  } catch (error) {
    console.error("Error:", error.message);
    res.status(500).json({ error: error.message });
  }
});

/**
 * Get tweet details only (without Gemini analysis)
 */
app.get("/api/tweet/:tweetId", async (req, res) => {
  try {
    const { tweetId } = req.params;
    
    if (!tweetId) {
      return res.status(400).json({ error: "Tweet ID is required" });
    }

    const tweetData = await getTweetDetails(tweetId);
    
    if (tweetData.errors) {
      return res.status(400).json({ error: "Twitter API Error", details: tweetData.errors });
    }

    res.json(tweetData);
  } catch (error) {
    console.error("Error:", error.message);
    res.status(500).json({ error: error.message });
  }
});

/**
 * Get user details by ID
 */
app.get("/api/user/:userId", async (req, res) => {
  try {
    const { userId } = req.params;
    
    if (!userId) {
      return res.status(400).json({ error: "User ID is required" });
    }

    const userData = await getUserById(userId);
    
    if (!userData) {
      return res.status(404).json({ error: "User not found" });
    }

    res.json(userData);
  } catch (error) {
    console.error("Error:", error.message);
    res.status(500).json({ error: error.message });
  }
});

/**
 * Analyze tweet with custom prompt
 */
app.post("/api/analyze/custom", async (req, res) => {
  try {
    const { tweetId, customPrompt } = req.body;
    
    if (!tweetId) {
      return res.status(400).json({ error: "Tweet ID is required" });
    }

    // Get tweet details
    const tweetData = await getTweetDetails(tweetId);
    
    if (!tweetData.data) {
      return res.status(400).json({ error: "Could not fetch tweet" });
    }

    const tweet = tweetData.data;
    const media = tweetData.includes?.media || [];

    if (media.length === 0) {
      return res.status(400).json({ error: "No media found in tweet" });
    }

    // Use Gemini with custom prompt
    if (!genAI) {
      return res.status(500).json({ error: "Gemini not configured" });
    }

    const model = genAI.getGenerativeModel({ model: "gemini-2.0-flash" });
    
    const mediaItem = media[0];
    let buffer;
    let mimeType;

    if (mediaItem.type === "video") {
      const videoUrl = getVideoMp4Url(mediaItem);
      buffer = await downloadVideo(videoUrl);
      mimeType = "video/mp4";
    } else {
      buffer = await downloadImage(mediaItem.url);
      mimeType = "image/jpeg";
    }

    const base64 = buffer.toString("base64");
    const prompt = customPrompt || "Analyze this media and describe what you see.";

    const result = await model.generateContent([
      { inlineData: { mimeType, data: base64 } },
      prompt,
    ]);

    res.json({
      tweet_id: tweetId,
      prompt: prompt,
      response: result.response.text(),
      analyzed_at: new Date().toISOString(),
    });
  } catch (error) {
    console.error("Error:", error.message);
    res.status(500).json({ error: error.message });
  }
});

// ============== START SERVER ==============

app.listen(PORT, () => {
  console.log(`\n${"═".repeat(60)}`);
  console.log(`🚀 PRAGMATIC API SERVER`);
  console.log(`${"═".repeat(60)}`);
  console.log(`\n   Server running on: http://localhost:${PORT}`);
  console.log(`\n   Twitter API: ${API_KEY ? "✓ Configured" : "✗ Not configured"}`);
  console.log(`   Gemini API:  ${GEMINI_API_KEY ? "✓ Configured" : "✗ Not configured"}`);
  console.log(`\n${"─".repeat(60)}`);
  console.log(`   API ENDPOINTS:`);
  console.log(`${"─".repeat(60)}`);
  console.log(`   GET  /                      - Health check`);
  console.log(`   GET  /api/health            - API status`);
  console.log(`   GET  /api/analyze/:tweetId  - Analyze tweet`);
  console.log(`   POST /api/analyze           - Analyze tweet (JSON body)`);
  console.log(`   GET  /api/tweet/:tweetId    - Get tweet details`);
  console.log(`   GET  /api/user/:userId      - Get user details`);
  console.log(`   POST /api/analyze/custom    - Custom Gemini prompt`);
  console.log(`${"─".repeat(60)}`);
  console.log(`\n   Example:`);
  console.log(`   curl http://localhost:${PORT}/api/analyze/1234567890`);
  console.log(`\n${"═".repeat(60)}\n`);
});
