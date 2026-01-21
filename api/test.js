import fetch from "node-fetch";
import dotenv from "dotenv";
import crypto from "crypto";
import OAuth from "oauth-1.0a";
import fs from "fs";
import path from "path";
import { GoogleGenerativeAI } from "@google/generative-ai";

dotenv.config();

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

// ============== GEMINI VIDEO ANALYSIS ==============

/**
 * Analyze video with Gemini 2.0 Flash
 * Extracts: visual content, text on screen, speech transcription
 */
async function analyzeVideoWithGemini(videoBuffer, tweetData) {
  if (!genAI) {
    console.log("❌ GEMINI_API_KEY not found in .env");
    return null;
  }

  console.log("\n🤖 Analyzing video with Gemini 2.0 Flash...");

  const model = genAI.getGenerativeModel({ model: "gemini-2.0-flash" });

  // Convert video buffer to base64
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

    // Parse JSON from response
    try {
      // Extract JSON from response (handle markdown code blocks)
      let jsonStr = responseText;
      const jsonMatch = responseText.match(/```json\n?([\s\S]*?)\n?```/);
      if (jsonMatch) {
        jsonStr = jsonMatch[1];
      } else {
        // Try to find JSON object directly
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
 * Analyze image with Gemini (for photo tweets)
 */
async function analyzeImageWithGemini(imageUrl, tweetData) {
  if (!genAI) {
    console.log("❌ GEMINI_API_KEY not found in .env");
    return null;
  }

  console.log("\n🤖 Analyzing image with Gemini 2.0 Flash...");

  const model = genAI.getGenerativeModel({ model: "gemini-2.0-flash" });

  // Download image
  const response = await fetch(imageUrl);
  const imageBuffer = await response.buffer();
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

// ============== MAIN WATCHER AGENT ==============

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

      analysisResult = await analyzeImageWithGemini(imageUrl, tweetContext);
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

// ============== RUN ==============

// Check for Gemini API key
if (!GEMINI_API_KEY) {
  console.log("\n⚠️  GEMINI_API_KEY not found in .env file!");
  console.log("   Add this to your .env file:");
  console.log("   GEMINI_API_KEY=your_gemini_api_key_here\n");
  console.log("   Get your key from: https://aistudio.google.com/app/apikey\n");
}

// Run the Watcher Agent on a tweet
watcherAgent("1994639065220878767");