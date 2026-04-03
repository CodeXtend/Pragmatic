import json
from flask import Flask, request, jsonify
from flask_cors import CORS
from fact_extracter import FactExtracter
from decision_maker import DecisionMaker

app = Flask(__name__)
CORS(app)

# Initialize the FactExtracter and DecisionMaker agents
extracter = FactExtracter()
decision_maker = DecisionMaker()


@app.route("/", methods=["GET"])
def health_check():
    """Health check endpoint."""
    return jsonify({"status": "ok", "message": "Pragmatic API is running"})


def extract_message_from_json(data):
    """
    Extract the message/query from various JSON formats.
    Handles unpredictable JSON structures from Agent 1.
    
    Args:
        data: The JSON data (dict, list, or string)
        
    Returns:
        A string representation of the message/query
    """
    if data is None:
        return None
    
    # If it's already a string, return it
    if isinstance(data, str):
        return data.strip() if data.strip() else None
    
    # If it's a list, try to extract content or convert to string
    if isinstance(data, list):
        # Try to find a meaningful item in the list
        for item in data:
            extracted = extract_message_from_json(item)
            if extracted:
                return extracted
        # If nothing found, convert the whole list to string
        return json.dumps(data, indent=2)
    
    # If it's a dict, try common keys first
    if isinstance(data, dict):
        # List of common keys that might contain the message (in priority order)
        common_keys = [
            "message", "msg", "query", "question", "text", "content",
            "input", "prompt", "claim", "statement", "body",
            "request", "user_input", "user_message", "payload"
        ]
        
        # First, try to find an exact match (case-insensitive)
        lower_keys = {k.lower(): k for k in data.keys()}
        for key in common_keys:
            if key in lower_keys:
                value = data[lower_keys[key]]
                if isinstance(value, str) and value.strip():
                    return value.strip()
                elif isinstance(value, (dict, list)):
                    extracted = extract_message_from_json(value)
                    if extracted:
                        return extracted
        
        # If this looks like structured analysis data (has keys like 'details', 'tweet_info', 
        # 'visual_summary', 'Gemini_Scan_Details', etc.), convert the entire thing to a 
        # formatted string for the agent to process
        analysis_indicators = [
            "details", "tweet_info", "visual_summary", "Gemini_Scan_Details",
            "detail_analysis", "is_potentially_misleading", "confidence_score",
            "post_caption", "scene_description", "objects_detected", "analyzed_at"
        ]
        
        lower_data_keys = [k.lower() for k in data.keys()]
        if any(indicator.lower() in lower_data_keys for indicator in analysis_indicators):
            # This is structured analysis data - format it nicely for the agent
            return format_analysis_data(data)
        
        # If no common key found, try the first substantial string value
        for key, value in data.items():
            if isinstance(value, str) and len(value.strip()) > 10:  # Require meaningful content
                return value.strip()
        
        # If still nothing, try nested extraction on dict/list values
        for value in data.values():
            if isinstance(value, (dict, list)):
                extracted = extract_message_from_json(value)
                if extracted and len(extracted) > 10:
                    return extracted
        
        # Last resort: convert the entire dict to a readable string
        return json.dumps(data, indent=2)
    
    # For any other type, convert to string
    return str(data)


def format_analysis_data(data):
    """
    Format structured analysis data into a readable string for the agent.
    
    Args:
        data: Dictionary containing analysis data from Agent 1
        
    Returns:
        A formatted string representation of the analysis
    """
    parts = []
    
    # Extract details section
    details = data.get("details", data)
    
    if isinstance(details, dict):
        if "User" in details or "author" in details:
            parts.append(f"Author: {details.get('User') or details.get('author', 'Unknown')}")
        
        if "post_caption" in details:
            parts.append(f"Post Caption: {details['post_caption']}")
        
        if "Gemini_Scan_Details" in details:
            parts.append(f"Image Analysis: {details['Gemini_Scan_Details']}")
        
        if "Person_talking_about" in details:
            parts.append(f"Audio/Speech: {details['Person_talking_about']}")
        
        # Visual summary
        visual = details.get("visual_summary", {})
        if isinstance(visual, dict):
            if visual.get("objects_detected"):
                parts.append(f"Objects Detected: {', '.join(visual['objects_detected']) if isinstance(visual['objects_detected'], list) else visual['objects_detected']}")
            if visual.get("scene_description"):
                parts.append(f"Scene: {visual['scene_description']}")
            if visual.get("text_overlays") and visual.get("text_overlays") != "None":
                parts.append(f"Text Overlays: {visual['text_overlays']}")
            if visual.get("editing_signs") and visual.get("editing_signs") != "None":
                parts.append(f"Editing Signs: {visual['editing_signs']}")
        
        if "detail_analysis" in details:
            parts.append(f"Analysis: {details['detail_analysis']}")
        
        if "is_potentially_misleading" in details:
            parts.append(f"Potentially Misleading: {details['is_potentially_misleading']}")
        
        if "confidence_score" in details:
            parts.append(f"Confidence Score: {details['confidence_score']}")
    
    # Tweet/post info
    tweet_info = data.get("tweet_info", {})
    if isinstance(tweet_info, dict):
        if tweet_info.get("url"):
            parts.append(f"Source URL: {tweet_info['url']}")
        if tweet_info.get("created_at"):
            parts.append(f"Posted At: {tweet_info['created_at']}")
    
    if data.get("analyzed_at"):
        parts.append(f"Analyzed At: {data['analyzed_at']}")
    
    # If we extracted meaningful parts, join them
    if parts:
        return "Social Media Post Analysis:\n" + "\n".join(parts)
    
    # Fallback to JSON dump
    return json.dumps(data, indent=2)


@app.route("/api/query", methods=["POST"])
def query():
    """
    Process a user query and return the agent's response.
    
    Request body:
        Accepts flexible JSON formats including:
        - {"message": "Your question here"}
        - {"query": "Your question here"}
        - {"text": "Your question here"}
        - {"content": "Your question here"}
        - Or any nested JSON structure
    
    Response:
        {
            "response": "Agent's response",
            "success": true
        }
    """
    data = request.get_json()
    
    if not data:
        return jsonify({"error": "Missing request body", "success": False}), 400
    
    user_input = extract_message_from_json(data)
    
    if not user_input or (isinstance(user_input, str) and len(user_input.strip()) == 0):
        return jsonify({"error": "Could not extract message from request body", "success": False}), 400
    
    try:
        # Agent 1: Extract facts and gather evidence
        fact_data = extracter.run(user_input)
        
        # Agent 2: Make final decision based on the extracted facts
        decision = decision_maker.make_decision(fact_data, user_input)
        
        return jsonify({"response": decision, "success": True})
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


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)