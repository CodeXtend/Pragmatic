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


@app.route("/api/query", methods=["POST"])
def query():
    """
    Process a user query and return the agent's response.
    
    Request body:
        {
            "message": "Your question here"
        }
    
    Response:
        {
            "response": "Agent's response",
            "success": true
        }
    """
    data = request.get_json()
    
    if not data or "message" not in data:
        return jsonify({"error": "Missing 'message' field in request body", "success": False}), 400
    
    user_input = data["message"]
    
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