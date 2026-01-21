import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from smolagents import CodeAgent, LiteLLMModel, DuckDuckGoSearchTool
from dotenv import load_dotenv
from typing import List, Optional, Dict, Any
from datetime import datetime
import json
import re
from tools.google_factcheck_tool import GoogleFactCheckTool


class FactExtracter:
    """
    A conversational agent for fact extraction and verification using LLM and search tools.
    
    This class provides a reusable interface to create an agent that can search for information
    and extract facts using DuckDuckGo search and an LLM model.
    """
    
    def __init__(self, model_id: Optional[str] = None, api_key: Optional[str] = None, max_steps: int = 4):
        """
        Initialize the FactExtracter agent.
        
        Args:
            model_id: The LLM model ID. If None, reads from LLM_MODEL env variable.
            api_key: The API key for the model. If None, reads from GEMINI_API_KEY env variable.
            max_steps: Maximum number of steps the agent can take (default: 4 for optimization).
        """
        load_dotenv()
        
        self.model_id = model_id or os.getenv("LLM_MODEL")
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.max_steps = max_steps
        
        if not self.model_id or not self.api_key:
            raise ValueError("Model ID and API key must be provided either as arguments or environment variables")
        
        self.model = LiteLLMModel(
            model_id=self.model_id,
            api_key=self.api_key
        )
        
        self.agent = CodeAgent(
            tools=[DuckDuckGoSearchTool(), GoogleFactCheckTool()],
            model=self.model,
            name="fact_extracter",
            max_steps=self.max_steps,
        )
        system_prompt = "system_prompt: You are a fact extraction agent. Always use the current date provided to verify if information is up-to-date. For any facts, ensure they are current as of the given date. Return long Structured fact data with all the references.\n\n"
        self.memory: List[str] = [system_prompt]
    
    def run(self, user_input: str) -> str:
        """
        Process a user input and return the agent's response.
        
        Args:
            user_input: The user's question or statement.
            
        Returns:
            The agent's response as a string.
        """
        self.memory.append(f"user: {user_input}")
        
        current_date = datetime.now().strftime("%Y-%m-%d")
        prompt = f"Current date: {current_date}\n\n" + "\n".join(self.memory)
        
        result = self.agent.run(prompt)
        
        # Extract the final answer string from the agent result
        if hasattr(result, 'content'):
            response = result.content
        elif isinstance(result, dict) and 'output' in result:
            response = result['output']
        elif isinstance(result, dict) and 'content' in result:
            response = result['content']
        else:
            response = str(result)
        
        self.memory.append(f"assistant: {response}")
        
        return response

    def fast_check(self, claim: str) -> Dict[str, Any]:
        """
        Fast fact-check verification without agent loop (skips ReAct).
        Uses a SINGLE LLM call to save quota.
        """
        # 1. Search directly (No LLM cost)
        search_tool = DuckDuckGoSearchTool()
        try:
            # Search for the claim
            search_results = search_tool.forward(f"{claim} fact check")
        except Exception as e:
            search_results = f"Search failed: {e}"
        
        # 2. Single LLM Call
        prompt = f"""You are a strict and skeptical fact-checking AI. Your job is to verify claims using ONLY the provided search evidence.

CLAIM: "{claim}"

EVIDENCE FROM SEARCH:
{search_results}

INSTRUCTIONS:
1. If the search results say the claim is FALSE, DEBUNKED, A RUMOR, or SATIRE, your verdict MUST be "False".
2. If the search results do NOT contain reputable news reports confirming the event, your verdict MUST be "False" or "Unverified".
3. Do NOT use your internal knowledge to validate breaking news or recent events. If it's not in the evidence, it's not verified.
4. If the evidence says "no reports found" or is unrelated, mark as "False" or "Unverified".
5. Be aggressive in flagging fake news.

Return ONLY a JSON object:
{{
    "verdict": "Verified" | "False" | "Unverified",
    "analysis": "<explanation citing the evidence>"
}}"""
        
        try:
            # Direct model call
            messages = [{"role": "user", "content": prompt}]
            response = self.model(messages)
            
            # Extract content
            content = ""
            if hasattr(response, 'content'):
                content = response.content
            elif isinstance(response, dict):
                content = response.get('content', str(response))
            else:
                content = str(response)
                
            # Parse JSON
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(0))
            else:
                return {"verdict": "Unverified", "analysis": content}
                
        except Exception as e:
            return {"verdict": "Error", "analysis": f"Fact-check error: {str(e)}"}

    def clear_memory(self):
        """Clear the conversation memory."""
        self.memory = []
    
    def get_memory(self) -> List[str]:
        """
        Get the current conversation memory.
        
        Returns:
            List of conversation history entries.
        """
        return self.memory.copy()
