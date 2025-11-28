from smolagents import CodeAgent, LiteLLMModel, DuckDuckGoSearchTool
from dotenv import load_dotenv
import os
from typing import List, Optional


class FactExtracter:
    """
    A conversational agent for fact extraction and verification using LLM and search tools.
    
    This class provides a reusable interface to create an agent that can search for information
    and extract facts using DuckDuckGo search and an LLM model.
    """
    
    def __init__(self, model_id: Optional[str] = None, api_key: Optional[str] = None, max_steps: int = 10):
        """
        Initialize the FactExtracter agent.
        
        Args:
            model_id: The LLM model ID. If None, reads from LLM_MODEL env variable.
            api_key: The API key for the model. If None, reads from GEMINI_API_KEY env variable.
            max_steps: Maximum number of steps the agent can take (default: 10).
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
            tools=[DuckDuckGoSearchTool()],
            model=self.model,
            name="fact_extracter",
            max_steps=self.max_steps,
        )
        
        self.memory: List[str] = []
    
    def run(self, user_input: str) -> str:
        """
        Process a user input and return the agent's response.
        
        Args:
            user_input: The user's question or statement.
            
        Returns:
            The agent's response as a string.
        """
        self.memory.append(f"user: {user_input}")
        
        prompt = "\n".join(self.memory)
        
        response = self.agent.run(prompt)
        
        self.memory.append(f"assistant: {response}")
        
        return response
    
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
