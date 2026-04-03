import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from smolagents import CodeAgent, LiteLLMModel, tool
from dotenv import load_dotenv
from typing import Optional, Dict, Any
from dataclasses import dataclass
import json


@dataclass
class FactDecision:
    """Data class to hold the final fact-check decision."""
    fact: str
    analysis: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "details": {
                "fact": self.fact,
                "analysis": self.analysis
            }
        }
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)


@tool
def format_decision(fact_status: str, analysis_text: str) -> str:
    """
    Format the final fact-check decision into a structured output.
    
    Args:
        fact_status: The verdict - either 'True' or 'Fake' indicating if the claim is true or false
        analysis_text: Detailed analysis including the reasoning and sources
        
    Returns:
        A JSON formatted string with the decision details
    """
    decision = FactDecision(
        fact=f"This is {fact_status}",
        analysis=analysis_text
    )
    return decision.to_json()


class DecisionMaker:
    """
    An agent that analyzes fact-check data and makes final decisions.
    
    This agent takes the output from fact extraction and verification tools,
    analyzes the information, and produces a final structured decision
    about whether a claim is true or fake.
    """
    
    def __init__(self, model_id: Optional[str] = None, api_key: Optional[str] = None, max_steps: int = 3):
        """
        Initialize the DecisionMaker agent.
        
        Args:
            model_id: The LLM model ID. If None, reads from LLM_MODEL env variable.
            api_key: The API key for the model. If None, reads from GEMINI_API_KEY env variable.
            max_steps: Maximum number of steps the agent can take (default: 3).
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
            tools=[format_decision],
            model=self.model,
            name="decision_maker",
            max_steps=self.max_steps,
        )
        
        self.system_prompt = """You are a fact-checking decision agent. Your role is to:
1. Analyze the provided fact-check data and evidence
2. Determine if the claim is TRUE or FAKE based on the evidence
3. Provide a clear analysis with sources

When making a decision, consider:
- The credibility of sources
- The consistency of evidence
- The recency of information
- Expert opinions and official statements

Always use the format_decision tool to output your final decision with:
- fact_status: 'True' if the claim is verified as true, 'Fake' if it's false/misleading
- analysis_text: A detailed explanation with sources cited

Example output format:
{
  "details": {
    "fact": "This is Fake",
    "analysis": "Drinking hot water does not prevent COVID-19. Sources: WHO states there is no evidence that hot water prevents coronavirus infection. CDC confirms that only vaccines and proper hygiene measures are effective preventive measures."
  }
}
"""

    def make_decision(self, fact_data: str, claim: str) -> Dict[str, Any]:
        """
        Analyze fact-check data and make a final decision.
        
        Args:
            fact_data: The collected fact-check data and evidence.
            claim: The original claim being verified.
            
        Returns:
            A dictionary containing the decision details.
        """
        prompt = f"""{self.system_prompt}

Original Claim: {claim}

Fact-Check Data and Evidence:
{fact_data}

Analyze the above information and make a final decision. Use the format_decision tool to output your structured decision."""

        result = self.agent.run(prompt)
        
        # Parse the result
        if hasattr(result, 'content'):
            response = result.content
        elif isinstance(result, dict) and 'output' in result:
            response = result['output']
        elif isinstance(result, dict) and 'content' in result:
            response = result['content']
        else:
            response = str(result)
        
        # Try to parse as JSON if it's a string
        try:
            if isinstance(response, str):
                # Find JSON in the response
                start_idx = response.find('{')
                end_idx = response.rfind('}') + 1
                if start_idx != -1 and end_idx > start_idx:
                    json_str = response[start_idx:end_idx]
                    return json.loads(json_str)
            return {"details": {"fact": "Unable to determine", "analysis": response}}
        except json.JSONDecodeError:
            return {"details": {"fact": "Unable to determine", "analysis": response}}
    
    def decide(self, fact_data: str, claim: str) -> str:
        """
        Make a decision and return as formatted JSON string.
        
        Args:
            fact_data: The collected fact-check data and evidence.
            claim: The original claim being verified.
            
        Returns:
            A JSON formatted string with the decision.
        """
        decision = self.make_decision(fact_data, claim)
        return json.dumps(decision, indent=2)


# Example usage
if __name__ == "__main__":
    # Initialize the decision maker
    decision_maker = DecisionMaker()
    
    # Example fact data (would normally come from FactExtracter)
    example_claim = "Drinking hot water prevents COVID-19"
    example_fact_data = """
    Source 1 (WHO): There is no evidence that drinking hot water can prevent or cure COVID-19.
    Hot water does not kill the virus inside your body.
    
    Source 2 (CDC): COVID-19 prevention includes vaccination, wearing masks, and hand hygiene.
    There is no scientific evidence supporting hot water consumption as a preventive measure.
    
    Source 3 (Reuters Fact Check): This claim has been debunked. The virus cannot be killed
    by drinking hot beverages as it infects the respiratory system.
    """
    
    # Make the decision
    result = decision_maker.decide(example_fact_data, example_claim)
    print(result)
