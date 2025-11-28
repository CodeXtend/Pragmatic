import requests
from smolagents import Tool
from typing import Optional
import os

class GoogleFactCheckTool(Tool):
    name = "google_fact_check"
    description = "Search for fact-checks on a given claim using Google Fact Check Tools API."
    inputs = {
        "query": {
            "type": "string",
            "description": "The claim or statement to fact-check"
        }
    }
    output_type = "string"

    def __init__(self):
        super().__init__()
        self.api_key = os.getenv("GOOGLE_FACT_CHECK_API_KEY")
       
    def forward(self, query: str) -> str:
        """
        Search for fact-checks on the given query.

        Args:
            query: The claim to fact-check

        Returns:
            A string containing the fact-check results
        """
        url = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
        params = {
            "query": query,
            "key": self.api_key
        }

        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()

            return data
        except requests.exceptions.RequestException as e:
            return f"Error querying Google Fact Check API: {str(e)}"
        except Exception as e:
            return f"Error processing fact-check results: {str(e)}"