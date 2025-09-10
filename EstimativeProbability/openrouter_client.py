"""
OpenRouter API client for probability estimation tasks.
Based on OpenRouter documentation: https://openrouter.ai/docs/quickstart#using-the-openrouter-api-directly
"""

import os
import json
import requests
from typing import Dict, List, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class OpenRouterClient:
    """Client for interacting with OpenRouter API."""
    
    def __init__(self, api_key: Optional[str] = None, site_url: Optional[str] = None, site_name: Optional[str] = None):
        """
        Initialize OpenRouter client.
        
        Args:
            api_key: OpenRouter API key (defaults to OPENROUTER_API_KEY env var)
            site_url: Site URL for rankings (defaults to YOUR_SITE_URL env var)
            site_name: Site name for rankings (defaults to YOUR_SITE_NAME env var)
        """
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        self.site_url = site_url or os.getenv("YOUR_SITE_URL", "")
        self.site_name = site_name or os.getenv("YOUR_SITE_NAME", "")
        self.base_url = "https://openrouter.ai/api/v1"
        
        if not self.api_key:
            raise ValueError("OpenRouter API key is required. Set OPENROUTER_API_KEY environment variable.")
    
    def _get_headers(self) -> Dict[str, str]:
        """Get headers for API requests."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # Optional headers for site attribution
        if self.site_url:
            headers["HTTP-Referer"] = self.site_url
        if self.site_name:
            headers["X-Title"] = self.site_name
            
        return headers
    
    def chat_completion(self, messages: List[Dict[str, str]], model: str = "openai/gpt-4o", **kwargs) -> Dict:
        """
        Make a chat completion request.
        
        Args:
            messages: List of message dicts with 'role' and 'content' keys
            model: Model to use (default: openai/gpt-4o)
            **kwargs: Additional parameters for the API
            
        Returns:
            API response as dictionary
        """
        url = f"{self.base_url}/chat/completions"
        
        data = {
            "model": model,
            "messages": messages,
            **kwargs
        }
        
        response = requests.post(
            url=url,
            headers=self._get_headers(),
            data=json.dumps(data)
        )
        
        response.raise_for_status()
        return response.json()
    
    def estimate_probability(self, question: str, model: str = "openai/gpt-4o") -> str:
        """
        Use AI to estimate probability for a given question.
        
        Args:
            question: The probability question to ask
            model: Model to use for estimation
            
        Returns:
            AI response with probability estimation
        """
        messages = [
            {
                "role": "system",
                "content": "You are an expert in probability theory and statistical analysis. Provide clear, well-reasoned probability estimates with explanations."
            },
            {
                "role": "user",
                "content": f"Please estimate the probability and explain your reasoning: {question}"
            }
        ]
        
        response = self.chat_completion(messages, model)
        return response["choices"][0]["message"]["content"]


# Example usage
if __name__ == "__main__":
    try:
        client = OpenRouterClient()
        
        # Example probability question
        question = "What is the probability of rolling a sum of 7 with two fair six-sided dice?"
        result = client.estimate_probability(question)
        
        print(f"Question: {question}")
        print(f"AI Response: {result}")
        
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure to set up your .env file with OPENROUTER_API_KEY")
