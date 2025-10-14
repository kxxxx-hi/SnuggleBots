import os
from typing import List, Dict, Any, Tuple
from openai import OpenAI
from pet_data import PetRecommendationSystem

class SimpleChatManager:
    """Tiny wrapper around OpenAI Chat Completions. No vector store. No init."""
    def __init__(self, system_prompt: str = "You are a helpful assistant.", model: str = "gpt-4o-mini", temperature: float = 0.2):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set")
        self.client = OpenAI(api_key=api_key)
        self.system_prompt = system_prompt
        self.model = model
        self.temperature = temperature
        self.history: List[Dict[str, str]] = [{"role": "system", "content": self.system_prompt}]
        self.pet_system = PetRecommendationSystem()

    def set_config(self, model: str = None, temperature: float = None):
        if model is not None:
            self.model = model
        if temperature is not None:
            self.temperature = temperature

    def ask(self, user_text: str) -> str:
        self.history.append({"role": "user", "content": user_text})
        resp = self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            messages=self.history[-20:]  # keep short
        )
        answer = resp.choices[0].message.content if resp and resp.choices else ""
        self.history.append({"role": "assistant", "content": answer})
        return answer
    
    def ask_with_pets(self, user_text: str) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Ask a question and return both text response and recommended pets
        Returns: (text_response, list_of_pets)
        """
        # Check if the query is asking for pet recommendations
        pet_keywords = [
            "adopt", "pet", "dog", "cat", "puppy", "kitten", "animal", 
            "recommend", "suggest", "available", "looking for", "want"
        ]
        
        query_lower = user_text.lower()
        is_pet_query = any(keyword in query_lower for keyword in pet_keywords)
        
        # Get regular text response
        text_response = self.ask(user_text)
        
        # Get pet recommendations if it's a pet-related query
        recommended_pets = []
        if is_pet_query:
            recommended_pets = self.pet_system.search_pets(user_text, max_results=3)
        
        return text_response, recommended_pets
