import os
from typing import List, Dict, Any
from openai import OpenAI

class SimpleChatManager:
    """Tiny wrapper around OpenAI Chat Completions. No vector store. No init."""
    def __init__(self, system_prompt: str = "You are a helpful assistant.", model: str = "gpt-4o-mini", temperature: float = 0.2):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.system_prompt = system_prompt
        self.model = model
        self.temperature = temperature
        self.history: List[Dict[str, str]] = [{"role": "system", "content": self.system_prompt}]

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
