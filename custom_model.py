import logging
from pydantic_ai.models.openai import OpenAIModel
from typing import Any, Dict, List, Optional

class OpenRouterModel(OpenAIModel):
    def __init__(self, model_id: str, **kwargs):
        super().__init__(model_id, **kwargs)
        self.model_id = model_id

    async def generate(self, messages, **kwargs):
        """Override the generate method to handle the OpenRouter response format"""
        response = await self.provider.chat_completion_create(
            model=self.model_id,
            messages=messages,
            **kwargs
        )
        
        # Extract the content from the response
        choices = getattr(response, 'choices', [])
        if not choices:
            raise ValueError("No choices in response")
        
        content = choices[0].message.content if choices[0].message else ""
        
        # Return the content directly - let the parent class handle formatting
        return content