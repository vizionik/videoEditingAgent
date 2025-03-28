import logging
from pydantic_ai.models.openai import OpenAIModel
from typing import Any, Dict, List, Optional

class OpenRouterModel(OpenAIModel):
    def __init__(self, model_id: str, **kwargs):
        super().__init__(model_id, **kwargs)
        self.model_id = model_id

    async def generate(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """
        Generate a response using the OpenRouter API.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            **kwargs: Additional parameters for the API call
        
        Returns:
            str: The generated response content
        
        Raises:
            ValueError: If no choices are present in the response
        """
        try:
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
            
        except Exception as e:
            logging.error(f"Error generating response: {str(e)}")
            raise