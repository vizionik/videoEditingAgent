from pydantic_ai import Agent
from pydantic_ai.providers.openai import OpenAIProvider
from tools.moviepy_tool import moviepy_tools
from custom_model import OpenRouterModel

import os
import sys
import asyncio
import inspect
from dotenv import load_dotenv
import logging
from openai import AsyncOpenAI

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables from .env file
load_dotenv()

# Check if API key is available
if not os.getenv("OPENROUTER_API_KEY"):
    raise ValueError("OPENROUTER_API_KEY environment variable not found. Please add it to your .env file.")

# Get configuration from environment variables with fallbacks
model_id = os.getenv("OPENROUTER_MODEL_ID", "anthropic/claude-3-haiku")
api_key = os.getenv("OPENROUTER_API_KEY")
api_base = os.getenv("OPENROUTER_API_BASE", "https://openrouter.ai/api/v1")

# Create a custom provider for OpenRouter that properly handles its response format
class OpenRouterProvider(OpenAIProvider):
    async def chat_completion_create(self, **kwargs):
        # Add our desired parameters here instead of at model creation
        kwargs["temperature"] = 0.7
        kwargs["max_tokens"] = 2000
        kwargs["stream"] = False
        
        # Log the kwargs for debugging
        logging.info(f"Creating chat completion with kwargs: {kwargs}")
        
        # Use the AsyncOpenAI client directly with proper configuration
        client = AsyncOpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
            default_headers={
                "HTTP-Referer": "https://videoeditor.example.com",
                "X-Title": "Video Editing Agent"
            }
        )
        
        # Make the request with proper parameters
        try:
            response = await client.chat.completions.create(**kwargs)
            # Log response details to help debug
            logging.info(f"OpenRouter response type: {type(response)}")
            logging.info(f"OpenRouter response attributes: {dir(response)}")
            
            # Add current timestamp if not present
            from time import time
            response.created = getattr(response, 'created', None) or int(time())
            
            # Log the actual content for debugging
            if hasattr(response, 'choices') and response.choices:
                message_content = response.choices[0].message.content
                logging.info(f"Message content: {message_content[:100]}...")
            return response
        except Exception as e:
            logging.error(f"Error in chat completion: {str(e)}")
            raise e

# Create provider with more specific configuration
provider = OpenRouterProvider(
    base_url=api_base,
    api_key=api_key
)

# Create the model with our custom OpenRouter implementation
model = OpenRouterModel(
    model_id,
    provider=provider
)

# Create the agent without the verbose parameter
agent = Agent(
    model=model,
    tools=moviepy_tools
)

# Entry point
if __name__ == "__main__":
    # Get user prompt from command line arguments
    if len(sys.argv) < 2:
        print("Usage: python main.py \"your video editing request here\"")
        sys.exit(1)
    prompt = sys.argv[1]
    
    # Print the agent's run method details for debugging
    logging.info(f"Agent run method signature: {inspect.signature(agent.run)}")
    
    # Run the agent and display the result
    async def run_and_display():
        result = await agent.run(prompt)
        # Print the result to make it visible to the user
        if hasattr(result, 'output'):
            print("\nAgent Response:")
            print(result.output)
        elif hasattr(result, 'content'):
            print("\nAgent Response:")
            print(result.content)
        else:
            print("\nAgent Response:")
            print(result)
            
    asyncio.run(run_and_display())