from pydantic_ai import Agent
from pydantic_ai.providers.openai import OpenAIProvider
from tools.moviepy_tool import moviepy_tools, VideoEffect, VideoFormat, TextPosition
from custom_model import OpenRouterModel

import os
import asyncio
import inspect
from dotenv import load_dotenv
import logging
from openai import AsyncOpenAI
from typing import List, Dict, Optional

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

class OpenRouterProvider(OpenAIProvider):
    async def chat_completion_create(self, **kwargs):
        kwargs["temperature"] = 0.7
        kwargs["max_tokens"] = 2000
        kwargs["stream"] = False
        
        logging.info(f"Creating chat completion with kwargs: {kwargs}")
        
        client = AsyncOpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
            default_headers={
                "HTTP-Referer": "https://videoeditor.example.com",
                "X-Title": "Video Editing Agent"
            }
        )
        
        try:
            response = await client.chat.completions.create(**kwargs)
            logging.info(f"OpenRouter response type: {type(response)}")
            
            from time import time
            response.created = getattr(response, 'created', None) or int(time())
            
            if hasattr(response, 'choices') and response.choices:
                message_content = response.choices[0].message.content
                logging.info(f"Message content: {message_content[:100]}...")
            return response
        except Exception as e:
            logging.error(f"Error in chat completion: {str(e)}")
            raise e

class ChatSession:
    def __init__(self, agent: Agent):
        self.agent = agent
        self.conversation_history: List[Dict[str, str]] = []
        self.available_commands = {
            'help': self.show_help,
            'list': self.list_tools,
            'tool': self.show_tool_help,
            'effects': self.list_effects,
            'formats': self.list_formats
        }

    def add_message(self, role: str, content: str):
        self.conversation_history.append({"role": role, "content": content})

    def format_message(self, content: str) -> str:
        return f"\n{content}\n"

    def list_tools(self) -> str:
        """List all available video editing tools and effects."""
        output = ["Available Video Editing Tools:"]
        
        # List main tools
        for tool in moviepy_tools:
            doc = tool.__doc__ or "No description available"
            doc = doc.strip().split('\n')[0]  # Get first line of docstring
            output.append(f"- {tool.__name__}: {doc}")
        
        # Add a blank line for readability
        output.append("")
        
        # List available effects
        output.append("Available Video Effects:")
        for effect in VideoEffect:
            effect_name = effect.name.replace('_', ' ').title()
            output.append(f"- {effect_name}: {effect.value}")
            
        # Add information about effect parameters
        output.append("\nEffect Parameters:")
        output.append("- duration: Time in seconds for fade effects")
        output.append("- factor: Speed multiplier for speedx effect (e.g., 2.0 = 2x speed)")
        output.append("- width/height: Dimensions for resize effect")
        output.append("- angle: Degrees for rotate effect")

        return '\n'.join(output)

    def show_tool_help(self, tool_name: Optional[str] = None) -> str:
        """Show detailed help for a specific tool."""
        if not tool_name:
            return "Usage: tool <tool_name> (e.g., 'tool trim_video')"

        for tool in moviepy_tools:
            if tool.__name__ == tool_name:
                doc = tool.__doc__ or "No documentation available"
                params = inspect.signature(tool).parameters
                param_docs = ["Parameters:"]
                for name, param in params.items():
                    annotation = param.annotation.__name__ if hasattr(param.annotation, '__name__') else str(param.annotation)
                    default = "" if param.default == inspect.Parameter.empty else f" (default: {param.default})"
                    param_docs.append(f"- {name}: {annotation}{default}")
                return f"{doc}\n\n" + '\n'.join(param_docs)
        return f"Tool '{tool_name}' not found"

    def list_effects(self) -> str:
        """List all available video effects."""
        effects = ["Available Video Effects:"]
        for effect in VideoEffect:
            effect_name = effect.name.replace('_', ' ').title()
            effects.append(f"- {effect_name}: {effect.value}")
        
        effects.append("\nEffect Parameters:")
        effects.append("- duration: Time in seconds for fade effects")
        effects.append("- factor: Speed multiplier for speedx effect (e.g., 2.0 = 2x speed)")
        effects.append("- width/height: Dimensions for resize effect")
        effects.append("- angle: Degrees for rotate effect")
        
        return '\n'.join(effects)

    def list_formats(self) -> str:
        """List all supported video formats."""
        formats = ["Supported Video Formats:"]
        for format in VideoFormat:
            formats.append(f"- {format.name}: {format.value}")
        return '\n'.join(formats)

    def show_help(self) -> str:
        """Show help information."""
        help_text = [
            "Video Editing Assistant Commands:",
            "- help: Show this help message",
            "- list: Show all available video editing tools and effects",
            "- tool <name>: Show detailed help for a specific tool",
            "- effects: List all available video effects",
            "- formats: List supported video formats",
            "- exit/quit: Exit the application",
            "\nExample commands:",
            "- tool trim_video",
            "- effects",
            "\nFor video editing, simply describe what you want to do with the video!"
        ]
        return '\n'.join(help_text)

    async def process_user_input(self, user_input: str) -> str:
        try:
            # Check if input is a command
            command_parts = user_input.lower().split()
            if command_parts[0] in self.available_commands:
                command = self.available_commands[command_parts[0]]
                args = command_parts[1:] if len(command_parts) > 1 else []
                return self.format_message(command(*args))

            # If not a command, process as normal request
            self.add_message("user", user_input)
            result = await self.agent.run(user_input)
            
            response_content = ""
            if hasattr(result, 'output'):
                response_content = result.output
            elif hasattr(result, 'content'):
                response_content = result.content
            else:
                response_content = str(result)

            self.add_message("assistant", response_content)
            return self.format_message(response_content)
        except Exception as e:
            error_message = f"Error processing your request: {str(e)}"
            logging.error(error_message)
            return self.format_message(error_message)

async def interactive_chat():
    provider = OpenRouterProvider(
        base_url=api_base,
        api_key=api_key
    )

    model = OpenRouterModel(
        model_id,
        provider=provider
    )

    agent = Agent(
        model=model,
        tools=moviepy_tools
    )

    chat_session = ChatSession(agent)

    print("\nWelcome to the Video Editing Assistant!")
    print("Type 'help' to see available commands.")
    print("Type 'exit' or 'quit' to end the conversation.")
    print("\nEnter your request or command:\n")

    while True:
        try:
            user_input = input("> ").strip()
            
            if user_input.lower() in ['exit', 'quit']:
                print("\nThank you for using the Video Editing Assistant. Goodbye!")
                break
            
            if not user_input:
                print("Please enter a valid request or command.")
                continue

            response = await chat_session.process_user_input(user_input)
            print(response)

        except KeyboardInterrupt:
            print("\nReceived interrupt signal. Shutting down...")
            break
        except Exception as e:
            print(f"\nAn error occurred: {str(e)}")
            print("Please try again or type 'exit' to quit.")

if __name__ == "__main__":
    try:
        asyncio.run(interactive_chat())
    except Exception as e:
        logging.error(f"Application error: {str(e)}")
        print("\nThe application encountered an error and needs to close.")