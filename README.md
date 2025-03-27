# VideoEditingAgent

An AI-powered video editing assistant that combines the pydantic-ai agentic framework with MoviePy for advanced video editing capabilities.

## Features

- AI-powered video editing assistance
- Supports multiple video editing operations:
  - Trimming videos
  - Applying effects (fade in, fade out, speed change, etc.)
  - Concatenating multiple videos
  - Adding text overlays
  - Converting video formats
  - Extracting audio from videos

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd videoEditingAgent

# Set up a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies using uv
uv pip install -e .
```

## Usage

### Basic Usage

```python
from pydantic_ai import Agent
from tools.moviepy_tool import MoviePyTool

# Initialize the agent
agent = Agent(your_model)
agent.add_tool(MoviePyTool())

# Use the agent for video editing tasks
response = agent.chat("Trim the video 'input.mp4' to be only 10 seconds long")
print(response)
```

### Available Operations

The MoviePyTool provides these main operations:

1. **load_video**: Load and get information about a video file
2. **trim_video**: Cut a video to a specific time range
3. **apply_effect**: Apply visual effects to videos
4. **concatenate_videos**: Join multiple videos together
5. **extract_audio**: Extract the audio track from a video
6. **add_text**: Add text overlays to videos
7. **convert_format**: Convert videos between different formats

## Examples

### Trimming a Video

```python
response = agent.chat(
    "Please trim the video 'my_video.mp4' to start at 10 seconds and end at 30 seconds. Save it as 'trimmed_video.mp4'."
)
```

### Adding Text Overlay

```python
response = agent.chat(
    "Add the text 'Hello World' to the center of 'input.mp4' starting at 2 seconds until 5 seconds. Save as 'text_video.mp4'."
)
```

### Applying Effects

```python
response = agent.chat(
    "Apply a fade out effect to the last 3 seconds of 'input.mp4' and save as 'fadeout_video.mp4'."
)
```

## Requirements

- Python 3.12 or higher
- MoviePy 2.1.2
- pydantic-ai 0.0.43 or higher
- pydantic-ai-slim[mcp] 0.0.43 or higher

Note: We use [uv](https://github.com/astral/uv) for dependency management instead of pip for better performance and reliability.
