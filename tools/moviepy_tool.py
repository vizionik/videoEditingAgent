from enum import Enum
from typing import List, Optional, Tuple, Union, Dict, Any, Literal
from pathlib import Path

from moviepy import *
from pydantic import Field, BaseModel
from pydantic_ai import Tool


class EffectParams(BaseModel):
    duration: Optional[float] = Field(None, description="Duration for fade effects")
    factor: Optional[float] = Field(None, description="Speed factor for speedx effect")
    width: Optional[int] = Field(None, description="Width for resize effect")
    height: Optional[int] = Field(None, description="Height for resize effect")
    angle: Optional[float] = Field(None, description="Angle for rotate effect")


class TextPosition(BaseModel):
    x: Literal['left', 'center', 'right'] = Field('center')
    y: Literal['top', 'center', 'bottom'] = Field('center')


class VideoEffect(str, Enum):
    FADEOUT = "fadeout"
    FADEIN = "fadein"
    CROSSFADEIN = "crossfadein"
    SPEEDX = "speedx"
    RESIZE = "resize"
    ROTATE = "rotate"
    MIRROR_X = "mirror_x"
    MIRROR_Y = "mirror_y"


class VideoFormat(str, Enum):
    MP4 = "mp4"
    GIF = "gif"
    AVI = "avi"
    MOV = "mov"
    WEBM = "webm"


def load_video(filepath: str) -> str:
    """
    Load a video file and return its information.
    """
    video = VideoFileClip(filepath)
    info = {
        "duration": video.duration,
        "fps": video.fps,
        "size": video.size,
        "filepath": filepath
    }
    video.close()
    return f"Video loaded successfully: {info}"


def trim_video(
    input_path: str, 
    output_path: str, 
    start_time: float = 0, 
    end_time: Optional[float] = None
) -> str:
    """
    Trim a video to the specified start and end times.
    """
    video = VideoFileClip(input_path)
    trimmed = video.subclip(start_time, end_time)
    trimmed.write_videofile(output_path)
    video.close()
    trimmed.close()
    return f"Video trimmed and saved to {output_path}"


def apply_effect(
    input_path: str,
    output_path: str,
    effect: VideoEffect,
    params: EffectParams = EffectParams()
) -> str:
    """
    Apply an effect to a video.
    """
    video = VideoFileClip(input_path)
    
    if effect == VideoEffect.FADEOUT:
        duration = params.duration or 1.0
        processed = video.fadeout(duration)
    elif effect == VideoEffect.FADEIN:
        duration = params.duration or 1.0
        processed = video.fadein(duration)
    elif effect == VideoEffect.SPEEDX:
        factor = params.factor or 2.0
        processed = video.speedx(factor)
    elif effect == VideoEffect.RESIZE:
        processed = video.resize(width=params.width, height=params.height)
    elif effect == VideoEffect.ROTATE:
        angle = params.angle or 90
        processed = video.rotate(angle)
    elif effect == VideoEffect.MIRROR_X:
        processed = video.fx(vfx.mirror_x)
    elif effect == VideoEffect.MIRROR_Y:
        processed = video.fx(vfx.mirror_y)
    else:
        video.close()
        return f"Effect {effect} not implemented."
        
    processed.write_videofile(output_path)
    video.close()
    processed.close()
    return f"Effect {effect} applied and saved to {output_path}"


def concatenate_videos(
    input_paths: List[str], 
    output_path: str
) -> str:
    """
    Concatenate multiple videos into one.
    """
    clips = [VideoFileClip(path) for path in input_paths]
    final_clip = concatenate_videoclips(clips)
    final_clip.write_videofile(output_path)
    for clip in clips:
        clip.close()
    final_clip.close()
    return f"Videos concatenated and saved to {output_path}"


def extract_audio(
    input_path: str, 
    output_path: str
) -> str:
    """
    Extract audio from a video.
    """
    video = VideoFileClip(input_path)
    audio = video.audio
    audio.write_audiofile(output_path)
    video.close()
    audio.close()
    return f"Audio extracted and saved to {output_path}"


def add_text(
    input_path: str,
    output_path: str,
    text: str,
    position: TextPosition = TextPosition(),
    fontsize: int = 30,
    color: str = 'white',
    start_time: float = 0,
    end_time: Optional[float] = None
) -> str:
    """
    Add text to a video.
    """
    video = VideoFileClip(input_path)
    txt_clip = TextClip(text, fontsize=fontsize, color=color)
    txt_clip = txt_clip.set_position((position.x, position.y)).set_start(start_time)
    if end_time is not None:
        txt_clip = txt_clip.set_end(end_time)
    final = CompositeVideoClip([video, txt_clip])
    final.write_videofile(output_path)
    video.close()
    txt_clip.close()
    final.close()
    return f"Text added to video and saved to {output_path}"


def convert_format(
    input_path: str,
    output_path: str,
    format: VideoFormat = VideoFormat.MP4
) -> str:
    """
    Convert a video to a different format.
    """
    video = VideoFileClip(input_path)
    
    if format == VideoFormat.GIF:
        video.write_gif(output_path)
    else:
        video.write_videofile(output_path)
        
    video.close()
    return f"Video converted to {format} and saved to {output_path}"


# Define the list of tools to be used by the agent
moviepy_tools = [
    load_video,
    trim_video,
    apply_effect,
    concatenate_videos,
    extract_audio,
    add_text,
    convert_format
]
