from enum import Enum
from typing import List, Optional, Tuple, Union, Dict, Any, Literal
from pathlib import Path

import numpy as np
from moviepy import (
    VideoFileClip, 
    CompositeVideoClip, 
    TextClip, 
    concatenate_videoclips, 
    vfx
)
from scipy.ndimage import gaussian_filter
from pydantic import Field, BaseModel
from pydantic_ai import Tool

class EffectParams(BaseModel):
    duration: Optional[float] = Field(None, description="Duration for fade effects")
    factor: Optional[float] = Field(None, description="Speed factor for speedx effect")
    width: Optional[int] = Field(None, description="Width for resize effect")
    height: Optional[int] = Field(None, description="Height for resize effect")
    angle: Optional[float] = Field(None, description="Angle for rotate effect")
    # Chromakey parameters
    key_color: Optional[str] = Field("green", description="Color to key out (e.g., 'green', 'blue')")
    color_tolerance: Optional[float] = Field(40.0, description="Color tolerance (0-100)")
    blur: Optional[float] = Field(1.0, description="Blur amount for edge softening")
    background_path: Optional[str] = Field(None, description="Path to background video/image")

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
    CHROMAKEY = "chromakey"

class VideoFormat(str, Enum):
    MP4 = "mp4"
    GIF = "gif"
    AVI = "avi"
    MOV = "mov"
    WEBM = "webm"

def rgb_to_hsv(rgb):
    """Convert RGB color space to HSV."""
    rgb_normalized = rgb.astype('float32') / 255.0
    maxc = np.maximum(np.maximum(rgb_normalized[..., 0], rgb_normalized[..., 1]), rgb_normalized[..., 2])
    minc = np.minimum(np.minimum(rgb_normalized[..., 0], rgb_normalized[..., 1]), rgb_normalized[..., 2])
    v = maxc
    
    zeros = np.zeros_like(maxc)
    ones = np.ones_like(maxc)
    
    s = np.where(maxc != 0, (maxc - minc) / maxc, zeros)
    rc = np.where(maxc != minc, (rgb_normalized[..., 1] - rgb_normalized[..., 2]) / (maxc - minc), zeros)
    gc = np.where(maxc != minc, 2.0 + (rgb_normalized[..., 2] - rgb_normalized[..., 0]) / (maxc - minc), zeros)
    bc = np.where(maxc != minc, 4.0 + (rgb_normalized[..., 0] - rgb_normalized[..., 1]) / (maxc - minc), zeros)
    
    h = np.where(rgb_normalized[..., 0] == maxc, bc,
                 np.where(rgb_normalized[..., 1] == maxc, rc, gc))
    h = np.where(maxc != minc, (h / 6.0) % 1.0, zeros)
    
    return np.dstack((h, s, v))

def create_chromakey_mask(frame, key_color: str, tolerance: float = 40.0, blur: float = 1.0) -> np.ndarray:
    """
    Create a mask for chromakey effect using both RGB and HSV color spaces.
    
    Args:
        frame: Input frame
        key_color: Color to key out ('green' or 'blue')
        tolerance: Color tolerance (0-100)
        blur: Blur amount for edge softening
    
    Returns:
        numpy.ndarray: Alpha mask
    """
    # Convert tolerance to 0-1 range
    tolerance = tolerance / 100.0
    
    # Define color ranges based on key_color
    if key_color.lower() == 'green':
        # HSV ranges for green
        hue_target = 0.33  # Green in HSV
        rgb_target = np.array([0, 255, 0])
    else:  # blue
        # HSV ranges for blue
        hue_target = 0.66  # Blue in HSV
        rgb_target = np.array([0, 0, 255])

    # Convert frame to HSV
    hsv = rgb_to_hsv(frame)
    
    # Create masks using both HSV and RGB
    hsv_mask = np.abs(hsv[..., 0] - hue_target) < tolerance
    
    # Normalize RGB values
    frame_normalized = frame.astype('float32') / 255.0
    rgb_target_normalized = rgb_target.astype('float32') / 255.0
    
    # Calculate RGB color difference
    rgb_diff = np.sqrt(np.sum((frame_normalized - rgb_target_normalized) ** 2, axis=-1))
    rgb_mask = rgb_diff < tolerance
    
    # Combine masks
    combined_mask = hsv_mask & rgb_mask
    
    # Apply saturation and value thresholds to reduce noise
    combined_mask = combined_mask & (hsv[..., 1] > 0.2) & (hsv[..., 2] > 0.2)
    
    # Convert to float
    mask = combined_mask.astype('float32')
    
    # Apply gaussian blur for smoother edges
    if blur > 0:
        mask = gaussian_filter(mask, sigma=blur)
    
    # Ensure mask is in correct range
    mask = np.clip(mask, 0, 1)
    
    return mask

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
    elif effect == VideoEffect.CHROMAKEY:
        if not params.background_path:
            video.close()
            return "Background path is required for chromakey effect"
        
        # Load background video/image
        background = VideoFileClip(params.background_path)
        
        # Resize background to match foreground if needed
        if background.size != video.size:
            background = background.resize(video.size)
        
        # Create chromakey effect
        def chromakey_frame(get_frame, t):
            frame = get_frame(t)
            mask = create_chromakey_mask(
                frame,
                params.key_color or "green",
                params.color_tolerance or 40.0,
                params.blur or 1.0
            )
            return frame * mask[..., np.newaxis]
        
        # Apply the effect
        processed = video.fl(chromakey_frame)
        
        # Composite with background
        processed = CompositeVideoClip([background, processed])
        background.close()
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
