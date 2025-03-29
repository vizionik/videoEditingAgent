import os
from enum import Enum
from typing import List, Optional, Tuple, Union, Dict, Any, Literal
from pathlib import Path

import numpy as np
from moviepy import VideoFileClip
from moviepy import TextClip
from moviepy import CompositeVideoClip
from moviepy import concatenate_videoclips
from moviepy import vfx
from scipy.ndimage import gaussian_filter
from pydantic import Field, BaseModel
from pydantic_ai import Agent, Tool, tools

# Video format and effect enums
class VideoFormat(str, Enum):
    MP4 = "mp4"
    GIF = "gif"
    AVI = "avi"
    MOV = "mov"
    WEBM = "webm"

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
    LUMINANCE_KEY = "luminance_key"

# Position and parameter models
class TextPosition(BaseModel):
    x: Literal['left', 'center', 'right'] = Field('center')
    y: Literal['top', 'center', 'bottom'] = Field('center')

class EffectParams(BaseModel):
    duration: Optional[float] = Field(None, description="Duration for fade effects")
    factor: Optional[float] = Field(None, description="Speed factor for speedx effect")
    width: Optional[int] = Field(None, description="Width for resize effect")
    height: Optional[int] = Field(None, description="Height for resize effect")
    angle: Optional[float] = Field(None, description="Angle for rotate effect")
    key_color: Optional[str] = Field("green", description="Color to key out (e.g., 'green', 'blue')")
    color_tolerance: Optional[float] = Field(40.0, description="Color tolerance (0-100)")
    blur: Optional[float] = Field(1.0, description="Blur amount for edge softening")
    background_path: Optional[str] = Field(None, description="Path to background video/image")
    luminance_min: Optional[float] = Field(0.0, description="Minimum brightness threshold (0-1)")
    luminance_max: Optional[float] = Field(1.0, description="Maximum brightness threshold (0-1)")
    softness: Optional[float] = Field(0.1, description="Softness of the luminance key transition (0-1)")

# Result models for tool operations
class VideoInfo(BaseModel):
    """Video information returned by load_video tool"""
    duration: float = Field(description="Duration of video in seconds")
    fps: float = Field(description="Frames per second")
    width: int = Field(description="Video width in pixels")
    height: int = Field(description="Video height in pixels")
    filepath: str = Field(description="Path to the video file")

class TrimResult(BaseModel):
    """Result of trimming a video"""
    output_path: str = Field(description="Path where the trimmed video was saved")
    start_time: float = Field(description="Start time of the trim in seconds")
    end_time: Optional[float] = Field(None, description="End time of the trim in seconds (None means end of video)")

class EffectResult(BaseModel):
    """Result of applying an effect to a video"""
    output_path: str = Field(description="Path where the processed video was saved")
    effect: VideoEffect = Field(description="The effect that was applied")
    params: Dict[str, Any] = Field(description="Parameters used for the effect")

class ConcatResult(BaseModel):
    """Result of concatenating videos"""
    output_path: str = Field(description="Path where the concatenated video was saved")
    input_count: int = Field(description="Number of input videos concatenated")

class AudioResult(BaseModel):
    """Result of audio extraction"""
    output_path: str = Field(description="Path where the audio was saved")
    duration: float = Field(description="Duration of the extracted audio in seconds")

class TextResult(BaseModel):
    """Result of adding text to a video"""
    output_path: str = Field(description="Path where the video with text was saved")
    text: str = Field(description="Text that was added")
    position: TextPosition = Field(description="Position where text was placed")

class FormatResult(BaseModel):
    """Result of converting video format"""
    output_path: str = Field(description="Path where the converted video was saved")
    format: VideoFormat = Field(description="Format the video was converted to")

class VoiceoverResult(BaseModel):
    """Result of adding voiceover"""
    output_path: str = Field(description="Path where the video with voiceover was saved")
    start_time: float = Field(description="Start time of the voiceover in seconds")
    volume: float = Field(description="Volume multiplier applied to voiceover")

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

def create_luminance_mask(frame, min_thresh: float = 0.0, max_thresh: float = 1.0, softness: float = 0.1) -> np.ndarray:
    """
    Create a mask for luminance key effect based on pixel brightness.
    
    Args:
        frame: Input frame
        min_thresh: Minimum brightness threshold (0-1)
        max_thresh: Maximum brightness threshold (0-1)
        softness: Softness of the transition (0-1)
    
    Returns:
        numpy.ndarray: Alpha mask
    """
    # Convert frame to grayscale using perceived luminance weights
    luminance = np.dot(frame[..., :3], [0.2989, 0.5870, 0.1140])
    
    # Normalize to 0-1 range
    luminance = luminance / 255.0
    
    # Create soft transition ranges
    soft_min = min_thresh - softness
    soft_max = max_thresh + softness
    
    # Create mask with soft transitions
    mask = np.clip((luminance - soft_min) / softness, 0, 1) * \
           np.clip((soft_max - luminance) / softness, 0, 1)
    
    # Apply gaussian blur for smoother edges
    if softness > 0:
        mask = gaussian_filter(mask, sigma=softness * 2)
    
    # Ensure mask is in correct range
    mask = np.clip(mask, 0, 1)
    
    return mask

async def load_video(ctx: tools.RunContext[None], filepath: str) -> VideoInfo:
    """Load a video file and return its metadata information."""
    if not filepath:
        raise ValueError("filepath is required")
    
    if not os.path.exists(filepath):
        raise ValueError(f"File not found: {filepath}")

    video = None
    try:
        video = VideoFileClip(filepath)
        return VideoInfo(
            duration=round(float(video.duration), 3),
            fps=round(float(video.fps), 3),
            width=int(video.size[0]),
            height=int(video.size[1]),
            filepath=filepath
        )
    except Exception as e:
        raise ValueError(f"Error loading video: {str(e)}")
    finally:
        if video is not None:
            try:
                video.close()
            except:
                pass

async def trim_video(
    ctx: tools.RunContext[None],
    input_path: str,
    output_path: str,
    start_time: float = 0,
    end_time: Optional[float] = None
) -> TrimResult:
    """Trim a video to the specified start and end times."""
    if not input_path or not output_path:
        raise ValueError("Both input_path and output_path are required")
    
    if not os.path.exists(input_path):
        raise ValueError(f"Input file not found: {input_path}")
    
    video = None
    trimmed = None
    try:
        video = VideoFileClip(input_path)
        trimmed = video.subclip(start_time, end_time)
        trimmed.write_videofile(output_path)
        
        return TrimResult(
            output_path=output_path,
            start_time=start_time,
            end_time=end_time
        )
    except Exception as e:
        raise ValueError(f"Error trimming video: {str(e)}")
    finally:
        if video is not None:
            video.close()
        if trimmed is not None:
            trimmed.close()

async def apply_effect(
    ctx: tools.RunContext[None],
    input_path: str,
    output_path: str,
    effect: VideoEffect,
    params: Dict[str, Any] = {}
) -> EffectResult:
    """Apply a video effect to the input video."""
    video = None
    processed = None
    background = None

    try:
        # Convert dict params to EffectParams for internal use
        effect_params = EffectParams(**params)
        
        # Ensure required parameters exist for specific effects
        if effect == VideoEffect.RESIZE and effect_params.width is None and effect_params.height is None:
            raise ValueError("Either width or height must be specified for resize effect")
        
        video = VideoFileClip(input_path)
        
        if effect == VideoEffect.FADEOUT:
            duration = effect_params.duration if effect_params.duration is not None else 1.0
            processed = video.fadeout(duration)
            
        elif effect == VideoEffect.FADEIN:
            duration = effect_params.duration if effect_params.duration is not None else 1.0
            processed = video.fadein(duration)
            
        elif effect == VideoEffect.SPEEDX:
            factor = effect_params.factor if effect_params.factor is not None else 2.0
            processed = video.speedx(factor)
            
        elif effect == VideoEffect.RESIZE:
            processed = video.resize(width=effect_params.width, height=effect_params.height)
            
        elif effect == VideoEffect.ROTATE:
            angle = effect_params.angle if effect_params.angle is not None else 90
            processed = video.rotate(angle)
            
        elif effect == VideoEffect.MIRROR_X:
            processed = video.fx(vfx.mirror_x)
            
        elif effect == VideoEffect.MIRROR_Y:
            processed = video.fx(vfx.mirror_y)
            
        elif effect == VideoEffect.CHROMAKEY:
            if not effect_params.background_path:
                raise ValueError("Background path is required for chromakey effect")
            
            # Load background video/image and apply effect
            background = VideoFileClip(effect_params.background_path)
            
            # Resize background to match foreground if needed
            if background.size != video.size:
                background = background.resize(video.size)
            
            # Create chromakey effect
            def chromakey_frame(get_frame, t):
                frame = get_frame(t)
                mask = create_chromakey_mask(
                    frame,
                    effect_params.key_color or "green",
                    effect_params.color_tolerance or 40.0,
                    effect_params.blur or 1.0
                )
                return frame * mask[..., np.newaxis]
            
            # Apply the effect
            processed = video.fl(chromakey_frame)
            processed = CompositeVideoClip([background, processed])
            
        elif effect == VideoEffect.LUMINANCE_KEY:
            if not effect_params.background_path:
                raise ValueError("Background path is required for luminance key effect")
            
            # Load background video/image and apply effect
            background = VideoFileClip(effect_params.background_path)
            
            # Resize background to match foreground if needed
            if background.size != video.size:
                background = background.resize(video.size)
            
            # Create luminance key effect
            def luminance_key_frame(get_frame, t):
                frame = get_frame(t)
                mask = create_luminance_mask(
                    frame,
                    effect_params.luminance_min or 0.0,
                    effect_params.luminance_max or 1.0,
                    effect_params.softness or 0.1
                )
                return frame * mask[..., np.newaxis]
            
            # Apply the effect
            processed = video.fl(luminance_key_frame)
            processed = CompositeVideoClip([background, processed])
            
        else:
            raise ValueError(f"Effect {effect} not implemented")
        
        # Write the processed video to the output file
        if processed is not None:
            processed.write_videofile(output_path)
            return EffectResult(
                output_path=output_path,
                effect=effect,
                params=params
            )
        else:
            raise ValueError("Failed to process video")
            
    except Exception as e:
        raise ValueError(f"Error applying effect: {str(e)}")
    
    finally:
        # Clean up resources
        if video is not None:
            try:
                video.close()
            except:
                pass
                
        if processed is not None:
            try:
                processed.close()
            except:
                pass
                
        if background is not None:
            try:
                background.close()
            except:
                pass

async def concatenate_videos(
    ctx: tools.RunContext[None],
    input_paths: List[str],
    output_path: str
) -> ConcatResult:
    """Concatenate multiple videos into a single video file."""
    if not input_paths:
        raise ValueError("No input paths provided")
    if not output_path:
        raise ValueError("Output path is required")

    clips = []
    final_clip = None
    try:
        clips = [VideoFileClip(path) for path in input_paths]
        final_clip = concatenate_videoclips(clips)
        final_clip.write_videofile(output_path)
        return ConcatResult(
            output_path=output_path,
            input_count=len(input_paths)
        )
    finally:
        for clip in clips:
            clip.close()
        if final_clip is not None:
            final_clip.close()

async def extract_audio(
    ctx: tools.RunContext[None],
    input_path: str,
    output_path: str
) -> AudioResult:
    """Extract the audio track from a video file and save it."""
    if not input_path:
        raise ValueError("Input path is required")
    if not output_path:
        raise ValueError("Output path is required")

    video = None
    audio = None
    try:
        video = VideoFileClip(input_path)
        if video.audio is None:
            raise ValueError("Video has no audio track")
            
        audio = video.audio
        audio.write_audiofile(output_path)
        
        return AudioResult(
            output_path=output_path,
            duration=float(audio.duration)
        )
    finally:
        if video is not None:
            video.close()
        if audio is not None:
            audio.close()

async def add_text(
    ctx: tools.RunContext[None],
    input_path: str,
    output_path: str,
    text: str,
    position: TextPosition = TextPosition(),
    fontsize: int = 30,
    color: str = 'white',
    start_time: float = 0,
    end_time: Optional[float] = None
) -> TextResult:
    """Add text overlay to a video file."""
    if not input_path or not output_path:
        raise ValueError("Both input_path and output_path are required")
    if not text:
        raise ValueError("Text is required")

    video = None
    txt_clip = None
    final = None
    try:
        video = VideoFileClip(input_path)
        txt_clip = TextClip(text, fontsize=fontsize, color=color)
        txt_clip = txt_clip.set_position((position.x, position.y)).set_start(start_time)
        if end_time is not None:
            txt_clip = txt_clip.set_end(end_time)
        final = CompositeVideoClip([video, txt_clip])
        final.write_videofile(output_path)
        
        return TextResult(
            output_path=output_path,
            text=text,
            position=position
        )
    except Exception as e:
        raise ValueError(f"Error adding text to video: {str(e)}")
    finally:
        if video is not None:
            video.close()
        if txt_clip is not None:
            txt_clip.close()
        if final is not None:
            final.close()

async def convert_format(
    ctx: tools.RunContext[None],
    input_path: str,
    output_path: str,
    format: VideoFormat = VideoFormat.MP4
) -> FormatResult:
    """Convert a video file to a different format."""
    if not input_path or not output_path:
        raise ValueError("Both input_path and output_path are required")
    if not os.path.exists(input_path):
        raise ValueError(f"Input file not found: {input_path}")

    video = None
    try:
        video = VideoFileClip(input_path)
        if format == VideoFormat.GIF:
            video.write_gif(output_path)
        else:
            video.write_videofile(output_path)
        
        return FormatResult(
            output_path=output_path,
            format=format
        )
    except Exception as e:
        raise ValueError(f"Error converting video: {str(e)}")
    finally:
        if video is not None:
            video.close()

async def add_voiceover(
    ctx: tools.RunContext[None],
    video_path: str,
    audio_path: str,
    output_path: str,
    start_time: float = 0,
    audio_volume: float = 1.0
) -> VoiceoverResult:
    """Add a voiceover audio track to a video."""
    if not video_path or not audio_path or not output_path:
        raise ValueError("All file paths are required")
    if not os.path.exists(video_path):
        raise ValueError(f"Video file not found: {video_path}")
    if not os.path.exists(audio_path):
        raise ValueError(f"Audio file not found: {audio_path}")

    video = None
    voiceover = None
    final_video = None
    try:
        # Load the video and audio
        video = VideoFileClip(video_path)
        voiceover = VideoFileClip(audio_path).audio
        
        # Set the start time and volume of the voiceover
        voiceover = voiceover.set_start(start_time).volumex(audio_volume)
        
        # Combine the original video with the voiceover
        if video.audio is not None:
            # If video has audio, mix it with the voiceover
            final_audio = CompositeVideoClip([
                video.set_audio(video.audio),
                video.set_audio(voiceover)
            ]).audio
        else:
            # If video has no audio, just use the voiceover
            final_audio = voiceover
        
        # Create and write final video
        final_video = video.set_audio(final_audio)
        final_video.write_videofile(output_path)
        
        return VoiceoverResult(
            output_path=output_path,
            start_time=start_time,
            volume=audio_volume
        )
    except Exception as e:
        raise ValueError(f"Error adding voiceover: {str(e)}")
    finally:
        if video is not None:
            video.close()
        if voiceover is not None:
            voiceover.close()
        if final_video is not None:
            final_video.close()

# Define the list of video editing tools
moviepy_tools = [
    Tool(load_video),
    Tool(trim_video),
    Tool(apply_effect),
    Tool(concatenate_videos),
    Tool(extract_audio),
    Tool(add_text),
    Tool(convert_format),
    Tool(add_voiceover)
]
