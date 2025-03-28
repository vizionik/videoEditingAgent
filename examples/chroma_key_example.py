"""
Example demonstrating advanced chroma key (green screen) effects using moviepy.
This shows how to effectively remove green backgrounds and handle common edge cases.
"""

import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from moviepy import VideoFileClip, ImageClip, CompositeVideoClip

from tools.moviepy_tool import create_chromakey_mask

def apply_chroma_key(
    foreground_path: str,
    background_path: str,
    output_path: str,
    key_color: str = "green",
    color_tolerance: float = 40.0,
    blur: float = 1.0,
    despill: bool = True
) -> None:
    """
    Apply chroma key effect to replace a colored background with a new background.
    
    Args:
        foreground_path: Path to the foreground video/image with green/blue screen
        background_path: Path to the background video/image
        output_path: Path to save the output video
        key_color: Color to key out ('green' or 'blue')
        color_tolerance: How much color variation to allow (0-100)
        blur: Amount of edge blur to apply (0.0+)
        despill: Whether to remove color spill on edges
    """
    # Load the clips
    foreground = VideoFileClip(foreground_path)
    is_bg_video = background_path.lower().endswith(('.mp4', '.mov', '.avi'))
    background = (VideoFileClip(background_path) if is_bg_video 
                 else ImageClip(background_path))
    
    # Resize background to match foreground if needed
    if background.size != foreground.size:
        background = background.resize(foreground.size)
    
    def process_frame(frame: np.ndarray, t: float) -> np.ndarray:
        # Create alpha mask from chroma key color
        mask = create_chromakey_mask(
            frame,
            key_color,
            color_tolerance,
            blur
        )
        
        # Apply despill effect if enabled
        if despill:
            # Convert to float for calculations
            frame = frame.astype(float)
            
            # Identify areas with potential color spill
            spill_mask = (mask > 0.1) & (mask < 0.9)
            
            if key_color.lower() == 'green':
                # Reduce green channel in spill areas
                green = frame[..., 1]
                other_channels = np.minimum(frame[..., 0], frame[..., 2])
                green_diff = green - other_channels
                green_diff = np.maximum(green_diff, 0) * spill_mask
                frame[..., 1] = np.clip(green - green_diff, 0, 255)
            else:  # blue
                # Reduce blue channel in spill areas
                blue = frame[..., 2]
                other_channels = np.minimum(frame[..., 0], frame[..., 1])
                blue_diff = blue - other_channels
                blue_diff = np.maximum(blue_diff, 0) * spill_mask
                frame[..., 2] = np.clip(blue - blue_diff, 0, 255)
            
            frame = frame.astype(np.uint8)
        
        # Apply the mask to the foreground frame
        return frame * mask[..., np.newaxis]
    
    # Apply chroma key processing to each frame
    processed = foreground.fl(process_frame)
    
    # Composite with background
    final = CompositeVideoClip([background, processed])
    
    # Write output file
    final.write_videofile(output_path)
    
    # Clean up
    foreground.close()
    background.close()
    processed.close()
    final.close()

def main():
    # Example usage
    examples_dir = Path(__file__).parent
    
    # Replace these paths with your actual video files
    foreground_path = str(examples_dir / "green_screen_video.mp4")
    background_path = str(examples_dir / "background.jpg")
    output_path = str(examples_dir / "output_chromakey.mp4")
    
    print("Applying chroma key effect...")
    
    # Example with default parameters (good for typical green screen footage)
    apply_chroma_key(
        foreground_path=foreground_path,
        background_path=background_path,
        output_path=output_path
    )
    
    # Example with custom parameters for challenging footage
    apply_chroma_key(
        foreground_path=foreground_path,
        background_path=background_path,
        output_path=str(examples_dir / "output_chromakey_fine_tuned.mp4"),
        key_color="green",
        color_tolerance=35.0,  # Stricter color matching
        blur=2.0,  # More edge softening
        despill=True  # Remove green spill on edges
    )

if __name__ == "__main__":
    main()