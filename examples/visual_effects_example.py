from tools.moviepy_tool import (
    VideoEffect, 
    EffectParams, 
    apply_effect,
    concatenate_videos
)

def create_effects_showcase():
    """
    Demonstrates various visual effects available in moviepy.
    
    This example shows how to:
    1. Rotate videos by specific angles
    2. Create mirror effects (horizontal and vertical)
    3. Change video playback speed
    4. Resize video dimensions
    
    The example assumes you have an input file:
    - input.mp4: Source video clip to apply effects to
    """
    # Example 1: Rotating video
    rotate_params = EffectParams(angle=45.0)  # 45 degree rotation
    apply_effect(
        "input.mp4",
        "rotated.mp4",
        VideoEffect.ROTATE,
        rotate_params
    )
    
    # Example 2: Mirror effects
    # Horizontal mirror
    apply_effect(
        "input.mp4",
        "mirror_x.mp4",
        VideoEffect.MIRROR_X,
        EffectParams()
    )
    
    # Vertical mirror
    apply_effect(
        "input.mp4",
        "mirror_y.mp4",
        VideoEffect.MIRROR_Y,
        EffectParams()
    )
    
    # Example 3: Speed changes
    # Double speed
    speed_params = EffectParams(factor=2.0)
    apply_effect(
        "input.mp4",
        "fast.mp4",
        VideoEffect.SPEEDX,
        speed_params
    )
    
    # Half speed (slow motion)
    slow_params = EffectParams(factor=0.5)
    apply_effect(
        "input.mp4",
        "slow.mp4",
        VideoEffect.SPEEDX,
        slow_params
    )
    
    # Example 4: Resizing
    resize_params = EffectParams(width=1280, height=720)  # Resize to 720p
    apply_effect(
        "input.mp4",
        "resized.mp4",
        VideoEffect.RESIZE,
        resize_params
    )
    
    # Combine all effects into a showcase
    concatenate_videos(
        [
            "rotated.mp4",
            "mirror_x.mp4",
            "mirror_y.mp4",
            "fast.mp4",
            "slow.mp4",
            "resized.mp4"
        ],
        "effects_showcase.mp4"
    )

if __name__ == "__main__":
    create_effects_showcase()