from tools.moviepy_tool import (
    VideoEffect, 
    EffectParams, 
    TextPosition,
    apply_effect,
    add_text,
    concatenate_videos
)

def create_advanced_composition():
    """
    Demonstrates how to create a complex video composition by combining
    multiple effects and transitions.
    
    This example shows how to:
    1. Chain multiple effects together
    2. Create picture-in-picture effects using chromakey
    3. Add text overlays with timing
    4. Create smooth transitions between complex scenes
    
    The example assumes you have these input files:
    - main.mp4: Main footage
    - overlay.mp4: Greenscreen overlay footage
    - background.mp4: Background footage
    """
    # Step 1: Create a slow-motion intro with fade
    intro_params = EffectParams(factor=0.5)  # Half speed
    apply_effect(
        "main.mp4",
        "intro_slow.mp4",
        VideoEffect.SPEEDX,
        intro_params
    )
    
    fade_params = EffectParams(duration=2.0)  # 2 second fade
    apply_effect(
        "intro_slow.mp4",
        "intro_fade.mp4",
        VideoEffect.FADEIN,
        fade_params
    )
    
    # Step 2: Create picture-in-picture with chromakey
    chromakey_params = EffectParams(
        key_color="green",
        color_tolerance=35.0,
        blur=2.0,
        background_path="background.mp4"
    )
    
    apply_effect(
        "overlay.mp4",
        "overlay_keyed.mp4",
        VideoEffect.CHROMAKEY,
        chromakey_params
    )
    
    # Step 3: Add rotation to the keyed footage
    rotate_params = EffectParams(angle=15.0)
    apply_effect(
        "overlay_keyed.mp4",
        "overlay_rotated.mp4",
        VideoEffect.ROTATE,
        rotate_params
    )
    
    # Step 4: Create mirrored effect for transition
    apply_effect(
        "main.mp4",
        "transition_mirror.mp4",
        VideoEffect.MIRROR_X,
        EffectParams()
    )
    
    # Step 5: Add text overlays
    add_text(
        "intro_fade.mp4",
        "intro_with_text.mp4",
        "Welcome to the Demo",
        TextPosition(x="center", y="center"),
        fontsize=60,
        color="white",
        start_time=1.0,
        end_time=4.0
    )
    
    # Step 6: Create final composition
    concatenate_videos(
        [
            "intro_with_text.mp4",
            "overlay_rotated.mp4",
            "transition_mirror.mp4"
        ],
        "final_composition.mp4"
    )

if __name__ == "__main__":
    create_advanced_composition()