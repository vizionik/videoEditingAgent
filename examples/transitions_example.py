from tools.moviepy_tool import (
    VideoEffect, 
    EffectParams, 
    apply_effect,
    concatenate_videos
)

def create_transition_demo():
    """
    Demonstrates various transition effects in moviepy.
    
    This example shows how to:
    1. Create fade in/out transitions between clips
    2. Apply crossfade transitions when concatenating videos
    3. Chain multiple effects together
    
    The example assumes you have these input files:
    - clip1.mp4: First video clip
    - clip2.mp4: Second video clip
    - clip3.mp4: Third video clip
    """
    # Step 1: Apply fade out to first clip
    fade_params = EffectParams(duration=1.0)  # 1 second fade
    apply_effect(
        "clip1.mp4",
        "clip1_fade.mp4",
        VideoEffect.FADEOUT,
        fade_params
    )
    
    # Step 2: Apply fade in to second clip
    apply_effect(
        "clip2.mp4",
        "clip2_fade.mp4",
        VideoEffect.FADEIN,
        fade_params
    )
    
    # Step 3: Apply crossfade to third clip
    apply_effect(
        "clip3.mp4",
        "clip3_fade.mp4",
        VideoEffect.CROSSFADEIN,
        fade_params
    )
    
    # Step 4: Concatenate all clips
    concatenate_videos(
        [
            "clip1_fade.mp4",
            "clip2_fade.mp4",
            "clip3_fade.mp4"
        ],
        "final_transitions.mp4"
    )

if __name__ == "__main__":
    create_transition_demo()