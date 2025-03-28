from tools.moviepy_tool import VideoEffect, EffectParams, apply_effect

def main():
    """
    Example demonstrating the luminance key effect.
    This will make bright areas of the video transparent and composite it over a background.
    """
    # Input parameters
    input_video = "input.mp4"  # Your foreground video
    background_video = "background.mp4"  # Your background video
    output_path = "output_luminance_key.mp4"

    # Create effect parameters
    params = EffectParams(
        # Make bright areas transparent (values above 0.7)
        luminance_min=0.7,
        luminance_max=1.0,
        # Add some softness to the transition
        softness=0.2,
        # Specify the background video
        background_path=background_video
    )

    # Apply luminance key effect
    result = apply_effect(
        input_video,
        output_path,
        VideoEffect.LUMINANCE_KEY,
        params
    )
    print(result)

if __name__ == "__main__":
    main()