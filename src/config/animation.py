"""
Holds fixed settings for the timing of animations. May be implemented via
dcc.Store elements or some other way in the future.
"""

class AnimationSettings:
    frames_per_second = 23
    time_for_animation_ms = 1000
    frames_count = frames_per_second * (time_for_animation_ms // 1000)
    interval_ms = max(time_for_animation_ms // frames_count, 1)  # Always at least 1ms
