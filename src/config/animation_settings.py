"""
Holds fixed settings for the timing of animations. May be implemented via
dcc.Store elements or some other way in the future.
"""

class AnimationSettings:
    def __init__(self) -> None:
        # Animation timing; implementable via dcc.store in the future.
        self.frames_per_second: int = 12
        self.time_for_animation_ms: int = 1000
        self.frames_count: int = self.frames_per_second * (
            self.time_for_animation_ms // 1000
        )
        interval_ms = self.time_for_animation_ms / self.frames_count
        self.interval_ms = max(int(interval_ms), 1)  # Always at least 1ms
