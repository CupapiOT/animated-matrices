"""
Holds fixed settings for the timing of animations. May be implemented via
dcc.Store elements or some other way in the future.
"""


class AnimationSettings:
    def __init__(self, frames_per_second, time_for_animation_ms=1000) -> None:
        self.frames_per_second = frames_per_second
        self.time_for_animation_ms = time_for_animation_ms
        self.frames_count = self.frames_per_second * (
            self.time_for_animation_ms // 1000
        )
        self.interval_ms = max(
            self.time_for_animation_ms // self.frames_count,
            1,
        )

    def __str__(self) -> str:
        return str(self.__dict__)
