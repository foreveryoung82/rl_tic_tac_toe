"""Learning Parameter Scheduler"""

from enum import Enum


class LearningParamPolicy(Enum):
    """Learning Parameter Policy"""

    EXPONENTIAL_DECAY = "exponential_decay"


class LearningParamScheduler:
    """Learning Parameter Scheduler"""

    def __init__(
        self,
        episodes: int,
        policy: LearningParamPolicy,
        start_value: float,
        min_value: float,
        decay_value: float,
    ):
        self._episodes = episodes
        self._policy = policy
        self._start_value = start_value
        self._min_value = min_value
        self._decay_value = decay_value
        self._current_value = start_value

    def update(self, episode_idx: int) -> float:
        """Update learning parameter"""
        if self._policy == LearningParamPolicy.EXPONENTIAL_DECAY:
            self._current_value = max(
                self._min_value, self._current_value * self._decay_value
            )
        return self._current_value
