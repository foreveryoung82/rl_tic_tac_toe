from enum import Enum, auto
from math import pow
from typing import Optional


class LearningParamSchedulerPolicy(Enum):
    """Learning Parameter Scheduler Policy"""

    EXPONENTIAL_DECAY = auto()


class LearningParamScheduler:
    """Learning Parameter Scheduler"""

    def __init__(
        self,
        episodes: int,
        start_value: float,
        min_value: float,
        decay_value: Optional[float] = None,
        policy: LearningParamSchedulerPolicy = LearningParamSchedulerPolicy.EXPONENTIAL_DECAY,
    ):
        assert isinstance(episodes, int) and episodes > 0
        self._episodes = episodes
        self._policy = policy
        assert isinstance(start_value, float) and 1.0 >= start_value >= min_value
        self._start = start_value
        assert isinstance(min_value, float) and 1.0 >= min_value >= 0.0
        self._min = min_value
        assert (
            isinstance(decay_value, float) and 1 >= decay_value > 0
        ) or decay_value is None
        self._decay = decay_value or pow(min_value / start_value, 1.0 / episodes)
        self._current = start_value

    @property
    def episodes(self) -> int:
        """Total number of episodes."""
        return self._episodes

    @property
    def policy(self) -> LearningParamSchedulerPolicy:
        """Learning parameter policy."""
        return self._policy

    @property
    def start(self) -> float:
        """Initial value before any decay."""
        return self._start

    @property
    def min(self) -> float:
        """Minimum allowed value after decay."""
        return self._min

    @property
    def decay(self) -> float:
        """Decay factor applied each update."""
        return self._decay

    @property
    def current(self) -> float:
        """Current parameter value."""
        return self._current

    def update(self, episode_idx: int) -> float:
        """Update learning parameter and return new value."""
        if self._policy == LearningParamSchedulerPolicy.EXPONENTIAL_DECAY:
            self._current = max(self._min, self._current * self._decay)
        return self._current
