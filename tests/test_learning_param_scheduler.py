# tests/test_learning_param_scheduler.py
import pytest

from rl_tic_tac_toe.learning_param_scheduler import (
    LearningParamScheduler,
    LearningParamSchedulerPolicy,
)


@pytest.fixture
def scheduler() -> LearningParamScheduler:
    """Create a default scheduler instance."""
    return LearningParamScheduler(
        episodes=10,
        start_value=0.8,
        min_value=0.2,
        policy=LearningParamSchedulerPolicy.EXPONENTIAL_DECAY,
    )


def test_episodes_property(scheduler: LearningParamScheduler) -> None:
    assert isinstance(scheduler.episodes, int)
    assert scheduler.episodes == 10


def test_policy_property(scheduler: LearningParamScheduler) -> None:
    assert isinstance(scheduler.policy, LearningParamSchedulerPolicy)
    assert scheduler.policy == LearningParamSchedulerPolicy.EXPONENTIAL_DECAY


def test_start_property(scheduler: LearningParamScheduler) -> None:
    assert isinstance(scheduler.start, float)
    assert scheduler.start == 0.8


def test_min_property(scheduler: LearningParamScheduler) -> None:
    assert isinstance(scheduler.min, float)
    assert scheduler.min == 0.2


def test_decay_property_default_calculation() -> None:
    """Verify that the default decay is computed correctly when None is passed."""
    start_value = 1.0
    min_value = 0.5
    episodes = 4
    # Expected decay: (min/start)^(1/episodes)
    expected_decay = (min_value / start_value) ** (1.0 / episodes)

    sched = LearningParamScheduler(
        episodes=episodes, start_value=start_value, min_value=min_value
    )
    assert pytest.approx(sched.decay, rel=1e-7) == expected_decay  # pyright: ignore[reportUnknownMemberType]


def test_current_property_initial(scheduler: LearningParamScheduler) -> None:
    # Current should equal the start value upon initialization
    assert scheduler.current == scheduler.start
