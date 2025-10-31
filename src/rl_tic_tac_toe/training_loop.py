"""A module implementing the training loop for a Q-learning Tic-Tac-Toe agent."""

import time
from dataclasses import dataclass
from math import pow
from random import Random
from typing import Optional

from .learning_param_scheduler import LearningParamPolicy, LearningParamScheduler
from .player import Player
from .q_learning_agent import QLearningAgent
from .snapshot_pool import SnapshotPool
from .training_episode import TrainingEpisode
from .training_reporter import TrainingReporter

EVALUATION_NUM = 20
SNAPSHOT_NUM = 10
HISTORICAL_OPPONENT_PROB = 0.3  # 挑战历史对手的概率
ALPHA_START = 0.5
ALPHA_MIN = 0.01
EPSILON_START = 1.0
EPSILON_MIN = 0.01


@dataclass(slots=True)
class TrainingLoopParams:
    episodes: int
    agent_x: QLearningAgent
    agent_o: QLearningAgent
    alpha_scheduler: Optional[LearningParamScheduler] = None
    epsilon_scheduler: Optional[LearningParamScheduler] = None
    snapshot_pool: Optional[SnapshotPool] = None


class TrainingLoop:
    """A class to manage the training loop of Q-learning agents."""

    def __init__(
        self,
        params: TrainingLoopParams,
        rng: Random = Random(0),
    ) -> None:
        assert params.episodes != 0
        self._episodes = params.episodes
        assert params.agent_x
        self._agent_x = params.agent_x
        assert params.agent_o
        self._agent_o = params.agent_o
        self.snapshot_pool = params.snapshot_pool or {
            "X": [params.agent_x.snapshot()],
            "O": [params.agent_o.snapshot()],
        }
        self.alpha_scheduler = params.alpha_scheduler or LearningParamScheduler(
            episodes=params.episodes,
            policy=LearningParamPolicy.EXPONENTIAL_DECAY,
            start_value=ALPHA_START,
            min_value=ALPHA_MIN,
            decay_value=pow(ALPHA_MIN / ALPHA_START, 1.0 / params.episodes),
        )
        self.epsilon_scheduler = params.epsilon_scheduler or LearningParamScheduler(
            episodes=params.episodes,
            policy=LearningParamPolicy.EXPONENTIAL_DECAY,
            start_value=EPSILON_START,
            min_value=EPSILON_MIN,
            decay_value=pow(EPSILON_MIN / EPSILON_START, 1.0 / params.episodes),
        )
        self.rng = rng
        self._reporter = TrainingReporter(
            self._agent_x, self._agent_o, self._episodes, self.snapshot_pool
        )

    def run(self) -> tuple[QLearningAgent, QLearningAgent]:
        """Run the training loop for a specified number of episodes."""
        start_time = time.time()

        for episode_idx in range(self._episodes):
            playing_x, playing_o = self.pick_opponents(episode_idx, self.rng)
            TrainingEpisode.run(playing_x, playing_o)
            self.update_learning_params(episode_idx)
            self._reporter.evaluate_and_snapshot_if_needed(episode_idx)

        end_time = time.time()
        report_training_result(start_time, end_time)
        return self._agent_x, self._agent_o

    def pick_opponents(
        self, episode_idx: int, rng: Random
    ) -> tuple[QLearningAgent, QLearningAgent]:
        """Pick two opponents from the snapshot pool."""
        playing_x = self._agent_x
        playing_o = self._agent_o

        if episode_idx % 2 == 0:
            pool = self.snapshot_pool[Player.PLAYER_X.value]
            playing_x = pick_agent(self._agent_x, pool, rng)
        else:
            pool = self.snapshot_pool[Player.PLAYER_O.value]
            playing_o = pick_agent(self._agent_o, pool, rng)

        return playing_x, playing_o

    def update_learning_params(self, episode_idx: int) -> None:
        """Adjust learning parameters for the agents."""
        agent_x = self._agent_x
        agent_o = self._agent_o
        alpha = self.alpha_scheduler.update(episode_idx)
        agent_x.alpha, agent_o.alpha = alpha, alpha
        epsilon = self.epsilon_scheduler.update(episode_idx)
        agent_x.epsilon, agent_o.epsilon = epsilon, epsilon


def pick_agent(
    active_agent: QLearningAgent, snapshot_pool: list[QLearningAgent], rng: Random
) -> QLearningAgent:
    """Pick an agent, either the active one or a historical snapshot."""
    picked = active_agent
    if snapshot_pool and rng.random() < HISTORICAL_OPPONENT_PROB:
        picked = rng.choice(snapshot_pool)
    return picked


def report_training_result(start_time: float, end_time: float) -> None:
    """Report the total training time."""
    print(f"\n训练完成，用时 {end_time - start_time:.2f} 秒。")
