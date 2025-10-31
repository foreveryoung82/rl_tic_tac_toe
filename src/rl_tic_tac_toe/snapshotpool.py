from typing import TypedDict

from .qlearningagent import QLearningAgent


class SnapshotPool(TypedDict, total=True):
    X: list[QLearningAgent]
    O: list[QLearningAgent]  # noqa: E741
