from typing import TypedDict

from .q_learning_agent import QLearningAgent


class SnapshotPool(TypedDict, total=True):
    X: list[QLearningAgent]
    O: list[QLearningAgent]  # noqa: E741
