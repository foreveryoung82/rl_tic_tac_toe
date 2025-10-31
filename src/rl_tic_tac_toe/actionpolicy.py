from enum import Enum, auto, unique


@unique
class ActionPolicy(Enum):
    GREEDY = auto()  # act as if epsilon == 0
    EPSILON_GREEDY = auto()  # act with agent's epsilon
    FULL_EXPLORATION = auto()  # act as if epsilon == 1.0
