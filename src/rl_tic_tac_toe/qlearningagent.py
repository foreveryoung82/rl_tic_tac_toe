"""A Tic-Tac-Toe Learning Agent by Q-learning algorithm."""

import copy
from random import Random

from .actionpolicy import ActionPolicy
from .player import Player
from .tictactoe import TicTacToe


class ImmutableSnapshotError(Exception):
    """Raised when attempting to modify a snapshot agent."""

    def __init__(self, message: str = "Cannot modify a snapshot agent."):
        self.message = message
        super().__init__(self.message)


class QLearningAgent:
    """A Tic-Tac-Toe Learning Agent by Q-learning algorithm."""

    def __init__(
        self,
        player: Player,
        alpha: float = 0.5,
        epsilon: float = 1.0,
        gamma: float = 0.9,
        rng: Random = Random(0),
    ):
        self._player: Player = player
        self._q_table: dict[str, dict[int, float]] = {}
        self._alpha = alpha
        self._gamma = gamma
        self._epsilon = epsilon
        self._rng = rng
        self._is_snapshot = False

    @property
    def player(self) -> Player:
        return self._player

    @property
    def epsilon(self) -> float:
        return self._epsilon

    @epsilon.setter
    def epsilon(self, value: float) -> None:
        if self.is_snapshot:
            raise ImmutableSnapshotError()
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{value}: Epsilon must be between 0 and 1.")
        self._epsilon = value

    @property
    def alpha(self) -> float:
        return self._alpha

    @alpha.setter
    def alpha(self, value: float) -> None:
        if self.is_snapshot:
            raise ImmutableSnapshotError()
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{value}: Alpha must be between 0 and 1.")
        self._alpha = value

    @property
    def gamma(self) -> float:
        return self._gamma

    @gamma.setter
    def gamma(self, value: float) -> None:
        if self.is_snapshot:
            raise ImmutableSnapshotError()
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{value}: Gamma must be between 0 and 1.")
        self._gamma = value

    @property
    def rng(self) -> Random:
        return self._rng

    @rng.setter
    def rng(self, value: Random) -> None:
        self._rng = value

    @property
    def is_snapshot(self) -> bool:
        return self._is_snapshot

    def get_state(self, game: TicTacToe) -> str:
        """Converts the current board state into a string representation."""
        return "".join(game.board)

    def get_q_value(self, state: str, action: int) -> float:
        """Retrieves the Q-value for a given state-action pair from the Q-table."""
        return self._q_table.get(state, {}).get(action, 0.0)

    def choose_action(self, game: TicTacToe, policy: ActionPolicy) -> int:
        """Select an action based on the current state and Q-values."""
        match policy:
            case ActionPolicy.GREEDY:
                return self.choose_action_greedy(game)
            case ActionPolicy.EPSILON_GREEDY:
                if self.rng.uniform(0, 1) < self.epsilon:
                    return self.choose_action_full_exploration(game)
                return self.choose_action_greedy(game)
            case ActionPolicy.FULL_EXPLORATION:
                return self.choose_action_full_exploration(game)

    def choose_action_greedy(self, game: TicTacToe) -> int:
        available = game.empty_cells()
        if not available:
            return self.choose_action_full_exploration(game)
        state = self.get_state(game)
        move_q_pairs = [(move, self.get_q_value(state, move)) for move in available]
        max_q = max(q for _, q in move_q_pairs)
        best_moves = [move for move, q in move_q_pairs if q == max_q]
        return self.rng.choice(best_moves)

    def choose_action_full_exploration(self, game: TicTacToe) -> int:
        available_moves = game.empty_cells()
        return self.rng.choice(available_moves)

    def update_q_table(
        self,
        state: str,
        action: int,
        reward: float,
        next_state: str,
        next_actions: list[int],
    ) -> None:
        """Updates the Q-value for a given state-action pair based on the reward and next state's maximum Q-value."""
        if self.is_snapshot:
            raise ImmutableSnapshotError()
        old_q = self.get_q_value(state, action)
        next_max_q = 0.0
        if next_actions:
            next_qs = (self.get_q_value(next_state, move) for move in next_actions)
            next_max_q = max(next_qs, default=0.0)
        new_q = old_q + self.alpha * (reward + self.gamma * next_max_q - old_q)
        if state not in self._q_table:
            self._q_table[state] = {}
        self._q_table[state][action] = new_q

    def snapshot(self) -> "QLearningAgent":
        """Creates a snapshot of the agent with exploration disabled."""
        snapshot = copy.deepcopy(self)
        snapshot._epsilon = 0  # 快照不进行探索
        snapshot._is_snapshot = True
        return snapshot
