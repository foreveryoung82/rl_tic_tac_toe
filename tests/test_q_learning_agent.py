from random import Random

import pytest

from rl_tic_tac_toe.action_policy import ActionPolicy
from rl_tic_tac_toe.player import Player
from rl_tic_tac_toe.q_learning_agent import (
    ImmutableSnapshotError,
    QLearningAgent,
)
from rl_tic_tac_toe.tic_tac_toe import TicTacToe


@pytest.fixture
def agent_x() -> QLearningAgent:
    """Return a QLearningAgent for player X with a fixed random seed."""
    return QLearningAgent(
        Player.PLAYER_X, alpha=0.1, epsilon=0.1, gamma=0.9, rng=Random(0)
    )


@pytest.fixture
def game() -> TicTacToe:
    """Return a fresh TicTacToe instance for each test."""
    return TicTacToe()


def test_initialization() -> None:
    """Test if the agent is initialized with the correct parameters."""
    agent_x = QLearningAgent(Player.PLAYER_X)
    assert agent_x.player == Player.PLAYER_X
    assert agent_x.epsilon == 1.0
    assert not agent_x.is_snapshot


def test_get_state(agent_x: QLearningAgent, game: TicTacToe) -> None:
    """Test the state representation of the board."""
    game.board = ["X", " ", "O", " ", "X", " ", "O", " ", " "]
    expected_state = "X O X O  "
    assert agent_x.get_state(game) == expected_state


def test_get_and_update_q_value(agent_x: QLearningAgent, game: TicTacToe) -> None:
    """Test getting and setting Q-values."""
    state = "         "
    action = 1
    reward = 1
    next_state = " X       "

    # Initially, Q-value should be 0
    assert agent_x.get_q_value(state, action) == 0.0

    # Update Q-table
    game.board = list(next_state)
    agent_x.update_q_table(state, action, reward, next_state, game.empty_cells())

    # Q-value should be updated based on the formula (reward + gamma * next_max_q - old_q)
    # Here, next_max_q is 0, old_q is 0, so new_q = alpha * reward = 0.1 * 1 = 0.1
    assert agent_x.get_q_value(state, action) == pytest.approx(0.1)  # pyright: ignore[reportUnknownMemberType]


def test_choose_action_greedy(agent_x: QLearningAgent, game: TicTacToe) -> None:
    """Test choose_action with a greedy policy (epsilon=0)."""
    state = agent_x.get_state(game)
    # Set Q-values to make action 3 the clear winner
    board = " X       "
    game.board = list(board)
    agent_x.update_q_table(state, 1, 0.5, board, game.empty_cells())
    board = "   X     "
    game.board = list(board)
    agent_x.update_q_table(state, 3, 1.0, board, game.empty_cells())

    # With a greedy policy, it should always choose the best action
    game.board = [" " for _ in range(9)]
    action = agent_x.choose_action(game, ActionPolicy.GREEDY)
    assert action == 3  # noqa: PLR2004


def test_choose_action_full_exploration(
    agent_x: QLearningAgent, game: TicTacToe, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test full exploration policy's exploration path."""
    available_moves = game.empty_cells()
    mock_rng = Random(0)

    # Mock the random generator to always choose the 3rd available move
    monkeypatch.setattr(mock_rng, "choice", lambda moves: moves[2])  # pyright: ignore[reportUnknownArgumentType, reportUnknownLambdaType]
    agent_x.rng = mock_rng
    action = agent_x.choose_action(game, ActionPolicy.FULL_EXPLORATION)

    # It should explore, and our mock forces the choice to be the 3rd available move (index 2)
    assert action == available_moves[2]


def test_choose_action_epsilon_greedy_exploitation(
    agent_x: QLearningAgent, game: TicTacToe, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test epsilon-greedy policy's exploitation path."""
    state = agent_x.get_state(game)
    # Set Q-values to make action 5 the clear winner
    board = " X       "
    game.board = list(board)
    agent_x.update_q_table(state, 1, 0.5, board, game.empty_cells())
    board = "     X   "
    game.board = list(board)
    agent_x.update_q_table(state, 5, 1.0, board, game.empty_cells())

    # Mock the random generator to ensure exploitation
    mock_rng = Random(0)
    monkeypatch.setattr(mock_rng, "uniform", lambda a, b: 0.5)  # pyright: ignore[reportUnknownLambdaType, reportUnknownArgumentType]
    agent_x.rng = mock_rng

    # Epsilon is 0.1, uniform(0,1) returns 0.5, so 0.5 is not < 0.1, leading to exploitation
    game.board = [" " for _ in range(9)]
    action = agent_x.choose_action(game, ActionPolicy.EPSILON_GREEDY)
    assert action == 5  # noqa: PLR2004


def test_update_q_table_logic(agent_x: QLearningAgent, game: TicTacToe) -> None:
    """Verify the Q-learning update formula."""
    state = "         "
    action = 0
    reward = 1
    next_state = "X        "
    # Let's assume the next state has some Q-values
    board = "XO       "
    game.board = list(board)
    agent_x.update_q_table(
        next_state, 1, 0.5, board, game.empty_cells()
    )  # Q(next, 1) = 0.05
    board = "X O      "
    game.board = list(board)
    agent_x.update_q_table(
        next_state, 2, 1.0, board, game.empty_cells()
    )  # Q(next, 2) = 0.1

    # The max Q-value for the next state is 0.1
    # next_max_q = 0.1

    # Update the Q-table for the original state-action
    game.board = list(next_state)
    agent_x.update_q_table(state, action, reward, next_state, game.empty_cells())

    # old_q = 0
    # new_q = old_q + alpha * (reward + gamma * next_max_q - old_q)
    # new_q = 0 + 0.1 * (1 + 0.9 * 0.1 - 0)
    # new_q = 0.1 * (1 + 0.09) = 0.1 * 1.09 = 0.109
    expected_q_value = 0.109
    assert agent_x.get_q_value(state, action) == pytest.approx(expected_q_value)  # pyright: ignore[reportUnknownMemberType]


def test_snapshot(agent_x: QLearningAgent) -> None:
    """Test creating a snapshot of an agent."""
    snapshot = agent_x.snapshot()

    # Snapshot should be a different object
    assert snapshot is not agent_x

    # Snapshot should have exploration disabled
    assert snapshot.epsilon == 0
    assert snapshot.is_snapshot

    # Original agent should be unchanged
    assert agent_x.epsilon > 0
    assert not agent_x.is_snapshot


def test_snapshot_is_immutable(agent_x: QLearningAgent) -> None:
    """Test that modifying a snapshot raises an error."""
    snapshot = agent_x.snapshot()
    with pytest.raises(ImmutableSnapshotError):
        snapshot.epsilon = 0.5
    with pytest.raises(ImmutableSnapshotError):
        snapshot.alpha = 0.5
    with pytest.raises(ImmutableSnapshotError):
        snapshot.gamma = 0.5


def test_property_validation(agent_x: QLearningAgent) -> None:
    """Test that property setters raise ValueError for invalid values."""
    with pytest.raises(ValueError):
        agent_x.epsilon = 1.1
    with pytest.raises(ValueError):
        agent_x.epsilon = -0.1
    with pytest.raises(ValueError):
        agent_x.alpha = 1.1
    with pytest.raises(ValueError):
        agent_x.gamma = -0.1
