from unittest.mock import MagicMock

import pytest

from rl_tic_tac_toe.evaluator import Evaluator
from rl_tic_tac_toe.player import Player
from rl_tic_tac_toe.q_learning_agent import QLearningAgent


@pytest.fixture
def mock_agents() -> tuple[QLearningAgent, QLearningAgent]:
    agent_x = MagicMock(spec=QLearningAgent)
    agent_o = MagicMock(spec=QLearningAgent)
    return agent_x, agent_o


def test_evaluate_agents(
    mock_agents: tuple[QLearningAgent, QLearningAgent], monkeypatch: pytest.MonkeyPatch
) -> None:
    mock_print = MagicMock()
    monkeypatch.setattr("builtins.print", mock_print)
    agent_x, agent_o = mock_agents

    # Let agent_x win in the first round and draw in the second round
    agent_x.choose_action.side_effect = [0, 1, 2, 0, 1, 8, 2, 5, 7]  # type: ignore
    agent_o.choose_action.side_effect = [3, 4, 5, 6, 7, 8]  # type: ignore

    result = Evaluator.evaluate_agents(agent_x, agent_o, num_games=2)

    assert result == {"x_wins": 2, "o_wins": 0, "draws": 0}


def test_evaluate_vs_random(monkeypatch: pytest.MonkeyPatch) -> None:
    mock_print = MagicMock()
    monkeypatch.setattr("builtins.print", mock_print)
    agent_to_test = MagicMock(spec=QLearningAgent)
    agent_letter = Player.PLAYER_X

    # Let the agent win in the first round and lose in the second round
    agent_to_test.choose_action.side_effect = [0, 1, 2, 0, 1, 5, 7, 8]
    monkeypatch.setattr(
        "rl_tic_tac_toe.random_player.RandomPlayer.choose_action",
        MagicMock(side_effect=[3, 4, 3, 4, 6, 8, 2, 5]),
    )

    result = Evaluator.evaluate_vs_random(agent_to_test, agent_letter, num_games=2)

    assert result == {"wins": 1, "losses": 1, "draws": 0}
