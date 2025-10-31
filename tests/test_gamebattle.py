from unittest.mock import MagicMock, call

import pytest

from rl_tic_tac_toe.gamebattle import (
    GameBattle,
    announce_game_result,
    ask_for_selecting_play_order,
    play_turn,
)
from rl_tic_tac_toe.player import Player
from rl_tic_tac_toe.qlearningagent import QLearningAgent
from rl_tic_tac_toe.tictactoe import TicTacToe


@pytest.fixture
def mock_game() -> TicTacToe:
    return MagicMock(spec=TicTacToe)


@pytest.fixture
def mock_agents() -> tuple[QLearningAgent, QLearningAgent]:
    agent_x = MagicMock(spec=QLearningAgent)
    agent_o = MagicMock(spec=QLearningAgent)
    return agent_x, agent_o


def test_ask_for_selecting_play_order(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("builtins.input", lambda _: "X")
    assert ask_for_selecting_play_order() == Player.PLAYER_O

    monkeypatch.setattr("builtins.input", lambda _: "O")
    assert ask_for_selecting_play_order() == Player.PLAYER_X


def test_announce_game_result(
    mock_game: TicTacToe, monkeypatch: pytest.MonkeyPatch
) -> None:
    mock_print = MagicMock()
    monkeypatch.setattr("builtins.print", mock_print)

    mock_game.current_winner = Player.PLAYER_X
    announce_game_result(mock_game)
    mock_print.assert_called_with(f"胜利者是 {Player.PLAYER_X}!")

    mock_game.current_winner = None
    announce_game_result(mock_game)
    mock_print.assert_called_with("平局!")


def test_play_turn_human(
    mock_game: TicTacToe,
    mock_agents: tuple[QLearningAgent, QLearningAgent],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mock_input = MagicMock(side_effect=["0"])
    monkeypatch.setattr("builtins.input", mock_input)
    mock_game.empty_cells.return_value = [0, 1, 2]

    play_turn(Player.PLAYER_X, Player.PLAYER_X, mock_agents, mock_game)

    mock_game.make_move.assert_called_with(0, Player.PLAYER_X)


def test_play_turn_ai(
    mock_game: TicTacToe,
    mock_agents: tuple[QLearningAgent, QLearningAgent],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mock_print = MagicMock()
    monkeypatch.setattr("builtins.print", mock_print)
    monkeypatch.setattr("time.sleep", lambda _: None)

    agent_x, agent_o = mock_agents
    agent_o.choose_action.return_value = 0
    play_turn(Player.PLAYER_O, Player.PLAYER_X, mock_agents, mock_game)

    agent_o.choose_action.assert_called()
    mock_game.make_move.assert_called_with(0, Player.PLAYER_O)


def test_ai_vs_ai(
    mock_agents: tuple[QLearningAgent, QLearningAgent], monkeypatch: pytest.MonkeyPatch
) -> None:
    mock_print = MagicMock()
    monkeypatch.setattr("builtins.print", mock_print)
    monkeypatch.setattr("time.sleep", lambda _: None)
    agent_x, agent_o = mock_agents

    def game_state_side_effect(*args, **kwargs):
        if agent_x.choose_action.call_count <= 3:
            return True
        return False

    agent_x.choose_action.side_effect = [0, 1, 2]
    agent_o.choose_action.side_effect = [3, 4]

    game = TicTacToe()
    GameBattle.ai_vs_ai(agent_x, agent_o)

    assert agent_x.choose_action.call_count == 3
    assert agent_o.choose_action.call_count == 2
    assert "胜利者是 X!" in [c[0][0] for c in mock_print.call_args_list]
