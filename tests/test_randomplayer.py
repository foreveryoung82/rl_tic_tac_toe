import pytest

from rl_tic_tac_toe.player import Player
from rl_tic_tac_toe.randomplayer import RandomPlayer
from rl_tic_tac_toe.tictactoe import TicTacToe


@pytest.fixture
def random_player_x() -> RandomPlayer:
    return RandomPlayer(Player.PLAYER_X)


@pytest.fixture
def game() -> TicTacToe:
    return TicTacToe()


class TestRandomPlayer:
    def test_choose_action_approximately_uniformly_random(
        self, game: TicTacToe, random_player_x: RandomPlayer
    ) -> None:
        """Test that RandomPlayer chooses actions approximately uniformly at random."""

        choice_num = 1000
        choice_count = [0 for _ in range(9)]
        for _ in range(choice_num):
            choice_count[random_player_x.choose_action(game)] += 1
        chances = (choice / choice_num for choice in choice_count)
        assert all((1.0 / 20) < chance < (1.0 / 4) for chance in chances)
