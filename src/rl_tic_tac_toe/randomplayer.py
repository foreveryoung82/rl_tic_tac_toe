from random import Random

from .player import Player
from .tictactoe import TicTacToe


class RandomPlayer:
    """
    A simple random player that makes random moves on the Tic-Tac-Toe board.
    This player is used as a baseline opponent during training and evaluation.
    It does not learn from the game state and always selects a random available move.
    """

    def __init__(self, player: Player, rng: Random = Random(0)) -> None:
        self.player: Player = player
        self.rng = rng

    def choose_action(self, game: "TicTacToe") -> int:
        return self.rng.choice(game.empty_cells())
