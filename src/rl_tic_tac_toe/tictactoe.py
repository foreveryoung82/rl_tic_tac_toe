"""Tic Tac Toe game logic module."""

from typing import Optional

from .player import Player


class TicTacToe:
    """A class to represent the Tic-Tac-Toe game."""

    # ... (代码未变)
    def __init__(self) -> None:
        self.board: list[str] = [" " for _ in range(9)]
        self._current_winner: Optional[Player] = None

    @property
    def current_winner(self) -> Optional[Player]:
        return self._current_winner

    def print_board(self) -> None:
        """Prints the current state of the board."""
        for row in [self.board[i * 3 : (i + 1) * 3] for i in range(3)]:
            print("| " + " | ".join(row) + " |")

    @staticmethod
    def print_board_nums() -> None:
        """Prints the board with number positions for reference."""
        number_board = [[str(i) for i in range(j * 3, (j + 1) * 3)] for j in range(3)]
        for row in number_board:
            print("| " + " | ".join(row) + " |")

    def empty_cells(self) -> list[int]:
        """return empty cells in the board"""
        return [i for i, spot in enumerate(self.board) if spot == " "]

    def has_empty_cell(self) -> bool:
        """return True if there are empty squares in the board"""
        return " " in self.board

    def make_move(self, square: int, letter: Player) -> bool:
        """Make a move on the board if the square is available."""
        if self.board[square] == " ":
            self.board[square] = letter
            if self.winner(square, letter):
                self._current_winner = letter
            return True
        return False

    def winner(self, square: int, player: Player) -> bool:
        """Check if the given player is the winner."""
        row_ind = square // 3
        row = self.board[row_ind * 3 : (row_ind + 1) * 3]
        if all([spot == player for spot in row]):
            return True
        col_ind = square % 3
        column = [self.board[col_ind + i * 3] for i in range(3)]
        if all([spot == player for spot in column]):
            return True
        if square % 2 == 0:
            diagonal1 = [self.board[i] for i in [0, 4, 8]]
            if all([spot == player for spot in diagonal1]):
                return True
            diagonal2 = [self.board[i] for i in [2, 4, 6]]
            if all([spot == player for spot in diagonal2]):
                return True
        return False

    def is_ended(self) -> bool:
        """Check if the game has ended (win or draw)."""
        return not self.has_empty_cell() or self.current_winner is not None

    def is_draw(self) -> bool:
        """Check if the game ended in a draw."""
        return not self.has_empty_cell() and self.current_winner is None
