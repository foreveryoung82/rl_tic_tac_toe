# tests/test_tictactoe.py
import pytest

from rl_tic_tac_toe.player import Player
from rl_tic_tac_toe.tictactoe import TicTacToe


@pytest.fixture
def game() -> TicTacToe:
    """Return a fresh TicTacToe instance for each test."""
    return TicTacToe()


@pytest.fixture
def two_games() -> tuple[TicTacToe, TicTacToe]:
    """Return two fresh TicTacToe instances for tests needing multiple games."""
    return TicTacToe(), TicTacToe()


@pytest.fixture
def three_games() -> tuple[TicTacToe, TicTacToe, TicTacToe]:
    """Return three fresh TicTacToe instances for tests needing multiple games."""
    return TicTacToe(), TicTacToe(), TicTacToe()


def test_initial_state(game: TicTacToe) -> None:
    # board should be empty
    assert game.board == [" "] * 9
    assert game.current_winner is None
    assert game.has_empty_cell()
    assert set(game.empty_cells()) == set(range(9))


def test_make_move_and_winner_row(game: TicTacToe) -> None:
    # X occupies the top row
    for i in range(3):
        assert game.make_move(i, Player.PLAYER_X)
    assert game.current_winner == Player.PLAYER_X
    # no further moves allowed on occupied squares
    assert not game.make_move(0, Player.PLAYER_O)


def test_make_move_and_winner_column(game: TicTacToe) -> None:
    for i in [0, 3, 6]:
        assert game.make_move(i, Player.PLAYER_O)
    assert game.current_winner == Player.PLAYER_O


def test_make_move_and_winner_diagonal(game: TicTacToe) -> None:
    for i in [0, 4, 8]:
        assert game.make_move(i, Player.PLAYER_X)
    assert game.current_winner == Player.PLAYER_X

    # reset board
    game = TicTacToe()
    for i in [2, 4, 6]:
        assert game.make_move(i, Player.PLAYER_O)
    assert game.current_winner == Player.PLAYER_O


def test_invalid_moves(game: TicTacToe) -> None:
    # Occupied square
    assert game.make_move(0, Player.PLAYER_X)
    assert not game.make_move(0, Player.PLAYER_O)

    # Out of range index raises IndexError (handled by caller)
    with pytest.raises(IndexError):
        game.make_move(9, Player.PLAYER_X)


def test_available_moves_and_empty_squares(game: TicTacToe) -> None:
    moves = set(range(9))
    for i in [1, 4, 7]:
        assert game.make_move(i, Player.PLAYER_X)
        moves.remove(i)
        assert set(game.empty_cells()) == moves
        assert game.has_empty_cell()
    # fill the rest
    for i in list(moves):
        game.make_move(i, Player.PLAYER_O)
    assert not game.has_empty_cell()
    assert game.empty_cells() == []


def test_winner_and_row_winner(game: TicTacToe) -> None:
    """A player wins by filling any complete row."""
    for start in (0, 3, 6):  # rows 0-2
        moves = [start + i for i in range(3)]
        for move in moves:
            game.make_move(move, Player.PLAYER_X)
        assert game.winner(moves[-1], Player.PLAYER_X) is True


def test_winner_and_column_winner(game: TicTacToe) -> None:
    """A player wins by filling any complete column."""
    for col_start in (0, 1, 2):  # columns 0-2
        moves = [col_start + i * 3 for i in range(3)]
        for move in moves:
            game.make_move(move, Player.PLAYER_O)
        assert game.winner(moves[-1], Player.PLAYER_O) is True


def test_winner_and_diagonal_winners(two_games: tuple[TicTacToe, TicTacToe]) -> None:
    """A player wins by filling either diagonal."""
    game_1, game_2 = two_games
    # main diagonal
    diag1 = [0, 4, 8]
    for move in diag1:
        game_1.make_move(move, Player.PLAYER_X)
    assert game_1.winner(diag1[-1], Player.PLAYER_X) is True

    # anti‑diagonal
    diag2 = [2, 4, 6]
    for move in diag2:
        game_2.make_move(move, Player.PLAYER_O)
    assert game_2.winner(diag2[-1], Player.PLAYER_O) is True


def test_winner_and_no_false_winner(
    three_games: tuple[TicTacToe, TicTacToe, TicTacToe],
) -> None:
    """Ensure that incomplete lines do not trigger a win."""
    game_1, game_2, game_3 = three_games
    # Horizontal half‑row
    game_1.make_move(0, Player.PLAYER_X)
    game_1.make_move(1, Player.PLAYER_X)
    assert game_1.winner(1, Player.PLAYER_X) is False

    # Vertical partial column
    game_2.make_move(3, Player.PLAYER_O)
    game_2.make_move(6, Player.PLAYER_O)
    assert game_2.winner(6, Player.PLAYER_O) is False

    # Diagonal missing one spot
    game_3.make_move(0, Player.PLAYER_X)
    game_3.make_move(4, Player.PLAYER_X)
    assert game_3.winner(4, Player.PLAYER_X) is False
