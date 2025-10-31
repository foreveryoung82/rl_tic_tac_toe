from rl_tic_tac_toe.player import Player


def test_opponent() -> None:
    assert Player.PLAYER_X.opponent() == Player.PLAYER_O
    assert Player.PLAYER_O.opponent() == Player.PLAYER_X
