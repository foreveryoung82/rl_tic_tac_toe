from enum import StrEnum


class Player(StrEnum):
    PLAYER_X = "X"
    PLAYER_O = "O"

    def opponent(self) -> "Player":
        match self:
            case Player.PLAYER_X:
                return Player.PLAYER_O
            case Player.PLAYER_O:
                return Player.PLAYER_X
