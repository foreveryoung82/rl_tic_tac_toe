import time

from .actionpolicy import ActionPolicy
from .player import Player
from .qlearningagent import QLearningAgent
from .tictactoe import TicTacToe


class GameBattle:
    @staticmethod
    def play_vs_ai(agent_x: QLearningAgent, agent_o: QLearningAgent) -> None:
        game = TicTacToe()
        player_letter: Player = ask_for_selecting_play_order()
        turn: Player = Player.PLAYER_X
        while game.has_empty_cell() and not game.current_winner:
            game.print_board()
            TicTacToe.print_board_nums()
            play_turn(turn, player_letter, (agent_x, agent_o), game)
            turn = turn.opponent()
        game.print_board()
        announce_game_result(game)

    @staticmethod
    def ai_vs_ai(agent_x: QLearningAgent, agent_o: QLearningAgent) -> None:
        game = TicTacToe()
        turn = Player.PLAYER_X
        while game.has_empty_cell() and not game.current_winner:
            if turn == Player.PLAYER_X:
                action = agent_x.choose_action(game, ActionPolicy.GREEDY)
                game.make_move(action, Player.PLAYER_X)
            else:
                action = agent_o.choose_action(game, ActionPolicy.GREEDY)
                game.make_move(action, Player.PLAYER_O)
            print(f"AI '{turn}' 落子:")
            game.print_board()
            print("-" * 10)
            time.sleep(0.5)
            turn = turn.opponent()
        if game.current_winner:
            print(f"胜利者是 {game.current_winner}!")
        else:
            print("平局!")


def ask_for_selecting_play_order() -> Player:
    """Asks the user to select whether they want to play as 'X' or 'O'."""
    while True:
        user_input = input("您想玩 'X' (先手)还是 'O' (后手)? ").upper()
        match user_input:
            case player if player in Player:
                assert isinstance(player, Player)
                return player.opponent()
            case _:
                continue


def play_turn(
    turn: Player,
    player_letter: Player,
    agents: tuple[QLearningAgent, QLearningAgent],
    game: TicTacToe,
) -> None:
    """Plays a single turn between the human player and the AI."""
    ai_letter: Player = player_letter.opponent()
    if turn == player_letter:
        move = -1
        while move not in game.empty_cells():
            try:
                move = int(input(f"您的回合 ({player_letter}). 请选择一个位置 (0-8): "))
                if move not in game.empty_cells():
                    print("无效的移动，请重试。")
            except ValueError:
                print("无效输入，请输入数字。")
        game.make_move(move, player_letter)
    else:
        print(f"\nAI的回合 ({ai_letter})...")
        agent = agents[0] if ai_letter == Player.PLAYER_X else agents[1]
        action = agent.choose_action(game, ActionPolicy.GREEDY)
        game.make_move(action, ai_letter)
        time.sleep(0.5)


def announce_game_result(game: TicTacToe) -> None:
    """Announces the result of the game."""
    if game.current_winner:
        print(f"胜利者是 {game.current_winner}!")
    else:
        print("平局!")
