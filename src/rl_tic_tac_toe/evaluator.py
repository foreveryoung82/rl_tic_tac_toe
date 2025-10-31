from .actionpolicy import ActionPolicy
from .player import Player
from .qlearningagent import QLearningAgent
from .randomplayer import RandomPlayer
from .tictactoe import TicTacToe


class Evaluator:
    @staticmethod
    def evaluate_agents(
        agent_x: QLearningAgent, agent_o: QLearningAgent, num_games: int = 1000
    ) -> dict[str, int]:
        """
        Evaluates two Q-learning agents playing against each other in num_games rounds.
        Disables exploration (epsilon=0) to assess performance under optimal play.
        Reports win/draw statistics.
        """
        # ... (代码未变)
        x_wins, o_wins, draws = 0, 0, 0
        for _ in range(num_games):
            game = TicTacToe()
            turn = Player.PLAYER_X
            while game.has_empty_cell() and not game.current_winner:
                if turn == Player.PLAYER_X:
                    action = agent_x.choose_action(game, ActionPolicy.GREEDY)
                    game.make_move(action, Player.PLAYER_X)
                else:
                    action = agent_o.choose_action(game, ActionPolicy.GREEDY)
                    game.make_move(action, Player.PLAYER_O)
                turn = turn.opponent()
            if game.current_winner == Player.PLAYER_X:
                x_wins += 1
            elif game.current_winner == Player.PLAYER_O:
                o_wins += 1
            else:
                draws += 1
        return {"x_wins": x_wins, "o_wins": o_wins, "draws": draws}

    @staticmethod
    def evaluate_vs_random(
        agent_to_test: QLearningAgent,
        agent_letter: Player,
        num_games: int = 1000,
    ) -> dict[str, int]:
        """Evaluates the AI's performance against a random player over num_games rounds."""
        random_player = RandomPlayer(agent_letter.opponent())
        wins, losses, draws = 0, 0, 0
        for _ in range(num_games):
            game = TicTacToe()
            turn = Player.PLAYER_X
            while game.has_empty_cell() and not game.current_winner:
                if turn == agent_letter:
                    action = agent_to_test.choose_action(game, ActionPolicy.GREEDY)
                    game.make_move(action, agent_letter)
                else:
                    action = random_player.choose_action(game)
                    game.make_move(action, random_player.player)
                turn = turn.opponent()
            if game.current_winner == agent_letter:
                wins += 1
            elif not game.current_winner:
                draws += 1
            else:
                losses += 1
        return {"wins": wins, "losses": losses, "draws": draws}
