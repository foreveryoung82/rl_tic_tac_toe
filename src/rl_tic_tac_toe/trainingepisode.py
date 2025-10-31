import itertools
from dataclasses import dataclass
from typing import Sequence

from .actionpolicy import ActionPolicy
from .player import Player
from .qlearningagent import QLearningAgent
from .tictactoe import TicTacToe

DRAW_GAME_REWARD = 0
WINNER_REWARD = 1
LOSER_REWARD = -1


@dataclass(slots=True)
class Decision:
    state: str
    action: int


History = list[tuple[Decision, QLearningAgent]]


class TrainingEpisode:
    @staticmethod
    def run(playing_x: QLearningAgent, playing_o: QLearningAgent) -> None:
        """Train a single episode between two agents."""
        game = TicTacToe()
        history: list[tuple[Decision, QLearningAgent]] = []
        agents = (playing_x, playing_o)

        # 游戏主循环
        for turn_idx, current_agent in enumerate(itertools.cycle(agents)):
            make_move_and_record(current_agent, game, history)
            # if is_ended(game):
            if game.is_ended():
                break
            opponent = agents[(turn_idx + 1) % 2]
            update_in_game_reward(opponent, game, history)

        # 处理游戏结束时的奖励
        if game.is_draw():
            update_draw_reward_for_both_agents(agents, game, history)
            return
        update_winner_reward(agents, game, history)
        update_loser_reward(agents, game, history)


def make_move_and_record(
    agent: QLearningAgent,
    game: TicTacToe,
    history: History,
) -> None:
    """Make a move for the agent and record the decision in history."""
    state = agent.get_state(game)
    action = agent.choose_action(game, ActionPolicy.EPSILON_GREEDY)
    game.make_move(action, agent.player)
    history.append((Decision(state, action), agent))


def has_moved_at_least_once(player: Player, history: History) -> bool:
    """Check if the specified player has made at least one move."""
    match player:
        case Player.PLAYER_X:
            if len(history) >= 1:
                return True
        case Player.PLAYER_O:
            # 因为轮流下棋，history>=2 可知：双方都下过至少一子
            if len(history) >= 2:  # noqa: PLR2004
                return True
    return False


def update_reward_if_active(
    agent: QLearningAgent,
    game: TicTacToe,
    decision: Decision,
    last_state: str,
    reward: float,
) -> None:
    """Update the reward for the agent if it is not a snapshot."""
    if agent.is_snapshot:
        return

    available_moves: list[int] = [] if game.current_winner else game.empty_cells()
    agent.update_q_table(
        decision.state, decision.action, reward, last_state, available_moves
    )


def update_winner_reward(
    agents: Sequence[QLearningAgent], game: TicTacToe, history: History
) -> None:
    """Update the reward for the winning agent."""
    winner_decision, winner = history[-1]
    winner_last_state = winner.get_state(game)
    update_reward_if_active(
        winner, game, winner_decision, winner_last_state, WINNER_REWARD
    )


def update_loser_reward(
    agents: Sequence[QLearningAgent], game: TicTacToe, history: History
) -> None:
    """Update the reward for the losing agent."""
    loser_decision, loser = history[-2]
    loser_last_state = loser.get_state(game)
    update_reward_if_active(loser, game, loser_decision, loser_last_state, LOSER_REWARD)


def update_draw_reward_for_both_agents(
    agents: Sequence[QLearningAgent], game: TicTacToe, history: History
) -> None:
    """Update the reward for both agents in case of a draw."""
    for i in range(2):
        decision, agent = history[-(i + 1)]
        last_state = agent.get_state(game)
        update_reward_if_active(agent, game, decision, last_state, DRAW_GAME_REWARD)


def update_in_game_reward(
    agent: QLearningAgent, game: TicTacToe, history: History
) -> None:
    """Update the in-game reward for the agent based on the current game state."""
    if not has_moved_at_least_once(agent.player, history):
        return
    decision, _ = history[-2]
    last_state = agent.get_state(game)
    update_reward_if_active(agent, game, decision, last_state, 0)
