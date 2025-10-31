from .evaluator import Evaluator
from .player import Player
from .qlearningagent import QLearningAgent
from .snapshotpool import SnapshotPool


class TrainingReporter:
    def __init__(self, agent_x: QLearningAgent, agent_o: QLearningAgent, episodes: int, snapshot_pool: SnapshotPool) -> None:
        self._agent_x = agent_x
        self._agent_o = agent_o
        self._episodes = episodes
        self._snapshot_pool = snapshot_pool
        self.evaluation_interval = max((1, episodes // 20))

    def evaluate_and_snapshot_if_needed(self, episode_idx: int) -> None:
        if (episode_idx + 1) % self.evaluation_interval != 0:
            return
        self._run_evaluation(episode_idx)
        self._run_snapshot(episode_idx, Player.PLAYER_X)
        self._run_snapshot(episode_idx, Player.PLAYER_O)

    def _run_evaluation(self, episode_idx: int) -> None:
        """Run evaluation of agents and report the results."""
        progress_percent: float = ((episode_idx + 1) / self._episodes) * 100
        print(
            f"\n{'=' * 15} 训练进度: {progress_percent:.0f}% (回合 {episode_idx + 1}/{self._episodes}) {'=' * 15}"
        )
        ai_vs_ai_results = Evaluator.evaluate_agents(self._agent_x, self._agent_o)
        print("\n--- 评估: AI vs. AI ---")
        total_games = sum(ai_vs_ai_results.values())
        print(
            f"X 胜率: {ai_vs_ai_results['x_wins']} ({(ai_vs_ai_results['x_wins'] / total_games) * 100:.2f}%)"
        )
        print(
            f"O 胜率: {ai_vs_ai_results['o_wins']} ({(ai_vs_ai_results['o_wins'] / total_games) * 100:.2f}%)"
        )
        print(
            f"平局率: {ai_vs_ai_results['draws']} ({(ai_vs_ai_results['draws'] / total_games) * 100:.2f}%)"
        )

        x_vs_random_results = Evaluator.evaluate_vs_random(
            self._agent_x, Player.PLAYER_X
        )
        print("\n--- 评估: AI (X) vs. 随机玩家 ---")
        total_games = sum(x_vs_random_results.values())
        print(
            f"AI 胜率: {x_vs_random_results['wins']} / {total_games} ({(x_vs_random_results['wins'] / total_games) * 100:.2f}%)"
        )
        print(
            f"AI 败率: {x_vs_random_results['losses']} / {total_games} ({(x_vs_random_results['losses'] / total_games) * 100:.2f}%)"
        )
        print(
            f"平局率: {x_vs_random_results['draws']} / {total_games} ({(x_vs_random_results['draws'] / total_games) * 100:.2f}%)"
        )

        o_vs_random_results = Evaluator.evaluate_vs_random(
            self._agent_o, Player.PLAYER_O
        )
        print("\n--- 评估: AI (O) vs. 随机玩家 ---")
        total_games = sum(o_vs_random_results.values())
        print(
            f"AI 胜率: {o_vs_random_results['wins']} / {total_games} ({(o_vs_random_results['wins'] / total_games) * 100:.2f}%)"
        )
        print(
            f"AI 败率: {o_vs_random_results['losses']} / {total_games} ({(o_vs_random_results['losses'] / total_games) * 100:.2f}%)"
        )
        print(
            f"平局率: {o_vs_random_results['draws']} / {total_games} ({(o_vs_random_results['draws'] / total_games) * 100:.2f}%)"
        )

        print(f"{'=' * 50}")

    def _run_snapshot(self, episode_idx: int, player: Player) -> None:
        """Create a snapshot of the agent and add it to the opponent pool."""
        print(f"--- [系统]: 在回合 {episode_idx + 1} 创建 '{player}' 代理的快照 ---")
        agent = self._agent_x if player == Player.PLAYER_X else self._agent_o
        self._snapshot_pool[player.value].append(agent.snapshot())
