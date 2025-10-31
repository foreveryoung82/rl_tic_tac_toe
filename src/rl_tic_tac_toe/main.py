from .gamebattle import GameBattle
from .player import Player
from .qlearningagent import QLearningAgent
from .trainingloop import TrainingLoop, TrainingLoopParams


def main() -> None:
    agent_x: QLearningAgent = QLearningAgent(Player.PLAYER_X)
    agent_o: QLearningAgent = QLearningAgent(Player.PLAYER_O)
    while True:
        print("\n--- 井字棋 AI 菜单 (联盟训练版) ---")
        print("1. 训练AI (首次使用推荐)")
        print("2. 与AI对战")
        print("3. 观看AI互相对战")
        print("4. 退出")
        choice = input("请输入您的选择: ")
        choice = choice.strip()
        if choice == "1":
            while True:
                episodes_str = input("请输入训练总局数 (默认: 200000): ")
                if episodes_str == "":
                    episodes = 200000
                    break
                try:
                    episodes = int(episodes_str)
                    if episodes > 0:
                        break
                    else:
                        print("请输入一个大于0的整数。")
                except ValueError:
                    print("无效输入，请输入一个数字。")
            params = TrainingLoopParams(episodes, agent_x, agent_o)
            train_loop = TrainingLoop(params)
            train_loop.run()
        elif choice == "2":
            GameBattle.play_vs_ai(agent_x, agent_o)
        elif choice == "3":
            GameBattle.ai_vs_ai(agent_x, agent_o)
        elif choice == "4":
            break
        else:
            print("无效选择，请重试。")


if __name__ == "__main__":
    main()
