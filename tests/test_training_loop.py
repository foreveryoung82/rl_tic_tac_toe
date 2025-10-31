from random import Random
from unittest.mock import MagicMock, patch

from pytest import fixture

from rl_tic_tac_toe.player import Player
from rl_tic_tac_toe.q_learning_agent import QLearningAgent
from rl_tic_tac_toe.snapshot_pool import SnapshotPool
from rl_tic_tac_toe.training_loop import TrainingLoop, TrainingLoopParams


@fixture
def agent_x() -> QLearningAgent:
    return QLearningAgent(Player.PLAYER_X)


@fixture
def agent_o() -> QLearningAgent:
    return QLearningAgent(Player.PLAYER_O)


@fixture
def params(agent_x: QLearningAgent, agent_o: QLearningAgent) -> TrainingLoopParams:
    return TrainingLoopParams(episodes=100, agent_x=agent_x, agent_o=agent_o)


@fixture
def snapshot_pool(agent_x: QLearningAgent, agent_o: QLearningAgent) -> SnapshotPool:
    return {"X": [agent_x.snapshot()], "O": [agent_o.snapshot()]}


@fixture
def training_loop(params: TrainingLoopParams) -> TrainingLoop:
    return TrainingLoop(params)


@patch("rl_tic_tac_toe.training_loop.TrainingReporter")
@patch("rl_tic_tac_toe.training_loop.TrainingEpisode")
def test_training_loop_run(
    mock_training_episode: MagicMock,
    mock_training_reporter: MagicMock,
    agent_x: QLearningAgent,
    agent_o: QLearningAgent,
    snapshot_pool: SnapshotPool,
) -> None:
    """Test the run method of the TrainingLoop."""
    params = TrainingLoopParams(episodes=100, agent_x=agent_x, agent_o=agent_o)
    loop = TrainingLoop(params, rng=Random(42))

    with patch("builtins.print") as mock_print:
        training_x, training_o = loop.run()

        # Check that the agents have been trained
        assert training_x.player == Player.PLAYER_X
        assert training_o.player == Player.PLAYER_O

        # Check that the learning parameters have decayed
        assert training_x.epsilon < 1.0
        assert training_o.epsilon < 1.0

        # Check that the evaluation methods were called
        assert mock_training_reporter.call_count == 1
        assert (
            mock_training_reporter.return_value.evaluate_and_snapshot_if_needed.call_count
            == 100
        )

        # Check that the print function was called for progress reports
        assert mock_print.call_count > 0

        assert mock_training_episode.run.call_count == 100


@patch("rl_tic_tac_toe.training_loop.pick_agent")
def test_pick_opponents(mock_pick_agent: MagicMock, params: TrainingLoopParams) -> None:
    """Test the pick_opponents method."""
    loop = TrainingLoop(params, rng=Random(42))

    # Mock pick_agent to return a snapshot
    snapshot_x = params.agent_x.snapshot()
    mock_pick_agent.return_value = snapshot_x
    picked_x, picked_o = loop.pick_opponents(0, loop.rng)
    assert picked_x.is_snapshot
    assert not picked_o.is_snapshot

    # Mock pick_agent to return the active agent
    mock_pick_agent.return_value = params.agent_x
    picked_x, picked_o = loop.pick_opponents(0, loop.rng)
    assert not picked_x.is_snapshot
    assert not picked_o.is_snapshot


def test_adjust_learning_params(params: TrainingLoopParams) -> None:
    """Test the adjust_learning_params method."""
    loop = TrainingLoop(params)
    agent_x = params.agent_x
    initial_alpha = agent_x.alpha
    initial_epsilon = agent_x.epsilon

    loop.update_learning_params(0)

    assert agent_x.alpha < initial_alpha
    assert agent_x.epsilon < initial_epsilon


def test_init_with_custom_args(
    agent_x: QLearningAgent, agent_o: QLearningAgent
) -> None:
    snapshot_pool: SnapshotPool = {"X": [agent_x.snapshot()], "O": [agent_o.snapshot()]}
    params = TrainingLoopParams(
        episodes=10000,
        agent_x=agent_x,
        agent_o=agent_o,
        snapshot_pool=snapshot_pool,
        alpha_scheduler=None,
        epsilon_scheduler=None,
    )
    assert TrainingLoop(params=params, rng=Random(0))
