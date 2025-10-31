from unittest.mock import MagicMock, patch

from pytest import fixture

from rl_tic_tac_toe.player import Player
from rl_tic_tac_toe.q_learning_agent import QLearningAgent
from rl_tic_tac_toe.snapshot_pool import SnapshotPool
from rl_tic_tac_toe.training_reporter import TrainingReporter


@fixture
def agent_x() -> QLearningAgent:
    return QLearningAgent(Player.PLAYER_X)


@fixture
def agent_o() -> QLearningAgent:
    return QLearningAgent(Player.PLAYER_O)


@fixture
def snapshot_pool(agent_x: QLearningAgent, agent_o: QLearningAgent) -> SnapshotPool:
    return {"X": [agent_x.snapshot()], "O": [agent_o.snapshot()]}


@fixture
def training_reporter(
    agent_x: QLearningAgent, agent_o: QLearningAgent, snapshot_pool: SnapshotPool
) -> TrainingReporter:
    return TrainingReporter(agent_x, agent_o, 100, snapshot_pool)


def test_run_snapshot(training_reporter: TrainingReporter) -> None:
    """Test the run_snapshot method."""
    reporter = training_reporter
    initial_snapshot_count = len(reporter._snapshot_pool[Player.PLAYER_X.value])

    with patch("builtins.print"):
        reporter._run_snapshot(0, Player.PLAYER_X)

    assert (
        len(reporter._snapshot_pool[Player.PLAYER_X.value])
        == initial_snapshot_count + 1
    )


@patch("rl_tic_tac_toe.training_reporter.Evaluator")
def test_evaluate_and_snapshot_if_needed(
    mock_evaluator: MagicMock,
    agent_x: QLearningAgent,
    agent_o: QLearningAgent,
    snapshot_pool: SnapshotPool,
) -> None:
    """Test the evaluate_and_snapshot_if_needed method of the TrainingLoop."""
    reporter = TrainingReporter(agent_x, agent_o, 100, snapshot_pool)

    # Mock the evaluator to avoid external dependencies
    mock_evaluator.evaluate_agents.return_value = {
        "x_wins": 1,
        "o_wins": 0,
        "draws": 0,
    }
    mock_evaluator.evaluate_vs_random.return_value = {
        "wins": 1,
        "losses": 0,
        "draws": 0,
    }

    with patch("builtins.print") as mock_print:
        reporter.evaluate_and_snapshot_if_needed(99)
        # Check that the evaluation methods were called
        assert mock_evaluator.evaluate_agents.call_count == 1
        assert mock_evaluator.evaluate_vs_random.call_count == 2

        # Check that the print function was called for progress reports
        assert mock_print.call_count > 0

        mock_evaluator.reset_mock()
        mock_print.reset_mock()

        reporter.evaluate_and_snapshot_if_needed(0)
        assert mock_evaluator.evaluate_agents.call_count == 0
        assert mock_evaluator.evaluate_vs_random.call_count == 0
        assert mock_print.call_count == 0
