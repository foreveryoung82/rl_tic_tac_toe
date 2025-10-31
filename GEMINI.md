# Project: RL Tic-Tac-Toe

## Project Overview

This project is a command-line implementation of Tic-Tac-Toe where a user can play against a Reinforcement Learning (RL) agent. The agent is trained using the Q-learning algorithm. The project also allows for training the AI agents and watching them play against each other.

The project is built in Python and uses `poetry` for dependency management and packaging.

**Key Technologies:**
- Python 3.12+
- Poetry
- Pytest (for testing)
- Ruff (for linting)
- MyPy (for static type checking)

**Architecture:**
- `src/rl_tic_tac_toe/main.py`: The main entry point of the application, providing a user menu to interact with the game.
- `src/rl_tic_tac_toe/tictactoe.py`: Contains the core logic for the Tic-Tac-Toe game board and rules.
- `src/rl_tic_tac_toe/qlearningagent.py`: Implements the Q-learning agent, which learns to play the game. It manages the Q-table and decision-making process.
- `src/rl_tic_tac_toe/trainingloop.py`: Orchestrates the training process where two AI agents play against each other for many episodes to learn the optimal strategy.
- `tests/`: Contains unit tests for the project, written using the `pytest` framework.

## Building and Running

### Installation

1.  Ensure you have Python 3.12+ and `poetry` installed.
2.  Install dependencies:
    ```bash
    poetry install
    ```

### Running the Application

To start the interactive game menu, run:
```bash
poetry run python src/rl_tic_tac_toe/main.py
```
From the menu, you can choose to:
1.  Train the AI agents.
2.  Play against a trained AI.
3.  Watch two AI agents play against each other.

### Running Tests

To execute the test suite, run:
```bash
poetry run pytest
```

## Development Conventions

### Code Style and Linting

This project uses `Ruff` for code formatting and linting. To check the code for style issues, run:
```bash
poetry run ruff check .
```

### Static Type Checking

This project uses `MyPy` for static type checking. To check for type errors, run:
```bash
poetry run mypy .
```
