# RL Tic-Tac-Toe

This project is a command-line implementation of Tic-Tac-Toe where a user can play against a Reinforcement Learning (RL) agent. The agent is trained using the Q-learning algorithm. The project also allows for training the AI agents and watching them play against each other.

## Features

*   Play Tic-Tac-Toe against an AI.
*   Train the AI through self-play.
*   Watch two AI agents play against each other.
*   Command-line interface.

## Installation

1.  Ensure you have Python 3.12+ and `poetry` installed.
2.  Clone the repository:
    ```bash
    git clone https://github.com/your-username/rl-tic-tac-toe.git
    ```
3.  Install dependencies:
    ```bash
    poetry install
    ```

## Usage

To start the interactive game menu, run:
```bash
poetry run python src/rl_tic_tac_toe/main.py
```
From the menu, you can choose to:
1.  Train the AI agents.
2.  Play against a trained AI.
3.  Watch two AI agents play against each other.

## Running Tests

To execute the test suite, run:
```bash
poetry run pytest
```

## Development

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
