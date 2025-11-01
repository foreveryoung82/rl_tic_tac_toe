# Development Roadmap

This document outlines the development roadmap for the RL Tic-Tac-Toe project.

## Release 0.1

*   Implement a basic training framework for the Q-learning algorithm.
*   Allow a human player to play against the AI.
*   Allow observation of two AI agents playing against each other.

## Release 0.2

*   Automatically generate game record files from AI vs. AI matches.
*   Use the generated game record files for training the AI agents.

## Release 0.3

*   Enable multi-threaded or multi-process generation of AI match records to improve performance.

## Release 0.4

*   Decouple learning from game record generation.
*   Perform learning in a single thread while game records are generated in parallel.

## Release 0.5

*   Focus on performance optimizations, including speed and memory usage.

## Release 1.0

*   Deliver a command-line framework that runs self-matches and learns in parallel using multiple threads or processes.
*   Achieve a stable and performant version of the RL Tic-Tac-Toe agent.
