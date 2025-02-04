# Multi-Agent Search: Pacman with Ghosts

## Introduction
This project involves designing agents for a multi-agent version of Pacman, where Pacman must navigate the maze while being pursued by ghosts. I implemented several adversarial search algorithms, including Minimax, Alpha-Beta Pruning, and Expectimax, along with an improved evaluation function to enhance Pacman's decision-making abilities.

## Table of Contents
- [Introduction](#introduction)
- [Q1: Reflex Agent](#q1-reflex-agent)
- [Q2: Minimax](#q2-minimax)
- [Q3: Alpha-Beta Pruning](#q3-alpha-beta-pruning)
- [Q4: Expectimax](#q4-expectimax)
- [Q5: Evaluation Function](#q5-evaluation-function)
- [Submission](#submission)

## Q1: Reflex Agent
Implemented a reflex agent that evaluates state-action pairs to make decisions. The agent considers the locations of food and ghosts, attempting to maximize score while avoiding danger. The agent successfully performs well on `testClassic` layout.

### Execution Commands:
```bash
python pacman.py -p ReflexAgent -l testClassic
python pacman.py -p ReflexAgent -k 1 --frameTime 0
python pacman.py -p ReflexAgent -k 2 --frameTime 0
```

## Q2: Minimax
Developed a Minimax agent that assumes ghosts play optimally. The agent recursively explores the game tree up to a given depth, selecting moves that minimize the worst-case outcome.

### Execution Commands:
```bash
python pacman.py -p MinimaxAgent -l minimaxClassic -a depth=4
python autograder.py -q q2 --no-graphics
```

## Q3: Alpha-Beta Pruning
Optimized the Minimax algorithm with Alpha-Beta pruning to reduce the number of states explored, leading to improved performance.

### Execution Commands:
```bash
python pacman.py -p AlphaBetaAgent -a depth=3 -l smallClassic
python autograder.py -q q3 --no-graphics
```

## Q4: Expectimax
Implemented the Expectimax algorithm to model the probabilistic behavior of ghosts. Unlike Minimax, this approach does not assume the worst-case scenario but rather considers the expected utility of each move.

### Execution Commands:
```bash
python pacman.py -p ExpectimaxAgent -l minimaxClassic -a depth=3
python autograder.py -q q4 --no-graphics
```

## Q5: Evaluation Function
Developed a more sophisticated evaluation function that evaluates states rather than actions, improving Pacman’s decision-making abilities. The function considers factors such as proximity to food, ghosts, and power pellets.

### Execution Commands:
```bash
python autograder.py -q q5 --no-graphics
```

## Submission
1. Submitted the project as a ZIP file `multiagent_sol.zip` on Canvas, containing all `.py` files.
2. Included details of any discussions or collaborations in the assignment comments.
3. Noted any late days used for submission.

This project enhanced my understanding of adversarial search algorithms and their practical applications in AI-driven decision-making.
