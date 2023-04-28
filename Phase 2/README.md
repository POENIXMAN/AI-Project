# Pacman Multi-Agent Competition

Welcome to the Pacman Multi-Agent Competition project!

In this phase of the project, the goal is to compete as Pacman against several Ghosts in a multi-agent environment. Pacman seeks to eat as many dots as possible while avoiding being eaten by Ghosts. Initially, several Ghosts appear on the maze, and only one of them starts moving. Then, every three seconds, an additional Ghost starts moving. The number of Ghosts is relative to the size of the maze, which could be small, medium, or large.

## Maze Selection

The code files include a sample of mazes of different sizes. Every maze has its specific number of ghosts to consider. The following is the command line to specify the maze name (after -l) and the ghost number (-k).

The maze names and their respective number of ghosts are:

- `smallClassic`, 10 ghosts
- `trappedClassic`, 2 ghosts
- `minimaxClassic`, 2 ghosts
- `mediumClassic`, 12 ghosts
- `capsuleClassic`, 7 ghosts
- `contestclassic`, 12 ghosts

## Evaluation

Each group will play three consecutive rounds of Pacman on randomly selected mazes. The assessment will be based on the quality of your proposed solution as well as the outcomes of the games your agent plays in the competition. That will be in terms of the number of times that you win out of three games, the score of each game, and the amount of time through which you manage to keep Pacman alive in each game.

## Getting Started

1. Clone this repository.
2. Install the required dependencies by running `pip install -r requirements.txt`.
3. Run the Pacman game by running `python pacman.py -p YourAgentName -l MazeName -k GhostNumber`. Replace `YourAgentName` with the name of your agent class, `MazeName` with the name of the maze, and `GhostNumber` with the number of ghosts in the maze.
4. Implement your agent class in the `multiAgents.py` file.

## Agent Class

The agent class should inherit from `MultiAgentSearchAgent` in the `multiAgents.py` file. The `getAction` method should return an action for Pacman to take based on the current game state. The game state is represented by a `GameState` object, which can be accessed through the `self.gameState` attribute.

You may implement any algorithms and techniques you see fit for your agent. However, keep in mind that the game will be played in real-time, so your agent should be able to make decisions quickly.



## Resources

- [Pacman Multi-Agent Project Description](http://ai.berkeley.edu/project_overview.html)
- [Pacman Multi-Agent Project FAQ](http://ai.berkeley.edu/contact.html)

