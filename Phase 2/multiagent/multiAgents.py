# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util
from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.
​
    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.
​
        getAction chooses among the best options according to the evaluation function.
​
        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.
​
        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.
​
        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.
​
        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
         # Calculate the distance to the nearest food
        foodDistances = [manhattanDistance(newPos, food) for food in newFood.asList()]
        minFoodDistance = min(foodDistances) if foodDistances else 0

        # Calculate the distance to the nearest ghost and check if it is in a scared state
        ghostDistances = []
        for i, ghostState in enumerate(newGhostStates):
            distance = manhattanDistance(newPos, ghostState.getPosition())
            if newScaredTimes[i] > 0:  # Ghost is scared
                distance *= -1
            ghostDistances.append(distance)

        minGhostDistance = min(ghostDistances) if ghostDistances else 0

        # Assign weights to different factors and calculate the final evaluation value
        foodWeight = 1.0
        ghostWeight = 2.0

        evaluationValue = successorGameState.getScore() - foodWeight * minFoodDistance + ghostWeight * minGhostDistance
        return evaluationValue
        return successorGameState.getScore()

def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.
​
    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.
​
    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.
​
    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState: GameState):
    #     """
    #     Returns the minimax action from the current gameState using self.depth
    #     and self.evaluationFunction.
    #
    #     Here are some method calls that might be useful when implementing minimax.
    #
    #     gameState.getLegalActions(agentIndex):
    #     Returns a list of legal actions for an agent
    #     agentIndex=0 means Pacman, ghosts are >= 1
    #
    #     gameState.generateSuccessor(agentIndex, action):
    #     Returns the successor game state after an agent takes an action
    #
    #     gameState.getNumAgents():
    #     Returns the total number of agents in the game
    #
    #     gameState.isWin():
    #     Returns whether or not the game state is a winning state
    #
    #     gameState.isLose():
    #     Returns whether or not the game state is a losing state
    #     """
    #     "*** YOUR CODE HERE ***"
    #     util.raiseNotDefined()
        def minimax(agent, depth, gameState):
            if gameState.isWin() or gameState.isLose() or depth == self.depth:
                return self.evaluationFunction(gameState), None
            
            if agent == 0:  # Pacman
                best_value = float('-inf')
                best_action = None
                for action in gameState.getLegalActions(agent):
                    successor = gameState.generateSuccessor(agent, action)
                    value, _ = minimax(1, depth, successor)
                    if value > best_value:
                        best_value = value
                        best_action = action
                return best_value, best_action
            else:  # Ghosts
                best_value = float('inf')
                next_agent = agent + 1
                if agent == gameState.getNumAgents() - 1:
                    next_agent = 0
                    depth += 1
                for action in gameState.getLegalActions(agent):
                    successor = gameState.generateSuccessor(agent, action)
                    value, _ = minimax(next_agent, depth, successor)
                    best_value = min(best_value, value)
                return best_value, None

        _, best_action = minimax(0, 0, gameState)
        return best_action

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        def alpha_beta(agent, depth, gameState, alpha, beta):
            if gameState.isWin() or gameState.isLose() or depth == self.depth:
                return self.evaluationFunction(gameState), None

            if agent == 0:  # Pacman
                best_value = float('-inf')
                best_action = None
                for action in gameState.getLegalActions(agent):
                    successor = gameState.generateSuccessor(agent, action)
                    value, _ = alpha_beta(1, depth, successor, alpha, beta)
                    if value > best_value:
                        best_value = value
                        best_action = action
                    alpha = max(alpha, best_value)
                    if beta <= alpha:
                        break
                return best_value, best_action
            else:  # Ghosts
                best_value = float('inf')
                next_agent = agent + 1
                if agent == gameState.getNumAgents() - 1:
                    next_agent = 0
                    depth += 1
                for action in gameState.getLegalActions(agent):
                    successor = gameState.generateSuccessor(agent, action)
                    value, _ = alpha_beta(next_agent, depth, successor, alpha, beta)
                    best_value = min(best_value, value)
                    beta = min(beta, best_value)
                    if beta <= alpha:
                        break
                return best_value, None

        _, best_action = alpha_beta(0, 0, gameState, float('-inf'), float('inf'))
        return best_action
class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction
​
        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        def expectimax(agent, depth, gameState):
            if gameState.isWin() or gameState.isLose() or depth == self.depth:
                return self.evaluationFunction(gameState), None
            if agent == 0:  # Pacman
                best_value = float('-inf')
                best_action = None
                for action in gameState.getLegalActions(agent):
                    successor = gameState.generateSuccessor(agent, action)
                    value, _ = expectimax(1, depth, successor)
                    if value > best_value:
                        best_value = value
                        best_action = action
                return best_value, best_action
            else:  # Ghosts
                value_sum = 0
                next_agent = agent + 1
                if agent == gameState.getNumAgents() - 1:
                    next_agent = 0
                    depth += 1
                legal_actions = gameState.getLegalActions(agent)
                num_actions = len(legal_actions)
                for action in legal_actions:
                    successor = gameState.generateSuccessor(agent, action)
                    value, _ = expectimax(next_agent, depth, successor)
                    value_sum += value
                return value_sum / num_actions, None

        _, best_action = expectimax(0, 0, gameState)
        return best_action

        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).
​
    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    pacmanPosition = currentGameState.getPacmanPosition()
    food = currentGameState.getFood()
    ghostStates = currentGameState.getGhostStates()
    scaredTimes = [ghostState.scaredTimer for ghostState in ghostStates]
    capsules = currentGameState.getCapsules()

    # Calculate the distance to the nearest food
    foodDistances = [manhattanDistance(pacmanPosition, foodPos) for foodPos in food.asList()]
    minFoodDistance = min(foodDistances) if foodDistances else 0

    # Calculate the distance to the nearest ghost and check if it is in a scared state
    ghostDistances = []
    for i, ghostState in enumerate(ghostStates):
        distance = manhattanDistance(pacmanPosition, ghostState.getPosition())
        if scaredTimes[i] > 0:  # Ghost is scared
            distance *= -1
        ghostDistances.append(distance)
    
    minGhostDistance = min(ghostDistances) if ghostDistances else 0

    # Calculate the distance to the nearest capsule
    capsuleDistances = [manhattanDistance(pacmanPosition, capsule) for capsule in capsules]
    minCapsuleDistance = min(capsuleDistances) if capsuleDistances else 0

    # Assign weights to different factors and calculate the final evaluation value
    # Original Values:
    # foodWeight = 1.0
    # ghostWeight = 2.0
    # capsuleWeight = 3.0
    # remainingFoodWeight = 10.0
    # remainingCapsuleWeight = 20.0
    
    
    foodWeight = 1.0
    ghostWeight = 2.0
    capsuleWeight = 3.0
    remainingFoodWeight = 10.0
    remainingCapsuleWeight = 20.0
    
    ghostThreshold = 3
    capsuleThreshold = 2
     
    # increase weight for ghosts if a ghost is nearby
    # increase weight for capsules if a ghost is nearby and a capsule is close
    if min(ghostDistances) < ghostThreshold and minCapsuleDistance < capsuleThreshold:
        capsuleWeight = 5.0  
        ghostWeight = 4.0
    elif min(ghostDistances) < ghostThreshold:
        ghostWeight = 5.0

    evaluationValue = currentGameState.getScore() \
        - foodWeight * minFoodDistance \
        + ghostWeight * minGhostDistance \
        - capsuleWeight * minCapsuleDistance \
        - remainingFoodWeight * len(foodDistances) \
        - remainingCapsuleWeight * len(capsules)

    return evaluationValue
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction