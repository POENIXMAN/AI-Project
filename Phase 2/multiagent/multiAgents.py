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
    DESCRIPTION: 
    
    This function evaluates the current state of the game by taking into account several factors,
    including the distance to the nearest food, the distance to the nearest ghost, the distance to the nearest capsule,
    the number of remaining food pellets, the number of remaining capsules, and the distance to the nearest junction.
    
    In this evaluation function, we consider the following factors:
    - The current score
    - The distance to the nearest food pellet
    - The number of remaining food pellets
    - The number of remaining capsules
    - The distance to the nearest ghost (or the nearest scared ghost)
    - The distance to the nearest capsule
    - The distance to the nearest junction (if no food is nearby)

    We adjust the weights of these factors depending on the game state. Specifically:
    - We increase the weight for ghosts and capsules if a ghost is nearby and a capsule is close.
    - We increase the weight for ghosts if a ghost is nearby and there are no capsules left.
    - we increase the weight of the capsules and decrease that of the ghost if the nearest capsule is closer than the nearest ghost

    We also modify the evaluation function to incentivize the agent to move towards the nearest
    junction when there is no food nearby. This encourages the agent to explore more of the maze
    and potentially find a closer source of food.
    
    """
    "*** YOUR CODE HERE ***"
    pacmanPosition = currentGameState.getPacmanPosition()
    food = currentGameState.getFood()
    ghostStates = currentGameState.getGhostStates()
    scaredTimes = [ghostState.scaredTimer for ghostState in ghostStates]
    capsules = currentGameState.getCapsules()
    walls = currentGameState.getWalls()

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
    
    # Calculate the distance to the nearest junction
    junctionDistances = []
    for x in range(walls.width):
        for y in range(walls.height):
            if not walls[x][y]:
                junctionDistances.append(manhattanDistance(pacmanPosition, (x, y)))
    minJunctionDistance = min(junctionDistances) if junctionDistances else 0

    # Assign weights to different factors and calculate the final evaluation value
    # Original Values:
    # foodWeight = 1.0
    # ghostWeight = 2.0
    # capsuleWeight = 3.0
    # remainingFoodWeight = 10.0
    # remainingCapsuleWeight = 20.0
    
    foodWeight = 4.0
    ghostWeight = 3.0
    capsuleWeight = 5.0
    junctionWeight = 2.0
    remainingFoodWeight = 15.0
    remainingCapsuleWeight = 10.0
    
    ghostThreshold = 4
    capsuleThreshold = 4
     
    # increase weight of ghosts if a ghost is nearby
    # increase weight of capsules if a ghost is nearby and a capsule is close
    if minGhostDistance < ghostThreshold and minCapsuleDistance < capsuleThreshold:
        # if the nearest capsule is closer than the nearest ghost increase the weight of the capsules and decrease that of the ghost
        if minCapsuleDistance < minGhostDistance:
            capsuleWeight = 12.0
            remainingCapsuleWeight = 30.0 
            ghostWeight = 2.0
        else:     
            capsuleWeight = 6.0
            remainingCapsuleWeight = 30.0  
            ghostWeight = 7.0
    elif minGhostDistance < ghostThreshold:
        ghostWeight = 8.0
        
    
    # increase weight of junction when there is no food nearby       
    if minFoodDistance == 0:
        junctionWeight = 10.0       

    evaluationValue = currentGameState.getScore() \
        - foodWeight * minFoodDistance \
        + ghostWeight * minGhostDistance \
        - capsuleWeight * minCapsuleDistance \
        - junctionWeight * minJunctionDistance \
        - remainingFoodWeight * len(foodDistances) \
        - remainingCapsuleWeight * len(capsules)

    return evaluationValue
    util.raiseNotDefined()    

# Abbreviation
better = betterEvaluationFunction


# # Code from phase 1 to use mazeDistance

# def mazeDistance(point1 , point2 , gameState: GameState) -> int:
#     """
#     Returns the maze distance between any two points, using the search functions
#     you have already built. The gameState can be any game state -- Pacman's
#     position in that state is ignored.

#     Example usage: mazeDistance( (2,4), (5,6), gameState)

#     This might be a useful helper function for your ApproximateSearchAgent.
#     """
#     x1, y1 = point1
#     x2, y2 = point2
#     walls = gameState.getWalls()
#     assert not walls[x1][y1], 'point1 is a wall: ' + str(point1)
#     assert not walls[x2][y2], 'point2 is a wall: ' + str(point2)
#     prob = PositionSearchProblem(gameState, start=point1, goal=point2, warn=False, visualize=False)
#     return len(search.astar(prob))

    
# class Actions:
#     """
#     A collection of static methods for manipulating move actions.
#     """
#     # Directions
#     _directions = {Directions.NORTH: (0, 1),
#                    Directions.SOUTH: (0, -1),
#                    Directions.EAST:  (1, 0),
#                    Directions.WEST:  (-1, 0),
#                    Directions.STOP:  (0, 0)}

#     _directionsAsList = _directions.items()

#     TOLERANCE = .001

#     def reverseDirection(action):
#         if action == Directions.NORTH:
#             return Directions.SOUTH
#         if action == Directions.SOUTH:
#             return Directions.NORTH
#         if action == Directions.EAST:
#             return Directions.WEST
#         if action == Directions.WEST:
#             return Directions.EAST
#         return action
#     reverseDirection = staticmethod(reverseDirection)

#     def vectorToDirection(vector):
#         dx, dy = vector
#         if dy > 0:
#             return Directions.NORTH
#         if dy < 0:
#             return Directions.SOUTH
#         if dx < 0:
#             return Directions.WEST
#         if dx > 0:
#             return Directions.EAST
#         return Directions.STOP
#     vectorToDirection = staticmethod(vectorToDirection)

#     def directionToVector(direction, speed = 1.0):
#         dx, dy =  Actions._directions[direction]
#         return (dx * speed, dy * speed)
#     directionToVector = staticmethod(directionToVector)

#     def getPossibleActions(config, walls):
#         possible = []
#         x, y = config.pos
#         x_int, y_int = int(x + 0.5), int(y + 0.5)

#         # In between grid points, all agents must continue straight
#         if (abs(x - x_int) + abs(y - y_int)  > Actions.TOLERANCE):
#             return [config.getDirection()]

#         for dir, vec in Actions._directionsAsList:
#             dx, dy = vec
#             next_y = y_int + dy
#             next_x = x_int + dx
#             if not walls[next_x][next_y]: possible.append(dir)

#         return possible

#     getPossibleActions = staticmethod(getPossibleActions)

#     def getLegalNeighbors(position, walls):
#         x,y = position
#         x_int, y_int = int(x + 0.5), int(y + 0.5)
#         neighbors = []
#         for dir, vec in Actions._directionsAsList:
#             dx, dy = vec
#             next_x = x_int + dx
#             if next_x < 0 or next_x == walls.width: continue
#             next_y = y_int + dy
#             if next_y < 0 or next_y == walls.height: continue
#             if not walls[next_x][next_y]: neighbors.append((next_x, next_y))
#         return neighbors
#     getLegalNeighbors = staticmethod(getLegalNeighbors)

#     def getSuccessor(position, action):
#         dx, dy = Actions.directionToVector(action)
#         x, y = position
#         return (x + dx, y + dy)
#     getSuccessor = staticmethod(getSuccessor)    
    
# # search.py
# # ---------
# # Licensing Information:  You are free to use or extend these projects for
# # educational purposes provided that (1) you do not distribute or publish
# # solutions, (2) you retain this notice, and (3) you provide clear
# # attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# # 
# # Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# # The core projects and autograders were primarily created by John DeNero
# # (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# # Student side autograding was added by Brad Miller, Nick Hay, and
# # Pieter Abbeel (pabbeel@cs.berkeley.edu).


# """
# In search.py, you will implement generic search algorithms which are called by
# Pacman agents (in searchAgents.py).
# """

# class SearchProblem:
#     """
#     This class outlines the structure of a search problem, but doesn't implement
#     any of the methods (in object-oriented terminology: an abstract class).

#     You do not need to change anything in this class, ever.
#     """

#     def getStartState(self):
#         """
#         Returns the start state for the search problem.
#         """
#         util.raiseNotDefined()

#     def isGoalState(self, state):
#         """
#           state: Search state

#         Returns True if and only if the state is a valid goal state.
#         """
#         util.raiseNotDefined()

#     def getSuccessors(self, state):
#         """
#           state: Search state

#         For a given state, this should return a list of triples, (successor,
#         action, stepCost), where 'successor' is a successor to the current
#         state, 'action' is the action required to get there, and 'stepCost' is
#         the incremental cost of expanding to that successor.
#         """
#         util.raiseNotDefined()

#     def getCostOfActions(self, actions):
#         """
#          actions: A list of actions to take

#         This method returns the total cost of a particular sequence of actions.
#         The sequence must be composed of legal moves.
#         """
#         util.raiseNotDefined()


# def tinyMazeSearch(problem):
#     """
#     Returns a sequence of moves that solves tinyMaze.  For any other maze, the
#     sequence of moves will be incorrect, so only use this for tinyMaze.
#     """
#     from game import Directions
#     s = Directions.SOUTH
#     w = Directions.WEST
#     return  [s, s, w, s, w, w, s, w]

# def depthFirstSearch(problem: SearchProblem):
#     """
#     Search the deepest nodes in the search tree first.
#     Your search algorithm needs to return a list of actions that reaches the
#     goal. Make sure to implement a graph search algorithm.
#     To get started, you might want to try some of these simple commands to
#     understand the search problem that is being passed in:
    
#     print("Start:", problem.getStartState())
#     print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
#     print("Start's successors:", problem.getSuccessors(problem.getStartState()))
#     """
    
#     # Stack to keep track of the current path being explored
#     stack = util.Stack()

#     # Set to keep track of visited states
#     visited = []

#     # Start state
#     start_state = problem.getStartState()

#     # Add the start state to the stack with an empty path and mark it as visited
#     stack.push((start_state, [],0))
#     visited.append(start_state)

#     while not stack.isEmpty():
#         # Get the next state and path to explore
#         state, path, cost = stack.pop()

#         # Check if this state is the goal state
#         if problem.isGoalState(state):
#             return path

#         # Get the successors of the current state and add them to the stack if they haven't been visited yet
#         for successor, action, toCost in problem.getSuccessors(state):
#             if successor not in visited:
#                 visited.append(successor)
#                 new_path = path + [action]
#                 stack.push((successor, new_path,cost+toCost))

#     # If no solution is found, return an empty list
#     return []
     
     
#     util.raiseNotDefined()

# def breadthFirstSearch(problem: SearchProblem):
#     """Search the shallowest nodes in the search tree first."""
    
#     # very similar to DFS but using queues instead of stack
#     queue = util.Queue()
#     visited = []
#     start_state = problem.getStartState()

#     queue.push((start_state, [],0))
#     visited.append(start_state)

#     while not queue.isEmpty():

#         state, path, cost = queue.pop()

#         if problem.isGoalState(state):
#             return path

#         # Get the successors of the current state and add them to the stack if they haven't been visited yet
       
#         for successor, action, toCost in problem.getSuccessors(state):
#             if successor not in visited:
#                 visited.append(successor)
#                 new_path = path + [action]
#                 queue.push((successor, new_path,toCost+cost))

#     # If no solution is found, return an empty list
#     return []
    
#     util.raiseNotDefined()

# def uniformCostSearch(problem: SearchProblem):
#     """Search the node of least total cost first."""
#     "*** YOUR CODE HERE ***"
    
#     frontier = util.PriorityQueue()
    
#     frontier.push((problem.getStartState(), [], 0), 0)  
    
#     explored = set()
    
#     while not frontier.isEmpty():
#     # Get the next node to expand from the frontier
#         node, actions, cost = frontier.pop()

#         # If the node is the goal state, return the actions taken to reach it
#         if problem.isGoalState(node):
#             return actions

#         # Add the node to the explored set
#         explored.add(node)

#         # Generate the successors of the current node
#         successors = problem.getSuccessors(node)

#         # Add each successor to the frontier if it hasn't been explored yet
#         for successor, action, step_cost in successors:
#             if successor not in explored:
#                 # Calculate the total cost of the path to the successor
#                 total_cost = cost + step_cost
#                 # Add the successor to the frontier with its total cost as priority
#                 frontier.push((successor, actions + [action], total_cost), total_cost)

#     util.raiseNotDefined()

# def nullHeuristic(state, problem=None):
#     """
#     A heuristic function estimates the cost from the current state to the nearest
#     goal in the provided SearchProblem.  This heuristic is trivial.
#     """
#     return 0

# def aStarSearch(problem: SearchProblem, heuristic=nullHeuristic):
#     """Search the node that has the lowest combined cost and heuristic first."""
#     "*** YOUR CODE HERE ***"
#     # Priority queue to keep track of the current path being explored
#     queue = util.PriorityQueue()

#     # Set to keep track of visited states
#     visited = []

#     # Start state
#     start_state = problem.getStartState()

#     # Add the start state to the queue with an empty path, cost and priority, and mark it as visited
#     queue.push((start_state, [], 0), 0 + heuristic(start_state, problem))
#     visited.append(start_state)

#     while not queue.isEmpty():
#         # Get the next state, path, cost and priority to explore
#         state, path, cost = queue.pop()

#         # Check if this state is the goal state
#         if problem.isGoalState(state):
#             return path

#         # Get the successors of the current state and add them to the queue if they haven't been visited yet
#         for successor, action, step_cost in problem.getSuccessors(state):
#             if successor not in visited:
#                 visited.append(successor)
#                 new_path = path + [action]
#                 new_cost = cost + step_cost
#                 priority = new_cost + heuristic(successor, problem)
#                 queue.push((successor, new_path, new_cost), priority)

#     # If no solution is found, return an empty list
#     return []
    
#     util.raiseNotDefined()
# class PositionSearchProblem(SearchProblem):
#     """
#     A search problem defines the state space, start state, goal test, successor
#     function and cost function.  This search problem can be used to find paths
#     to a particular point on the pacman board.

#     The state space consists of (x,y) positions in a pacman game.

#     Note: this search problem is fully specified; you should NOT change it.
#     """

#     def __init__(self, gameState, costFn = lambda x: 1, goal=(1,1), start=None, warn=True, visualize=True):
#         """
#         Stores the start and goal.

#         gameState: A GameState object (pacman.py)
#         costFn: A function from a search state (tuple) to a non-negative number
#         goal: A position in the gameState
#         """
#         self.walls = gameState.getWalls()
#         self.startState = gameState.getPacmanPosition()
#         if start != None: self.startState = start
#         self.goal = goal
#         self.costFn = costFn
#         self.visualize = visualize
#         if warn and (gameState.getNumFood() != 1 or not gameState.hasFood(*goal)):
#             print('Warning: this does not look like a regular search maze')

#         # For display purposes
#         self._visited, self._visitedlist, self._expanded = {}, [], 0 # DO NOT CHANGE

#     def getStartState(self):
#         return self.startState

#     def isGoalState(self, state):
#         isGoal = state == self.goal

#         # For display purposes only
#         if isGoal and self.visualize:
#             self._visitedlist.append(state)
#             import __main__
#             if '_display' in dir(__main__):
#                 if 'drawExpandedCells' in dir(__main__._display): #@UndefinedVariable
#                     __main__._display.drawExpandedCells(self._visitedlist) #@UndefinedVariable

#         return isGoal

#     def getSuccessors(self, state):
#         """
#         Returns successor states, the actions they require, and a cost of 1.

#          As noted in search.py:
#              For a given state, this should return a list of triples,
#          (successor, action, stepCost), where 'successor' is a
#          successor to the current state, 'action' is the action
#          required to get there, and 'stepCost' is the incremental
#          cost of expanding to that successor
#         """

#         successors = []
#         for action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
#             x,y = state
#             dx, dy = Actions.directionToVector(action)
#             nextx, nexty = int(x + dx), int(y + dy)
#             if not self.walls[nextx][nexty]:
#                 nextState = (nextx, nexty)
#                 cost = self.costFn(nextState)
#                 successors.append( ( nextState, action, cost) )

#         # Bookkeeping for display purposes
#         self._expanded += 1 # DO NOT CHANGE
#         if state not in self._visited:
#             self._visited[state] = True
#             self._visitedlist.append(state)

#         return successors

#     def getCostOfActions(self, actions):
#         """
#         Returns the cost of a particular sequence of actions. If those actions
#         include an illegal move, return 999999.
#         """
#         if actions == None: return 999999
#         x,y= self.getStartState()
#         cost = 0
#         for action in actions:
#             # Check figure out the next state and see whether its' legal
#             dx, dy = Actions.directionToVector(action)
#             x, y = int(x + dx), int(y + dy)
#             if self.walls[x][y]: return 999999
#             cost += self.costFn((x,y))
#         return cost
    


# # Abbreviations
# bfs = breadthFirstSearch
# dfs = depthFirstSearch
# astar = aStarSearch
# ucs = uniformCostSearch
    