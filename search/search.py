# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem: SearchProblem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:
    """
    # print("Start:", problem.getStartState())
    # print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    # print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    
    # Stack to keep track of the current path being explored
    stack = util.Stack()

    # Set to keep track of visited states
    visited = set()

    # Start state
    start_state = problem.getStartState()

    # Add the start state to the stack with an empty path and mark it as visited
    stack.push((start_state, []))
    visited.add(start_state)

    while not stack.isEmpty():
        # Get the next state and path to explore
        state, path = stack.pop()

        # Check if this state is the goal state
        if problem.isGoalState(state):
            return path

        # Get the successors of the current state and add them to the stack if they haven't been visited yet
        for successor, action, cost in problem.getSuccessors(state):
            if successor not in visited:
                visited.add(successor)
                new_path = path + [action]
                stack.push((successor, new_path))

    # If no solution is found, return an empty list
    return []
     
     
    util.raiseNotDefined()

def breadthFirstSearch(problem: SearchProblem):
    """Search the shallowest nodes in the search tree first."""
    
    queue = util.Queue()
    visited = set()
    start_state = problem.getStartState()

    queue.push((start_state, []))
    visited.add(start_state)

    while not queue.isEmpty():

        state, path = queue.pop()

        if problem.isGoalState(state):
            return path

        # Get the successors of the current state and add them to the stack if they haven't been visited yet
        for successor, action, cost in problem.getSuccessors(state):
            if successor not in visited:
                visited.add(successor)
                new_path = path + [action]
                queue.push((successor, new_path))

    # If no solution is found, return an empty list
    return []
    
    util.raiseNotDefined()

def uniformCostSearch(problem: SearchProblem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    
    frontier = util.PriorityQueue()
    
    frontier.push((problem.getStartState(), [], 0), 0)  
    
    explored = set()
    
    while not frontier.isEmpty():
    # Get the next node to expand from the frontier
        node, actions, cost = frontier.pop()

        # If the node is the goal state, return the actions taken to reach it
        if problem.isGoalState(node):
            return actions

        # Add the node to the explored set
        explored.add(node)

        # Generate the successors of the current node
        successors = problem.getSuccessors(node)

        # Add each successor to the frontier if it hasn't been explored yet
        for successor, action, step_cost in successors:
            if successor not in explored:
                # Calculate the total cost of the path to the successor
                total_cost = cost + step_cost
                # Add the successor to the frontier with its total cost as priority
                frontier.push((successor, actions + [action], total_cost), total_cost)

    util.raiseNotDefined()

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem: SearchProblem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch