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



class Node():
    """ 
    Node class that contains useful information for returning the correct path to the goal.
    """
    def __init__(self,state,path_cost,action=None,parent=None):
        self.state = state
        self.action = action
        self.parent = parent
        self.path_cost = path_cost

    def __eq__(self,other):
        """
        Defining equal nodes to be nodes containing the same state, needed for UCS update
        """
        return self.state == other.state

def getSolution(node):
    """
    Returns all actions taken from the start state up to the goal
    """
    solution = []
    curr = node
    while curr.parent != None :
        solution = solution + [curr.action]
        curr = curr.parent
    solution.reverse()                                                              # Solution list needs to be reversed, since the earliest actions were appended last
    return solution

def stateNotInFrontier(frontier,node):
    """
    Helper function that checks whether a Node in the frontier has the same state as 'node'
    """
    for i in frontier.list:
        if i.state == node.state:
            return False
    return True



def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    """
    
    curr = Node(problem.getStartState(),0)          

    if problem.isGoalState(curr.state):                                             # If the starting state is the goal state, no actions needed
        return []

    frontier = util.Stack()                                                         # Frontier is a Stack (DFS)
    frontier.push(curr)
    explored = set()                                                                # Explored set is a python set()

    while(True):
        if frontier.isEmpty():                                                      # If no nodes in frontier : Failure/No goal found
            return None
        
        curr = frontier.pop()                                                       # First node in the frontier is removed from the Stack
        # if problem.isGoalState(curr.state):                                    	 # Goal check for autograder 
        #     return getSolution(curr)

        explored.add(curr.state)                                                    # and its state is added to the explored set

        for i in problem.getSuccessors(curr.state):                                 # For each of the current Node's successors:
            child = Node(i[0],curr.path_cost + 1,i[1],curr)

            if problem.isGoalState(child.state):                                    # If successor is a goal state, the solution is returned
                return getSolution(child)

            if child.state not in explored and stateNotInFrontier(frontier,child):  # If the successor's state is neither in the explored set nor
                frontier.push(child)                                                # the frontier, the node is pushed in the frontier


def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""

    curr = Node(problem.getStartState(),0)          

    if problem.isGoalState(curr.state):                                             # If the starting state is the goal state, no actions needed
        return []

    frontier = util.Queue()                                                         # Frontier is a Queue (BFS)
    frontier.push(curr)
    explored = set()                                                                # Explored set is a python set()

    while(True):
        if frontier.isEmpty():                                                      # If no nodes in frontier : Failure/No goal found
            return None
        
        curr = frontier.pop()                                                       # First node in the frontier is removed from the Queue
        # if problem.isGoalState(curr.state):                                    	# Goal check for autograder 
        #     return getSolution(curr)

        explored.add(curr.state)                                                    # and its state is added to the explored set

        for i in problem.getSuccessors(curr.state):                                 # For each of the current Node's successors:
            child = Node(i[0],curr.path_cost + 1,i[1],curr)

            if problem.isGoalState(child.state):									# If successor is a goal state, the solution is returned
            	return getSolution(child)

            if child.state not in explored and stateNotInFrontier(frontier,child):  # If the successor's state is neither in the explored set nor
                frontier.push(child)                                                # the frontier, the Node is pushed in the frontier


def uniformCostSearch(problem):
    """Search the node of least total cost first."""

    curr = Node(problem.getStartState(),0)

    frontier = util.PriorityQueue()                                                 # Frontier is a PriorityQueue (UCS)
    frontier.push(curr,curr.path_cost)                                              # A Node's priority is the cost from the Starting State to that Node 
    explored = set()                                                                # Explored set is a python set()

    while(True):
        if frontier.isEmpty():                                                      # If no nodes in frontier : Failure/No goal found
            return None
        
        curr = frontier.pop()                                                       # Node with the highest priority is removed from the PriorityQueue

        if problem.isGoalState(curr.state):                                         # If the current Node is a goal state, the solution is returned 
            return getSolution(curr)

        explored.add(curr.state)                                                    # Current Node is added to the explored set

        for i in problem.getSuccessors(curr.state):                                 # For each of the current Node's successors:
            child = Node(i[0],curr.path_cost + i[2],i[1],curr)                      
            if child.state not in explored:                                         # If the successor's state is not in the explored set
                frontier.update(child,child.path_cost)                              # If the successor's state exists in the frontier:
                                                                                    # Replace it with the new Node if the new Node's cost is lower than the one in the froniter
                                                                                    # Do nothing if the new Node's cost is higher than the one in the frontier
                                                                                    # Push it into the priorityQueue if the Node's state doesn't exist in the frontier

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    curr = Node(problem.getStartState(),0)

    frontier = util.PriorityQueue()                                                     # Frontier is a PriorityQueue (UCS)
    frontier.push(curr,curr.path_cost + heuristic(curr.state,problem))                  # A Node's priority is the cost from the Starting State to that Node + the heuristic's value for that state
    explored = set()                                                                    # Explored set is a python set()

    while(True):
        if frontier.isEmpty():                                                          # If no nodes in frontier : Failure/No goal found
            return None
        
        curr = frontier.pop()                                                           # Node with the highest priority is removed from the PriorityQueue

        if problem.isGoalState(curr.state):                                             # If the current Node is a goal state, the solution is returned 
            return getSolution(curr)

        explored.add(curr.state)                                                        # Current Node is added to the explored set

        for i in problem.getSuccessors(curr.state):                                     # For each of the current Node's successors:
            child = Node(i[0],curr.path_cost + i[2],i[1],curr)                      
            if child.state not in explored:                                             # If the successor's state is not in the explored set
                frontier.update(child,child.path_cost + heuristic(child.state,problem)) # If the successor's state exists in the frontier:
                                                                                        # Replace it with the new Node if the new Node's cost + the heuristic's value is lower than the one in the frontier
                                                                                        # Do nothing if the new Node's cost + the heuristic's value is higher than the one in the frontier
                                                                                        # Push it into the priorityQueue if the Node's state doesn't exist in the frontier


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
