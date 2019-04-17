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

def mandist(x,y):
	"""
	Calculates the manhattan Distance between points x and y
	"""
	return abs(x[0]-y[0]) + abs(x[1]-y[1])

class ReflexAgent(Agent):
		"""
			A reflex agent chooses an action at each choice point by examining
			its alternatives via a state evaluation function.

			The code below is provided as a guide.  You are welcome to change
			it in any way you see fit, so long as you don't touch our method
			headers.
		"""


		def getAction(self, gameState):
				"""
				You do not need to change this method, but you're welcome to.

				getAction chooses among the best options according to the evaluation function.

				Just like in the previous project, getAction takes a GameState and returns
				some Directions.X for some X in the set {North, South, West, East, Stop}
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

		def evaluationFunction(self, currentGameState, action):
				"""
				Design a better evaluation function here.

				The evaluation function takes in the current and proposed successor
				GameStates (pacman.py) and returns a number, where higher numbers are better.

				The code below extracts some useful information from the state, like the
				remaining food (newFood) and Pacman position after moving (newPos).
				newScaredTimes holds the number of moves that each ghost will remain
				scared because of Pacman having eaten a power pellet.

				Print out these variables to see what you're getting, then combine them
				to create a masterful evaluation function.
				"""
				# Useful information you can extract from a GameState (pacman.py)

				successorGameState = currentGameState.generatePacmanSuccessor(action)
				newPos = successorGameState.getPacmanPosition()
				newFood = successorGameState.getFood()
				newGhostStates = successorGameState.getGhostStates()
				newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

				dist = []
				x = 0
				score = successorGameState.getScore()					# -- Score bonus based on the next state's score --
				count = 0												# -- Score penalty based on the remaining food on the board --
				mink = 0												# -- Score penalty based on Pacman's distance from the ghosts --
				mind = 0												# -- Score penalty based on the closest food on the board - used to help pacman navigate --

				for i in newFood:										# Calculates the number of food in the board -> (count)
					y = 0												# And the manhattanDistance between Pacman and each food -> (dist)
					for j in i:
						if j:
							dist += [mandist(newPos,(x,y))]
							count+=1
						y+=1
					x+=1

				if dist == []:											# Edge case for no food 
					mind = 0
				else:													# Distance to closest food
					mind = min(dist)

				ghostPos = []
				for i in newGhostStates:								# Saves positions of ghosts in (ghostPos)
					ghostPos += [i.getPosition()]


				for i in ghostPos:
					k = mandist(newPos,i)
					if k == 1:											# If a ghost is right next to pacman that can't be good
						mink += 2										# Score modifier is lowered a lot
					elif k == 0:										# If Pacman walks right into a ghost
						mink += 10000									# Score modifier is lowered A LOT
					else:												# The further away a ghost is from Pacman, the lower its impact on the score
						mink += 1/(k*k)

				if ghostPos == []:										# If no ghosts on board, don't consider the distance to closest food
					mind = 0											# as it may lead to lockups

				return score - 20*count - 25*mink - mind				# Returns a linear function of the score bonuses and penalties 

def scoreEvaluationFunction(currentGameState):
		"""
			This default evaluation function just returns the score of the state.
			The score is the same one displayed in the Pacman GUI.

			This evaluation function is meant for use with adversarial search agents
			(not reflex agents).
		"""
		return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
		"""
			This class provides some common elements to all of your
			multi-agent searchers.  Any methods defined here will be available
			to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

			You *do not* need to make any changes here, but you can if you want to
			add functionality to all your adversarial search agents.  Please do not
			remove anything, however.

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

		def getAction(self, gameState):
				"""
					Returns the minimax action using self.depth and self.evaluationFunction
				"""

				action = None
				v = None

				for act in gameState.getLegalActions(0):

					succ = gameState.generateSuccessor(0,act)

					if v == None:													# Basically run maxvalue on agent 0
						v = self.minValue(succ,1,0)									# but return the action corresponding
						action = act 												# to the maximum value instead of the 
					else:															# value itself
						test = self.minValue(succ,1,0)
						if v < test:
							v = test
							action = act

				return action

		def minValue(self, gameState, agentIndex, currDepth):
			"""
				Returns the minimum value of the next agent's minimax values
			"""

			v = float('inf')

			if agentIndex == gameState.getNumAgents()-1:							# If the next agent is agent 0

				if self.depth == currDepth:											# If maximum depth has been reached, Stop
					return self.evaluationFunction(gameState)						# Evaluate this state and return its value
				else:

					for act in gameState.getLegalActions(agentIndex):
						succ = gameState.generateSuccessor(agentIndex,act)
						v = min(self.maxValue(succ,currDepth+1),v)					# Value is the minimum of agent 0's minimax values, also increment depth by 1

			else:																	# If there are agents left that haven't made a move

				for act in gameState.getLegalActions(agentIndex):
					succ = gameState.generateSuccessor(agentIndex,act)
					v = min(self.minValue(succ,agentIndex+1,currDepth),v)			# Value is the minimum of the next agent's minimax values

			if v != float('inf'):													# If there have been any successors, v will have changed
				return v 															# Return it
			return self.evaluationFunction(gameState)								# Else there are no successors, so evaluate this state and return its value


		def maxValue(self, gameState, currDepth):
			"""
				Returns the minimum value of the next agent's minimax values
				(No agent argument, since only agent 0 will be using this)
			"""
			v = float('-inf')

			if self.depth == currDepth:												# If maximum depth has been reached, Stop
				return self.evaluationFunction(gameState)							# Evaluate this state and return its value

			for act in gameState.getLegalActions(0):
				succ = gameState.generateSuccessor(0,act)
				v = max(self.minValue(succ,1,currDepth),v)							# Value is the maximum of the next agent's minimax values

			if v != float('-inf'):													# If there have been any successors, v will have changed
				return v 															# Return it
			return self.evaluationFunction(gameState)								# Else there are no successors, so evaluate this state and return its value


class AlphaBetaAgent(MultiAgentSearchAgent):
		"""
			Your minimax agent with alpha-beta pruning (question 3)
		"""

		# Very similar to minimax
		# Except for the a and b values that get passed on 
		# through minValue and maxValue

		def getAction(self, gameState):
				"""
					Returns the minimax action using self.depth and self.evaluationFunction
				"""

				action = None
				v = None
				a = float('-inf')
				b = float('inf')
				for act in gameState.getLegalActions(0):

					succ = gameState.generateSuccessor(0,act)

					if v == None:
						v = self.minValue(succ,1,0,a,b)
						action = act
					else:
						test = self.minValue(succ,1,0,a,b)
						if v < test:
							v = test
							action = act

					a = max(a,v)

				return action

		def minValue(self, gameState, agentIndex, currDepth, a, b):

			v = float('inf')

			if agentIndex == gameState.getNumAgents()-1:						# If the next agent is agent 0

				if self.depth == currDepth:										# If maximum depth has been reached, Stop
					return self.evaluationFunction(gameState)					# Evaluate this state and return its value
				else:

					for act in gameState.getLegalActions(agentIndex):
						succ = gameState.generateSuccessor(agentIndex,act)
						v = min(self.maxValue(succ,currDepth+1,a,b),v)			# Value is the minimum of agent 0's minimax values, also increment depth by 1
						if v < a:												# Prune
							return v
						b = min(b,v)

			else:

				for act in gameState.getLegalActions(agentIndex):
					succ = gameState.generateSuccessor(agentIndex,act)
					v = min(self.minValue(succ,agentIndex+1,currDepth,a,b),v)	# Value is the minimum of the next agent's minimax values
					if v < a:													# Prune
						return v
					b = min(b,v)

			if v != float('inf'):												# If there have been any successors, v will have changed
				return v 														# Return it
			return self.evaluationFunction(gameState)							# Else there are no successors, so evaluate this state and return its value


		def maxValue(self, gameState, currDepth, a, b):

			v = float('-inf')

			if self.depth == currDepth:											# If maximum depth has been reached, Stop
				return self.evaluationFunction(gameState)						# Evaluate this state and return its value

			for act in gameState.getLegalActions(0):
				succ = gameState.generateSuccessor(0,act)
				v = max(self.minValue(succ,1,currDepth,a,b),v)					# Value is the maximum of the next agent's minimax values
				if v > b:														# Prune
					return b
				a = max(a,v)

			if v != float('-inf'):												# If there have been any successors, v will have changed
				return v 														# Return it
			return self.evaluationFunction(gameState)							# Else there are no successors, so evaluate this state and return its value

class ExpectimaxAgent(MultiAgentSearchAgent):
		"""
			Your expectimax agent (question 4)
		"""

		# Very similar to minimax
		# But instead of minValue, maxValue calls chanceVal
		def getAction(self, gameState):
				"""
					Returns the expectimax action using self.depth and self.evaluationFunction

					All ghosts should be modeled as choosing uniformly at random from their
					legal moves.
				"""

				action = None
				v = None

				for act in gameState.getLegalActions(0):

					succ = gameState.generateSuccessor(0,act)

					if v == None:
						v = self.chanceVal(succ,1,0)
						action = act
					else:
						test = self.chanceVal(succ,1,0)
						if v < test:
							v = test
							action = act

				return action

		def chanceVal(self, gameState, agentIndex, currDepth):

			v = 0

			actions = gameState.getLegalActions(agentIndex)
			if actions != []:
				probability = (1/(float(len(actions))))							# All next actions are equally probable
				if agentIndex == gameState.getNumAgents() -1:					# If agent is the last agent

					if self.depth == currDepth:									# If maximum depth has been reached, Stop
						return float(self.evaluationFunction(gameState))		# Evaluate this state and return its value
					else:

						for act in actions:										# Agent 0 is up next
							succ = gameState.generateSuccessor(agentIndex,act)	# Return weighted average of each successor action's max value
							v += probability*(self.maxValue(succ,currDepth+1))	# using each action's probability as weight
				else:															# If agent is not the last agent

					for act in actions:											# Another agent's turn
						succ = gameState.generateSuccessor(agentIndex,act)		# Return weighted average of each successor action's max value
						v += probability*(self.chanceVal(succ,agentIndex+1,currDepth))	# using each action's probability as weight
			else:																# If agent is not the last agent
				return float(self.evaluationFunction(gameState))
			return v

		def maxValue(self, gameState, currDepth):

			v = float('-inf')

			if self.depth == currDepth:											# If maximum depth has been reached, Stop
				return self.evaluationFunction(gameState)						# Evaluate this state and return its value

			for act in gameState.getLegalActions(0):
				succ = gameState.generateSuccessor(0,act)
				v = max(self.chanceVal(succ,1,currDepth),v)						# Value is the maximum of the next agent's expectimax values

			if v != float('-inf'):												# If there have been any successors, v will have changed
				return v 														# Return it
			return self.evaluationFunction(gameState)							# Else there are no successors, so evaluate this state and return its value



def betterEvaluationFunction(currentGameState):
		"""
			Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
			evaluation function (question 5).

			DESCRIPTION: My reflex agent worked well enough so I used a similar
			approach in my evaluation function with some minor differences.
		"""
		position = currentGameState.getPacmanPosition()
		food = currentGameState.getFood()
		ghostStates = currentGameState.getGhostStates()
		caps = currentGameState.getCapsules()
		ScaredTimes = [ghostState.scaredTimer for ghostState in ghostStates]

		ghostPos = []
		
		score = currentGameState.getScore()						# -- Score bonus based on the next state's score --
		count = 0												# -- Score penalty based on the remaining food on the board --
		dist = []
		x = 0

		for i in food:
			y = 0
			for j in i:
				if j:
					dist += [mandist(position,(x,y))]			# Create a list containing the distances of pacman to every pellet
					count+=1
				y+=1
			x+=1

		if dist == []:											# -- Score penalty based on the closest food on the board - used to help pacman navigate --
			mind = 0											# If there are none left, no penalty
		else:
			mind = min(dist)									# Else gravitate towards the closest food

		for i in ghostStates:
			ghostPos += [i.getPosition()]

		mink = 0												# -- Score penalty based on Pacman's distance from the ghosts --

		for i in ghostPos:
			k = mandist(position,i)								# Just like in reflexAgent
			if k == 1:
				mink +=2
			elif k != 0:
				mink += 1/k*k
			else:
				mink += 10000

		maxt = 0
		for i in ScaredTimes:									# Checking if all ghosts are currently
			if i > maxt:
				maxt = i

		if maxt > 0:											# If they are
			mink = -2*mink										# Turn ghost penalty into bonus

		if ghostPos == []:										# If no ghosts on board, don't consider the distance to closest food
			mind = 0											# as it may lead to lockups

		return score - 20*count - 25*mink - mind - len(caps)	# Returns a linear function of the score bonuses and penalties 

# Abbreviation
better = betterEvaluationFunction

