# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for 
# educational purposes provided that (1) you do not distribute or publish 
# solutions, (2) you retain this notice, and (3) you provide clear 
# attribution to UC Berkeley, including a link to 
# http://inst.eecs.berkeley.edu/~cs188/pacman/pacman.html
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero 
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and 
# Pieter Abbeel (pabbeel@cs.berkeley.edu).
#
# Modified by Eugene Agichtein for CS325 Sp 2014 (eugene@mathcs.emory.edu)
#

from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

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
        Note that the successor game state includes updates such as available food,
        e.g., would *not* include the food eaten at the successor state's pacman position
        as that food is no longer remaining.
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        currentFood = currentGameState.getFood() #food available from current state
        newFood = successorGameState.getFood() #food available from successor state (excludes food@successor) 
        currentCapsules=currentGameState.getCapsules() #power pellets/capsules available from current state
        newCapsules=successorGameState.getCapsules() #capsules available from successor (excludes capsules@successor)
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        # Calculate the distance to the nearest food
        foodDistances = [manhattanDistance(newPos, foodPos) for foodPos in newFood.asList()]
        minFoodDistance = min(foodDistances) if foodDistances else 1

        # Calculate the distance to the nearest active ghost
        ghostDistances = [manhattanDistance(newPos, ghostState.getPosition()) for ghostState in newGhostStates]
        minGhostDistance = min(ghostDistances) if ghostDistances else 1

        # If ghosts are too closed
        if min(ghostDistances) < 2 and min(newScaredTimes) == 0:
            return -float('inf')

        # Combine the components of the evaluation function
        return successorGameState.getScore() + (1.0 / (1.0 + minFoodDistance))


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
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        "*** YOUR CODE HERE ***"
        #util.raiseNotDefined()

        # Minimax algorithm to decide Pac-Man's move
        def minimax(agentIndex, depth, gameState):
            if gameState.isWin() or gameState.isLose() or depth == 0:
                return self.evaluationFunction(gameState), None

            if agentIndex == 0:  # Pac-Man's turn (maximizer)
                bestScore = float("-inf")
                bestAction = None
                for action in gameState.getLegalActions(agentIndex):
                    nextGameState = gameState.generateSuccessor(agentIndex, action)
                    score, _ = minimax(1, depth, nextGameState)
                    if score > bestScore:
                        bestScore, bestAction = score, action
                return bestScore, bestAction
            else:  # Ghosts' turn (minimizer)
                nextAgent = agentIndex + 1
                if nextAgent >= gameState.getNumAgents():  # Cycle back to Pac-Man
                    nextAgent = 0
                    depth -= 1
                bestScore = float("inf")
                bestAction = None
                for action in gameState.getLegalActions(agentIndex):
                    nextGameState = gameState.generateSuccessor(agentIndex, action)
                    score, _ = minimax(nextAgent, depth, nextGameState)
                    if score < bestScore:
                        bestScore, bestAction = score, action
                return bestScore, bestAction

        _, action = minimax(0, self.depth, gameState)
        return action


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        #util.raiseNotDefined()
        def alphaBeta(agentIndex, depth, gameState, alpha, beta):
            if gameState.isWin() or gameState.isLose() or depth == self.depth:
                return self.evaluationFunction(gameState), None

            if agentIndex == 0:  # Pac-Man's turn
                value = float("-inf")
                bestAction = None
                for action in gameState.getLegalActions(agentIndex):
                    successor = gameState.generateSuccessor(agentIndex, action)
                    successorValue = alphaBeta(1, depth, successor, alpha, beta)[0]
                    if successorValue > value:
                        value, bestAction = successorValue, action
                    if value > beta:
                        return value, bestAction
                    alpha = max(alpha, value)
                return value, bestAction
            else:  # Ghosts' turn
                value = float("inf")
                nextAgent = (agentIndex + 1) % gameState.getNumAgents()
                nextDepth = depth + 1 if nextAgent == 0 else depth
                for action in gameState.getLegalActions(agentIndex):
                    successor = gameState.generateSuccessor(agentIndex, action)
                    successorValue = alphaBeta(nextAgent, nextDepth, successor, alpha, beta)[0]
                    if successorValue < value:
                        value = successorValue
                    if value < alpha:
                        return value, None
                    beta = min(beta, value)
                return value, None

        _, action = alphaBeta(0, 0, gameState, float("-inf"), float("inf"))
        return action

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        #util.raiseNotDefined()
        def expectimaxSearch(agentIndex, depth, gameState):
            if agentIndex >= gameState.getNumAgents():
                agentIndex = 0
                depth += 1
            if depth == self.depth or gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState), None
            if agentIndex == 0:
                return maxValue(agentIndex, depth, gameState)
            else:
                return expValue(agentIndex, depth, gameState)

        def maxValue(agentIndex, depth, gameState):
            v = float("-inf"), None
            for action in gameState.getLegalActions(agentIndex):
                nextValue = expectimaxSearch(agentIndex + 1, depth, gameState.generateSuccessor(agentIndex, action))[0]
                v = max(v, (nextValue, action))
            return v

        def expValue(agentIndex, depth, gameState):
            v = 0, None
            actions = gameState.getLegalActions(agentIndex)
            prob = 1.0 / len(actions) if actions else 1
            for action in actions:
                nextValue = expectimaxSearch(agentIndex + 1, depth, gameState.generateSuccessor(agentIndex, action))[0]
                v = (v[0] + nextValue * prob, action)
            return v[0], None

        _, action = expectimaxSearch(0, 0, gameState)
        return action

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>

    """
    "*** YOUR CODE HERE ***"
    #util.raiseNotDefined()
    if currentGameState.isWin():
        return float("inf")
    elif currentGameState.isLose():
        return -float("inf")

    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood().asList()
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
    score = currentGameState.getScore()
    totalScaredTimes = sum(newScaredTimes)

    foodDistances = [util.manhattanDistance(newPos, foodPos) for foodPos in newFood]
    minFoodDistance = min(foodDistances) if foodDistances else 1

    ghostDistances = [util.manhattanDistance(newPos, ghost.getPosition()) for ghost in newGhostStates if
                      ghost.scaredTimer == 0]
    minGhostDistance = min(ghostDistances) if ghostDistances else 10



    return score + totalScaredTimes + 10.0 / minFoodDistance + -10.0 / minGhostDistance



# Abbreviation
better = betterEvaluationFunction

class ContestAgent(MultiAgentSearchAgent):
    """
      Your agent for the mini-contest
    """

    def getAction(self, gameState):
        """
          Returns an action.  You can use any method you want and search to any depth you want.
          Just remember that the mini-contest is timed, so you have to trade off speed and computation.

          Ghosts don't behave randomly anymore, but they aren't perfect either -- they'll usually
          just make a beeline straight towards Pacman (or away from him if they're scared!)
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

