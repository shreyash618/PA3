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

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

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
        successorGameState = currentGameState.generatePacmanSuccessor(action)   #the succesor game state
        newPos = successorGameState.getPacmanPosition() #the new position
        newFood = successorGameState.getFood()  #the new food
        newGhostStates = successorGameState.getGhostStates()    #the new ghost states
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]  #the new scared times
        
        # calculate the manhattan distance from the current position to the closest food item
        # the net benefit/desirability of a food item is inversely proportional to the distance to it
        # using list iteration

        foodProximity = [util.manhattanDistance(newPos, food) for food in newFood.asList()]
        closestFood =  10 / min(foodProximity) + 1 if foodProximity else 0

        currentDistanceToGhost = min ([util.manhattanDistance(currentGameState.getPacmanPosition(), ghost) for ghost in currentGameState.getGhostPositions()])

        penalty = 0

        for ghost, scaredTimer in zip(newGhostStates, newScaredTimes):
            distanceToGhost = util.manhattanDistance(newPos, ghost.getPosition())

            #if the ghost isn't scared and the new position is close to the ghost,
            # then pacman should apply a penalty to that position to get further from it
            if (scaredTimer == 0):
                if distanceToGhost <= 1:
                    penalty -= 500
                elif distanceToGhost <= 2:
                    penalty -= 200
                else:
                    penalty -= 10/distanceToGhost

            #if the ghost is scared, pacman should try to get closer to the ghost and select the new position
            if (scaredTimer > 0):
                if (scaredTimer <= 2 and distanceToGhost >= 3):
                    penalty -= 500
                if(distanceToGhost < currentDistanceToGhost):
                    penalty += 100
                else:
                     penalty += 10/distanceToGhost
        
        if action == 'Stop':
            penalty -= 10
        totalScore = closestFood + penalty + successorGameState.getScore()
        return totalScore

def scoreEvaluationFunction(currentGameState: GameState):
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

#things I changed in Medhasri's minimax function
#1. the current score initialization! pacman has a double negative in front of infinity and the ghost is set to positive infiinty?!
# pacman needs to be negative infinity
# since the goal of the ghost is to MINIMIZE the max player's benefit, it should be set to positive infinity

#2. depth shouldn't decrease for each function call, it should only decrease when we've finished a cycle
# since we have multiple ghosts and only one pacman, decreasing depth at each call would mean we don't finish the function

#3. syntax issue in ghost section compaing tempScore to currScore

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState: GameState):
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

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        # util.raiseNotDefined()

        # calling on recursive function
        _, best_action = self.minimax_recursion(gameState, self.depth, 0)
        return best_action
    
    def minimax_recursion(self, curGameState, depth, index):
        # if no win, loss or legal actions not possible, base case!
        if curGameState.isWin() or curGameState.isLose() or len(curGameState.getLegalActions(index))==0 or depth==0:
            return self.evaluationFunction(curGameState), None
        
        if index==0:
            currScore = float('-inf') #-float("-inf") #there are two negative signs which make this positive! #should be negative!
            currAction = None
            
            for action in curGameState.getLegalActions(index):
                successorTemp = curGameState.generateSuccessor(index, action)
                tempScore, _ = self.minimax_recursion(successorTemp, depth, 1)
                if tempScore > currScore:
                    currAction = action
                    currScore = tempScore
            return currScore, currAction
        else:
            currScore = float('inf') #-float("-inf") #should be positive infinity
            currAction = None

            nAgent = (index + 1) % (curGameState.getNumAgents())
            for action in curGameState.getLegalActions(index):
                successorTemp = curGameState.generateSuccessor(index, action)
                newDepth = depth - 1 if nAgent == 0 else depth
                tempScore, _ = self.minimax_recursion(successorTemp, newDepth, nAgent)
                if tempScore < currScore: #flipped the sign (tempScore > currScore) #remember the goal of ghosts is MINIMIZE!
                    currAction = action
                    currScore = tempScore
            return currScore, currAction
            

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        
        if max_value: 
            V= float('-inf')
            for succcesor in gameState.generateSuccessor(index, action):
                v = max (v, succcesor.getScore())

        if min_value: 
            V= float('-inf')
            for succcesor in gameState.generateSuccessor(index, action):
                v = max (v, succcesor.getScore())

        best_score = float('-inf')
        best_action = None

        # Loop through Pacman's possible actions
        for action in gameState.getLegalActions(0):  # Pacman has index 0
            successor_state = gameState.generateSuccessor(0, action)
            score, _ = self.minimax_recursion(successor_state, self.depth - 1, 1)
            
            # Update the best action based on the score
            if score > best_score:
                best_score = score
                best_action = action
        
        return best_action


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
