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
    
    def minimax_recursion(self, curGameState, currdepth, currIndex):
        # if no win, loss or legal actions not possible, base case!
        if curGameState.isWin() or curGameState.isLose() or len(curGameState.getLegalActions(currIndex))==0 or currdepth==0:
            return self.evaluationFunction(curGameState), None
        
        if currIndex==0:
            currScore = float('-inf') 
            currAction = None
            
            for action in curGameState.getLegalActions(currIndex):
                successorTemp = curGameState.generateSuccessor(currIndex, action)
                tempScore, _ = self.minimax_recursion(successorTemp, currdepth, 1)

                if tempScore > currScore:
                    currAction = action
                    currScore = tempScore
            return currScore, currAction
        else:
            currScore = float('inf')
            currAction = None

            nextIndex = (currIndex + 1) % (curGameState.getNumAgents())
            for action in curGameState.getLegalActions(currIndex):
                successorTemp = curGameState.generateSuccessor(currIndex, action)
                newDepth = currdepth - 1 if nextIndex == 0 else currdepth
                tempScore, _ = self.minimax_recursion(successorTemp, newDepth, nextIndex)

                if tempScore < currScore:
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
        #defining the recursive alpha-beta function
        #parameters we need to know: which player is playing? (0 for pacman, 1 or beyond for ghost), depth, alpha-value, beta-value, and obviously the game state
       
        def alphaBeta(gameState: GameState, index, depth, alpha, beta):
            if (gameState.isWin() or gameState.isLose() or depth == self.depth): #base case #checking if the current state is a terminal node
                return self.evaluationFunction(gameState)
            
            #else go to the recursive step
            if index == 0: #meaning pacman is playing, so we use the maximizer function
                bestScore = float('-inf')
                for action in gameState.getLegalActions(index):
                    successorState = gameState.generateSuccessor(index, action)
                    score = alphaBeta (successorState, index + 1, depth, alpha, beta)
                    bestScore = max (score, bestScore)
                    alpha = max(alpha, bestScore)
                    if beta <= alpha:
                        break
                return bestScore

            elif index > 0: #meaning a ghost is playing, so we use the minimizer function
                bestScore = float('inf')
                if index == gameState.getNumAgents() - 1:
                    index = 0
                for action in gameState.getLegalActions(index):
                    successorState = gameState.generateSuccessor(index, action)
                    nextDepth = depth + 1 if nextAgent == 0 else depth
                    score = alphaBeta(successorState, index + 1, nextDepth, alpha, beta)
                    bestScore = min (score, bestScore)
                    beta = min(beta, bestScore)
                    if beta <= alpha: #prune if the beta value is less than the alpha value
                        break
                return bestScore
        
        bestAction = None
        alpha, beta = float('-inf'), float('inf')
        bestScore = float('-inf')
        for action in gameState.getLegalActions(0):  # Pacman's turn at the root
            successorState = gameState.generateSuccessor(0, action)
            score = alphaBeta(successorState, 1, 0, alpha, beta)
            if score > bestScore:
                bestScore = score
                bestAction = action
            alpha = max(alpha, bestScore)

        return bestAction
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
        best_action, __ = self.expectimax(gameState, self.depth * gameState.getNumAgents(), 0, "")
        return best_action

    def expectimax(self, gameState, currDepth, currIndex, currAction):
        if currDepth==0 or gameState.isLose() or gameState.isWin():
            return currAction, self.evaluationFunction(gameState)
        elif currIndex!=0:
            currScore = 0

            for a in gameState.getLegalActions(currIndex):
                nextIndex = (currIndex + 1) % gameState.getNumAgents()
                successorTemp = gameState.generateSuccessor(currIndex, a)
                
                _, tempAction = self.expectimax(successorTemp, currDepth-1, nextIndex, currAction)
                currScore += tempAction * (1/len(gameState.getLegalActions(currIndex)))
            return (currAction, currScore)
        else:  
            nextAction = tuple(("max", float('-inf')))
            for a in gameState.getLegalActions(currIndex):
                nextIndex = (currIndex + 1) % gameState.getNumAgents()
                successorTemp = gameState.generateSuccessor(currIndex, a)
                if currDepth != self.depth * gameState.getNumAgents():
                    successor = self.expectimax(successorTemp, currDepth - 1, nextIndex, currAction)
                else:
                    successor = self.expectimax(gameState.generateSuccessor(currIndex, a),
                                            currDepth - 1,nextIndex, a)
                nextAction = max(nextAction,successor,key = lambda x:x[1])
            return nextAction
        
        # util.raiseNotDefined()

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    #util.raiseNotDefined()
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

# Abbreviation
better = betterEvaluationFunction
