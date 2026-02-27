# valueIterationAgents.py
# -----------------------
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


# valueIterationAgents.py
# -----------------------
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


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        #loop over specified num iterations
        for currIter in range(self.iterations):
            currVals = util.Counter()
            #loop through all states in mdp
            for currState in self.mdp.getStates():
                #if terminal, value is 0
                if self.mdp.isTerminal(currState):
                    currVals[currState] = 0
                else:
                    #get possible actions
                    possibleActions = self.mdp.getPossibleActions(currState)
                    #compute the max q val from all those actions
                    maxQVal = -float("inf")
                    for currAction in possibleActions:
                        currQVal = self.computeQValueFromValues(currState, currAction)
                        maxQVal = max(maxQVal,currQVal)
                    #update the best value for this state
                    currVals[currState] = maxQVal
            self.values = currVals #update for next iter



    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        res = 0 #will hold q val
        #get all potential transitions
        potentialTransitions = self.mdp.getTransitionStatesAndProbs(state, action)
        #loop through possibilities
        for currNext, currProb in potentialTransitions:
            #get reward and compute bellman
            currReward = self.mdp.getReward(state,action,currNext)
            res+= currProb * (currReward + self.discount * self.values[currNext])
        return res

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        #if we are at a terminal state, no next action
        if self.mdp.isTerminal(state):
            return None
        
        bestAction = None
        bestVal = -float("inf")

        #loop through actions
        for currAction in self.mdp.getPossibleActions(state):
            currQVal = self.computeQValueFromValues(state,currAction)
            #choose actions with highest q val
            if currQVal > bestVal:
                bestVal = currQVal
                bestAction = currAction

        return bestAction

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        #get list of all states in mdp
        states = self.mdp.getStates()
        #loop over num iterations
        for currIter in range(self.iterations):
            #pick state based on cyclic order
            currState = states[currIter % len(states)]
            #check terminal
            if self.mdp.isTerminal(currState):
                continue
            #get all possible actions
            possibleActions = self.mdp.getPossibleActions(currState)
            #skip if noa ctions
            if not possibleActions:
                continue
            maxQVal = -float("inf")
            for currAction in possibleActions:
                #cmpute q val and update max
                currQVal = self.computeQValueFromValues(currState, currAction)
                if currQVal > maxQVal:
                    maxQVal = currQVal
            
            #update value only of this state
            self.values[currState] = maxQVal
                

class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        #get predecessots of all states
        pred = {}
        #initialize empty set for each state
        for currState in self.mdp.getStates():
            pred[currState] = set()
        
        #loop through all states to get predecessors
        for currState in self.mdp.getStates():
            #get possible actions
            possibleActions = self.mdp.getPossibleActions(currState)    
            for currAction in possibleActions:
                #get transitons for curr state and action
                possibleTransitions = self.mdp.getTransitionStatesAndProbs(currState,currAction)
                #loop over transitions
                #if probability is greater than 0, it is a predecessor
                for currNext, currProb in possibleTransitions:
                    if currProb >0:
                        pred[currNext].add(currState)

        pq = util.PriorityQueue()

        #loop over all states and push diff into pq
        for currState in self.mdp.getStates():
            #check terminal
            if self.mdp.isTerminal(currState):
                continue
            #find max q val
            maxQVal = -float("inf")
            for currAction in self.mdp.getPossibleActions(currState):
                currQVal = self.computeQValueFromValues(currState,currAction)
                if currQVal > maxQVal:
                    maxQVal = currQVal
            
            #compute difference between curr val and max q val
            currDiff = abs(self.values[currState] - maxQVal)
            #push with negative priority
            pq.update(currState,-currDiff)

        #main loop
        for currIter in range(self.iterations):
            #check pq is empty
            if pq.isEmpty():
                return
            #pop highest priority
            currState = pq.pop()

            #update val if not terminal
            if not self.mdp.isTerminal(currState):
                maxQVal = -float("inf")
                for currAction in self.mdp.getPossibleActions(currState):
                    currQVal = self.computeQValueFromValues(currState,currAction)
                    if currQVal > maxQVal:
                        maxQVal = currQVal
                self.values[currState] = maxQVal
            
            #update predecessors
            for currPred in pred[currState]:
                #check terminal
                if self.mdp.isTerminal(currPred):
                    continue
                
                maxQVal = -float("inf")
                for currAction in self.mdp.getPossibleActions(currPred):
                    currQVal = self.computeQValueFromValues(currPred,currAction)
                    if currQVal > maxQVal:
                        maxQVal = currQVal
                
                #compute difference
                currDiff = abs(self.values[currPred] - maxQVal)
                #if different large enough, push into queue
                if currDiff > self.theta:
                    pq.update(currPred, -currDiff)

