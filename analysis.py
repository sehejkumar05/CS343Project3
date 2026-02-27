# analysis.py
# -----------
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


######################
# ANALYSIS QUESTIONS #
######################

# Set the given parameters to obtain the specified policies through
# value iteration.

def question2():
    answerDiscount = 0.5
    #changing this value so agent always moves where intended 
    answerNoise = 0.0
    return answerDiscount, answerNoise

def question3a():
    answerDiscount = .25 # low so future rewards dont matter as much
    answerNoise = 0.0 #no risk of falling
    answerLivingReward = -1 #makes agent finish quicker
    #this will cause the agent to take shortest path and ignore danger
    return answerDiscount, answerNoise, answerLivingReward
    # If not possible, return 'NOT POSSIBLE'

def question3b():
    answerDiscount = .5
    answerNoise = .25 #high noise to make cliff dangerous
    answerLivingReward = -1 #still prefer closer exit
    #the gaent will avoid risky path and prefer close exit
    return answerDiscount, answerNoise, answerLivingReward
    # If not possible, return 'NOT POSSIBLE'

def question3c():
    answerDiscount = .9 #makes +10 valuable
    answerNoise = 0
    answerLivingReward = -1 
    #this will have the agent go to the +1= and take shortest route
    return answerDiscount, answerNoise, answerLivingReward
    # If not possible, return 'NOT POSSIBLE'

def question3d():
    answerDiscount = .75 #makes +10 valuable
    answerNoise = .5 #shows the cliff is dangerous
    answerLivingReward = -.5
    #now this will not prefer exiting that early but go to +10 safely
    return answerDiscount, answerNoise, answerLivingReward
    # If not possible, return 'NOT POSSIBLE'

def question3e():
    #positive living rate makes it go on forever
    answerDiscount = .9
    answerNoise = .25
    answerLivingReward = 1.0
    return answerDiscount, answerNoise, answerLivingReward
    # If not possible, return 'NOT POSSIBLE'

def question8():
    answerEpsilon = None
    answerLearningRate = None
    return answerEpsilon, answerLearningRate
    # If not possible, return 'NOT POSSIBLE'

if __name__ == '__main__':
    print('Answers to analysis questions:')
    import analysis
    for q in [q for q in dir(analysis) if q.startswith('question')]:
        response = getattr(analysis, q)()
        print('  Question %s:\t%s' % (q, str(response)))
