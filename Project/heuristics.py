import util
from util import manhattanDistance, euclideanDistance
# A hueristics know the end goal and get the current state and return a value
def null_heuristic(state,node):
    return 0

def manhattanDistance_heuristic(state,node):
    return manhattanDistance(node, state.getPacmanPosition())
def euclideanDistance_heuristic(state,node):
    return euclideanDistance(node, state.getPacmanPosition())