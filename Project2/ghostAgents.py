# ghostAgents.py
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


from game import Agent
from game import Actions
from game import Directions
import random
from util import manhattanDistance
import util

class GhostAgent(Agent):
    def __init__(self, index, state=None):
        self.index = index

    def getAction(self, state):
        dist = self.getDistribution(state)
        if len(dist) == 0:
            return Directions.STOP
        else:
            return util.chooseFromDistribution(dist)

    def getDistribution(self, state):
        "Returns a Counter encoding a distribution over actions from the provided state."
        util.raiseNotDefined()


class RandomGhost(GhostAgent):
    "A ghost that chooses a legal action uniformly at random."

    def getDistribution(self, state):
        dist = util.Counter()
        for a in state.getLegalActions(self.index): dist[a] = 1.0
        dist.normalize()
        return dist


class DirectionalGhost(GhostAgent):
    "A ghost that prefers to rush Pacman, or flee when scared."

    def __init__(self, index, state=None, prob_attack=0.8, prob_scaredFlee=0.8):
        print("diractionl ghost Index: ", index)
        self.index = index
        self.prob_attack = prob_attack
        self.prob_scaredFlee = prob_scaredFlee

    def getDistribution(self, state):
        # Read variables from state
        ghostState = state.getGhostState(self.index)
        legalActions = state.getLegalActions(self.index)
        pos = state.getGhostPosition(self.index)
        isScared = ghostState.scaredTimer > 0
        speed_rnd = lambda: random.uniform(0.2, 1.0)
        speed = speed_rnd()
        if isScared: speed = speed_rnd() * 0.5
        actionVectors = [Actions.directionToVector(a, speed) for a in legalActions]
        newPositions = [(pos[0] + a[0], pos[1] + a[1]) for a in actionVectors]
        pacmanPosition = state.getPacmanPosition()

        # Select best actions given the state
        distancesToPacman = [manhattanDistance(pos, pacmanPosition) for pos in newPositions]
        if isScared:
            bestScore = max(distancesToPacman)
            bestProb = self.prob_scaredFlee
        else:
            bestScore = min(distancesToPacman)
            bestProb = self.prob_attack
        bestActions = [action for action, distance in zip(legalActions, distancesToPacman) if distance == bestScore]

        # Construct distribution
        dist = util.Counter()
        for a in bestActions: dist[a] = bestProb / len(bestActions)
        for a in legalActions: dist[a] += (1 - bestProb) / len(legalActions)
        dist.normalize()
        return dist


##### Arbel's ghost - PRM #####
from My_PRM import Roadmap
from math import ceil, floor


class PRMGhost(GhostAgent):
    """
    A ghost that only know the world via PRM
    """

    def __init__(self, index, layout=None, prob_attack=0.99, prob_scaredFlee=0.99, samples=50, degree=5):
        self.index = index
        self.layout = layout
        self.degree = degree
        print("PRM ghost Index: ", index)
        self.start = layout.agentPositions[index][1]
        self.start = (round(self.start[0], 3), round(self.start[1], 3))
        print(self.start)
        self.prob_attack = prob_attack
        self.prob_scaredFlee = prob_scaredFlee
        self.buildPRM(samples)
        self.prm.add([self.start])
        self.establish_edges()
        self.next_node = self.start

    def getDistribution(self, state):
        ghost_state = state.getGhostState(self.index)
        legal_actions = state.getLegalActions(self.index)
        pos = state.getGhostPosition(self.index)
        is_scared = ghost_state.scaredTimer > 0
        speed_rnd = lambda: random.uniform(0.2, 1.0)
        speed = speed_rnd()
        if is_scared: speed = speed_rnd() * 0.5
        action_vectors = [Actions.directionToVector(a, speed) for a in legal_actions]
        new_positions = [(pos[0] + a[0], pos[1] + a[1]) for a in action_vectors]
        pacman_position = state.getPacmanPosition()
        pacman_position = (round(pacman_position[0], 3), round(pacman_position[1], 3))
        self.add_to_prm(pacman_position, self.degree)
        # TODO: currently we add too many points to the PRM which make long games run very slow. maybe forget old locations or places that aren't visited?
        # TODO: additionally I can speed up the game for multiple ghosts maybe by adding Pacman locations to a
        #  different, common (global?) PRM that only joins to the ghosts when needed
        if self.is_in_node(pos):
            self.next_node = self.find_next_node(pos, pacman_position)
        print "current next node is ", self.next_node
        # Select best actions given the state
        distances_to_next_node = [manhattanDistance(pos, self.next_node) for pos in new_positions]
        print "distances to next nodes", distances_to_next_node
        if is_scared:
            best_score = max(distances_to_next_node)
            best_prob = self.prob_scaredFlee
        else:
            best_score = min(distances_to_next_node)
            best_prob = self.prob_attack
        best_actions = [action for action, distance in zip(legal_actions, distances_to_next_node) if
                        distance == best_score]
        # Construct distribution
        dist = util.Counter()
        for a in best_actions: dist[a] = best_prob / len(best_actions)
        for a in legal_actions: dist[a] += (1 - best_prob) / len(legal_actions)
        dist.normalize()
        open('prm_edges_for_ghost_' + str(self.index) + '.txt', 'w').write(str(self.prm.edges))
        open('prm_vertices_for_ghost_' + str(self.index) + '.txt', 'w').write(str(self.prm.vertices))
        return dist

    def find_next_node(self, pos, pacman_position):
        path = self.prm.dijkstra(pos, pacman_position)
        if path == None:
            return self.next_node
        return path[1]

    #### PRM ####
    def sample_space(self, width, height, n):
        samples = []
        for i in range(n):
            samples.append((round(random.uniform(1, width - 1), 3), round(random.uniform(1, height - 1),
                                                                          3)))  # round to 3 decimal places means a tolerance of 0.001
        return samples

    def order_by_distance(self, v):
        return sorted(self.prm.vertices, key=lambda x: manhattanDistance(v, x))

    def establish_edges(self, degree=5):  # try to connect all vertices which have viable path between them
        for v in self.prm.vertices:
            d = degree
            for w in self.order_by_distance(v):
                if d < 1:
                    break
                if not self.collision(v, w):
                    self.prm.connect(self.prm.vertices[v], self.prm.vertices[w])
                    d -= 1

    def buildPRM(self, num_samples=100):
        samples = self.sample_space(self.layout.width, self.layout.height, num_samples)
        print("samples: ", samples)
        self.prm = Roadmap(samples)
        print("prm: ", self.prm.vertices)
        print(self.prm.vertices[samples[0]].edges)
        self.establish_edges()
        # save prm edges to a file to view
        with open('prm_edges_for_ghost_' + str(self.index) + '.txt', 'w') as f:
            f.write(str(self.prm.edges))

    def add_to_prm(self, v, degree=5):
        """In order to avoid adding too many nodes (slows the game) we only add a node it if's far enough from the
        closest node or if they have a wall between them"""
        v = (round(v[0], 3), round(v[1], 3))
        self.prm.add([v])
        neighbors = self.order_by_distance(v)
        d = degree
        for w in neighbors:
            if d < 1:
                break
            if not self.collision(v, w):
                # print("adding edge: ", v, w)
                self.prm.connect(self.prm.vertices[v], self.prm.vertices[w])
                d -= 1

    def not_wall(self, (x, y)):
        """it's probably not a wall"""
        if x == int(x) and y == int(y):
            if self.layout.isWall((x, y)):
                return True
        return False

    def is_in_node(self, v, tolerance=1.5):
        """check if a vertex is in a node"""
        x, y = round(v[0], 3), round(v[1], 3)
        xs = map(lambda x: round(x[0], 3), self.prm.vertices.keys())
        ys = map(lambda x: round(x[1], 3), self.prm.vertices.keys())
        for i in range(len(xs)):
            if manhattanDistance((x, y), (xs[i], ys[i])) < tolerance:
                return True
        return False

    def collision(self, start,
                  end):  # TODO make this approxmiate walking in a stright line (along the dots on the map)
        """
        Returns true if there is a wall between the two points going in two stright lines (kinda, i think.)
        """
        walls = self.layout.walls
        x1, y1 = start
        x2, y2 = end
        x1 = int(floor(x1))
        x2 = int(floor(x2))
        y1 = int(floor(y1))
        y2 = int(floor(y2))

        if walls[x1][y1] or walls[x2][y2]:
            return True

        p1 = (x1, y1)
        p2 = (x2, y2)
        line_pix = self.bresenham(p1, p2)

        for p in line_pix:
            (x, y) = p
            if walls[x][y]:
                return True

        return False

    def bresenham(self, start, end):
        """Bresenham's Line Algorithm
        Produces a list of tuples from start and end
        """

        # Setup initial conditions
        x1, y1 = start
        x2, y2 = end
        dx = x2 - x1
        dy = y2 - y1

        # Determine how steep the line is
        is_steep = abs(dy) > abs(dx)

        # Rotate line
        if is_steep:
            x1, y1 = y1, x1
            x2, y2 = y2, x2

        # Swap start and end points if necessary and store swap state
        swapped = False
        if x1 > x2:
            x1, x2 = x2, x1
            y1, y2 = y2, y1
            swapped = True

        # Recalculate differentials
        dx = x2 - x1
        dy = y2 - y1

        # Calculate error
        error = int(dx / 2.0)
        ystep = 1 if y1 < y2 else -1

        # Iterate over bounding box generating points between start and end
        y = y1
        points = []
        for x in range(x1, x2 + 1):
            coord = (y, x) if is_steep else (x, y)
            points.append(coord)
            error -= abs(dy)
            if error < 0:
                y += ystep
                error += dx

        # Reverse the list if the coordinates were swapped
        if swapped:
            points.reverse()
        return points
