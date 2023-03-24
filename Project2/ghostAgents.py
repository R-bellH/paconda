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
import numpy as np

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
    A ghost that only know the world via PRM    """
    def __init__(self, index, layout=None, prob_attack=0.99, prob_scaredFlee=0.99, samples=100, degree=5):
        GhostAgent.__init__(self, index)
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
        # TODO: currently we add too many points to the PRM which make long games run very slow. maybe forget old locations or places that aren't visited?
        # TODO: additionally I can speed up the game for multiple ghosts maybe by adding Pacman locations to a
        #  different, common (global?) PRM that only joins to the ghosts when needed
        if self.is_in_node(pos):
            self.add_to_prm(pacman_position, self.degree)
            self.next_node = self.find_next_node(pos, (next_x, next_y))
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

    def is_in_node(self, v, tolerance=1):
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


#####  forgetful ghost #####
'''A ghost that can't remember where pacman was'''
class forgetfulGhost(PRMGhost):
    def __init__(self, index, layout=None, prob_attack=0.99, prob_scaredFlee=0.99, samples=30, degree=5):
        PRMGhost.__init__(self,index, layout, prob_attack, prob_scaredFlee, samples, degree)
        self.remember_to_forget= util.Queue()
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
        # TODO: currently we add too many points to the PRM which make long games run very slow. maybe forget old locations or places that aren't visited?
        # TODO: additionally I can speed up the game for multiple ghosts maybe by adding Pacman locations to a
        #  different, common (global?) PRM that only joins to the ghosts when needed
        if self.is_in_node(pos):
            self.add_to_prm(pacman_position, self.degree)
            self.next_node = self.find_next_node(pos, pacman_position)
        # Select best actions given the state
        distances_to_next_node = [manhattanDistance(pos, self.next_node) for pos in new_positions]
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
    def add_to_prm(self, v, degree=1, tolerance=0.5):
        """In order to avoid adding too many nodes (slows the game) we only add a node it if's far enough from the
        closest node or if they have a wall between them"""
        v = (round(v[0], 3), round(v[1], 3))
        closest = self.order_by_distance(v)[0]
        if manhattanDistance(v, closest) > tolerance or self.collision_fn(v, closest):
            self.prm.add([v])
            self.remember_to_forget.push(v)
            if len(self.remember_to_forget) > 5:
                self.prm.remove_vertex(self.remember_to_forget.pop())
            neighbors = self.order_by_distance(v)
            for w in neighbors:
                if v != w and not self.collision_fn(v, w):
                    print("adding edge: ", v, w)
                    self.prm.connect(self.prm.vertices[v], self.prm.vertices[w])
                    degree -= 1
                    if degree == 0:
                        break
#### Flank ghost ####
'''A ghost that tries to flank pacman'''
class FlankGhost(PRMGhost):
    """
    A ghost that only know the world via PRM
    """

    def __init__(self, index, state=None, prob_attack=0.99, prob_scaredFlee=0.99, samples=100, degree=5):
        PRMGhost.__init__(self, index, state, prob_attack, prob_scaredFlee, samples, degree)

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
        # TODO: currently we add too many points to the PRM which make long games run very slow. maybe forget old locations or places that aren't visited?
        # TODO: additionally I can speed up the game for multiple ghosts maybe by adding Pacman locations to a
        #  different, common (global?) PRM that only joins to the ghosts when needed
        if self.is_in_node(pos):
            self.add_to_prm(pacman_position, self.degree)
            self.next_node = self.find_next_node(pos, pacman_position)
        # Select best actions given the state
        distances_to_next_node = [manhattanDistance(pos, self.next_node) for pos in new_positions]
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

    def bresenham(self, start, end):
        """
        Bresenham's Line Generation Algorithm
        https://www.youtube.com/watch?v=76gp2IAazV4
        """

        # step 1 get end-points of line
        (x0, y0) = start
        (x1, y1) = end

        line_pixel = []
        line_pixel.append((x0, y0))

        # step 2 calculate difference
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)

        if dx == 0 :
            for y in range(y0, y1):
                line_pixel.append((x0, y))
            line_pixel = np.array(line_pixel)
            return line_pixel

        m = dy / dx

        # step 3 perform test to check if pk < 0
        flag = True


        step = 1
        if x0 > x1 or y0 > y1:
            step = -1

        mm = False
        if m < 1:
            x0, x1, y0, y1 = y0, y1, x0, x1
            dx = abs(x1 - x0)
            dy = abs(y1 - y0)
            mm = True

        p0 = 2 * dx - dy
        x = x0
        y = y0

        for i in range(abs(y1 - y0)):
            if flag:
                x_previous = x0
                p_previous = p0
                p = p0
                flag = False
            else:
                x_previous = x
                p_previous = p

            if p >= 0:
                x = x + step

            p = p_previous + 2 * dx - 2 * dy * (abs(x - x_previous))
            y = y + 1

            if mm:
                line_pixel.append((y, x))
            else:
                line_pixel.append((x, y))

        line_pixel = np.array(line_pixel)

        return line_pixel