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
from util import manhattanDistance, bresenham
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


##### PRM ghost #####
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
        print([pos[1] for pos in layout.agentPositions if pos[0] == False])
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
        other_locations = []
        for agent in range(1, len(state.data.agentStates)):
            if agent != self.index:
                other_locations.append(state.data.agentStates[agent].configuration.pos)
                # print "other locations: ", other_locations
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
            closest_pos = (10e5, 10e5)
            for other in other_locations + [pos]:
                if manhattanDistance(pacman_position, other) < manhattanDistance(pacman_position, closest_pos):
                    closest_pos = other
            if (pacman_position[0] >= closest_pos[0]):
                next_x = pacman_position[0] + 0.1
            else:
                next_x = pacman_position[0] - 0.1
            if (pacman_position[1] >= closest_pos[1]):
                next_y = pacman_position[1] + 0.1
            else:
                next_y = pacman_position[1] - 0.1
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
            samples.append((round(random.uniform(0, width - 1), 3), round(random.uniform(0, height - 1),
                                                                          3)))  # round to 3 decimal places means a tolerance of 0.001
        return samples

    def order_by_distance(self, v):
        return sorted(self.prm.vertices, key=lambda x: manhattanDistance(v, x))

    def establish_edges(self):  # try to connect all vertices which have viable path between them
        for v in self.prm.vertices:
            for w in self.order_by_distance(v):
                if not self.collision_fn(v, w):
                    self.prm.connect(self.prm.vertices[v], self.prm.vertices[w])

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

    def add_to_prm(self, v, degree=1, tolerance=0.5):
        """In order to avoid adding too many nodes (slows the game) we only add a node it if's far enough from the
        closest node or if they have a wall between them"""
        v = (round(v[0], 3), round(v[1], 3))
        closest = self.order_by_distance(v)[0]
        if manhattanDistance(v, closest) > tolerance or self.collision_fn(v, closest):
            self.prm.add([v])
            neighbors = self.order_by_distance(v)
            for w in neighbors:
                if v != w and not self.collision_fn(v, w):
                    print("adding edge: ", v, w)
                    self.prm.connect(self.prm.vertices[v], self.prm.vertices[w])
                    degree -= 1
                    if degree == 0:
                        break

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

    def collision_fn(self, start,end):  # TODO make this approxmiate walking in a stright line (along the dots on the map)
        """
        Returns true if there is a wall between the two points going in two stright lines (kinda, i think.)
        """
        walls = self.layout.walls
        x1, y1 = start
        x2, y2 = end
        x_s = ceil(min(x1, x2))
        x_e = floor(max(x1, x2))
        y_s = ceil(min(y1, y2))
        y_e = floor(max(y1, y2))
        while x_s <= x_e or y_s <= y_e:
            if x_s <= x_e:
                if walls[int(x_s)][int(y1)]:
                    return True
                x_s += 1
            if y_s <= y_e:
                if walls[int(x1)][int(y_s)]:
                    return True
                y_s += 1
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