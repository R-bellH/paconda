from math import floor, ceil
from ghostAgents import GhostAgent
from util import manhattanDistance
import numpy as np
import ghostAgents
import util

class GridGhost(GhostAgent):
    """
     A ghost that only knows the world via a Grid, but not the actual grid   """

    def __init__(self, index, layout=None, prob_attack=0.99, prob_flee=0.99, grid_size=5):
        GhostAgent.__init__(self, index)
        self.index = index
        print("Grid ghost Index: ", index)
        self.layout = layout
        self.start = layout.agentPositions[index][1]
        print(self.start)
        self.prob_attack = prob_attack
        self.prob_flee = prob_flee
        self.grid_size = grid_size
        self.grid = None
        self.build_grid()
        self.next_tile = self.position_to_grid(self.start)

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
        if self.position_to_grid(pos) == self.next_tile :
            self.next_tile = self.find_next_tile(pos, pacman_position)
        # print "current next node is ", self.next_node
        # Select best actions given the state
        distances_to_next_node = [manhattanDistance(pos, self.grid_to_position(self.next_tile)) for pos in new_positions]
        # print "distances to next nodes", distances_to_next_node
        if is_scared:
            best_score = max(distances_to_next_node)
            best_prob = self.prob_flee
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
        return dist

    def build_grid(self):                               # Create a Grid and find which tiles have obstacles to avoid
        height = self.grid_size
        width = self.grid_size * self.layout.width / self.layout.height
        if height < 1 or width < 1:
            return
        self.grid = np.ones((height, width), bool)
        for i in range(height):
            for j in range(width):
                for m in range(int(floor(i*self.layout.height/height)), int(ceil((i+1)*self.layout.height/height))):
                    for n in range(int(floor(j*self.layout.width/width)), int(ceil((j+1)*self.layout.width/width))):
                        if self.layout.isWall((m, n)):
                            self.grid[i, j] = False

    def find_next_tile(self, pos, pacman_position):     # Call dfs on the grid to find the next tile to move to
        next_tile = self.dfs_on_grid(pos, pacman_position)
        if next is None:
            self.grid_size += 5                         # If no tile is found make grid finer
            self.build_grid()
            return self.next_tile
        return next_tile

    def position_to_grid(self, pos):    # Find which tile contains a position on the board
        m = self.grid_size
        n = self.grid_size * self.layout.width / self.layout.height
        x = floor(pos[0] * n / self.layout.height)
        y = floor(pos[1] * m / self.layout.width)
        return x, y

    def grid_to_position(self, pos):    # Find the positions on the board in the middle of a tile
        m = self.grid_size
        n = self.grid_size * self.layout.width / self.layout.height
        x = floor((pos[0]+0.5) * self.layout.height / n)
        y = floor((pos[1]+0.5) * self.layout.width / m)
        return x, y

    def dfs_on_grid(self, start, end):
        queue = [(end[0], end[1])]
        visited = set()
        while queue:
            v = queue.pop(0)
            if v in visited:
                continue
            visited.add(v)
            if manhattanDistance(v, start) < 2:
                return v
            neighbor_cells = ((1, 0), (0, 1), (-1, 0), (0, -1))
            neighbors = list()
            for n in neighbor_cells:
                if self.grid[v[0]+n[0], v[1]+n[1]]:
                    neighbors.append((v[0]+n[0], v[1]+n[1]))
            for u in neighbors:
                if u not in visited:
                    queue.append(v)
        # print('No path found between {} and {}'.format(start, end))
        return None