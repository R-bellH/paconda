import heapq
from collections import namedtuple, Mapping
from heapq import heappop, heappush

from util import INF, get_pairs, merge_dicts, flatten, RED, default_selector, apply_alpha,manhattanDistance


class Vertex(object):

    def __init__(self, q):
        self.q = q
        self.edges = {}
        self._handle = None
        self.weight = 0

    def clear(self):
        self._handle = None

    # def draw(self, env, color=apply_alpha(RED, alpha=0.5)):
    #     if self._path is None:
    #         return
    #     # https://github.mit.edu/caelan/lis-openrave
    #     from manipulation.primitives.display import draw_edge
    #     #self._handle = draw_edge(env, self.v1.q, self.v2.q, color=color)
    #     for q1, q2 in get_pairs(self.configs()):
    #         self._handles.append(draw_edge(env, q1, q2, color=color))

    def __str__(self):
        return 'Vertex(' + str(self.q) + ')'

    __repr__ = __str__


class Edge(object):

    def __init__(self, v1, v2, path):
        self.v1, self.v2 = v1, v2
        self.v1.edges[v2], self.v2.edges[v1] = self, self
        self._path = path
        self._handles = []

    def end(self, start):
        if self.v1 == start:
            return self.v2
        if self.v2 == start:
            return self.v1
        assert False

    def path(self, start):
        if self._path is None:
            return [self.end(start).q]
        if self.v1 == start:
            return self._path + [self.v2.q]
        if self.v2 == start:
            return self._path[::-1] + [self.v1.q]
        assert False

    def configs(self):
        if self._path is None:
            return []
        return [self.v1.q] + self._path + [self.v2.q]

    def clear(self):
        # self._handle = None
        self._handles = []

    # def draw(self, env, color=apply_alpha(RED, alpha=0.5)):
    #     if self._path is None:
    #         return
    #     # https://github.mit.edu/caelan/lis-openrave
    #     from manipulation.primitives.display import draw_edge
    #     # self._handle = draw_edge(env, self.v1.q, self.v2.q, color=color)
    #     for q1, q2 in get_pairs(self.configs()):
    #         self._handles.append(draw_edge(env, q1, q2, color=color))

    def __str__(self):
        return 'Edge(' + str(self.v1.q) + ' - ' + str(self.v2.q) + ')'

    __repr__ = __str__

SearchNode = namedtuple('SearchNode', ['cost', 'parent'])


class Roadmap(Mapping, object):

    def __init__(self, samples=[]):
        self.vertices = {}
        self.edges = []
        self.add(samples)
        # added by Arbel
        self.distance= manhattanDistance

    def __getitem__(self, q):
        return self.vertices[q]

    def __len__(self):
        return len(self.vertices)

    def __iter__(self):
        return iter(self.vertices)

    def add(self, samples):
        new_vertices = []
        for q in samples:
            if q not in self:
                self.vertices[q] = Vertex(q)
                new_vertices.append(self[q])
        return new_vertices

    def connect(self, v1, v2, path=None):
        if v1==v2:
            return None
        if v1 not in v2.edges:
            edge = Edge(v1, v2, path)
            self.edges.append(edge)
            return edge
        return None

    def clear(self):
        for v in self.vertices.values():
            v.clear()
        for e in self.edges:
            e.clear()


    def remove_vertex(self,v): # currently disfunctional
        print "number of vertices", len(self.vertices)
        print "number of edges", len(self.edges)
        v=self.vertices[(round(v[0], 3),round(v[1], 3))]
        for e in v.edges.values():
            e.clear()
        del self.vertices[v.q]
        print "number of vertices", len(self.vertices)
        print "number of edges", len(self.edges)
    @staticmethod
    def merge(*roadmaps):
        new_roadmap = Roadmap()
        new_roadmap.vertices = merge_dicts(
            *[roadmap.vertices for roadmap in roadmaps])
        new_roadmap.edges = list(
            flatten(roadmap.edges for roadmap in roadmaps))
        return new_roadmap
    ##### Arbel's code #####
    def neighbors(self, v):
        neighbors = []
        if v in self.vertices:
            v=self.vertices[(round(v[0], 3),round(v[1], 3))]
        else:
            v=self.closest_node(v)
        for edge in self.edges:
            if edge.v1 == v:
                neighbors.append(edge.v2.q)
            if edge.v2 == v:
                neighbors.append(edge.v1.q)
        neighbors = list(set(neighbors))
        return neighbors

    def dijkstra(self, v1, v2, tolerance=0.5):
        heap = [(0, v1, [])]
        visited = set()
        while heap:
            (d, v, path) = heapq.heappop(heap)
            v=(float(v[0]),float(v[1]))
            if v in visited:
                continue
            visited.add(v)
            path = path + [v]
            if self.distance(v,v2)<tolerance:#v == v2:
                return path
            neighbors = self.neighbors(v)
            for u in neighbors:
                if u not in visited:
                    heapq.heappush(heap, (d + self.distance(v, u), u, path))
        print('No path found between {} and {}'.format(v1, v2))
        return None

    def closest_node(self, v):
        """Returns the closest node to v"""
        min_dist = float('inf')
        closest_node = None
        for node in self.vertices.values():
            dist = self.distance(node.q, v)
            if dist < min_dist:
                min_dist = dist
                closest_node = node
        return closest_node

    def a_star(self, start_node, stop_node, h=lambda n: 0, distance=manhattanDistance):
        # open_list is a list of nodes which have been visited, but who's neighbors
        # haven't all been inspected, starts off with the start node
        # closed_list is a list of nodes which have been visited
        # and who's neighbors have been inspected
        open_list = {start_node}
        closed_list = set([])

        # g contains current distances from start_node to all other nodes
        # the default value (if it's not found in the map) is +infinity
        g = {}

        g[start_node] = 0

        # parents contains an adjacency map of all nodes
        parents = {}
        parents[start_node] = start_node

        while len(open_list) > 0:
            n = None

            # find a node with the lowest value of f() - evaluation function
            for v in open_list:
                if n == None or g[v] + h(v) < g[n] + h(n):
                    n = v

            if n == None:
                print('Path does not exist!')
                return None

            # if the current node is the stop_node
            # then we begin reconstruction the path from it to the start_node
            if n == stop_node:
                reconst_path = []

                while parents[n] != n:
                    reconst_path.append(n)
                    n = parents[n]

                reconst_path.append(start_node)

                reconst_path.reverse()

                print('Path found: {}'.format(reconst_path))
                return reconst_path

            # for all neighbors of the current node do
            for m in self.neighbors(n):
                weight = distance(n,m) #m.weight
                # if the current node isn't in both open_list and closed_list
                # add it to open_list and note n as it's parent
                if m not in open_list and m not in closed_list:
                    open_list.add(m)
                    parents[m] = n
                    g[m] = g[n] + weight

                # otherwise, check if it's quicker to first visit n, then m
                # and if it is, update parent data and g data
                # and if the node was in the closed_list, move it to open_list
                else:
                    if g[m] > g[n] + weight:
                        g[m] = g[n] + weight
                        parents[m] = n

                        if m in closed_list:
                            closed_list.remove(m)
                            open_list.add   (m)

            # remove n from the open_list, and add it to closed_list
            # because all of his neighbors were inspected
            open_list.remove(n)
            closed_list.add(n)

        print('Path does not exist!')
        return None