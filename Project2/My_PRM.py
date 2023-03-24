import heapq
from collections import namedtuple, Mapping
from heapq import heappop, heappush

from util import INF, get_pairs, merge_dicts, flatten, RED, default_selector, apply_alpha,manhattanDistance


class Vertex(object):

    def __init__(self, q):
        self.q = q
        self.edges = {}
        self._handle = None

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

    def __call__(self, q1, q2):
        if q1 not in self or q2 not in self:
            return None
        start, goal = self[q1], self[q2]
        queue = [(0, start)]
        nodes, processed = {start: SearchNode(0, None)}, set()

        def retrace(v):
            pv = nodes[v].parent
            if pv is None:
                return [v.q]
            return retrace(pv) + v.edges[pv].path(pv)

        while len(queue) != 0:
            _, cv = heappop(queue)
            if cv in processed:
                continue
            processed.add(cv)
            if cv == goal:
                return retrace(cv)
            for nv, edge in cv.edges.items():
                cost = nodes[cv].cost + len(edge.path(cv))
                if nv not in nodes or cost < nodes[nv].cost:
                    nodes[nv] = SearchNode(cost, cv)
                    heappush(queue, (cost, nv))
        print('No path found')
        return None

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


    def remove_vertex(self,v):
        print "number of vertices", len(self.vertices)
        print "number of edges", len(self.edges)
        v=self.vertices[(round(v[0], 3),round(v[1], 3))]
        for e in v.edges.values():
            print "edges of v1" , e.v1.edges
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