import networkx as nx
import matplotlib.pyplot as plt
def show_RRT(ghost_index):
    with open("rrt_tree_for_ghost_"+str(ghost_index)+".txt") as f:
        G=nx.Graph()
        for line in f.read().splitlines():
            print(len(line))
            trre=eval(line)
            print(trre)
            points =[tuple(x[0]) for x in trre]
            edges = [tuple((tuple(x[0]),points[x[1]])) for x in trre]
            for i,node in enumerate(points):
                G.add_node(node,x=node[0],y=node[1])
            for i,edge in enumerate(edges):
                G.add_edge(*edge)
        pos = {node: (G.nodes[node]['x'], G.nodes[node]['y']) for node in G.nodes()}
        nx.draw(G, pos, with_labels=False, node_size=1, node_color='r', width=0.5, alpha=0.3, edge_color='b')
    path_nx = nx.Graph()
    with open('rrt_current_path_for_ghost_'+str(ghost_index)+'.txt') as f:
        path = eval(f.read())
        path_nx.add_node(path[0], x=path[0][0], y=path[0][1])
        for i in range(len(path) - 1):
            path_nx.add_node(path[i + 1], x=path[i][0], y=path[i][1])
            path_nx.add_edge(path[i], path[i + 1])
    pos_path = {node: (path_nx.nodes[node]['x'], path_nx.nodes[node]['y']) for node in path_nx.nodes()}
    nx.draw(path_nx, pos_path, with_labels=False, node_size=1, node_color='g', witdh=.5, alph=0.7, edge_color='g')
    plt.text(0.5, 0.5, "RRT")
    plt.show()
if __name__ == '__main__':
    show_RRT(1)