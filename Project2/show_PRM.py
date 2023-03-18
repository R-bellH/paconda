import networkx as nx
import matplotlib.pyplot as plt


def show_PRM(ghost_num):
    G = nx.Graph()
    # read the vertices and edges
    with open ('prm_edges_for_ghost_'+ghost_num+'.txt') as f:
        edges = f.read()
        # replace '-' with ',' to make it a tuple
        edges = edges.replace('-', ',')
        # remove letters and brackets
        edges = edges.replace('Edge', '')
        # parse the string as a list of tuples
        edges = eval(edges)
    with open ('prm_vertices_for_ghost_'+ghost_num+'.txt') as f:
        vertices = f.read()
        # remove letters and brackets
        vertices = vertices.replace('Vertex', '')
        vertices=eval(vertices)
        vertices=[v for v in vertices]
        print(vertices)
    for i,node in enumerate(vertices):
        G.add_node(node,x=node[0],y=node[1])
    print(G.nodes(data=True))
    for i,edge in enumerate(edges):
        G.add_edge(*edge)
    print(len(G.nodes))
    # Draw the graph
    pos = {node: (G.nodes[node]['x'],G.nodes[node]['y']) for node in G.nodes()}
    nx.draw(G, pos, with_labels=False, node_size=1, node_color='r', width=0.5, alpha=0.5,edge_color='b')
    plt.show()

show_PRM('4')
