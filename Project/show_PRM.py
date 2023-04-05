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
        print(edge)
    print(len(G.nodes))
    # Draw the graph
    pos = {node: (G.nodes[node]['x'],G.nodes[node]['y']) for node in G.nodes()}
    nx.draw(G, pos, with_labels=False, node_size=1, node_color='r', width=0.5, alpha=0.3,edge_color='b')
    # color the chosen path
    path_nx=nx.Graph()
    with open ('PRM_current_path_of'+ghost_num+'.txt') as f:
        path=eval(f.read())
        if len(path)>=1:
            path_nx.add_node(path[0],x=path[0][0],y=path[0][1])
            for i in range(len(path)-1):
                path_nx.add_node(path[i+1],x=path[i][0],y=path[i][1])
                path_nx.add_edge(path[i],path[i+1])
        pos_path = {node: (path_nx.nodes[node]['x'],path_nx.nodes[node]['y']) for node in path_nx.nodes()}
        nx.draw(path_nx,pos_path, with_labels=False,node_size=1,node_color='g',witdh=.5,alph=0.7,edge_color='g')
    plt.text(0.5,0.5,"PRM")
    plt.show()

if __name__ == '__main__':
    show_PRM('1')
