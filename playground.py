import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import gene_functions as gf
fig=plt.figure()
vertices=5
edgeProb=0.4
G=nx.erdos_renyi_graph(vertices,edgeProb,directed=True)
G.add_edge(0, 0)
nx.draw_shell(G,with_labels=True)

duplicationGenes = [0,1,2]
G_dup=gf.duplicate_genes_fast(G,duplicationGenes)
nx.draw_shell(G_dup,with_labels=True)
fig.show()