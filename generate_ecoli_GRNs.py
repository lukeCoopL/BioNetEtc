import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from Genetc import utilities as ut
from Genetc import duplication as dup

import pickle


out_deg_dist=dict()
in_deg_dist=dict()
G_col=nx.read_edgelist("E_coli_GRN_edges_only.txt",create_using=nx.DiGraph())
G_col=max(ut.connected_component_subgraphs(G_col),key=len)
print(G_col.nodes)
G_col=nx.convert_node_labels_to_integers(G_col)
G_coli=nx.DiGraph()
for i in G_col.nodes:
    G_coli.add_node(i)
for i in G_col.edges:
    G_coli.add_edge(i[0],i[1])
for i in G_coli.nodes:
    in_deg_dist[i]=0
    out_deg_dist[i]=0
    for j in G_coli.predecessors(i):
        in_deg_dist[i]=in_deg_dist[i]+1
    for j in G_coli.successors(i):
        out_deg_dist[i]=out_deg_dist[i]+1

inList=list(in_deg_dist.values())
outList=list(out_deg_dist.values())
inList=sorted(inList)
outList=sorted(outList)

G=dup.GRN_seed_graph()
steps=len(G_coli.nodes)-len(G.nodes)
print(steps)
node_remover=[]
G_dict=dict()
for r in [0,0.1,0.2,0.5,1,2,5]:
    for q in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]:
        G_dict[(r,q)]=[]
        for i in range(100):
            print(r,q,i)
            G_q=dup.ped_pea_single_lineage(G,steps,r,q,iteration=0,isolated_nodes_allowed=False)
            
            #for i in G_q.nodes():
            #    if len(list(G_q.predecessors(i)))==0 and len(list(G_q.successors(i)))==0:
            #        node_remover.append(i)
            #if len(node_remover)!=0:
            #    G_q.remove_nodes_from(node_remover)
            G_dict[(r,q)].append(G_q)

with open('G_dict_rq.pkl', 'wb') as f:
    pickle.dump(G_dict, f)
