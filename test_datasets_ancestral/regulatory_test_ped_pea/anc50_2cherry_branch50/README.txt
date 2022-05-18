Graphs generated using the PED_PEA model with ancestor defined as:

G=gf.GRN_seed_graph_ped_pea(steps=41,r=0.2,q=0.4)

On a 2 cherry 4 leaf tree explicitly defined by 

t=nx.DiGraph()
t.add_edge(0,1)
t.add_edge(1,2)
t.add_edge(1,3)
t.add_edge(0,4)
t.add_edge(4,5)
t.add_edge(4,6)

and with branchLength=50

The r and q parameters of the PED_PEA model are defined as r=3, q=0.4

The graphs are then generated using
graphLeaves,internalGraphs=gf.ped_pea_graphs_from_tree(ancestor,t,3,0.4)

This stores the graphs on the leaves as well as the internal nodes

Read these in with:


leafGraphs=dict()
internalGraphs=dict()
t=nx.DiGraph()
t.add_edge(0,1)
t.add_edge(1,2)
t.add_edge(1,3)
t.add_edge(0,4)
t.add_edge(4,5)
t.add_edge(4,6)
root = [n for n,d in t.in_degree() if d==0]
leaves = [n for n,d in t.out_degree() if d==0]
for i in range(6):
    if i in leaves:
        leafGraphs[i]= nx.read_edgelist("test_datasets_ancestral/regulatory_test_ped_pea/anc50_2cherry_branch"+str(branchLength)+"/LEAF"+str(i)+"_two_cherry_branchLength"+str(branchLength)+".txt",create_using=nx.DiGraph)
    else:
        internalGraphs[i]= nx.read_edgelist("test_datasets_ancestral/regulatory_test_ped_pea/anc50_2cherry_branch"+str(branchLength)+"/INTERNAL"+str(i)+"_two_cherry_branchLength"+str(branchLength)+".txt",create_using=nx.DiGraph)