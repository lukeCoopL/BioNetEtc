Graphs generated using the DMC model with ancestor defined as:

G=gf.GRN_seed_graph_ped_pea(steps=41,qMod=0.4,qCon=0.1)

On a 2 cherry 4 leaf tree explicitly defined by 

t=nx.DiGraph()
t.add_edge(0,1)
t.add_edge(1,2)
t.add_edge(1,3)
t.add_edge(0,4)
t.add_edge(4,5)
t.add_edge(4,6)

and with branchLength=5

The qMod and qCon parameters of the DMC model are defined as qCon=0.1, qMod=0.4

The graphs are then generated using
graphLeaves,internalGraphs=gf.dmc_graphs_from_tree(ancestor,t,0.1,0.4)

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
        leafGraphs[i]= nx.read_edgelist("test_datasets_ancestral/regulatory_test/anc50_2cherry_branch"+str(branchLength)+"/LEAF"+str(i)+"_two_cherry_branchLength"+str(branchLength)+".txt",create_using=nx.DiGraph)
    else:
        internalGraphs[i]= nx.read_edgelist("test_datasets_ancestral/regulatory_test/anc50_2cherry_branch"+str(branchLength)+"/INTERNAL"+str(i)+"_two_cherry_branchLength"+str(branchLength)+".txt",create_using=nx.DiGraph)