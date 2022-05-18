
#from re import M, S
from time import time
from matplotlib.pyplot import xcorr
import networkx as nx
import itertools
import copy
from networkx.readwrite.json_graph import tree
import numpy as np
from networkx.algorithms.shortest_paths.unweighted import all_pairs_shortest_path, predecessor
import math
from collections import defaultdict
from networkx.utils import py_random_state
import random as random
import json
import matplotlib.pyplot as plt
from scipy.stats import expon as ex 
#from utilities import connected_component_subgraphs
from Genetc.utilities import *
#------------------------------------------------------
#Network evolution models

def duplicate_genes(G,genes,iteration=0,self_loops=True):
  mapping = {}
  ##print(iteration)
  for i in genes:
    
    mapping[i] = str(i)+"_"+str(iteration)
    
  G_sub=G.subgraph(genes)
  G_sub=nx.relabel_nodes(G_sub,mapping)
  G_dup=nx.compose(G,G_sub)

  
  for j in genes:
      
    for i in list(G.predecessors(j)):
      if i!=j:  
        G_dup.add_edge(i,str(j)+"_"+str(iteration))
      
    for i in list(G.successors(j)):
      if i!=j:
        G_dup.add_edge(str(j)+"_"+str(iteration),i)
    if self_loops:
      if (j,j) in G_dup.edges():
        G_dup.add_edge(str(j)+"_"+str(iteration),str(j)+"_"+str(iteration))
        G_dup.add_edge(j,str(j)+"_"+str(iteration))
        G_dup.add_edge(str(j)+"_"+str(iteration),j)
  
  return G_dup

def PED_PEA(G,r,q,iteration=0):
  ###print(iteration)
  G=copy.deepcopy(G)
  n=len(list(G.nodes))
  nodeNum=n-1
  rando=np.random.random()
  nodeList=list(G.nodes)
  ###print(rando,nodeNum,round(nodeNum*rando))
  dupNode=nodeList[round(nodeNum*rando)]
  
  
  G=duplicate_genes(G,[dupNode],iteration=iteration)
  
  
  parents=copy.deepcopy(list(G.predecessors(dupNode)))
  children=copy.deepcopy(list(G.successors(dupNode)))
  parentsCopy=set(copy.deepcopy(parents))
  childrenCopy=set(copy.deepcopy(children))
 
  theRemovalists=childrenCopy.union(set(parentsCopy))
  nodeListRemoved=set(list(nodeList))-theRemovalists
  if (dupNode,dupNode) in G.edges():
    rando=np.random.rand(1)  
    if rando <= q:
      rando=np.random.rand(1)
        
      if rando >0.5:
        G.remove_edge(str(dupNode)+"_"+str(iteration),dupNode)
      else:
        G.remove_edge(dupNode,dupNode)
  if (str(dupNode)+"_"+str(iteration),str(dupNode)+"_"+str(iteration)) in G.edges():
    rando=np.random.rand(1)  
    if rando <= q:
      rando=np.random.rand(1)
      if rando >0.5:
        G.remove_edge(str(dupNode)+"_"+str(iteration),str(dupNode)+"_"+str(iteration))
      else:
        G.remove_edge(dupNode,str(dupNode)+"_"+str(iteration))
  for i in parents:
    if i!=dupNode and i!= str(dupNode)+"_"+str(iteration):
      rando=np.random.rand(1)
      if rando <= q:
        rando=np.random.rand(1)
        if rando >0.5:
          G.remove_edge(i,str(dupNode)+"_"+str(iteration))
        else:
          G.remove_edge(i,dupNode)
  children=copy.deepcopy(list(G.successors(dupNode)))
  for i in children:
    
    if i!= dupNode and i!= str(dupNode)+"_"+str(iteration):
      rando=np.random.rand(1)
      
      if rando <= q:
        rando=np.random.rand(1)
        
        if rando >0.5:
          G.remove_edge(str(dupNode)+"_"+str(iteration),i)
        else:
          G.remove_edge(dupNode,i)
  for i in nodeListRemoved:
    rando = np.random.rand(1)
    if rando<r/nodeNum:
      rando=np.random.rand(1)
      if rando>0.5:
        G.add_edge(str(dupNode)+"_"+str(iteration),i)
      else:
        G.add_edge(dupNode,i)
  for i in nodeListRemoved:
    rando = np.random.rand(1)
    if rando<r/nodeNum:
      rando=np.random.rand(1)
      if rando>0.5:
        G.add_edge(i,str(dupNode)+"_"+str(iteration))
      else:
        G.add_edge(i,dupNode)
  
  return G

def ped_pea_network_birth(G,steps1,steps2,r,q,iteration=0):
    G1=copy.deepcopy(G)
    G2=copy.deepcopy(G)
    ###print(G1.nodes)
    #G1history=[]
    #G2history=[]
    for i in range(0,steps1):
        #G1history.append(G1)
        G1=PED_PEA(G1,r,q,iteration=iteration+i+1)
    for j in range(0,steps2):
        #G2history.append(G2)
        G2=PED_PEA(G2,r,q,iteration=iteration+j+1)
    return G1,G2
def ped_pea_single_lineage(G,steps,r,q,iteration=0,isolated_nodes_allowed=True):
    G1=copy.deepcopy(G)

    #G1history=[]
    i=0
    while i<steps:
        
        #G1history.append(G1)
        G1=PED_PEA(G1,r,q,iteration=iteration+i+1)
        if isolated_nodes_allowed:
          i=i+1
        else:
          node_remover=[]
          for k in G1.nodes():
            if len(list(G1.predecessors(k)))==0 and len(list(G1.successors(k)))==0:
                node_remover.append(k)
          if len(node_remover)!=0:
              G1.remove_nodes_from(node_remover)
          else:
            i=i+1
        
        
    return G1
def network_birth(G,steps1,steps2,qCon,qMod,iteration=0):
    G1=copy.deepcopy(G)
    G2=copy.deepcopy(G)
    ###print(G1.nodes)
    #G1history=[]
    #G2history=[]
    for i in range(0,steps1):
        #G1history.append(G1)
        G1=dmc(G1,qCon,qMod,iteration=iteration+i+1)
    for j in range(0,steps2):
        #G2history.append(G2)
        G2=dmc(G2,qCon,qMod,iteration=iteration+j+1)
    return G1,G2
def dmc_single_lineage(G,steps,qCon,qMod,iteration=0):
    G1=copy.deepcopy(G)

    #G1history=[]
    
    for i in range(0,steps):
        #G1history.append(G1)
        G1=dmc(G1,qCon,qMod,iteration=iteration+i+1)
    
    return G1
def dmc(G,qCon,qMod,iteration=0):
  ###print(iteration)
  G=copy.deepcopy(G)
  nodeNum=len(list(G.nodes))-1
  rando=np.random.random()
  nodeList=list(G.nodes)
  ###print(rando,nodeNum,round(nodeNum*rando))
  dupNode=nodeList[round(nodeNum*rando)]
  
  
  G=duplicate_genes(G,[dupNode],iteration=iteration)
  parents=copy.deepcopy(list(G.predecessors(dupNode)))
  children=copy.deepcopy(list(G.successors(dupNode)))
  parentsCopy=set(copy.deepcopy(parents))
  childrenCopy=set(copy.deepcopy(children))
  if (str(dupNode)+"_"+str(iteration),dupNode) in G.edges():
    G.remove_edge(str(dupNode)+"_"+str(iteration),dupNode)
  if (dupNode,str(dupNode)+"_"+str(iteration)) in G.edges():
    G.remove_edge(dupNode,str(dupNode)+"_"+str(iteration))
  for i in parents:
    if i==dupNode:
      rando=np.random.rand(1)
    
      if rando <= qMod:
        rando=np.random.rand(1)
        
        if rando >0.5:
          G.remove_edge(str(dupNode)+"_"+str(iteration),str(dupNode)+"_"+str(iteration))
        else:
          G.remove_edge(dupNode,dupNode)
    else:
      rando=np.random.rand(1)
      
      if rando <= qMod:
        rando=np.random.rand(1)
        
        if rando >0.5:
          G.remove_edge(i,str(dupNode)+"_"+str(iteration))
        else:
          G.remove_edge(i,dupNode)
  children=copy.deepcopy(list(G.successors(dupNode)))
  for i in children:
    if i==dupNode and ((i,i) in G.edges() and (str(dupNode)+"_"+str(iteration),str(dupNode)+"_"+str(iteration)) in G.edges() ):
      rando=np.random.rand(1)
    
      if rando <= qMod:
        rando=np.random.rand(1)
        
        if rando >0.5:

          G.remove_edge(str(dupNode)+"_"+str(iteration),str(dupNode)+"_"+str(iteration))
        else:
          G.remove_edge(dupNode,dupNode)
    elif i!= dupNode:
      rando=np.random.rand(1)
      if rando <= qMod:
        rando=np.random.rand(1)
        if rando >0.5:
          G.remove_edge(str(dupNode)+"_"+str(iteration),i)
        else:
          G.remove_edge(dupNode,i)
  rando=np.random.rand(1)
  if rando<qCon:
    G.add_edge(dupNode,str(dupNode)+"_"+str(iteration))
  rando=np.random.rand(1)
  if rando<qCon:
    G.add_edge(str(dupNode)+"_"+str(iteration),dupNode)
  return G

def duplication_forest(G,mostRecent):
  G_forest=nx.DiGraph()
  G_forest.add_nodes_from(G)
  treeComplete=False
  #mostRecent=0
  addedToTree=[]
  branchLength=1
  ancestorLength1 = branchLength
  ancestorLength2 = branchLength
  while not treeComplete:
    for i in list(G.nodes):
        
        nodeTemp=str(i)
        ###print("mostRecent",mostRecent)
        ###print("check of dup number",nodeTemp[-len(str(mostRecent)):len(nodeTemp)])
        
        if len(nodeTemp)>=len(str(mostRecent))+1 and mostRecent>0:
          if nodeTemp[-1-len(str(mostRecent))]=='_' and nodeTemp[-len(str(mostRecent)):len(nodeTemp)]==str(mostRecent):
            nodeTempAncestor=nodeTemp[:-1-len(str(mostRecent))]
            ###print("Temp node and ancestor",nodeTemp,nodeTempAncestor)
            if len(nodeTempAncestor)==1:
              nodeTempAncestor=int(nodeTempAncestor)
              addedToTree.append(nodeTempAncestor)
            if nodeTemp not in addedToTree:
              j=nodeTempAncestor
              m=nodeTemp
              ###print(list(G_forest.predecessors(j)))
              #if len(list(G_forest.predecessors(j)))==0:
                #addedToTree.append(nodeTempAncestor)
              while len(list(G_forest.predecessors(j)))!=0:
                for k in G_forest.predecessors(j):
                  ###print(G_forest[k][j]['weight'])
                  ancestorLength1=ancestorLength1-G_forest[k][j]['weight']
                  j=k
                  
              while len(list(G_forest.predecessors(m)))!=0:
                for k in G_forest.predecessors(m):
                  ###print(G_forest[k][m]['weight'])
                  ancestorLength2=ancestorLength2-G_forest[k][m]['weight']
                  m=k
                  
              
              G_forest.add_node(str(j)+"Anc")
              G_forest.add_edge(str(j)+"Anc",m,weight=ancestorLength2)
              G_forest.add_edge(str(j)+"Anc",j,weight=ancestorLength1)
              
              addedToTree.append(nodeTemp)
              ###print("added to tree",addedToTree)
              ###print("most Recent",mostRecent)
              #if len(addedToTree)>=len(G.nodes):
              #  treeComplete=True
    mostRecent=mostRecent-1
    branchLength=branchLength+1
    ancestorLength1=branchLength
    ancestorLength2=branchLength
    if mostRecent<0:
      treeComplete=True
      ##print(addedToTree)
  return G_forest
def tree_distance_loop(x,y,G_forest,treeDepth=1):
  
  inGraph = False
  if x in G_forest.nodes and y in G_forest.nodes:
    inGraph=True
  if not inGraph:
    #print("Either " +str(x) + " or " + str(y) + " not in graph")
    return 0
  components = connected_component_subgraphs(G_forest)
  shared_tree=False
  for graph in components:
    if x in graph.nodes and y in graph.nodes:
      shared_tree= True
  if not shared_tree:
    ##print("Nodes " + str(x) +" and " + str(y) + " share no duplication history")
    return 0
  distance=0
  if len(list(G_forest.predecessors(x)))!=0 and len(list(G_forest.predecessors(y)))!=0:
    xAnc=list(G_forest.predecessors(x))[0]
    yAnc=list(G_forest.predecessors(y))[0]
    xBranchLength=G_forest[xAnc][x]['weight']
    yBranchLength=G_forest[yAnc][y]['weight']
  if x!=y:
    x=xAnc
    y=yAnc
    distance = xBranchLength+yBranchLength
  else:
    return 0
  while x!=y:
    ##print(x,y)
    if xBranchLength>yBranchLength and len(list(G_forest.predecessors(y)))!=0:
      yAnc=list(G_forest.predecessors(y))[0]
      yBranchLength=G_forest[yAnc][y]['weight']
      y=yAnc
      
      distance=distance+yBranchLength
    elif yBranchLength>=xBranchLength and len(list(G_forest.predecessors(x)))!=0:
      xAnc=list(G_forest.predecessors(x))[0]
      xBranchLength=G_forest[xAnc][x]['weight']
      x=xAnc
      
      distance=distance+xBranchLength
    elif yBranchLength>=xBranchLength and len(list(G_forest.predecessors(y)))!=0:
      yAnc=list(G_forest.predecessors(y))[0]
      yBranchLength=G_forest[yAnc][y]['weight']
      y=yAnc
      
      distance=distance+yBranchLength
    elif xBranchLength>yBranchLength and len(list(G_forest.predecessors(x)))!=0:
      xAnc=list(G_forest.predecessors(x))[0]
      xBranchLength=G_forest[xAnc][x]['weight']
      x=xAnc
      
      distance=distance+xBranchLength
  return np.round(distance/2)
  
def closest_neighbour_distance(x,y,G1_forest,G2_forest,treeDepth=1):
  
  ancestor=0
  
  leafList = [i for i in G2_forest.nodes() if G2_forest.out_degree(i)==0 and G2_forest.in_degree(i)==1]
  if y not in leafList:
    #print(str(y)+" not present in second duplication forest, returned 0")
    return 0
  if x in leafList:
    
    ancestor=x
    return tree_distance_loop(ancestor,y,G2_forest,treeDepth=treeDepth)
  leafListG1 = [i for i in G1_forest.nodes() if G1_forest.out_degree(i)==0 and G1_forest.in_degree(i)==1]
  minDistance=treeDepth
  for j in leafListG1:
    if j in leafList:
      tempDistance = tree_distance_loop(x,j,G1_forest,treeDepth=treeDepth)
      if tempDistance<minDistance:
        minDistance=tempDistance
        ancestor=j
  if ancestor==0:
    ###print("Nothing in common between duplication trees")
    return 0
  return np.round((minDistance + tree_distance_loop(ancestor,y,G2_forest,treeDepth=treeDepth))/2)

def NC_scorer(alignment,mapped,G1,G2,G1_forest,G2_forest,labelsConserved=True,DMCSteps=0,childDistance=0):
  
  if DMCSteps == 0:
    maxTreeDepth=0
    treeDepth=0
    
    for G in [G1_forest,G2_forest]:
      treeDepth=0
      treeTraversed=False
      for x in [i for i in G.nodes() if G.in_degree(i)==0 and G.out_degree(i)!=0]:
        treeDepth=0
        treeTraversed=False
        while not treeTraversed:
          if len(list(G.successors(x)))==0:
            treeTraversed=True
          else:
            
            xChild = list(G.successors(x))[0]
            
            treeDepth = treeDepth + G[x][xChild]['weight']
            ###print("x and child ",x,xChild,treeDepth)
            x=xChild
            
      if treeDepth>maxTreeDepth:
        maxTreeDepth=treeDepth
    maxTreeDepth=2*maxTreeDepth
    ###print('maxTreeDepth',maxTreeDepth)
  else:
    maxTreeDepth=2*DMCSteps
  alignment=dict(alignment)
  NCScore=0
  if labelsConserved:
    for i in mapped:
      if G2.nodes[alignment[i]]['orig_label']==G1.nodes[i]['orig_label']:
        tempScore=(childDistance)
        if tempScore!=0:
          NCScore=NCScore+1/tempScore
        
      elif G2.nodes[alignment[i]]['orig_label'] in G1_forest.nodes and G1.nodes[i]['orig_label'] in G1_forest.nodes:
        tempScore=tree_distance_loop(G1.nodes[i]['orig_label'],G2.nodes[alignment[i]]['orig_label'],G1_forest,maxTreeDepth)
        if tempScore!=0:
          NCScore = NCScore+1/tempScore
        ##print("treedist",tree_distance_loop(G1.nodes[i]['orig_label'],G2.nodes[alignment[i]]['orig_label'],G1_forest,maxTreeDepth)/maxTreeDepth)
      elif G2.nodes[alignment[i]]['orig_label'] in G2_forest.nodes:
        tempScore=closest_neighbour_distance(G1.nodes[i]['orig_label'],G2.nodes[alignment[i]]['orig_label'],G1_forest,G2_forest,maxTreeDepth)
        if tempScore!=0:
          NCScore=NCScore+1/tempScore
        ##print("difftreedist",closest_neighbour_distance(G1.nodes[i]['orig_label'],G2.nodes[alignment[i]]['orig_label'],G1_forest,G2_forest,maxTreeDepth)/maxTreeDepth)
  else:
    for i in mapped:
      if alignment[i]==i:
        NCScore=NCScore+1 - 2*childDistance/maxTreeDepth
      elif alignment[i] in G1_forest.nodes and i in G1_forest.nodes:
        NCScore = NCScore+1-tree_distance_loop(i,alignment[i],G1_forest,maxTreeDepth)/maxTreeDepth
      elif alignment[i] in G2_forest.nodes:
        NCScore=NCScore+1-closest_neighbour_distance(i,alignment[i],G1_forest,G2_forest,maxTreeDepth)/maxTreeDepth
  return NCScore/len(mapped)

def original_networks_NC_score(G1,G2,G1_forest,G2_forest,DMCSteps=0,childDistance=0):
  
  NCScore=0
  if DMCSteps==0:
    maxTreeDepth=0
    treeDepth=0
    
    for G in [G1_forest,G2_forest]:
      treeDepth=0
      treeTraversed=False
      for x in [i for i in G.nodes() if G.in_degree(i)==0 and G.out_degree(i)!=0]:
        treeDepth=0
        treeTraversed=False
        while not treeTraversed:
          if len(list(G.successors(x)))==0:
            treeTraversed=True
          else:
            
            xChild = list(G.successors(x))[0]
            
            treeDepth = treeDepth + G[x][xChild]['weight']
            ##print("x and child ",x,xChild,treeDepth)
            x=xChild
            
      if treeDepth>maxTreeDepth:
        maxTreeDepth=treeDepth
    maxTreeDepth=2*maxTreeDepth
    ##print('maxTreeDepth',maxTreeDepth)
  else:
    maxTreeDepth=2*DMCSteps
  for i in G1.nodes():
    minMatcher=len(G1.nodes)**2
    if i in G2.nodes():
      minMatcher= childDistance
      matcher=minMatcher
    else:
      matcher=minMatcher
    
    for j in G2.nodes():
      if j!=i:
        if j in G1.nodes():
          matcher = tree_distance_loop(i,j,G1_forest,maxTreeDepth)
        else:
          matcher = closest_neighbour_distance(i,j,G1_forest,G2_forest,maxTreeDepth)
      if matcher<minMatcher and matcher !=0:
        
        minMatcher=matcher
    ##print("minmatcher",minMatcher)
    NCScore=NCScore+1/minMatcher
  return NCScore/len(G1.nodes)

def label_conserver(G):
  labelConserver=dict()
  origGList=list(G.nodes)
  for i in origGList:
      labelConserverTemp = {"orig_label":i}
      labelConserver[i]=labelConserverTemp
  nx.set_node_attributes(G, labelConserver)
  ##print(labelConserver)
  ##print(G.nodes[0]['orig_label'])
  return G



def dmc_graphs_from_tree(ancestor,t,qMod,qCon,constant_edge_length=0):
  dfsEdges=nx.dfs_edges(t,source=0)
  internalNodes=dict()
  #iterationRec=dict()
  internalNodes[0]=ancestor
  #iterationRec[0]=0
  iterationRec=0
  leafGraphs=dict()
  root = [n for n,d in t.in_degree() if d==0]
  leaves = [n for n,d in t.out_degree() if d==0]
  if len(root) !=1:
    print("not a rooted tree")
    return [nx.empty_graph()]
  if constant_edge_length>0:
    for i in dfsEdges:
      G=dmc_single_lineage(G,constant_edge_length,qCon,qMod,iteration=i[0]*constant_edge_length)
  else:
    
    for i in dfsEdges:
      
      internalNodes[i[1]]=dmc_single_lineage(internalNodes[i[0]],t[i[0]][i[1]]["weight"],qCon,qMod,iteration=iterationRec)
      iterationRec=iterationRec+t[i[0]][i[1]]["weight"]
      #iterationRec[i[1]]=iterationRec[i[0]]+t[i[0]][i[1]]["weight"]
      
      if i[1] in leaves:
        leafGraphs[i[1]]=(internalNodes[i[1]])
  return leafGraphs,internalNodes

def ped_pea_graphs_from_tree(ancestor,t,r,q,constant_edge_length=0):
  dfsEdges=nx.dfs_edges(t,source=0)
  internalNodes=dict()
  #iterationRec=dict()
  internalNodes[0]=ancestor
  #iterationRec[0]=0
  iterationRec=0
  leafGraphs=dict()
  root = [n for n,d in t.in_degree() if d==0]
  leaves = [n for n,d in t.out_degree() if d==0]
  if len(root) !=1:
    print("not a rooted tree")
    return [nx.empty_graph()]
  if constant_edge_length>0:
    for i in dfsEdges:
      G=ped_pea_single_lineage(G,constant_edge_length,r,q,iteration=i[0]*constant_edge_length)
  else:
    
    for i in dfsEdges:
      print("itrec",iterationRec)
      internalNodes[i[1]]=ped_pea_single_lineage(internalNodes[i[0]],t[i[0]][i[1]]["weight"],r,q,iteration=iterationRec)
      iterationRec=iterationRec+t[i[0]][i[1]]["weight"]
      #iterationRec[i[1]]=iterationRec[i[0]]+t[i[0]][i[1]]["weight"]
      
      if i[1] in leaves:
        leafGraphs[i[1]]=(internalNodes[i[1]])
  return leafGraphs,internalNodes

def GRN_seed_graph():
  G=nx.DiGraph()
  G.add_edge(0,1)
  G.add_edge(0,2)
  G.add_edge(0,3)
  G.add_edge(0,4)
  G.add_edge(0,9)
  G.add_edge(0,10)
  G.add_edge(0,11)
  G.add_edge(0,12)
  G.add_edge(0,13)
  G.add_edge(0,14)
  G.add_edge(0,15)
  G.add_edge(0,16)
  G.add_edge(4,5)
  G.add_edge(4,6)
  G.add_edge(4,7)
  G.add_edge(8,4)
  G.add_edge(0,8)
  G.add_edge(0,0)
  G.add_edge(8,8)
  G.add_edge(4,4)
  G.add_edge(4,17)
  G.add_edge(4,18)
  G.add_edge(4,19)
  G.add_edge(4,20)
  G.add_edge(4,21)
  G.add_edge(4,22)
  G.add_edge(4,23)
  G.add_edge(4,24)
  return G
def GRN_seed_graph_dmc(steps,qMod,qCon):
  G=GRN_seed_graph()
  G=dmc_single_lineage(G,steps,qCon,qMod,iteration=0)
  isoNodes=[i for i in list(G.nodes) if len(list(G.predecessors(i)))==0 and len(list(G.successors(i)))==0]
  iteration=steps
  while len(isoNodes)!=0:
    replace=len(isoNodes)
    G.remove_nodes_from(isoNodes)
    
    G=dmc_single_lineage(G,replace,qCon,qMod,iteration=iteration)
    iteration=iteration+replace
    isoNodes=[i for i in list(G.nodes) if len(list(G.predecessors(i)))==0 and len(list(G.successors(i)))==0]

  return G
def GRN_seed_graph_ped_pea(steps,r,q):
  G=GRN_seed_graph()
  G=ped_pea_single_lineage(G,steps,r,q,iteration=0)
  isoNodes=[i for i in list(G.nodes) if len(list(G.predecessors(i)))==0 and len(list(G.successors(i)))==0]
  iteration=steps
  while len(isoNodes)!=0:
    replace=len(isoNodes)
    G.remove_nodes_from(isoNodes)
    
    G=ped_pea_single_lineage(G,replace,r,q,iteration=iteration)
    iteration=iteration+replace
    isoNodes=[i for i in list(G.nodes) if len(list(G.predecessors(i)))==0 and len(list(G.successors(i)))==0]

  return G  
def hormozdiari_seed_graph(edgeProb):
  G1=nx.complete_graph(10)
  G2=nx.complete_graph(7)
  G1=nx.DiGraph(G1)
  G2=nx.DiGraph(G2)
  G=nx.union(G1,G2,rename=('G-','H-'))
  for i in G.nodes():
    for j in G.nodes():
      if (i,j) not in G.edges():
        rando=random.random()
        if rando<edgeProb:
          G.add_edge(i,j)
  
  for i in range(33):
    
    G_cop=copy.deepcopy(G)
    G_cop.add_node(i)
    
    for j in G_cop.nodes():
      rando=random.random()
      if rando<edgeProb:
        G_cop.add_edge(i,j)
      rando=random.random()
      if rando<edgeProb:
        G_cop.add_edge(j,i)
    G=G_cop
  G=nx.convert_node_labels_to_integers(G)
  return G
