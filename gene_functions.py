from matplotlib.pyplot import xcorr
import networkx as nx
import itertools
import copy
import numpy as np
from networkx.algorithms.shortest_paths.unweighted import all_pairs_shortest_path, predecessor
#
def duplicate_genes(G,genes,iteration=0):
  mapping = {}
  print(iteration)
  for i in genes:
    if iteration!=0:
      mapping[i] = str(i)+"_"+str(iteration)
    else:
      mapping[i] = str(i)+"_c"
  G_sub=G.subgraph(genes)
  G_sub=nx.relabel_nodes(G_sub,mapping)
  G_dup=nx.compose(G,G_sub)
  if iteration!=0:
    for i in list(G.nodes()):
      for j in list(G.nodes()):
        if (i,j) in list(G.edges()):
          if (i,str(i)+"_"+str(iteration)) not in list(G_dup.edges()) and j in genes:
            G_dup.add_edge(i,str(j)+"_"+str(iteration))
          if (str(i)+"_"+str(iteration),j) not in list(G_dup.edges()) and i in genes:
            G_dup.add_edge(str(i)+"_"+str(iteration),j)
  else:
    for i in list(G.nodes()):
      for j in list(G.nodes()):
        if (i,j) in list(G.edges()):
          if (i,str(j)+"_c") not in list(G_dup.edges()) and j in genes:
            G_dup.add_edge(i,str(j)+"_c")
          if (str(i)+"_c",j) not in list(G_dup.edges()) and i in genes:
            G_dup.add_edge(str(i)+"_c",j)
  return G_dup

#Here PA and PB are the sets of parents (children) of the nodes a in graph 1 and b in graph 2 respectively
#Let n=min(|PA|,|PB|). This function returns a list of all n-permutations of the 
# elements of the larger of PA and PB
def NF_permlist(PA,PB):
  
  m=max(len(PA),len(PB))
  n=min(len(PA),len(PB))
  if m ==0 or n ==0:
    return []
  if len(PA)>len(PB):
    return [list(j) for i in itertools.combinations(PA,len(PB)-1) for j in itertools.permutations(list(i))]
  else:
    return [list(j) for i in itertools.combinations(PB,len(PA)-1) for j in itertools.permutations(list(i))]

#Here PA and PB are the sets of parents (children) of the nodes a in graph 1 and b in graph 2 respectively
#Construct the list of all tuples (a_i,b_j) where a_i is a parent (child) of node a in graph 1 and
# b_j is a parent (child) of node b in graph 2
def NF_tupleList(PA,PB):
  tempTupleList=[]
  tupleListList=[]
  pMin = min(len(PA),len(PB))
  pMax=max(len(PA),len(PB))
  if len(PA)==pMin:
    minGraph=0
  else:
    minGraph=1
  if minGraph==0:
    for l in NF_permlist(PA,PB):
      tempTupleList=[]
      for i in range(0,len(l)):
        
        tempTupleList.append((PA[i],l[i]))
      tupleListList.append(tempTupleList)
  if minGraph==1:
    for l in NF_permlist(PA,PB):
      tempTupleList=[]
      for i in range(0,len(l)):
        tempTupleList.append((l[i],PB[i]))
      tupleListList.append(tempTupleList)
  return tupleListList

#Here PA and PB are the sets of parents (children) of the nodes a in graph 1 and b in graph 2 respectively
# G1 and G2 and the two graphs to be aligned
#Returns the 'optimal matching' by finding the set of tuples (a_i,b_j) that minimise the sum given
# in the Node Fingerprinting paper
def NF_summer(G1,G2,PA,PB):
  tupleListList=NF_tupleList(PA,PB)
  summand=0
  minList=[]
  minSummand=10000000
  
  for l in tupleListList:
    summand=0
    for i in l:
      lister=list(i)
      summand = summand + np.abs(G1.out_degree(lister[0])-G2.out_degree(lister[1]))+np.abs(G1.in_degree(lister[0])-G2.in_degree(lister[1]))
    if summand<minSummand:
      minSummand=summand
      minList=l
  
  return minList

#Returns the score function for a pair of nodes (x,y), x in graph 1 and y in graph 2
def NF_scorer(x,y,G1,G2,PA,PB,CA,CB,aligned,alpha,beta):
  minListOut=NF_summer(G1,G2,PA,PB)
  minListIn=NF_summer(G1,G2,CA,CB)
  productOut=1
  productIn=1
  
  for i in minListOut:
    lister=list(i)
    productOut = productOut*NF_delta(i[0],i[1],G1,G2,aligned,alpha)
  
 
  for i in minListIn:
    lister=list(i)
    productIn = productIn*NF_delta(i[0],i[1],G1,G2,aligned,alpha)
  return beta**(np.abs(G1.out_degree(x)-G2.out_degree(y)))*productOut + beta**(np.abs(G1.in_degree(x)-G2.in_degree(y)))*productIn

#Determines the value of the delta function for a pair of nodes (a,b), a in graph 1, b in graph 2
def NF_delta(a,b,G1,G2,aligned,alpha):
  if (a,b) in aligned or max(G1.out_degree(a),G2.out_degree(b))==0 or max(G1.in_degree(a),G2.in_degree(b))==0:
    return alpha
  else:
    return min(G1.out_degree(a),G2.out_degree(b))/max(G1.out_degree(a),G2.out_degree(b))+min(G1.in_degree(a),G2.in_degree(b))/max(G1.in_degree(a),G2.in_degree(b))

#Performs node fingerprinting for two graphs G1 and G2, with parameters alpha and beta (default 
# alpha=32, beta =0.8

#Performs a one-to-one alignment. This can be easily modified to give a one-to-many,
# many-to-many, many-to-one map
def NF(G1,G2,alpha=32,beta=0.8):
  aligned=[]
  alignedVert1=[]
  alignedVert2=[]
  maxScore=0
  while len(aligned) <min(len(list(G1.nodes())),len(list(G2.nodes()))):
    maxScore=0
    score=0
    for x in list(G1.nodes()):
      
      PA=list(G1.predecessors(x))
      
      CA=list(G1.successors(x))
      
      for y in list(G2.nodes()):
        
        PB=list(G2.predecessors(y))
        
        CB=list(G2.successors(y))
        
        if (x,y) not in aligned and x not in alignedVert1 and y not in alignedVert2:
          
          score = NF_scorer(x,y,G1,G2,PA,PB,CA,CB,aligned,alpha,beta)
          
          if score>maxScore:
            maxScore=score
            maxX=x
            maxY=y
    aligned.append((maxX,maxY))
    print(maxX,maxY)
    alignedVert1.append(maxX)
    alignedVert2.append(maxY)
  return aligned

def NF_many_to_one(G1,G2,alpha=32,beta=0.8):
  aligned=[]
  alignedVert1=[]
  alignedVert2=[]
  maxScore=0
  while len(aligned) <min(len(list(G1.nodes())),len(list(G2.nodes()))):
    maxScore=0
    score=0
    for x in list(G1.nodes()):
      
      PA=list(G1.predecessors(x))
      
      CA=list(G1.successors(x))
      
      for y in list(G2.nodes()):
        
        PB=list(G2.predecessors(y))
        
        CB=list(G2.successors(y))
        
        if (x,y) not in aligned and x not in alignedVert1:
          
          score = NF_scorer(x,y,G1,G2,PA,PB,CA,CB,aligned,alpha,beta)
          
          if score>maxScore:
            maxScore=score
            maxX=x
            maxY=y
    aligned.append((maxX,maxY))
    print(maxX,maxY)
    alignedVert1.append(maxX)
    alignedVert2.append(maxY)
  return aligned,alignedVert1
def NF_many_to_many(G1,G2,alpha=32,beta=0.8,gamma=20):
  aligned=[]
  alignedVert1=set()
  alignedVert2=set()
  maxScore=0
  scoreLimit=False
  while not scoreLimit:
    maxScore=0
    
    score=0
    for x in list(G1.nodes()):
      
      PA=list(G1.predecessors(x))
      
      CA=list(G1.successors(x))
      
      for y in list(G2.nodes()):
        
        PB=list(G2.predecessors(y))
        
        CB=list(G2.successors(y))
        
        if (x,y) not in aligned:
          
          score = NF_scorer(x,y,G1,G2,PA,PB,CA,CB,aligned,alpha,beta)
          
          if score>maxScore:
            #print(score)
            maxScore=score
            maxX=x
            maxY=y
    aligned.append((maxX,maxY))
    #print(maxX,maxY)
    alignedVert1.add(maxX)
    alignedVert2.add(maxY)
    if maxScore <gamma:
      scoreLimit=True
  return aligned,alignedVert1
def network_birth(G,steps1,steps2,qCon,qMod):
  #works but has issue that if a gene is duplicated twice then the second copy disappears as it doesn;t have a unique name
    G1=copy.deepcopy(G)
    G2=copy.deepcopy(G)
    print(G1.nodes)
    G1history=[]
    G2history=[]
    for i in range(0,steps1):
        G1history.append(G1)
        G1=dmc(G1,qCon,qMod,iteration=i+1)
    for j in range(0,steps2):
        G2history.append(G2)
        G2=dmc(G2,qCon,qMod,iteration=j+1)
    return G1,G2
def dmc(G,qCon,qMod,iteration=0):
  print(iteration)
  #works but has issue that if a gene is duplicated twice then the second copy disappears as it doesn;t have a unique name
  G=copy.deepcopy(G)
  nodeNum=len(list(G.nodes))-1
  rando=np.random.random()
  nodeList=list(G.nodes)
  dupNode=nodeList[round(nodeNum*rando)]
  parents=copy.deepcopy(G.predecessors(dupNode))
  
  G=duplicate_genes(G,[dupNode],iteration=iteration)
  
  
  for i in parents:
    rando=np.random.rand(1)
    if rando > qMod:
      rando=np.random.rand(1)
      if rando >0.5:
        G.remove_edge(i,str(dupNode)+"_"+str(iteration))
      else:
        G.remove_edge(i,dupNode)
  children=copy.deepcopy(G.successors(dupNode))
  for i in children:
    rando=np.random.rand(1)
    if rando > qMod:
      rando=np.random.rand(1)
      if rando >0.5:
        G.remove_edge(str(dupNode)+"_"+str(iteration),i)
      else:
        G.remove_edge(dupNode,i)
  rando=np.random.rand(1)
  if rando>qCon:
    rando=np.random.rand(1)
    if rando>0.5:
      G.add_edge(dupNode,str(dupNode)+"_"+str(iteration))
    else:
      G.add_edge(str(dupNode)+"_"+str(iteration),dupNode)
  return G
