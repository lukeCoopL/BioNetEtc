from matplotlib.pyplot import xcorr
import networkx as nx
import itertools
import copy
import numpy as np
from networkx.algorithms.shortest_paths.unweighted import all_pairs_shortest_path, predecessor
#
def connected_component_subgraphs(G):
    G=copy.deepcopy(G)
    G=G.to_undirected()
    for c in nx.connected_components(G):
        yield G.subgraph(c)
            
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
          if(x==y):
            print("true pair",score,x,y)
          #if score<maxScore:
          #  print("lower",score,x,y)
          if score==maxScore:
            print("equal",score,x,y)
          if score>maxScore:
            print("greater",score,x,y)
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
def network_birth(G,steps1,steps2,qCon,qMod,iteration=0):
    G1=copy.deepcopy(G)
    G2=copy.deepcopy(G)
    print(G1.nodes)
    G1history=[]
    G2history=[]
    for i in range(0,steps1):
        G1history.append(G1)
        G1=dmc(G1,qCon,qMod,iteration=iteration+i+1)
    for j in range(0,steps2):
        G2history.append(G2)
        G2=dmc(G2,qCon,qMod,iteration=iteration+j+1)
    return G1,G2
def dmc(G,qCon,qMod,iteration=0):
  print(iteration)
  G=copy.deepcopy(G)
  nodeNum=len(list(G.nodes))-1
  rando=np.random.random()
  nodeList=list(G.nodes)
  print(rando,nodeNum,round(nodeNum*rando))
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
        print("mostRecent",mostRecent)
        print("check of dup number",nodeTemp[-len(str(mostRecent)):len(nodeTemp)])
        if mostRecent==0:
          print("m")
        elif len(nodeTemp)>=len(str(mostRecent))+1:
          if nodeTemp[-1-len(str(mostRecent))]=='_' and nodeTemp[-len(str(mostRecent)):len(nodeTemp)]==str(mostRecent):
            nodeTempAncestor=nodeTemp[:-1-len(str(mostRecent))]
            print("Temp node and ancestor",nodeTemp,nodeTempAncestor)
            if len(nodeTempAncestor)==1:
              nodeTempAncestor=int(nodeTempAncestor)
              addedToTree.append(nodeTempAncestor)
            if nodeTemp not in addedToTree:
              j=nodeTempAncestor
              m=nodeTemp
              print(list(G_forest.predecessors(j)))
              #if len(list(G_forest.predecessors(j)))==0:
                #addedToTree.append(nodeTempAncestor)
              while len(list(G_forest.predecessors(j)))!=0:
                for k in G_forest.predecessors(j):
                  print(G_forest[k][j]['weight'])
                  ancestorLength1=ancestorLength1-G_forest[k][j]['weight']
                  j=k
                  
              while len(list(G_forest.predecessors(m)))!=0:
                for k in G_forest.predecessors(m):
                  print(G_forest[k][m]['weight'])
                  ancestorLength2=ancestorLength2-G_forest[k][m]['weight']
                  m=k
                  
              
              G_forest.add_node(str(j)+"Anc")
              G_forest.add_edge(str(j)+"Anc",m,weight=ancestorLength2)
              G_forest.add_edge(str(j)+"Anc",j,weight=ancestorLength1)
              
              addedToTree.append(nodeTemp)
              print("added to tree",addedToTree)
              #print("most Recent",mostRecent)
              if len(addedToTree)>=len(G.nodes):
                treeComplete=True
    mostRecent=mostRecent-1
    branchLength=branchLength+1
    ancestorLength1=branchLength
    ancestorLength2=branchLength
    if mostRecent<0:
      treeComplete=True
      print(addedToTree)
  return G_forest
    
def NC_scorer(alignment,mapped,G1,G2,G1_forest,G2_forest):
  alignment=dict(alignment)
  NCScore=0
  for i in mapped:
    if alignment[i]==i:
      NCScore=NCScore+1
    elif alignment[i] in G1_forest.nodes and i in G1_forest.nodes:
      NCScorer = NCScorer+1-tree_distance(i,alignment[i])
    #elif alignment[i] in G2_forest.nodes and i in G1_forest.nodes and alignment[i] not in G2_forest.nodes and i not in G1_forest.nodes:
      #NCScorer=NCScorer+1-closest_neighbour_distance(i,alignment[i])

def tree_distance_rec(x,y,G1_forest):
  #print(list(G1_forest.predecessors(x))[0])
  if x==y:
    
    return 0
  if x!=y and len(list(G1_forest.predecessors(x)))!=0 and len(list(G1_forest.predecessors(y)))!=0:
    return G1_forest[list(G1_forest.predecessors(x))[0]][x]['weight']+G1_forest[list(G1_forest.predecessors(y))[0]][y]['weight'] + tree_distance_rec(list(G1_forest.predecessors(x))[0],y,G1_forest)+tree_distance_rec(x,list(G1_forest.predecessors(y))[0],G1_forest)
  elif x!=y and len(list(G1_forest.predecessors(x)))!=0:
    return G1_forest[list(G1_forest.predecessors(x))[0]][x]['weight']+tree_distance_rec(list(G1_forest.predecessors(x))[0],y,G1_forest)
  elif x!=y and len(list(G1_forest.predecessors(y)))!=0:
    return G1_forest[list(G1_forest.predecessors(y))[0]][y]['weight']+tree_distance_rec(x,list(G1_forest.predecessors(y))[0],G1_forest)
  else:
    return 0
def tree_distance(x,y,G1_forest):
  if x!=y:
    return G1_forest[list(G1_forest.predecessors(x))[0]][x]['weight']+G1_forest[list(G1_forest.predecessors(y))[0]][y]['weight']+tree_distance_rec(list(G1_forest.predecessors(x))[0],list(G1_forest.predecessors(y))[0],G1_forest)
  else:
    return 0
def tree_distance_loop(x,y,G_forest):
  inGraph = False
  if x in G_forest.nodes and y in G_forest.nodes:
    inGraph=True
  if not inGraph:
    return "Either " +str(x) + " or " + str(y) + " not in graph"
  components = connected_component_subgraphs(G_forest)
  shared_tree=False
  for graph in components:
    if x in graph.nodes and y in graph.nodes:
      shared_tree= True
  if not shared_tree:
    return "Nodes " + str(x) +" and " + str(y) + " share no duplication history"
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
    print(x,y)
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
  
  return distance  
  
#def closest_neighbour_distance(x,y,G1_forest,G2_forest):

def label_conserver(G):
  labelConserver=dict()
  origGList=list(G.nodes)
  for i in origGList:
      labelConserverTemp = {"orig_label":i}
      labelConserver[i]=labelConserverTemp
  nx.set_node_attributes(G, labelConserver)
  print(labelConserver)
  print(G.nodes[0]['orig_label'])
  return G




    
  