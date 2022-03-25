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



#------------------------------------------------
#Network alignment

#Here PA and PB are the sets of parents (children) of the nodes a in graph 1 and b in graph 2 respectively
#Let n=min(|PA|,|PB|). This function returns a list of all n-permutations of the 
# elements of the larger of PA and PB
def NF_permlist(PA,PB):
  
  m=max(len(PA),len(PB))
  n=min(len(PA),len(PB))
  if m ==0 or n ==0:
    return []
  #print("pre list make")
  if len(PA)>len(PB):
    #print("PA,PB",PA,PB)
    permListGen=itertools.permutations(list(PA),len(list(PB)))
    
    #print("post list make")
    
    return permListGen
  else:
    #print("PA,PB",PA,PB)
    permListGen=itertools.permutations(list(PB),len(list(PA)))
    
    #print("post list make")
    
    return permListGen
  


#Determines the value of the delta function for a pair of nodes (a,b), a in graph 1, b in graph 2
def NF_delta(a,b,G1,G2,aligned,alpha):
  ###print("delta stRT")
  outDict1=dict(G1.out_degree())
  outDict2=dict(G2.out_degree())
  inDict1=dict(G1.in_degree())
  inDict2=dict(G2.in_degree())
  
  if (a,b) in aligned or max(outDict1[a],outDict2[b])==0 or max(inDict1[a],inDict2[b])==0:
    ###print("delta end")
    return alpha
  else:
    ###print("delta end")
    return min(outDict1[a],outDict2[b])/max(outDict1[a],outDict2[b])+min(inDict1[a],inDict2[b])/max(inDict1[a],inDict2[b])

#Here PA and PB are the sets of parents (children) of the nodes a in graph 1 and b in graph 2 respectively
#Construct the list of all tuples (a_i,b_j) where a_i is a parent (child) of node a in graph 1 and
# b_j is a parent (child) of node b in graph 2
def NF_tupleList(PA,PB):
  ##print("tuplelist stRT")
  tempTupleList=[]
  tupleListList=[]
  pMin = min(len(PA),len(PB))
  pMax=max(len(PA),len(PB))
  if len(PA)==pMin:
    minGraph=0
  else:
    minGraph=1
  #print("tuple pre list make")
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
  #print("tuplelist end")
  return tupleListList

def NF_summer(G1,G2,PA,PB):
  
  minList=[]
  scoreMatrix=np.zeros((len(PA),len(PB)))
  matchMatrixi=np.zeros((len(PA),len(PB)))
  matchMatrixj=np.zeros((len(PA),len(PB)))
  for i in range(len(list(PA))):
      
    for j in range(len(list(PB))):
        iScore=np.abs(G1.out_degree(list(PA)[i])-G2.out_degree(list(PB)[j]))+np.abs(G1.in_degree(list(PA)[i])-G2.in_degree(list(PB)[j]))
        scoreMatrix[i,j]=iScore
        matchMatrixi[i,j]=list(PA)[i]
        matchMatrixj[i,j]=list(PB)[j]
  minAxisLen=min(len(list(PA)),len(list(PB)))
    
  for i in range(minAxisLen):
      #print("min Score",scoreMatrix.min())
      minLoc=scoreMatrix.argmin()
      
      #print("Location of min score",minLoc)
      #print(list(scoreMatrix.shape))
      xLoc=minLoc%list(scoreMatrix.shape)[0]
      yLoc=minLoc%list(scoreMatrix.shape)[1]
      #print("x,y location of min score",xLoc,yLoc)
      lister=[int(matchMatrixi[xLoc,yLoc]),int(matchMatrixj[xLoc,yLoc])]
      minList.append(lister)
      
      scoreMatrix=np.delete(scoreMatrix,(xLoc),axis=0)
      matchMatrixi=np.delete(matchMatrixi,(xLoc),axis=0)
      matchMatrixj=np.delete(matchMatrixj,(xLoc),axis=0)
        
      scoreMatrix=np.delete(scoreMatrix,(yLoc),axis=1)
      matchMatrixi=np.delete(matchMatrixi,(yLoc),axis=1)
      matchMatrixj=np.delete(matchMatrixj,(yLoc),axis=1)
        
      
  return minList

#Returns the NF score function for a pair of nodes (x,y), x in graph 1 and y in graph 2
def NF_scorer(x,y,G1,G2,aligned,alpha,beta,pairingDictParents,pairingDictChildren):
  ###print("scorer stRT")
  
  productOut=1
  productIn=1
  
  for i in pairingDictParents[(x,y)]:
    
    productOut = productOut*NF_delta(i[0],i[1],G1,G2,aligned,alpha)
  
 
  for i in pairingDictChildren[(x,y)]:
    
    productIn = productIn*NF_delta(i[0],i[1],G1,G2,aligned,alpha)
  ###print("scorer end")
  return beta**(np.abs(G1.out_degree(x)-G2.out_degree(y)))*productOut + beta**(np.abs(G1.in_degree(x)-G2.in_degree(y)))*productIn

#Performs node finger##printing for two graphs G1 and G2, with parameters alpha and beta (default 
# alpha=32, beta =0.8

#Performs a one-to-one alignment. This can be easily modified to give a one-to-many,
# many-to-many, many-to-one map
def NF(G1,G2,alpha=32,beta=0.8):
  aligned=[]
  alignedVert1=[]
  alignedVert2=[]
  maxScore=0
  pairingDictParents=dict()
  pairingDictChildren=dict()
  for x in list(G1.nodes()):
      
      PA=list(G1.predecessors(x))
      
      CA=list(G1.successors(x))
      
      for y in list(G2.nodes()):
        
        PB=list(G2.predecessors(y))
        
        CB=list(G2.successors(y))
        pairingDictParents[(x,y)]=NF_summer(G1,G2,PA,PB)
        
        pairingDictChildren[(x,y)]=NF_summer(G1,G2,CA,CB)
        
        #print(x,y,"prepped")
 
  while len(aligned) <min(len(list(G1.nodes())),len(list(G2.nodes()))):
    maxScore=0
    score=0
    for x in list(G1.nodes()):
      for y in list(G2.nodes()):
        if (x,y) not in aligned and x not in alignedVert1 and y not in alignedVert2:
          
          score = NF_scorer(x,y,G1,G2,aligned,alpha,beta,pairingDictParents,pairingDictChildren)
          print(x,y,score)
          if score>maxScore:
            maxScore=score
            maxX=x
            maxY=y
    aligned.append((maxX,maxY))
    print(maxX,maxY, "paired")
    alignedVert1.append(maxX)
    alignedVert2.append(maxY)
  return aligned,alignedVert1

def NF_gene_family(G1,G2,P1,P2,alpha=32,beta=0.8,thresh=5):
  aligned=[]
  alignedVert1=[]
  alignedVert2=[]
  maxScore=0
  pairingDictParents=dict()
  pairingDictChildren=dict()
  
  for x in list(G1.nodes()):
      
      PA=list(G1.predecessors(x))
      
      CA=list(G1.successors(x))
      
      for y in list(G2.nodes()):
        
        PB=list(G2.predecessors(y))
        
        CB=list(G2.successors(y))
        pairingDictParents[(x,y)]=NF_summer(G1,G2,PA,PB)
        
        pairingDictChildren[(x,y)]=NF_summer(G1,G2,CA,CB)
        
        #print(x,y,"prepped")
  maxScore=thresh+1
  while maxScore>thresh:
    maxScore=0
    score=0
    for fam in P1:
      for x in P1[fam]:
        for y in P2[fam]:
          if (x,y) not in aligned and x not in alignedVert1 and y not in alignedVert2:
            
            score = NF_scorer(x,y,G1,G2,aligned,alpha,beta,pairingDictParents,pairingDictChildren)
            
            if score>maxScore:
              maxScore=score
              maxX=x
              maxY=y
    #print(maxX,maxY,maxScore)
    if maxScore>thresh:
      aligned.append((maxX,maxY))
      ###print(maxX,maxY)
      alignedVert1.append(maxX)
      alignedVert2.append(maxY)
  return aligned,alignedVert1

def NF_many_to_one(G1,G2,alpha=32,beta=0.8):
  aligned=[]
  alignedVert1=[]
  alignedVert2=[]
  maxScore=0
  pairingDictParents=dict()
  pairingDictChildren=dict()
  for x in list(G1.nodes()):
      
      PA=list(G1.predecessors(x))
      
      CA=list(G1.successors(x))
      
      for y in list(G2.nodes()):
        
        PB=list(G2.predecessors(y))
        
        CB=list(G2.successors(y))
        pairingDictParents[(x,y)]=NF_summer(G1,G2,PA,PB)
        
        pairingDictChildren[(x,y)]=NF_summer(G1,G2,CA,CB)
        
        #print(x,y,"prepped")
 
  while len(aligned) <min(len(list(G1.nodes())),len(list(G2.nodes()))):
    maxScore=0
    score=0
    for x in list(G1.nodes()):
      for y in list(G2.nodes()):
        if (x,y) not in aligned and x not in alignedVert1:
          
          score = NF_scorer(x,y,G1,G2,aligned,alpha,beta,pairingDictParents,pairingDictChildren)
          
          if score>maxScore:
            maxScore=score
            maxX=x
            maxY=y
    aligned.append((maxX,maxY))
    ###print(maxX,maxY)
    alignedVert1.append(maxX)
    alignedVert2.append(maxY)
  return aligned,alignedVert1

#------------------------------------------------------
#Network evolution models

def duplicate_genes(G,genes,iteration=0):
  mapping = {}
  ##print(iteration)
  for i in genes:
    if iteration!=0:
      mapping[i] = str(i)+"_"+str(iteration)
    else:
      mapping[i] = str(i)+"_c"
  G_sub=G.subgraph(genes)
  G_sub=nx.relabel_nodes(G_sub,mapping)
  G_dup=nx.compose(G,G_sub)

  if iteration!=0:
    for j in genes:
      
      for i in list(G.predecessors(j)):
        if i!=j:  
          G_dup.add_edge(i,str(j)+"_"+str(iteration))
      
      for i in list(G.successors(j)):
        if i!=j:
          G_dup.add_edge(str(j)+"_"+str(iteration),i)
      if (j,j) in G_dup.edges():
        G_dup.add_edge(str(j)+"_"+str(iteration),str(j)+"_"+str(iteration))
  else:
    for i in list(G.nodes()):
      for j in genes:
        if (i,j) in list(G.edges()):
          
          G_dup.add_edge(i,str(j)+"_c")
        if (j,i) in list(G.edges()):
            G_dup.add_edge(str(j)+"_c",i)
  return G_dup

def PED_PEA(G,r,q,iteration=0):
   ###print(iteration)
  G=copy.deepcopy(G)
  nodeNum=len(list(G.nodes))-1
  rando=np.random.random()
  nodeList=list(G.nodes)
  ###print(rando,nodeNum,round(nodeNum*rando))
  dupNode=nodeList[round(nodeNum*rando)]
  
  
  G=duplicate_genes(G,[dupNode],iteration=iteration)
  
  
  parents=copy.deepcopy(G.predecessors(dupNode))
  children=copy.deepcopy(G.successors(dupNode))
  theRemovalists=set(children).union(set(parents))

  nodeListRemoved=set(list(nodeList))-theRemovalists
  
  for i in parents:
    if i==dupNode:
      rando=np.random.rand(1)
    
      if rando <= q:
        rando=np.random.rand(1)
        
        if rando >0.5:
          G.remove_edge(str(dupNode)+"_"+str(iteration),str(dupNode)+"_"+str(iteration))
        else:
          G.remove_edge(dupNode,dupNode)
    else:
      rando=np.random.rand(1)
      
      if rando <= q:
        rando=np.random.rand(1)
        
        if rando >0.5:
          G.remove_edge(i,str(dupNode)+"_"+str(iteration))
        else:
          G.remove_edge(i,dupNode)
  children=copy.deepcopy(G.successors(dupNode))
  for i in children:
    if i==dupNode and ((i,i) in G.edges() and (str(dupNode)+"_"+str(iteration),str(dupNode)+"_"+str(iteration)) in G.edges() ):
      rando=np.random.rand(1)
    
      if rando <= q:
        rando=np.random.rand(1)
        
        if rando >0.5:

          G.remove_edge(str(dupNode)+"_"+str(iteration),str(dupNode)+"_"+str(iteration))
        else:
          G.remove_edge(dupNode,dupNode)
    elif i!= dupNode:
      rando=np.random.rand(1)
      if rando <= q:
        rando=np.random.rand(1)
        if rando >0.5:
          G.remove_edge(str(dupNode)+"_"+str(iteration),i)
        else:
          G.remove_edge(dupNode,i)
  for i in nodeListRemoved:
    rando = np.random.rand(1)
    if rando<r/len(nodeListRemoved):
      rando=np.random.rand(1)
      if rando>0.5:
        G.add_edge(str(dupNode)+"_"+str(iteration),i)
      else:
        G.add_edge(dupNode,i)
  for i in nodeListRemoved:
    rando = np.random.rand(1)
    if rando<r/len(nodeListRemoved):
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
def ped_pea_single_lineage(G,steps,r,q,iteration=0):
    G1=copy.deepcopy(G)

    #G1history=[]
    
    for i in range(0,steps):
        #G1history.append(G1)
        G1=PED_PEA(G1,r,q,iteration=iteration+i+1)
    
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
  
  
  parents=copy.deepcopy(G.predecessors(dupNode))
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
  children=copy.deepcopy(G.successors(dupNode))
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

#DMC model where nodes of degree 0 are removed as they appear
def dmc_modified(G,qCon,qMod,iterations):
  for i in range(0,iterations):
    G=dmc_modified_helper(G,qCon,qMod,iteration=i+1)
    ##print(G.out_degree(0)+G.in_degree(0))
    for j in list(G.nodes()):
      if (G.out_degree(j)==0 and G.in_degree(j)==0):
        G.remove_node(j)
      elif G.out_degree(j)==1 and G.in_degree(j)==1 and (j,j) in list(G.edges):
        G.remove_node(j)
  return G
  
def dmc_modified_helper(G,qCon,qMod,iteration):
  ###print(iteration)
  ##print(iteration)
  G=copy.deepcopy(G)
  nodeNum=len(list(G.nodes))-1
  rando=np.random.random()
  nodeList=list(G.nodes)
  ###print(rando,nodeNum,round(nodeNum*rando))
  dupNode=nodeList[round(nodeNum*rando)]
 
  G=duplicate_genes(G,[dupNode],iteration=iteration)

  parents=copy.deepcopy(G.predecessors(dupNode))
  children=copy.deepcopy(G.successors(dupNode))
  for i in parents:
    rando=np.random.rand(1)
    
    if rando < qMod:
      rando=np.random.rand(1)
      
      if rando >0.5:
        G.remove_edge(i,str(dupNode)+"_"+str(iteration))
      else:
        G.remove_edge(i,dupNode)
  
  for i in children:
    rando=np.random.rand(1)
    if rando < qMod:
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

def print_var_name(variable):
  for theName in globals():
    if eval(theName) == variable:
      return theName

def read_list_as_float(strang):
  vecc=[]
  with open(strang) as f:
    lines = f.read().splitlines()
  for i in lines:
    vecc.append(float(i))
  
  return vecc

def write_list_to_file(thelist,name):
  textfile = open(str(name)+".txt", "w")
  for element in thelist:
      textfile.write(str(element) + "\n")
  textfile.close()

def write_dict_to_file(theDict,name):

  with open(name, 'w') as convert_file:
    convert_file.write(json.dumps(theDict))
def read_dict_float(strang):
  # reading the data from the file
  with open(strang) as f:
      data = f.read()
  js = json.loads(data)
  return js
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


#-----------------------------------------------------------
#Ancestral Reconstruction

def ancestral_likelihood_dmc(G,i,j,qMod,qCon):
  G=copy.deepcopy(G)
  prod=1
  anchorNeighboursIn = list(G.predecessors(i))
  anchorNeighboursOut = list(G.successors(i))
  duplicateNeighboursIn=list(G.predecessors(j))
  duplicateNeighboursOut=list(G.successors(j))
  intersectingNeighboursIn = list(set(anchorNeighboursIn)&set(duplicateNeighboursIn))
  intersectingNeighboursOut = list(set(anchorNeighboursOut)&set(duplicateNeighboursOut))
  intersectingNeighbours=intersectingNeighboursIn+intersectingNeighboursOut
  
  symmdiffNeighboursIn = list(set(anchorNeighboursIn)^set(duplicateNeighboursIn))
  symmdiffNeighboursOut = list(set(anchorNeighboursOut)^set(duplicateNeighboursOut))
  symmdiffNeighbours=symmdiffNeighboursIn+symmdiffNeighboursOut
  neighboursTot=len(intersectingNeighbours+symmdiffNeighbours)
  for k in intersectingNeighbours:
    if k!=i and k!=j:
      prod = prod*(1-qMod)
  for k in symmdiffNeighbours:
    if k!=i and k!=j:
      prod = prod*qMod/2
  if (i,j) in list(G.edges):
    prod=prod*qCon
  else:
    prod=prod*(1-qCon)
  if (j,i) in list(G.edges):
    prod=prod*qCon
  else:
    prod=prod*(1-qCon)
  #prod=prod/len(list(G.nodes))
  #prod=prod*neighboursTot
  return prod
def ancestral_likelihood_ped_pea(G,i,j,r,q):
  G=copy.deepcopy(G)
  prod=1
  anchorNeighboursIn = list(G.predecessors(i))
  anchorNeighboursOut = list(G.successors(i))
  duplicateNeighboursIn=list(G.predecessors(j))
  duplicateNeighboursOut=list(G.successors(j))
  intersectingNeighboursIn = list(set(anchorNeighboursIn)&set(duplicateNeighboursIn))
  intersectingNeighboursOut = list(set(anchorNeighboursOut)&set(duplicateNeighboursOut))
  intersectingNeighbours=intersectingNeighboursIn+intersectingNeighboursOut
  
  symmdiffNeighboursIn = list(set(anchorNeighboursIn)^set(duplicateNeighboursIn))
  symmdiffNeighboursOut = list(set(anchorNeighboursOut)^set(duplicateNeighboursOut))
  symmdiffNeighbours=symmdiffNeighboursIn+symmdiffNeighboursOut
  neighboursTot=len(intersectingNeighbours+symmdiffNeighbours)
  intersectNo=len(intersectingNeighbours)
  theConst=intersectNo/(q*len(list(G.nodes)))
  for k in intersectingNeighbours:
    if k!=i and k!=j:
      prod = prod*(1-q)
  for k in symmdiffNeighbours:
    if k!=i and k!=j:
      prod = prod*(theConst*q/2+(1-theConst)*r/len(list(G.nodes)))
  
  #prod=prod/len(list(G.nodes))
  
  return prod

def one_step_NK(G,qMod,qCon):
  G=copy.deepcopy(G)
  arrival=dict()
  anchor=dict()
  V=list(G.nodes)
  
  Vlen=len(V)
  G_anc=nx.DiGraph(G)
  
  maxLikelihood = -1
  pairList=[]
  for i in V:
    for j in V:
      if j!=i:
        tempLikelihood = ancestral_likelihood_dmc(G,i,j,qMod,qCon)
        if tempLikelihood==maxLikelihood:
          pairList.append((i,j))
        if tempLikelihood>maxLikelihood:
          pairList=[(i,j)]
          maxLikelihood=tempLikelihood
  if len(pairList)==0:
    return G,arrival,anchor
  rando =np.random.random()
  rando = int(np.round(rando*len(pairList)))
    
  selectedNodes=pairList[rando-1]
  ##print("Merge nodes",selectedNodes)
  arrival[selectedNodes[1]]=len(V)
  anchor[selectedNodes[1]]=selectedNodes[0]
  anchorNeighboursIn = list(G.predecessors(selectedNodes[0]))
  anchorNeighboursOut=list(G.successors(selectedNodes[0]))
  duplicateNeighboursIn=list(G.predecessors(selectedNodes[1]))
  duplicateNeighboursOut=list(G.successors(selectedNodes[1]))
  s = set(anchorNeighboursIn)
  anchorNeighboursIn = [x for x in duplicateNeighboursIn if x not in s]
  for i in anchorNeighboursIn:
    if i!=selectedNodes[1] and i in V:
      G_anc.add_edge(i,selectedNodes[0])
  s = set(anchorNeighboursOut)
  anchorNeighboursOut = [x for x in duplicateNeighboursOut if x not in s]
  for i in anchorNeighboursOut:
    if i!=selectedNodes[1] and i in V:
      G_anc.add_edge(selectedNodes[0],i)
  G_anc.remove_node(selectedNodes[1])
  V.remove(selectedNodes[1])
  Vlen=len(V)
  ##print("G_anc modified nodes",G_anc.nodes)
  return G_anc,arrival,anchor
def NK(G,k,qMod,qCon):
  G=copy.deepcopy(G)
  arrival=dict()
  anchor=dict()
  V=list(G.nodes)
  
  Vlen=len(V)
  G_anc=nx.DiGraph(G)
  ##print("G_anc original nodes",G_anc.nodes)
  while Vlen>k:
    maxLikelihood = -1
    pairList=[]
    for i in V:
      for j in V:
        if j!=i:
          tempLikelihood = ancestral_likelihood_dmc(G,i,j,qMod,qCon)
          
          if tempLikelihood==maxLikelihood:
            
            pairList.append((i,j))
            
            
          if tempLikelihood>maxLikelihood:
            pairList=[(i,j)]
            
            maxLikelihood=tempLikelihood
    if len(pairList)==0:
      return G,arrival,anchor
    rando =np.random.random()
    rando = int(np.round(rando*len(pairList)))
    
    selectedNodes=pairList[rando-1]
    ##print("Merge nodes",selectedNodes)
    arrival[selectedNodes[1]]=len(V)
    anchor[selectedNodes[1]]=selectedNodes[0]
    anchorNeighboursIn = list(G.predecessors(selectedNodes[0]))
    anchorNeighboursOut=list(G.successors(selectedNodes[0]))
    duplicateNeighboursIn=list(G.predecessors(selectedNodes[1]))
    duplicateNeighboursOut=list(G.successors(selectedNodes[1]))
    s = set(anchorNeighboursIn)
    anchorNeighboursIn = [x for x in duplicateNeighboursIn if x not in s]
    for i in anchorNeighboursIn:
      if i!=selectedNodes[1] and i in V:
        G_anc.add_edge(i,selectedNodes[0])
    s = set(anchorNeighboursOut)
    anchorNeighboursOut = [x for x in duplicateNeighboursOut if x not in s]
    for i in anchorNeighboursOut:
      if i!=selectedNodes[1] and i in V:
        G_anc.add_edge(selectedNodes[0],i)
    G_anc.remove_node(selectedNodes[1])
    V.remove(selectedNodes[1])
    Vlen=len(V)
    ##print("G_anc modified nodes",G_anc.nodes)
  return G_anc,arrival,anchor
def ancestral_pair(G1,G2,U1,U2,qMod,qCon):
  G1=copy.deepcopy(G1)
  G2=copy.deepcopy(G2)
  G1_induced=nx.induced_subgraph(G1,U1)
  G2_induced=nx.induced_subgraph(G2,U2)
  G1_reduced=G1_induced
  G2_reduced=G2_induced
  x=2
  t=0
  if len(U1)<len(U2):
    ##print("G2")
    t=len(G1_reduced.nodes)-len(G2_reduced.nodes)
    G2_reduced,arrival,anchor=NK(G2_induced,len(U1),qMod,qCon)
    
  if len(U1)>len(U2):
    ##print("G1")
    t=len(G2_reduced.nodes)-len(G1_reduced.nodes)
    G1_reduced,arrival,anchor=NK(G1_induced,len(U2),qMod,qCon)
    x=1
  
  return G1_reduced,G2_reduced,x,t
def gene_family_partitioner(G,orig_label=False):
  if orig_label:
    origin=""
    partition = dict()
    for i in list(G.nodes):
      if "_" in str(G.nodes[i]['orig_label']):
        origin=""
        s=str(G.nodes[i]['orig_label'])[0]
        j=0
        while s!='_':
          
          origin=origin+s
          j=j+1
          
          s=str(G.nodes[i]['orig_label'])[j]
          
      else:
        origin=str(G.nodes[i]['orig_label'])
      if origin not in partition:
        partition[origin]=[]
      partition[origin].append(i)
    
    return partition
  else:
    origin=""
    partition = dict()
    for i in list(G.nodes):
      if "_" in str(i):
        origin=""
        s=str(i)[0]
        j=0
        while s!='_':
          
          origin=origin+s
          j=j+1
          
          s=str(i)[j]
          
      else:
        origin=str(i)
      if origin not in partition:
        partition[origin]=[]
      partition[origin].append(i)
    
    return partition
def gene_family_relabeller(G):
  origin=""
  mapping = dict()
  originCounter=dict()
  leRand=copy.deepcopy(G.nodes())
  random.shuffle(list(G.nodes()))
  print(leRand)
  for i in leRand:
    if "_" in str(i):
      origin=""
      s=str(i)[0]
      j=0
      while s!='_':
        
        origin=origin+s
        j=j+1
        
        s=str(i)[j]
        
    else:
      origin=str(i)
    if origin not in originCounter:
      originCounter[origin]=0
    originCounter[origin]=originCounter[origin]+1
    mapping[i]=str(origin)+"_"+str(originCounter[origin])
  return mapping
def dmc_anc_rec(G1,G2,qMod,qCon):
  P=[]
  E0=[]
  P1 = gene_family_partitioner(G1)
  ##print("Partition G1",P1)
  P2 = gene_family_partitioner(G2)
  ##print("Partition G2",P2)
  if len(P1)<len(P2):
    bigP=P2
    smallP=P1
    Plen=len(smallP)
  else:
    bigP=P1
    smallP=P2
    Plen=len(smallP)
  for i in bigP:
    if i in smallP:
      
      P.append(ancestral_pair(G1,G2,P1[i],P2[i],qMod,qCon))
  B=sorted(P, key=lambda tup: tup[3])
  ##print(B,"B")
  V0=[]
  for i in range(0,len(B)):
    x=B[i][2]
    V0=V0+list((B[i][x-1]).nodes)
    ##print(B[i][x-1].nodes)
    for j in V0:
      if x==1:
        
        G1_induced=nx.induced_subgraph(G1,list(B[i][x-1].nodes))
        
        for u in list(B[i][x-1].nodes):
          if (u,j) in list(G1.edges):
            E0.append((u,j))
          if (j,u) in list(G1.edges):
            E0.append((j,u))
      if x==2:
        G2_induced=nx.induced_subgraph(G2,list(B[i][x-1].nodes))
        for u in list(B[i][x-1].nodes):
          if (u,j) in list(G2.edges):
            E0.append((u,j))
          if(j,u) in list(G2.edges):
            E0.append((j,u))
  
  gAncestral=nx.DiGraph()
  gAncestral.add_nodes_from(V0)
  gAncestral.add_edges_from(E0)
  return gAncestral
def ec_score(G1,G2):
        sourceEdges= len(G1.edges())
        
        conservedEdge=0

        for x in list(G1.nodes):
            for y in list(G1.nodes):
                if (x,y) in list(G1.edges) and (x,y) in list(G2.edges):
                    conservedEdge=conservedEdge+1
                
        G2_ind = nx.induced_subgraph(G2,list(G1.nodes))
        #nx.draw_circular(G2_ind,with_labels=True)
        inducedEdges=len(list(G2_ind.edges))
        
        return conservedEdge/sourceEdges
def normalised_ec_score(G1,G2):
        sourceEdges= len(G1.edges())
        targetEdges=len(G2.edges())
        conservedEdge=0

        for x in list(G1.nodes):
            for y in list(G1.nodes):
                if (x,y) in list(G1.edges) and (x,y) in list(G2.edges):
                    conservedEdge=conservedEdge+1
                
        
        if targetEdges==0 or sourceEdges==0:
            return 0
        else:
            return 0.5*(conservedEdge/sourceEdges+conservedEdge/targetEdges)
def getMCS(G_source, G_new):
  #USER: BONSON STACK OVERFLOW MAY 9 2017
    matching_graph=nx.Graph()

    for n1,n2,attr in G_new.edges(data=True):
        if G_source.has_edge(n1,n2) :
            matching_graph.add_edge(n1,n2,weight=1)

    graphs = list(connected_component_subgraphs(matching_graph))

    mcs_length = 0
    mcs_graph = nx.Graph()
    for i, graph in enumerate(graphs):

        if len(graph.nodes()) > mcs_length:
            mcs_length = len(graph.nodes())
            mcs_graph = graph

    return mcs_graph
def geo_mean(iterable):

    a = np.array(iterable)
    return a.prod()**(1.0/len(a))
def LCCS(G1,G2):
  
  lccs=getMCS(G1,G2)
  G1=nx.induced_subgraph(G1,list(lccs.nodes))
  G2=nx.induced_subgraph(G2,list(lccs.nodes))
  minEdge=min(len(list(G1.edges())),len(list(G2.edges())))
  nodes=len(list(lccs.nodes))
  lister=[minEdge,nodes]
  lccsScore=geo_mean(lister)
  return lccsScore
def between_family_conserved_edges(G1,G2,P1,P2):
        sourceEdges= len(G1.edges())
        targetEdges=len(G2.edges())
        conservedEdge=0
        for fam1 in P1:
          for fam2 in P1:
            if fam2!=fam1:
              for x in list(P1[fam1]):
                for y in list(P1[fam2]):
                  if (x,y) in list(G1.edges) and (x,y) in list(G2.edges):
                    conservedEdge=conservedEdge+1
                
        
        if targetEdges==0 or sourceEdges==0:
            return 0
        else:
            return conservedEdge
                
        
def conserved_edges(G1,G2):
        sourceEdges= len(G1.edges())
        targetEdges=len(G2.edges())
        conservedEdge=0
        
        for x in list(G1.nodes):
            for y in list(G1.nodes):
                if (x,y) in list(G1.edges) and (x,y) in list(G2.edges):
                    conservedEdge=conservedEdge+1
                
        
        if targetEdges==0 or sourceEdges==0:
            return 0
        else:
            return conservedEdge
                
        
def ics_score(G1,G2):
        sourceEdges= len(list(G1.edges()))
        
        conservedEdge=0

        for x in list(G1.nodes):
            for y in list(G1.nodes):
                if (x,y) in list(G1.edges) and (x,y) in list(G2.edges):
                    conservedEdge=conservedEdge+1
                
        G2_ind = nx.induced_subgraph(G2,list(G1.nodes))
        #nx.draw_circular(G2_ind,with_labels=True)
        inducedEdges=len(list(G2_ind.edges))
        if inducedEdges==0:
            return 0
        else:
          return conservedEdge/inducedEdges
def s3_score(G1,G2):
        sourceEdges= len(G1.edges())
        
        conservedEdge=0

        for x in list(G1.nodes):
            for y in list(G1.nodes):
                if (x,y) in list(G1.edges) and (x,y) in list(G2.edges):
                    conservedEdge=conservedEdge+1
                
        G2_ind = nx.induced_subgraph(G2,list(G1.nodes))
        #nx.draw_circular(G2_ind,with_labels=True)
        inducedEdges=len(list(G2_ind.edges))
        if sourceEdges+inducedEdges-conservedEdge==0:
            return 0
        else:
            return conservedEdge/(sourceEdges+inducedEdges-conservedEdge)

def normalised_s3_score(G1,G2):
        sourceEdges1= len(G1.edges())
        sourceEdges2=len(G2.edges())
        conservedEdge=0

        for x in list(G1.nodes):
            for y in list(G1.nodes):
                if (x,y) in list(G1.edges) and (x,y) in list(G2.edges):
                    conservedEdge=conservedEdge+1
                
        G2_ind = nx.induced_subgraph(G2,list(G1.nodes))
        G1_ind = nx.induced_subgraph(G1,list(G2.nodes))
        #nx.draw_circular(G2_ind,with_labels=True)
        inducedEdges2=len(list(G2_ind.edges))
        inducedEdges1=len(list(G1_ind.edges))
        if sourceEdges1+inducedEdges2-conservedEdge==0 or sourceEdges2+inducedEdges1-conservedEdge==0:
            return 0
        else:
            return 0.5*(conservedEdge/(sourceEdges1+inducedEdges2-conservedEdge)+conservedEdge/(sourceEdges2+inducedEdges1-conservedEdge))
def node_merger(G,i,j,self_loops=True):
  G=copy.deepcopy(G)
  G_anc=copy.deepcopy(G)
  anchorNeighboursIn = list(G.predecessors(i))
  anchorNeighboursOut=list(G.successors(i))
  duplicateNeighboursIn=list(G.predecessors(j))
  duplicateNeighboursOut=list(G.successors(j))

  s = set(anchorNeighboursIn)
  anchorNeighboursIn = [x for x in duplicateNeighboursIn if x not in s]
  for k in anchorNeighboursIn:
    if k!=j and k in list(G.nodes):
      if k==i:
        if self_loops:
          G_anc.add_edge(k,i)
      else:
        G_anc.add_edge(k,i)
  
  s = set(anchorNeighboursOut)
  anchorNeighboursOut = [x for x in duplicateNeighboursOut if x not in s]
  for k in anchorNeighboursOut:
    if k!=j and k in list(G.nodes):
      if k==i:
        if self_loops:
          G_anc.add_edge(k,i)
      else:
        G_anc.add_edge(k,i)
  G_anc.remove_node(j)
  return G_anc



def ancestor_finder_without_alignment_alt(G1,G2,qMod,qCon,tolerance=0.05):
  """Returns a pair of networkx graphs G1' and G2' based off of the input graphs G1 and G2
    with nodes merged as to maximise their EC similarity.

    This is a heuristic algorithm that finds pairs of nodes, either in G1 and G2, that are the most likely to have been the most
    recently duplicated pairs and determines which of these pairs, when merged, will increase the
    EC similarity of the two graphs.

    The best pair under these conditions is chosen and is merged, and the process is repeated until the EC score 
    can no longer be improved. That is, a local optimum is found.

    Parameters
    ----------
    G1 : networkx graph
        One of the two graphs
    G2 : networkx graph
        One of the two graphs
    qMod : float
        The probability of edge deletion in the assumed forward DMC model.
        In cases where this is not known it may need to be estimated.
    qCon : float
        The probability of creating an edge between duplicated pairs of vertices in the assumed forward DMC model.
        In cases where this is not known it may need to be estimated.
    tolerance : float, optional (default=0.05)
        A tolerance value used to determine the acceptable likelihood a pair of nodes to have
        been the most recently duplicated pair. A non-zero tolerance allows less likely pairs to be chosen, possibly 
        increasing the EC score more than the best pair.

    Notes
    -----
    This algorithm is not guaranteed to find the global optimum. 

    This algorithm comprises 4 main steps:
    1. The likelihood of all pairs of nodes in both G1 and G2 are calculated.
    2. A list of the 'best' pairs is generated. These pairs are those that have the highest likelihood, 
    depending on the tolerance value.
    3. From these pairs, the pair(s) that improve(s) the EC score the most is chosen and merged. Most of the time there
    is more than one such pair, in which case a random one is chosen.
    4. The graph in which the most recent merge has occured is updated and the process is repeated until the EC score 
    can no longer be improved.
    
    Issues 
    ------
    1. For the first go of the loop, the maxScore is -1 and so all the scores are greater than it
    2. It is assumed that likelihoods are comparable across the graphs G1 and G2 for different sized graphs. 
    This very well may not be the case and should be followed up.
    3. Many pairs seem to have the give the same EC score. This could be fine, but possibly something is going on here.

    """
  #graphPair is a repeatedly updated tuple containing the present state of graphs G1 and G2, the EC score of the current pair,
  # and an int that represents which of G1 and G2 the most recent merge occured in.
  graphPair=(G1,G2,-1,0) 
  #signal is a boolean that is set to False when the EC score of the current pair is no longer improving.
  signal =True 
  
  #maxScore records the current best EC score
  maxScore=-1

  while signal:
    #step 1: calculate the likelihood of all pairs of nodes in both G1 and G2

    #alignedPairs is a dictionary, keyed by (i,j,k) that records the EC score of graphs resulting from merging 
    # each pair of nodes i,j in graph G_k (k=1 or k=2).
    alignedPairs=dict()
    #maxLikelihood is updated each loop and records the likelihood score of the most likely node pair for the current loop
    maxLikelihood = -1
    #pairList is a list of tuples (i,j,k,likelihood) where i and j are the nodes in graph G_k
    # and likelihood is the likelihood of the pair (i,j) being most recently duplicated in graph G_k
    pairList1=[]
    pairList2=[]
    #G1 and G2 are updated to be the most recent pair of graphs
    G1=graphPair[0]
    G2=graphPair[1]

    #Consider each node pair (i,j) in G1
    for i in list(G1.nodes):
      for j in list(G1.nodes):
        if j!=i:
          #Construct the graph G1_temp resulting from merging (i,j) in G1
          #This pair of if statements ensures that a node merge keeps a label common to both graphs if possible
          if j not in G2.nodes and i in G2.nodes:
            G1_temp=node_merger(G1,i,j)
          else:
            G1_temp=node_merger(G1,j,i)
          
          #measure the EC score between G1_temp and G2
          tempScore= normalised_ec_score(G1_temp,G2)
          
          #check the likelihood function for the pair (i,j)
          tempLikelihood = ancestral_likelihood_dmc(G1,i,j,qMod,qCon)
          
          #Update the pairList and the alignedPairs dictionary with the likelihood of the pair (i,j) and the EC score
          # of the merged node graph G1_temp and G2
          pairList1.append((i,j,1,tempLikelihood))
          alignedPairs[(i,j,1)]=(G1_temp,G2,tempScore,1)
         
    #Consider each node pair (i,j) in G2
    for i in list(G2.nodes):  
      for j in list(G2.nodes):
        if j!=i:
          #Construct the graph G2_temp resulting from merging (i,j) in G2
          #This pair of if statements ensures that a node merge keeps a label common to both graphs if possible
          if j not in G1.nodes and i in G1.nodes:
            G2_temp=node_merger(G2,i,j)
          else:
            G2_temp=node_merger(G2,j,i)
          #measure the EC score between G2_temp and G1
          tempScore= normalised_ec_score(G2_temp,G1)

          #check the likelihood function for the pair (i,j)
          tempLikelihood = ancestral_likelihood_dmc(G2,i,j,qMod,qCon)
          
          #Update the pairList and the alignedPairs dictionary with the likelihood of the pair (i,j) and the EC score
          # of the merged node graph G2_temp and G1
          pairList2.append((i,j,2,tempLikelihood))
          alignedPairs[(i,j,2)]=(G1,G2_temp,tempScore,2)
          
    #step 2: choose the best pair(s) based on the likelihood function and the tolerance value
    #End algorithm if pairList is empty. Only occurs if both graphs are single node or empty.
    if pairList1==[] and pairList2==[]:
      signal=False
    #if pairList isn't empty, choose the best pair(s) based on the likelihood function and the tolerance value  
    else:
      pairListFinal=[]
      maxLikelihood1=max(pairList1,key=lambda x:x[3])[3] #find the maximum likelihood over all pairs
      
      for i in pairList1:
        if i[3]>=maxLikelihood1-tolerance*maxLikelihood1:
          pairListFinal.append((i[0],i[1],i[2])) #add the pair to the final list if the likelihood is greater than the maximum-tolerance
      
      maxLikelihood2=max(pairList2,key=lambda x:x[3])[3] #find the maximum likelihood over all pairs
      
      for i in pairList2:
        if i[3]>=maxLikelihood2-tolerance*maxLikelihood2:
          pairListFinal.append((i[0],i[1],i[2])) #add the pair to the final list if the likelihood is greater than the maximum-tolerance
    
    #step 3: from the most likely pair(s) choose the best pair(s) based on the EC score    
      
      highScoreList=[] #highScoreList refines the list of node pairs even further, by selecting only those that give the highest EC score
      #pairListFinal should never be empty, but is included to be safe
      if pairListFinal==[]:
        signal=False
        
      else:
        #the maximum of all scores of this iteration is recorded in maxScoreOfList to set the maxScore for the next iteration
        maxScoreOfList=0
        #for each pair in pairListFinal, find the EC score given by merging them. If this score improves upon the score in the previous
        # iteration, add the pair to the highScoreList.
        for i in pairListFinal:
          graphPairs=alignedPairs[i]
          tempScore=graphPairs[2]
          if tempScore>=maxScore: #issue here that for the first merge, the maxScore is -1 and so all the scores are greater than it
            if tempScore>=maxScoreOfList: 
              maxScoreOfList=tempScore
            
            highScoreList.append(graphPairs)
    #If highScoreList is empty, then a local optimum is reached and the algorithm has finished.
        if highScoreList==[]:
          signal=False
          print("highscorelist empty")
        else:
    #For each of the best pairs, weight the pairs by their EC score and randomly pick a pair according to these weights
          weights=[]
          for i in range(len(highScoreList)):
            if highScoreList[i][2]!=0:
              weights.append(highScoreList[i][2])
    #If an EC score is 0 then it is weighted with a very small amount (0 messes with the random.choices function),
    #but a pair with EC score 0 should theoretically not appear anyway.
            else:
              weights.append(0.0000001)
          print(pairListFinal)
          print(weights)  
          #Pick the pair to merge and update the graph pair with the merged graph (either G1 or G2)
          graphPair=random.choices(highScoreList,weights=weights)
          #step 4: the graph pair is updated
          #output of random.choices is a single entry list, so the first entry is used
          graphPair=graphPair[0]
          print(graphPair)
          #update the maxScore for the next iteration
          maxScore=graphPair[2]
          
  return graphPair[0],graphPair[1]
def takeThird(elem):
  return elem[2]
def ancestor_finder_without_alignment_alt_alt(G1,G2,qMod,qCon,tolerance=0.05):
  """Returns a pair of networkx graphs G1' and G2' based off of the input graphs G1 and G2
    with nodes merged as to maximise their EC similarity.

    This is a heuristic algorithm that finds pairs of nodes, either in G1 and G2, that are the most likely to have been the most
    recently duplicated pairs and determines which of these pairs, when merged, will increase the
    EC similarity of the two graphs.

    The best pair under these conditions is chosen and is merged, and the process is repeated until the EC score 
    can no longer be improved. That is, a local optimum is found.

    Parameters
    ----------
    G1 : networkx graph
        One of the two graphs
    G2 : networkx graph
        One of the two graphs
    qMod : float
        The probability of edge deletion in the assumed forward DMC model.
        In cases where this is not known it may need to be estimated.
    qCon : float
        The probability of creating an edge between duplicated pairs of vertices in the assumed forward DMC model.
        In cases where this is not known it may need to be estimated.
    tolerance : float, optional (default=0.05)
        A tolerance value used to determine the acceptable likelihood a pair of nodes to have
        been the most recently duplicated pair. A non-zero tolerance allows less likely pairs to be chosen, possibly 
        increasing the EC score more than the best pair.

    Notes
    -----
    This algorithm is not guaranteed to find the global optimum. 

    This algorithm comprises 4 main steps:
    1. The likelihood of all pairs of nodes in both G1 and G2 are calculated.
    2. A list of the 'best' pairs is generated. These pairs are those that have the highest likelihood, 
    depending on the tolerance value.
    3. From these pairs, the pair(s) that improve(s) the EC score the most is chosen and merged. Most of the time there
    is more than one such pair, in which case a random one is chosen.
    4. The graph in which the most recent merge has occured is updated and the process is repeated until the EC score 
    can no longer be improved.
    
    Issues 
    ------
    1. For the first go of the loop, the maxScore is -1 and so all the scores are greater than it
    2. It is assumed that likelihoods are comparable across the graphs G1 and G2 for different sized graphs. 
    This very well may not be the case and should be followed up.
    3. Many pairs seem to have the give the same EC score. This could be fine, but possibly something is going on here.

    """
  #graphPair is a repeatedly updated tuple containing the present state of graphs G1 and G2, the EC score of the current pair,
  # and an int that represents which of G1 and G2 the most recent merge occured in.
  graphPair=(G1,G2,-1,0) 
  #signal is a boolean that is set to False when the EC score of the current pair is no longer improving.
  signal =True 
  
  #maxScore records the current best EC score
  maxScore=-1

  while signal:
    #step 1: calculate the likelihood of all pairs of nodes in both G1 and G2

    #alignedPairs is a dictionary, keyed by (i,j,k) that records the EC score of graphs resulting from merging 
    # each pair of nodes i,j in graph G_k (k=1 or k=2).
    alignedPairs=dict()
    #maxLikelihood is updated each loop and records the likelihood score of the most likely node pair for the current loop
    maxLikelihood = -1
    #pairList is a list of tuples (i,j,k,likelihood) where i and j are the nodes in graph G_k
    # and likelihood is the likelihood of the pair (i,j) being most recently duplicated in graph G_k
    pairList1=[]
    pairList2=[]
    #G1 and G2 are updated to be the most recent pair of graphs
    G1=graphPair[0]
    G2=graphPair[1]

    #Consider each node pair (i,j) in G1
    for i in list(G1.nodes):
      for j in list(G1.nodes):
        if j!=i:
          #Construct the graph G1_temp resulting from merging (i,j) in G1
          #This pair of if statements ensures that a node merge keeps a label common to both graphs if possible
          if j not in G2.nodes and i in G2.nodes:
            G1_temp=node_merger(G1,i,j)
          else:
            G1_temp=node_merger(G1,j,i)
          
          #measure the EC score between G1_temp and G2
          tempScore= normalised_ec_score(G1_temp,G2)
          
          #check the likelihood function for the pair (i,j)
          tempLikelihood = ancestral_likelihood_dmc(G1,i,j,qMod,qCon)
          
          #Update the pairList and the alignedPairs dictionary with the likelihood of the pair (i,j) and the EC score
          # of the merged node graph G1_temp and G2
          pairList1.append((i,j,1,tempLikelihood))
          alignedPairs[(i,j,1)]=(i,j,tempScore,1)
         
    #Consider each node pair (i,j) in G2
    for i in list(G2.nodes):  
      for j in list(G2.nodes):
        if j!=i:
          #Construct the graph G2_temp resulting from merging (i,j) in G2
          #This pair of if statements ensures that a node merge keeps a label common to both graphs if possible
          if j not in G1.nodes and i in G1.nodes:
            G2_temp=node_merger(G2,i,j)
          else:
            G2_temp=node_merger(G2,j,i)
          #measure the EC score between G2_temp and G1
          tempScore= normalised_ec_score(G2_temp,G1)

          #check the likelihood function for the pair (i,j)
          tempLikelihood = ancestral_likelihood_dmc(G2,i,j,qMod,qCon)
          
          #Update the pairList and the alignedPairs dictionary with the likelihood of the pair (i,j) and the EC score
          # of the merged node graph G2_temp and G1
          pairList2.append((i,j,2,tempLikelihood))
          alignedPairs[(i,j,2)]=(i,j,tempScore,2)
          
    #step 2: choose the best pair(s) based on the likelihood function and the tolerance value
    #End algorithm if pairList is empty. Only occurs if both graphs are single node or empty.
    if pairList1==[] and pairList2==[]:
      signal=False
    #if pairList isn't empty, choose the best pair(s) based on the likelihood function and the tolerance value  
    else:
      pairListFinal=[]
      maxLikelihood1=max(pairList1,key=lambda x:x[3])[3] #find the maximum likelihood over all pairs
      
      for i in pairList1:
        if i[3]>=maxLikelihood1-tolerance*maxLikelihood1:
          pairListFinal.append((i[0],i[1],i[2])) #add the pair to the final list if the likelihood is greater than the maximum-tolerance
      
      maxLikelihood2=max(pairList2,key=lambda x:x[3])[3] #find the maximum likelihood over all pairs
      
      for i in pairList2:
        if i[3]>=maxLikelihood2-tolerance*maxLikelihood2:
          pairListFinal.append((i[0],i[1],i[2])) #add the pair to the final list if the likelihood is greater than the maximum-tolerance
    
    #step 3: from the most likely pair(s) choose the best pair(s) based on the EC score    
      
      highScoreList=[] #highScoreList refines the list of node pairs even further, by selecting only those that give the highest EC score
      #pairListFinal should never be empty, but is included to be safe
      if pairListFinal==[]:
        signal=False
        
      else:
        #the maximum of all scores of this iteration is recorded in maxScoreOfList to set the maxScore for the next iteration
        maxScoreOfList=0
        #for each pair in pairListFinal, find the EC score given by merging them. If this score improves upon the score in the previous
        # iteration, add the pair to the highScoreList.
        for i in pairListFinal:
          graphPairs=alignedPairs[i]
          tempScore=graphPairs[2]
          if tempScore>=maxScore: #issue here that for the first merge, the maxScore is -1 and so all the scores are greater than it
            if tempScore>=maxScoreOfList: 
              maxScoreOfList=tempScore
            
            highScoreList.append(graphPairs)
    #If highScoreList is empty, then a local optimum is reached and the algorithm has finished.
        if highScoreList==[]:
          signal=False
          print("highscorelist empty")
        else:
    #For each of the best pairs, weight the pairs by their EC score and randomly pick a pair according to these weights
          
          highScoreList.sort(reverse=True,key=takeThird)

            
          print(pairListFinal)
          print(highScoreList)
          for i in highScoreList:
            if i[3]==1:
              if i[0] in G1.nodes() and i[1] in G1.nodes():
                print(i[0],i[1])
                if i[1] not in G2.nodes and i[0] in G2.nodes:
                  G1_temp=node_merger(G1,i[0],i[1])
                else:
                  
                  G1_temp=node_merger(G1,i[1],i[0])
                print('1',normalised_ec_score(G1_temp,G2))
                if normalised_ec_score(G1_temp,G2)>=maxScore:
                  print("more gooder 1")
                  G1=G1_temp
                  maxScore=normalised_ec_score(G1_temp,G2)
                  
            elif i[3]==2:
              if i[0] in G2.nodes() and i[1] in G2.nodes():
                print(i[0],i[1])
                if i[1] not in G1.nodes and i[0] in G1.nodes:
                  
                  G2_temp=node_merger(G2,i[0],i[1])
                else:
                  G2_temp=node_merger(G2,i[1],i[0])
                print('2',normalised_ec_score(G2_temp,G1))
                if normalised_ec_score(G2_temp,G1)>=maxScore:
                  print("more gooder 2")
                  G2=G2_temp
                  maxScore=normalised_ec_score(G2_temp,G1)
          #step 4: the graph pair is updated
          #output of random.choices is a single entry list, so the first entry is used
          graphPair=(G1,G2,normalised_ec_score(G1,G2),1)
          print(graphPair)
          #update the maxScore for the next iteration
          maxScore=normalised_ec_score(G1,G2)
  print(graphPair[0].edges(),graphPair[1].edges())
  return graphPair[0],graphPair[1]

def ancestor_finder_without_alignment_alt_alt_alt(G1,G2,qMod,qCon,tolerance=0,toleranceEC=0.02):
  """Returns a pair of networkx graphs G1' and G2' based off of the input graphs G1 and G2
    with nodes merged as to maximise their similarity.

    This is a heuristic algorithm that finds pairs of nodes, either in G1 and G2, that are the most likely to have been the most
    recently duplicated pairs and determines which of these pairs, when merged, will increase the
    similarity of the two graphs.

    The best pair under these conditions is chosen and is merged, and the process is repeated until the S3 score 
    can no longer be improved. That is, a local optimum is found.

    Parameters
    ----------
    G1 : networkx graph
        One of the two graphs
    G2 : networkx graph
        One of the two graphs
    qMod : float
        The probability of edge deletion in the assumed forward DMC model.
        In cases where this is not known it may need to be estimated.
    qCon : float
        The probability of creating an edge between duplicated pairs of vertices in the assumed forward DMC model.
        In cases where this is not known it may need to be estimated.
    toleranceLikelihood : float, optional (default=0.05)
        A tolerance value used to determine the acceptable likelihood a pair of nodes to have
        been the most recently duplicated pair. A non-zero tolerance allows less likely pairs to be chosen, possibly 
        increasing the EC score more than the best pair.
    toleranceScore : float, optional (default=0.05)
        A tolerance value used to determine the acceptable decrease in similarity score compared to
        the previous step. With a tolerance value of 0, the algorithm will typically find a local optimum too early.
        With a tolerance value too large, the algorithm will produce the a single node graph for G1' and G2'.

    Notes
    -----
    
    This algorithm comprises 4 main steps:
    1. The likelihood of all pairs of nodes in both G1 and G2 are calculated.
    2. A list of the 'best' pairs is generated. These pairs are those that have the highest likelihood, 
    depending on the likelihood tolerance value.
    3. From these pairs, the pair(s) of node pairs (one pair in G1 and one pair in G2) that most
    improve(s) the similarity score when merged (in G1 and G2 respectively) is chosen and merged. Most of the time there
    is more than one such pair, in which case a random one is chosen.
    4. G1 and G2 are updated with the chosen merged nodes and the process is repeated until the similarity score 
    can no longer be improved, dependent on the similarity tolerance value toleranceScore.
    
    Issues 
    ------
    1. For the first go of the loop, the maxScore is -1 and so all the scores are greater than it
    
    2. Many pairs seem to have the give the same similarity score. This could be fine, but possibly something is going on here.

    """
  
  #graphPair is a repeatedly updated tuple containing the present state of graphs G1 and G2, the similarity score of the current pair,
  # and an int that represents which of G1 and G2 the most recent merge occured in.
  graphPair=(G1,G2,-1,0) 
  #signal is a boolean that is set to False when the EC score of the current pair is no longer improving.
  signal =True 
  
  #maxScore records the current best similarity score for the current loop
  maxScore=-1
  #totalMax records the previous best similarity score
  totalMax=-1
  while signal:
    #step 1: calculate the likelihood of all pairs of nodes in both G1 and G2

    #alignedPairs is a list that records the similarity score of graphs resulting from merging 
    # each pair of nodes i,j in graph G_k (k=1 or k=2).
    alignedPairs=[]
    
    #pairList1 and pairList2 are lists of tuples (i,j,k,likelihood) where i and j are the nodes in graph G_k
    # and likelihood is the likelihood of the pair (i,j) being most recently duplicated in graph G_k
    pairList1=[]
    pairList2=[]
    #G1 and G2 are updated to be the most recent pair of graphs
    G1=graphPair[0]
    G2=graphPair[1]

    #Consider each node pair (i,j) in G1
    for i in list(G1.nodes):
      for j in list(G1.nodes):
        if j!=i:
          
          #check the likelihood function for the pair (i,j)
          tempLikelihood = ancestral_likelihood_dmc(G1,i,j,qMod,qCon)
          
          #Update the pairList and the alignedPairs dictionary with the likelihood of the pair (i,j)
          pairList1.append((i,j,1,tempLikelihood))
        
    #Consider each node pair (i,j) in G2
    for i in list(G2.nodes):  
      for j in list(G2.nodes):
        if j!=i:
          
          #check the likelihood function for the pair (i,j)
          tempLikelihood = ancestral_likelihood_dmc(G2,i,j,qMod,qCon)
          
          #Update the pairList and the alignedPairs dictionary with the likelihood of the pair (i,j) 
          pairList2.append((i,j,2,tempLikelihood))
              
    #step 2: choose the best pair(s) based on the likelihood function and the tolerance value
    #End algorithm if pairList is empty. Only occurs if both graphs are single node or empty.
    if pairList1==[] and pairList2==[]:
      signal=False
    #if pairList isn't empty, choose the best pair(s) based on the likelihood function and the tolerance value  
    else:
      pairListFinal1=[]
      pairListFinal2=[]
      maxLikelihood1=max(pairList1,key=lambda x:x[3])[3] #find the maximum likelihood over all pairs
      
      for i in pairList1:
        if i[3]>=maxLikelihood1-tolerance*maxLikelihood1:
          pairListFinal1.append((i[0],i[1],i[2])) #add the pair to the final list if the likelihood is greater than the maximum-tolerance
      
      maxLikelihood2=max(pairList2,key=lambda x:x[3])[3] #find the maximum likelihood over all pairs
      
      for i in pairList2:
        if i[3]>=maxLikelihood2-tolerance*maxLikelihood2:
          pairListFinal2.append((i[0],i[1],i[2])) #add the pair to the final list if the likelihood is greater than the maximum-tolerance

      for i in pairListFinal1:
        #construct the graph G1_temp resulting from merging (i,j) in G1
        #This pair of if statements ensures that a node merge keeps a label common to both graphs if possible
          
        if i[1] not in G2.nodes and i[0] in G2.nodes:
            G1_temp=node_merger(G1,i[0],i[1])
        else:
            G1_temp=node_merger(G1,i[1],i[0])
        for j in pairListFinal2:
          #Construct the graph G2_temp resulting from merging (i,j) in G2
          #This pair of if statements ensures that a node merge keeps a label common to both graphs if possible
          
          if j[1] not in G1.nodes and j[0] in G1.nodes:
            G2_temp=node_merger(G2,j[0],j[1])
          else:
            G2_temp=node_merger(G2,j[1],j[0])
          tempScore= normalised_ec_score(G1_temp,G2_temp)
          alignedPairs.append((G1_temp,G2_temp,tempScore))
    #step 3: from the most likely pair(s) choose the best pair(s) based on the similarity score    
      
      highScoreList=[] #highScoreList refines the list of node pairs even further, by selecting only those that give the highest EC score
      #pairListFinal should never be empty, but is included to be safe
      if alignedPairs==[]:
        signal=False
        
      else:
        #the maximum of all scores of this iteration is recorded in maxScoreOfList to set the maxScore for the next iteration
        
        #for each pair in pairListFinal, find the EC score given by merging them. If this score improves upon the score in the previous
        # iteration, add the pair to the highScoreList.
        maxScore=0
        for i in alignedPairs:
          graphPairs=i
          tempScore=graphPairs[2]
          if tempScore>=maxScore:
            maxScore=tempScore #issue here that for the first merge, the maxScore is -1 and so all the scores are greater than it
        if maxScore>=totalMax-toleranceEC*totalMax:
          for i in alignedPairs:
            graphPairs=i
            tempScore=graphPairs[2]
            if tempScore>=maxScore:
                
              highScoreList.append(graphPairs)
          if highScoreList==[]:
            signal=False
            print("highscorelist empty")
          else:
      #For each of the best pairs, weight the pairs by their EC score and randomly pick a pair according to these weights
            weights=[]
            for i in range(len(highScoreList)):
              if highScoreList[i][2]!=0:
                weights.append(highScoreList[i][2])
      #If an EC score is 0 then it is weighted with a very small amount (0 messes with the random.choices function),
      #but a pair with EC score 0 should theoretically not appear anyway.
              else:
                weights.append(0.0000001)
            print(weights)
            
            #Pick the pair to merge and update the graph pair with the merged graph (either G1 or G2)
            graphPair=random.choices(highScoreList,weights=weights)
            #step 4: the graph pair is updated
            #output of random.choices is a single entry list, so the first entry is used
            graphPair=graphPair[0]
            print(graphPair)
            #update the maxScore for the next iteration
            totalMax=graphPair[2]
        else:
          signal=False
    #If highScoreList is empty, then a local optimum is reached and the algorithm has finished.
          
        
  return graphPair[0],graphPair[1]

def ancestor_finder_without_alignment_alt_alt_alt_alt(G1,G2,qMod,qCon,tolerance=0,toleranceEC=0):
  """Returns a pair of networkx graphs G1' and G2' based off of the input graphs G1 and G2
    with nodes merged as to maximise their EC similarity.

    This is a heuristic algorithm that finds pairs of nodes, either in G1 and G2, that are the most likely to have been the most
    recently duplicated pairs and determines which of these pairs, when merged, will increase the
    EC similarity of the two graphs.

    The best pair under these conditions is chosen and is merged, and the process is repeated until the EC score 
    can no longer be improved. That is, a local optimum is found.

    Parameters
    ----------
    G1 : networkx graph
        One of the two graphs
    G2 : networkx graph
        One of the two graphs
    qMod : float
        The probability of edge deletion in the assumed forward DMC model.
        In cases where this is not known it may need to be estimated.
    qCon : float
        The probability of creating an edge between duplicated pairs of vertices in the assumed forward DMC model.
        In cases where this is not known it may need to be estimated.
    tolerance : float, optional (default=0.05)
        A tolerance value used to determine the acceptable likelihood a pair of nodes to have
        been the most recently duplicated pair. A non-zero tolerance allows less likely pairs to be chosen, possibly 
        increasing the EC score more than the best pair.

    Notes
    -----
    This algorithm is not guaranteed to find the global optimum. 

    This algorithm comprises 4 main steps:
    1. The likelihood of all pairs of nodes in both G1 and G2 are calculated.
    2. A list of the 'best' pairs is generated. These pairs are those that have the highest likelihood, 
    depending on the likelihood tolerance value.
    3. From these pairs, the pair(s) that improve(s) the EC score the most is chosen and merged. Most of the time there
    is more than one such pair, in which case a random one is chosen.
    4. The graph in which the most recent merge has occured is updated and the process is repeated until the EC score 
    can no longer be improved.
    
    Issues 
    ------
    1. For the first go of the loop, the maxScore is -1 and so all the scores are greater than it
    2. It is assumed that likelihoods are comparable across the graphs G1 and G2 for different sized graphs. 
    This very well may not be the case and should be followed up.
    3. Many pairs seem to have the give the same EC score. This could be fine, but possibly something is going on here.

    """
  
  #graphPair is a repeatedly updated tuple containing the present state of graphs G1 and G2, the EC score of the current pair,
  # and an int that represents which of G1 and G2 the most recent merge occured in.
  graphPair=(G1,G2,-1,0) 
  #signal is a boolean that is set to False when the EC score of the current pair is no longer improving.
  signal =True 
  
  #maxScore records the current best EC score
  maxScore=-1
  n=len(G1.nodes())
  theGraphList=[]
  while signal and len(graphPair[0].nodes)>30:
    #step 1: calculate the likelihood of all pairs of nodes in both G1 and G2

    #alignedPairs is a dictionary, keyed by (i,j,k) that records the EC score of graphs resulting from merging 
    # each pair of nodes i,j in graph G_k (k=1 or k=2).
    alignedPairs=[]
    #maxLikelihood is updated each loop and records the likelihood score of the most likely node pair for the current loop
    maxLikelihood = -1
    #pairList is a list of tuples (i,j,k,likelihood) where i and j are the nodes in graph G_k
    # and likelihood is the likelihood of the pair (i,j) being most recently duplicated in graph G_k
    pairList1=[]
    pairList2=[]
    #G1 and G2 are updated to be the most recent pair of graphs
    G1=graphPair[0]
    G2=graphPair[1]

    #Consider each node pair (i,j) in G1
    for num1,i in enumerate(list(G1.nodes)):
      for num2,j in enumerate(list(G1.nodes)):
        if num2>num1:
          #Construct the graph G1_temp resulting from merging (i,j) in G1
          #This pair of if statements ensures that a node merge keeps a label common to both graphs if possible
          
          #check the likelihood function for the pair (i,j)
          tempLikelihood = ancestral_likelihood_dmc(G1,i,j,qMod,qCon)
          
          #Update the pairList and the alignedPairs dictionary with the likelihood of the pair (i,j) and the EC score
          # of the merged node graph G1_temp and G2
          pairList1.append((i,j,1,tempLikelihood))
          
         
    #Consider each node pair (i,j) in G2
    for num1,i in enumerate(list(G2.nodes)):  
      for num2,j in enumerate(list(G2.nodes)):
        if num2>num1:
          #Construct the graph G2_temp resulting from merging (i,j) in G2
          #This pair of if statements ensures that a node merge keeps a label common to both graphs if possible
          
          
          #check the likelihood function for the pair (i,j)
          tempLikelihood = ancestral_likelihood_dmc(G2,i,j,qMod,qCon)
          
          #Update the pairList and the alignedPairs dictionary with the likelihood of the pair (i,j) and the EC score
          # of the merged node graph G2_temp and G1
          pairList2.append((i,j,2,tempLikelihood))
          
          
    #step 2: choose the best pair(s) based on the likelihood function and the tolerance value
    #End algorithm if pairList is empty. Only occurs if both graphs are single node or empty.
    if pairList1==[] and pairList2==[]:
      signal=False
    #if pairList isn't empty, choose the best pair(s) based on the likelihood function and the tolerance value  
    else:
      pairListFinal1=[]
      pairListFinal2=[]
      maxLikelihood1=max(pairList1,key=lambda x:x[3])[3] #find the maximum likelihood over all pairs
      
      for i in pairList1:
        if i[3]>=maxLikelihood1-tolerance*maxLikelihood1:
          pairListFinal1.append((i[0],i[1],i[2])) #add the pair to the final list if the likelihood is greater than the maximum-tolerance
      
      maxLikelihood2=max(pairList2,key=lambda x:x[3])[3] #find the maximum likelihood over all pairs
      
      for i in pairList2:
        if i[3]>=maxLikelihood2-tolerance*maxLikelihood2:
          pairListFinal2.append((i[0],i[1],i[2])) #add the pair to the final list if the likelihood is greater than the maximum-tolerance
      print(pairListFinal1,pairListFinal2)
      for i in pairListFinal1:
        if i[1] not in G2.nodes and i[0] in G2.nodes:
            G1_temp=node_merger(G1,i[0],i[1],self_loops=False)
        else:
            G1_temp=node_merger(G1,i[1],i[0],self_loops=False)
        for j in pairListFinal2:
          
          if j[1] not in G1.nodes and j[0] in G1.nodes:
            G2_temp=node_merger(G2,j[0],j[1],self_loops=False)
          else:
            G2_temp=node_merger(G2,j[1],j[0],self_loops=False)
          G1_tempp=copy.deepcopy(G1_temp)
          G2_tempp=copy.deepcopy(G2_temp)
          G_intersect=nx.intersection(G1_tempp,G2_tempp)
          #G1_new=nx.induced_subgraph(G1_tempp,G_intersect.nodes())
          #G2_new=nx.induced_subgraph(G2_tempp,G_intersect.nodes())
          G1_new=G1_tempp
          G2_new=G2_tempp
          tempScore= normalised_ec_score(G1_new,G2_new)
          
          alignedPairs.append((G1_temp,G2_temp,tempScore,len(G_intersect.nodes)))
    #step 3: from the most likely pair(s) choose the best pair(s) based on the EC score    
      
      highScoreList=[] #highScoreList refines the list of node pairs even further, by selecting only those that give the highest EC score
      #pairListFinal should never be empty, but is included to be safe
      
        
      
        #the maximum of all scores of this iteration is recorded in maxScoreOfList to set the maxScore for the next iteration
        
        #for each pair in pairListFinal, find the EC score given by merging them. If this score improves upon the score in the previous
        # iteration, add the pair to the highScoreList.
      maxScore=0
      for i in alignedPairs:
        graphPairs=i
        tempScore=graphPairs[2]
        if tempScore>=maxScore:
          maxScore=tempScore #issue here that for the first merge, the maxScore is -1 and so all the scores are greater than it
      for i in alignedPairs:
        graphPairs=i
        tempScore=graphPairs[2]
        if tempScore>=maxScore-toleranceEC*maxScore:
             
          highScoreList.append(graphPairs)
    #If highScoreList is empty, then a local optimum is reached and the algorithm has finished.
      
      
    #For each of the best pairs, weight the pairs by their EC score and randomly pick a pair according to these weights
      weights=[]
      for i in range(len(highScoreList)):
        if highScoreList[i][2]!=0:
          weights.append(highScoreList[i][2])
    #If an EC score is 0 then it is weighted with a very small amount (0 messes with the random.choices function),
    #but a pair with EC score 0 should theoretically not appear anyway.
        else:
          weights.append(0.0000001)
      print(weights,len(G1_temp.nodes))
          
          #Pick the pair to merge and update the graph pair with the merged graph (either G1 or G2)
      graphPair=random.choices(highScoreList,weights=weights)
          #step 4: the graph pair is updated
          #output of random.choices is a single entry list, so the first entry is used
      graphPair=graphPair[0]
      print(graphPair)
          #update the maxScore for the next iteration
        #maxScore=graphPair[2]
      theGraphList.append(graphPair)
  theNewGraphList=[]
  for i in theGraphList:
    G1_temp=copy.deepcopy(i[0])
    G2_temp=copy.deepcopy(i[1])
    G_intersect=nx.intersection(G1_temp,G2_temp)
    #G1_temp=nx.induced_subgraph(G1_temp,G_intersect.nodes())
    #G2_temp=nx.induced_subgraph(G2_temp,G_intersect.nodes())
    G1_new=G1_temp.copy()
    G2_new=G2_temp.copy()
    outKeys1,deg_seq_out1 = zip(*G1_new.out_degree())
    outKeys2,deg_seq_out2 = zip(*G2_new.out_degree())
    inKeys1,deg_seq_in1 = zip(*G1_new.in_degree())
    inKeys2,deg_seq_in2 = zip(*G2_new.in_degree())
    #G2_ind = nx.induced_subgraph(G2_new,list(G1_new.nodes))
    #G1_ind= nx.induced_subgraph(G1_new,list(G2_new.nodes))
    #outKeys1,deg_seq_out_ind1 = zip(*G1_ind.out_degree())
    #outKeys2,deg_seq_out_ind2 = zip(*G2_ind.out_degree())
    #inKeys1,deg_seq_in_ind1 = zip(*G1_ind.in_degree())
    #inKeys2,deg_seq_in_ind2 = zip(*G2_ind.in_degree())
    
    preMeaner=random_number_of_conserved_edges_mean(len(G1_new.nodes),deg_seq_out1,deg_seq_out2,deg_seq_in1,deg_seq_in2,self_loops=False)
    preStddev=random_number_of_conserved_edges_std(len(G1_new.nodes),deg_seq_out1,deg_seq_out2,deg_seq_in1,deg_seq_in2,self_loops=False)
    #preMeaner_ind=random_number_of_conserved_edges_mean(len(G1_ind.nodes),deg_seq_out_ind1,deg_seq_out_ind2,deg_seq_in_ind1,deg_seq_in_ind2)
    #preStddev_ind=random_number_of_conserved_edges_std(len(G1_ind.nodes),deg_seq_out_ind1,deg_seq_out_ind2,deg_seq_in_ind1,deg_seq_in_ind2)
    #nullMean=0.5*(preMeaner/(len(G1_new.edges())+len(G2_ind.edges())-preMeaner)+preMeaner/(len(G2_new.edges())+len(G1_ind.edges())-preMeaner))
    #nullStd=0.5*(preStddev/(len(G1_new.edges())+len(G2_ind.edges())-preStddev)+preStddev/(len(G2_new.edges())+len(G1_ind.edges())-preStddev))
    nullMean=0.5*(preMeaner/len(G1_new.edges())+preMeaner/len(G2_new.edges()))
    nullStd=0.5*(preStddev/len(G1_new.edges())+preStddev/len(G2_new.edges()))
    #nullMean=preMeaner
    #nullStd=preStddev
    tempScore= i[2]
    if nullStd!=0:
      zScore=(tempScore-nullMean)/nullStd
    else:
      zScore=0
    print("paired")
    theNewGraphList.append((i[0],i[1],zScore))
    print(zScore)
    '''
    nullVec=null_distribution_ec_score(G1_temp,G2_temp,resolution=10000)
    nullStd=np.std(nullVec,ddof=1)
    nullMean=np.mean(nullVec)
    tempScore= normalised_ec_score(G1_temp,G2_temp)
    zScore=(tempScore-nullMean)/nullStd
    print("paired")
    theNewGraphList.append((G1_temp,G2_temp,zScore))
    print(zScore)
    '''
    
  graphPair=max(theNewGraphList,key=lambda x:x[2])
  print("the big winner score is",graphPair[2],"and its graphs with",len(graphPair[0].nodes),"nodes")
  return graphPair[0],graphPair[1]

def ancestor_finder_without_alignment_the_fifth(G1,G2,qMod,qCon,tolerance=0,toleranceEC=0):
  """Returns a pair of networkx graphs G1' and G2' based off of the input graphs G1 and G2
    with nodes merged as to maximise their EC similarity.

    This is a heuristic algorithm that finds pairs of nodes, either in G1 and G2, that are the most likely to have been the most
    recently duplicated pairs and determines which of these pairs, when merged, will increase the
    EC similarity of the two graphs.

    The best pair under these conditions is chosen and is merged, and the process is repeated until the EC score 
    can no longer be improved. That is, a local optimum is found.

    Parameters
    ----------
    G1 : networkx graph
        One of the two graphs
    G2 : networkx graph
        One of the two graphs
    qMod : float
        The probability of edge deletion in the assumed forward DMC model.
        In cases where this is not known it may need to be estimated.
    qCon : float
        The probability of creating an edge between duplicated pairs of vertices in the assumed forward DMC model.
        In cases where this is not known it may need to be estimated.
    tolerance : float, optional (default=0.05)
        A tolerance value used to determine the acceptable likelihood a pair of nodes to have
        been the most recently duplicated pair. A non-zero tolerance allows less likely pairs to be chosen, possibly 
        increasing the EC score more than the best pair.

    Notes
    -----
    This algorithm is not guaranteed to find the global optimum. 

    This algorithm comprises 4 main steps:
    1. The likelihood of all pairs of nodes in both G1 and G2 are calculated.
    2. A list of the 'best' pairs is generated. These pairs are those that have the highest likelihood, 
    depending on the likelihood tolerance value.
    3. From these pairs, the pair(s) that improve(s) the EC score the most is chosen and merged. Most of the time there
    is more than one such pair, in which case a random one is chosen.
    4. The graph in which the most recent merge has occured is updated and the process is repeated until the EC score 
    can no longer be improved.
    
    Issues 
    ------
    1. For the first go of the loop, the maxScore is -1 and so all the scores are greater than it
    2. It is assumed that likelihoods are comparable across the graphs G1 and G2 for different sized graphs. 
    This very well may not be the case and should be followed up.
    3. Many pairs seem to have the give the same EC score. This could be fine, but possibly something is going on here.

    """
  
  #graphPair is a repeatedly updated tuple containing the present state of graphs G1 and G2, the EC score of the current pair,
  # and an int that represents which of G1 and G2 the most recent merge occured in.
  graphPair=(G1,G2,-1,0) 
  #signal is a boolean that is set to False when the EC score of the current pair is no longer improving.
  signal =True 
  
  #maxScore records the current best EC score
  maxScore=-1
  n=len(G1.nodes())
  theGraphList=[]
  G_internal=nx.intersection(G1,G2)
  while signal and len(graphPair[0].nodes)>len(G_internal.nodes):
    #step 1: calculate the likelihood of all pairs of nodes in both G1 and G2

    #alignedPairs is a dictionary, keyed by (i,j,k) that records the EC score of graphs resulting from merging 
    # each pair of nodes i,j in graph G_k (k=1 or k=2).
    alignedPairs=[]
    #maxLikelihood is updated each loop and records the likelihood score of the most likely node pair for the current loop
    maxLikelihood = -1
    #pairList is a list of tuples (i,j,k,likelihood) where i and j are the nodes in graph G_k
    # and likelihood is the likelihood of the pair (i,j) being most recently duplicated in graph G_k
    pairList1=[]
    pairList2=[]
    #G1 and G2 are updated to be the most recent pair of graphs
    G1=graphPair[0]
    G2=graphPair[1]
    G_internal=nx.intersection(G1,G2)
    external_nodes_1=[i for i in list(G1.nodes) if i not in list(G_internal.nodes)]
    #Consider each node pair (i,j) in G1
    for i in external_nodes_1:
      for j in list(G1.nodes):
        if j!=i:
          #Construct the graph G1_temp resulting from merging (i,j) in G1
          #This pair of if statements ensures that a node merge keeps a label common to both graphs if possible
          
          #check the likelihood function for the pair (i,j)
          tempLikelihood = ancestral_likelihood_dmc(G1,i,j,qMod,qCon)
          
          #Update the pairList and the alignedPairs dictionary with the likelihood of the pair (i,j) and the EC score
          # of the merged node graph G1_temp and G2
          pairList1.append((i,j,1,tempLikelihood))
          
    external_nodes_2=[i for i in list(G2.nodes) if i not in list(G_internal.nodes)]     
    #Consider each node pair (i,j) in G2
    for i in external_nodes_2:  
      for j in list(G2.nodes):
        if j!=i:
          #Construct the graph G2_temp resulting from merging (i,j) in G2
          #This pair of if statements ensures that a node merge keeps a label common to both graphs if possible
          
          
          #check the likelihood function for the pair (i,j)
          tempLikelihood = ancestral_likelihood_dmc(G2,i,j,qMod,qCon)
          
          #Update the pairList and the alignedPairs dictionary with the likelihood of the pair (i,j) and the EC score
          # of the merged node graph G2_temp and G1
          pairList2.append((i,j,2,tempLikelihood))
          
          
    #step 2: choose the best pair(s) based on the likelihood function and the tolerance value
    #End algorithm if pairList is empty. Only occurs if both graphs are single node or empty.
    
    if pairList1==[] and pairList2==[]:
      signal=False
    #if pairList isn't empty, choose the best pair(s) based on the likelihood function and the tolerance value  
    else:
      pairListFinal1=[]
      pairListFinal2=[]
      maxLikelihood1=max(pairList1,key=lambda x:x[3])[3] #find the maximum likelihood over all pairs
      
      for i in pairList1:
        if i[3]>=maxLikelihood1-tolerance*maxLikelihood1:
          pairListFinal1.append((i[0],i[1],i[2])) #add the pair to the final list if the likelihood is greater than the maximum-tolerance
      
      maxLikelihood2=max(pairList2,key=lambda x:x[3])[3] #find the maximum likelihood over all pairs
      
      for i in pairList2:
        if i[3]>=maxLikelihood2-tolerance*maxLikelihood2:
          pairListFinal2.append((i[0],i[1],i[2])) #add the pair to the final list if the likelihood is greater than the maximum-tolerance
      print(pairListFinal1,pairListFinal2)
      for i in pairListFinal1:
        if i[1] not in G2.nodes and i[0] in G2.nodes:
            G1_temp=node_merger(G1,i[0],i[1])
        else:
            G1_temp=node_merger(G1,i[1],i[0])
        for j in pairListFinal2:
          
          if j[1] not in G1.nodes and j[0] in G1.nodes:
            G2_temp=node_merger(G2,j[0],j[1])
          else:
            G2_temp=node_merger(G2,j[1],j[0])
          G1_tempp=copy.deepcopy(G1_temp)
          G2_tempp=copy.deepcopy(G2_temp)
          G_intersect=nx.intersection(G1_tempp,G2_tempp)
          #G1_new=nx.induced_subgraph(G1_tempp,G_intersect.nodes())
          #G2_new=nx.induced_subgraph(G2_tempp,G_intersect.nodes())
          G1_new=G1_tempp
          G2_new=G2_tempp
          #tempScore= normalised_ec_score(G1_new,G2_new)
          tempScore= normalised_ec_score(G1_new,G2_new)
          
          alignedPairs.append((G1_temp,G2_temp,tempScore,len(G1_new.nodes)))
    #step 3: from the most likely pair(s) choose the best pair(s) based on the EC score    
      print(pairListFinal1,pairListFinal2)
      highScoreList=[] #highScoreList refines the list of node pairs even further, by selecting only those that give the highest EC score
      #pairListFinal should never be empty, but is included to be safe
      
        
      
        #the maximum of all scores of this iteration is recorded in maxScoreOfList to set the maxScore for the next iteration
        
        #for each pair in pairListFinal, find the EC score given by merging them. If this score improves upon the score in the previous
        # iteration, add the pair to the highScoreList.
      maxScore=0
      for i in alignedPairs:
        graphPairs=i
        tempScore=graphPairs[2]
        if tempScore>=maxScore:
          maxScore=tempScore #issue here that for the first merge, the maxScore is -1 and so all the scores are greater than it
      for i in alignedPairs:
        graphPairs=i
        tempScore=graphPairs[2]
        if tempScore>=maxScore-toleranceEC*maxScore:
             
          highScoreList.append(graphPairs)
    #If highScoreList is empty, then a local optimum is reached and the algorithm has finished.
      print(highScoreList)
      
    #For each of the best pairs, weight the pairs by their EC score and randomly pick a pair according to these weights
      weights=[]
      for i in range(len(highScoreList)):
        if highScoreList[i][2]!=0:
          weights.append(highScoreList[i][2])
    #If an EC score is 0 then it is weighted with a very small amount (0 messes with the random.choices function),
    #but a pair with EC score 0 should theoretically not appear anyway.
        else:
          weights.append(0.0000001)
      print(weights,len(G1_temp.nodes))
         
          #Pick the pair to merge and update the graph pair with the merged graph (either G1 or G2)
      graphPair=random.choices(highScoreList,weights=weights)
          #step 4: the graph pair is updated
          #output of random.choices is a single entry list, so the first entry is used
      graphPair=graphPair[0]
      print(graphPair)
          #update the maxScore for the next iteration
        #maxScore=graphPair[2]
      theGraphList.append(graphPair)
  theNewGraphList=[]
  for i in theGraphList:
    G1_temp=copy.deepcopy(i[0])
    G2_temp=copy.deepcopy(i[1])
    G_intersect=nx.intersection(G1_temp,G2_temp)
    #G1_temp=nx.induced_subgraph(G1_temp,G_intersect.nodes())
    #G2_temp=nx.induced_subgraph(G2_temp,G_intersect.nodes())
    G1_new=G1_temp.copy()
    G2_new=G2_temp.copy()
    outKeys1,deg_seq_out1 = zip(*G1_new.out_degree())
    outKeys2,deg_seq_out2 = zip(*G2_new.out_degree())
    inKeys1,deg_seq_in1 = zip(*G1_new.in_degree())
    inKeys2,deg_seq_in2 = zip(*G2_new.in_degree())
    #G2_ind = nx.induced_subgraph(G2_new,list(G1_new.nodes))
    #G1_ind= nx.induced_subgraph(G1_new,list(G2_new.nodes))
    #outKeys1,deg_seq_out_ind1 = zip(*G1_ind.out_degree())
    #outKeys2,deg_seq_out_ind2 = zip(*G2_ind.out_degree())
    #inKeys1,deg_seq_in_ind1 = zip(*G1_ind.in_degree())
    #inKeys2,deg_seq_in_ind2 = zip(*G2_ind.in_degree())
    
    preMeaner=random_number_of_conserved_edges_mean(len(G1_new.nodes),deg_seq_out1,deg_seq_out2,deg_seq_in1,deg_seq_in2)
    preStddev=random_number_of_conserved_edges_std(len(G1_new.nodes),deg_seq_out1,deg_seq_out2,deg_seq_in1,deg_seq_in2)
    #preMeaner_ind=random_number_of_conserved_edges_mean(len(G1_ind.nodes),deg_seq_out_ind1,deg_seq_out_ind2,deg_seq_in_ind1,deg_seq_in_ind2)
    #preStddev_ind=random_number_of_conserved_edges_std(len(G1_ind.nodes),deg_seq_out_ind1,deg_seq_out_ind2,deg_seq_in_ind1,deg_seq_in_ind2)
    #nullMean=0.5*(preMeaner/(len(G1_new.edges())+len(G2_ind.edges())-preMeaner)+preMeaner/(len(G2_new.edges())+len(G1_ind.edges())-preMeaner))
    #nullStd=0.5*(preStddev/(len(G1_new.edges())+len(G2_ind.edges())-preStddev)+preStddev/(len(G2_new.edges())+len(G1_ind.edges())-preStddev))
    nullMean=0.5*(preMeaner/len(G1_new.edges())+preMeaner/len(G2_new.edges()))
    nullStd=0.5*(preStddev/len(G1_new.edges())+preStddev/len(G2_new.edges()))
    #nullMean=preMeaner
    #nullStd=preStddev
    tempScore= i[2]
    if nullStd!=0:
      zScore=(tempScore-nullMean)/nullStd
    else:
      zScore=0
    print("paired")
    theNewGraphList.append((i[0],i[1],zScore))
    print(zScore)
    '''
    nullVec=null_distribution_ec_score(G1_temp,G2_temp,resolution=10000)
    nullStd=np.std(nullVec,ddof=1)
    nullMean=np.mean(nullVec)
    tempScore= normalised_ec_score(G1_temp,G2_temp)
    zScore=(tempScore-nullMean)/nullStd
    print("paired")
    theNewGraphList.append((G1_temp,G2_temp,zScore))
    print(zScore)
    '''
    
  graphPair=max(theNewGraphList,key=lambda x:x[2])
  print("the big winner score is",graphPair[2],"and its graphs with",len(graphPair[0].nodes),"nodes")
  return graphPair[0],graphPair[1]

def ancestor_finder_without_alignment_branching(G1,G2,qMod,qCon,tolerance=0,toleranceEC=0):
  """Returns a pair of networkx graphs G1' and G2' based off of the input graphs G1 and G2
    with nodes merged as to maximise their EC similarity.

    This is a heuristic algorithm that finds pairs of nodes, either in G1 and G2, that are the most likely to have been the most
    recently duplicated pairs and determines which of these pairs, when merged, will increase the
    EC similarity of the two graphs.

    The best pair under these conditions is chosen and is merged, and the process is repeated until the EC score 
    can no longer be improved. That is, a local optimum is found.

    Parameters
    ----------
    G1 : networkx graph
        One of the two graphs
    G2 : networkx graph
        One of the two graphs
    qMod : float
        The probability of edge deletion in the assumed forward DMC model.
        In cases where this is not known it may need to be estimated.
    qCon : float
        The probability of creating an edge between duplicated pairs of vertices in the assumed forward DMC model.
        In cases where this is not known it may need to be estimated.
    tolerance : float, optional (default=0.05)
        A tolerance value used to determine the acceptable likelihood a pair of nodes to have
        been the most recently duplicated pair. A non-zero tolerance allows less likely pairs to be chosen, possibly 
        increasing the EC score more than the best pair.

    Notes
    -----
    This algorithm is not guaranteed to find the global optimum. 

    This algorithm comprises 4 main steps:
    1. The likelihood of all pairs of nodes in both G1 and G2 are calculated.
    2. A list of the 'best' pairs is generated. These pairs are those that have the highest likelihood, 
    depending on the likelihood tolerance value.
    3. From these pairs, the pair(s) that improve(s) the EC score the most is chosen and merged. Most of the time there
    is more than one such pair, in which case a random one is chosen.
    4. The graph in which the most recent merge has occured is updated and the process is repeated until the EC score 
    can no longer be improved.
    
    Issues 
    ------
    1. For the first go of the loop, the maxScore is -1 and so all the scores are greater than it
    2. It is assumed that likelihoods are comparable across the graphs G1 and G2 for different sized graphs. 
    This very well may not be the case and should be followed up.
    3. Many pairs seem to have the give the same EC score. This could be fine, but possibly something is going on here.

    """
  
  #graphPair is a repeatedly updated tuple containing the present state of graphs G1 and G2, the EC score of the current pair,
  # and an int that represents which of G1 and G2 the most recent merge occured in.
  graphPair=(G1,G2,-1,0) 
  #signal is a boolean that is set to False when the EC score of the current pair is no longer improving.
  signal =True 
  
  #maxScore records the current best EC score
  maxScore=-1
  n=len(G1.nodes())
  theGraphList=[]
  graphPair=branch_helper(graphPair,theGraphList,tolerance,toleranceEC,qMod,qCon,lastScore=0,maxPair=graphPair)
  
  print("the big winner score is",graphPair[2],"and its graphs with",len(graphPair[0].nodes),"nodes")
  return graphPair[0],graphPair[1]
def branch_helper(graphPair,theGraphList,tolerance,toleranceEC,qMod,qCon,lastScore,maxPair):
    if len(graphPair[0].nodes)<=40:
          
          return graphPair
    #step 1: calculate the likelihood of all pairs of nodes in both G1 and G2

    #alignedPairs is a dictionary, keyed by (i,j,k) that records the EC score of graphs resulting from merging 
    # each pair of nodes i,j in graph G_k (k=1 or k=2).
    alignedPairs=[]
    #maxLikelihood is updated each loop and records the likelihood score of the most likely node pair for the current loop
    maxLikelihood = -1
    #pairList is a list of tuples (i,j,k,likelihood) where i and j are the nodes in graph G_k
    # and likelihood is the likelihood of the pair (i,j) being most recently duplicated in graph G_k
    pairList1=[]
    pairList2=[]
    #G1 and G2 are updated to be the most recent pair of graphs
    G1=graphPair[0]
    G2=graphPair[1]

    #Consider each node pair (i,j) in G1
    for num1,i in enumerate(list(G1.nodes)):
      for num2,j in enumerate(list(G1.nodes)):
        if num2>num1:
          #Construct the graph G1_temp resulting from merging (i,j) in G1
          #This pair of if statements ensures that a node merge keeps a label common to both graphs if possible
          
          #check the likelihood function for the pair (i,j)
          tempLikelihood = ancestral_likelihood_dmc(G1,i,j,qMod,qCon)
          
          #Update the pairList and the alignedPairs dictionary with the likelihood of the pair (i,j) and the EC score
          # of the merged node graph G1_temp and G2
          pairList1.append((i,j,1,tempLikelihood))
          
         
    #Consider each node pair (i,j) in G2
    for num1,i in enumerate(list(G2.nodes)):  
      for num2,j in enumerate(list(G2.nodes)):
        if num2>num1:
          #Construct the graph G2_temp resulting from merging (i,j) in G2
          #This pair of if statements ensures that a node merge keeps a label common to both graphs if possible
          
          
          #check the likelihood function for the pair (i,j)
          tempLikelihood = ancestral_likelihood_dmc(G2,i,j,qMod,qCon)
          
          #Update the pairList and the alignedPairs dictionary with the likelihood of the pair (i,j) and the EC score
          # of the merged node graph G2_temp and G1
          pairList2.append((i,j,2,tempLikelihood))
          
          
    #step 2: choose the best pair(s) based on the likelihood function and the tolerance value
    #End algorithm if pairList is empty. Only occurs if both graphs are single node or empty.
    if pairList1==[] and pairList2==[]:
      signal=False
    #if pairList isn't empty, choose the best pair(s) based on the likelihood function and the tolerance value  
    else:
      pairListFinal1=[]
      pairListFinal2=[]
      maxLikelihood1=max(pairList1,key=lambda x:x[3])[3] #find the maximum likelihood over all pairs
      
      for i in pairList1:
        if i[3]>=maxLikelihood1-tolerance*maxLikelihood1:
          pairListFinal1.append((i[0],i[1],i[2])) #add the pair to the final list if the likelihood is greater than the maximum-tolerance
      
      maxLikelihood2=max(pairList2,key=lambda x:x[3])[3] #find the maximum likelihood over all pairs
      
      for i in pairList2:
        if i[3]>=maxLikelihood2-tolerance*maxLikelihood2:
          pairListFinal2.append((i[0],i[1],i[2])) #add the pair to the final list if the likelihood is greater than the maximum-tolerance

      for i in pairListFinal1:
        if i[1] not in G2.nodes and i[0] in G2.nodes:
            G1_temp=node_merger(G1,i[0],i[1])
        else:
            G1_temp=node_merger(G1,i[1],i[0])
        for j in pairListFinal2:
          
          if j[1] not in G1.nodes and j[0] in G1.nodes:
            G2_temp=node_merger(G2,j[0],j[1])
          else:
            G2_temp=node_merger(G2,j[1],j[0])
          G1_tempp=copy.deepcopy(G1_temp)
          G2_tempp=copy.deepcopy(G2_temp)
          G_intersect=nx.intersection(G1_tempp,G2_tempp)
          #G1_new=nx.induced_subgraph(G1_tempp,G_intersect.nodes())
          #G2_new=nx.induced_subgraph(G2_tempp,G_intersect.nodes())
          G1_new=G1_tempp
          G2_new=G2_tempp
          tempScore= normalised_ec_score(G1_new,G2_new)
          
          outKeys1,deg_seq_out1 = zip(*G1_new.out_degree())
          outKeys2,deg_seq_out2 = zip(*G2_new.out_degree())
          inKeys1,deg_seq_in1 = zip(*G1_new.in_degree())
          inKeys2,deg_seq_in2 = zip(*G2_new.in_degree())
          
          
          preMeaner=random_number_of_conserved_edges_mean(len(G1_new.nodes),deg_seq_out1,deg_seq_out2,deg_seq_in1,deg_seq_in2)
          preStddev=random_number_of_conserved_edges_std(len(G1_new.nodes),deg_seq_out1,deg_seq_out2,deg_seq_in1,deg_seq_in2)
          nullMean=0.5*(preMeaner/len(G1_new.edges())+preMeaner/len(G2_new.edges()))
          nullStd=0.5*(preStddev/len(G1_new.edges())+preStddev/len(G2_new.edges()))
        
          if nullStd!=0:
            zScore=(tempScore-nullMean)/nullStd
          else:
            zScore=0
          
          
          
          alignedPairs.append((G1_temp,G2_temp,zScore,len(G_intersect.nodes)))
    #step 3: from the most likely pair(s) choose the best pair(s) based on the EC score    
      print(alignedPairs)
      highScoreList=[] #highScoreList refines the list of node pairs even further, by selecting only those that give the highest EC score
      #pairListFinal should never be empty, but is included to be safe
      
        
      
        #the maximum of all scores of this iteration is recorded in maxScoreOfList to set the maxScore for the next iteration
        
        #for each pair in pairListFinal, find the EC score given by merging them. If this score improves upon the score in the previous
        # iteration, add the pair to the highScoreList.
      maxxScore=0
      for i in alignedPairs:
        graphPairs=i
        tempScore=graphPairs[2]
        if tempScore>=maxxScore:
          maxxScore=tempScore #issue here that for the first merge, the maxScore is -1 and so all the scores are greater than it

      for i in alignedPairs:
        graphPairs=i
        tempScore=graphPairs[2]
        if tempScore>=maxxScore-toleranceEC*maxxScore:
             
          highScoreList.append(graphPairs)
      if len(highScoreList)==0:
        
        
        return graphPair
     
      print(highScoreList,len(G1_temp.nodes))
      tempMax=(1,1,0)
      for i in highScoreList:
        
        print(i[2]) 
        if i[2] >tempMax[2]:
          tempMax=i
        tempScore=branch_helper(i,theGraphList,tolerance,toleranceEC,qMod,qCon,i[2],maxPair)
        if tempScore[2]>tempMax[2]:
          tempMax=tempScore
      return tempMax
        
def ancestor_finder_without_alignment_the_seventh(G1,G2,qMod,qCon,tolerance=0,toleranceEC=0):
  """Returns a pair of networkx graphs G1' and G2' based off of the input graphs G1 and G2
    with nodes merged as to maximise their EC similarity.

    This is a heuristic algorithm that finds pairs of nodes, either in G1 and G2, that are the most likely to have been the most
    recently duplicated pairs and determines which of these pairs, when merged, will increase the
    EC similarity of the two graphs.

    The best pair under these conditions is chosen and is merged, and the process is repeated until the EC score 
    can no longer be improved. That is, a local optimum is found.

    Parameters
    ----------
    G1 : networkx graph
        One of the two graphs
    G2 : networkx graph
        One of the two graphs
    qMod : float
        The probability of edge deletion in the assumed forward DMC model.
        In cases where this is not known it may need to be estimated.
    qCon : float
        The probability of creating an edge between duplicated pairs of vertices in the assumed forward DMC model.
        In cases where this is not known it may need to be estimated.
    tolerance : float, optional (default=0.05)
        A tolerance value used to determine the acceptable likelihood a pair of nodes to have
        been the most recently duplicated pair. A non-zero tolerance allows less likely pairs to be chosen, possibly 
        increasing the EC score more than the best pair.

    Notes
    -----
    This algorithm is not guaranteed to find the global optimum. 

    This algorithm comprises 4 main steps:
    1. The likelihood of all pairs of nodes in both G1 and G2 are calculated.
    2. A list of the 'best' pairs is generated. These pairs are those that have the highest likelihood, 
    depending on the likelihood tolerance value.
    3. From these pairs, the pair(s) that improve(s) the EC score the most is chosen and merged. Most of the time there
    is more than one such pair, in which case a random one is chosen.
    4. The graph in which the most recent merge has occured is updated and the process is repeated until the EC score 
    can no longer be improved.
    
    Issues 
    ------
    1. For the first go of the loop, the maxScore is -1 and so all the scores are greater than it
    2. It is assumed that likelihoods are comparable across the graphs G1 and G2 for different sized graphs. 
    This very well may not be the case and should be followed up.
    3. Many pairs seem to have the give the same EC score. This could be fine, but possibly something is going on here.

    """
  
  #graphPair is a repeatedly updated tuple containing the present state of graphs G1 and G2, the EC score of the current pair,
  # and an int that represents which of G1 and G2 the most recent merge occured in.
  graphPair=(G1,G2,-1,0) 
  #signal is a boolean that is set to False when the EC score of the current pair is no longer improving.
  signal =True 
  
  #maxScore records the current best EC score
  maxScore=-1
  n=len(G1.nodes())
  theGraphList=[]
  prevScore=0
  countUp=False
  while signal and prevScore<0.99:
    #step 1: calculate the likelihood of all pairs of nodes in both G1 and G2

    #alignedPairs is a dictionary, keyed by (i,j,k) that records the EC score of graphs resulting from merging 
    # each pair of nodes i,j in graph G_k (k=1 or k=2).
    alignedPairs=[]
    #maxLikelihood is updated each loop and records the likelihood score of the most likely node pair for the current loop
    maxLikelihood = -1
    #pairList is a list of tuples (i,j,k,likelihood) where i and j are the nodes in graph G_k
    # and likelihood is the likelihood of the pair (i,j) being most recently duplicated in graph G_k
    pairList1=[]
    pairList2=[]
    #G1 and G2 are updated to be the most recent pair of graphs
    G1=graphPair[0]
    G2=graphPair[1]

    #Consider each node pair (i,j) in G1
    for num1,i in enumerate(list(G1.nodes)):
      for num2,j in enumerate(list(G1.nodes)):
        if num2>num1:
          #Construct the graph G1_temp resulting from merging (i,j) in G1
          #This pair of if statements ensures that a node merge keeps a label common to both graphs if possible
          
          #check the likelihood function for the pair (i,j)
          tempLikelihood = ancestral_likelihood_dmc(G1,i,j,qMod,qCon)
          
          #Update the pairList and the alignedPairs dictionary with the likelihood of the pair (i,j) and the EC score
          # of the merged node graph G1_temp and G2
          pairList1.append((i,j,1,tempLikelihood))
          
         
    #Consider each node pair (i,j) in G2
    for num1,i in enumerate(list(G2.nodes)):  
      for num2,j in enumerate(list(G2.nodes)):
        if num2>num1:
          #Construct the graph G2_temp resulting from merging (i,j) in G2
          #This pair of if statements ensures that a node merge keeps a label common to both graphs if possible
          
          
          #check the likelihood function for the pair (i,j)
          tempLikelihood = ancestral_likelihood_dmc(G2,i,j,qMod,qCon)
          
          #Update the pairList and the alignedPairs dictionary with the likelihood of the pair (i,j) and the EC score
          # of the merged node graph G2_temp and G1
          pairList2.append((i,j,2,tempLikelihood))
          
    maxLikelihood1=max(pairList1,key=lambda x:x[3])[3]
    maxLikelihood2=max(pairList2,key=lambda x:x[3])[3]
    maxCreen=np.sqrt(maxLikelihood1*maxLikelihood2)
    pairPairList=[]      
    #step 2: choose the best pair(s) based on the likelihood function and the tolerance value
    #End algorithm if pairList is empty. Only occurs if both graphs are single node or empty.
    if pairList1==[] and pairList2==[]:
      signal=False
    #if pairList isn't empty, choose the best pair(s) based on the likelihood function and the tolerance value  
    else:
      print("list begins")
      for i in pairList1:
        for j in pairList2:
            if np.sqrt(i[3]*j[3])>= maxCreen-tolerance*maxCreen:
              pairPairList.append(((i[0],i[1]),(j[0],j[1]),np.sqrt(i[3]*j[3])))
      print("sorting begins")
      pairPairList=sorted(pairPairList, key=lambda x: x[2],reverse=True) #find the maximum likelihood over all pairs
      
      print("merging begins")
      count=0
      signal=False
      for i in pairPairList:
        count=count+1
        #if count>20:
        #  countUp=True
        #  break
        
        if i[0][1] not in G2.nodes and i[0][0] in G2.nodes:
            G1_temp=node_merger(G1,i[0][0],i[0][1],self_loops=False)
        else:
            G1_temp=node_merger(G1,i[0][1],i[0][0],self_loops=False)
      
          
        if i[1][1] not in G1.nodes and i[1][0] in G1.nodes:
          G2_temp=node_merger(G2,i[1][0],i[1][1],self_loops=False)
        else:
          G2_temp=node_merger(G2,i[1][1],i[1][0],self_loops=False)
        G1_tempp=copy.deepcopy(G1_temp)
        G2_tempp=copy.deepcopy(G2_temp)
        G_intersect=nx.intersection(G1_tempp,G2_tempp)
        #G1_new=nx.induced_subgraph(G1_tempp,G_intersect.nodes())
        #G2_new=nx.induced_subgraph(G2_tempp,G_intersect.nodes())
        G1_new=G1_tempp
        G2_new=G2_tempp
        tempScore= normalised_ec_score(G1_new,G2_new)
        print(i,tempScore)
        if tempScore>prevScore:
          print("chosen",i)
          graphPair=(G1_temp,G2_temp,tempScore)
          prevScore=tempScore
          signal=True
      #if countUp:
      #  break
      print("out of the merge")
      
      theGraphList.append(graphPair)
      print(graphPair,len(graphPair[0].nodes))  
  theNewGraphList=[]
  for i in theGraphList:
    G1_temp=copy.deepcopy(i[0])
    G2_temp=copy.deepcopy(i[1])
    G_intersect=nx.intersection(G1_temp,G2_temp)
    #G1_temp=nx.induced_subgraph(G1_temp,G_intersect.nodes())
    #G2_temp=nx.induced_subgraph(G2_temp,G_intersect.nodes())
    G1_new=G1_temp.copy()
    G2_new=G2_temp.copy()
    outKeys1,deg_seq_out1 = zip(*G1_new.out_degree())
    outKeys2,deg_seq_out2 = zip(*G2_new.out_degree())
    inKeys1,deg_seq_in1 = zip(*G1_new.in_degree())
    inKeys2,deg_seq_in2 = zip(*G2_new.in_degree())
    #G2_ind = nx.induced_subgraph(G2_new,list(G1_new.nodes))
    #G1_ind= nx.induced_subgraph(G1_new,list(G2_new.nodes))
    #outKeys1,deg_seq_out_ind1 = zip(*G1_ind.out_degree())
    #outKeys2,deg_seq_out_ind2 = zip(*G2_ind.out_degree())
    #inKeys1,deg_seq_in_ind1 = zip(*G1_ind.in_degree())
    #inKeys2,deg_seq_in_ind2 = zip(*G2_ind.in_degree())
    
    preMeaner=random_number_of_conserved_edges_mean(len(G1_new.nodes),deg_seq_out1,deg_seq_out2,deg_seq_in1,deg_seq_in2,self_loops=False)
    preStddev=random_number_of_conserved_edges_std(len(G1_new.nodes),deg_seq_out1,deg_seq_out2,deg_seq_in1,deg_seq_in2,self_loops=False)
    #preMeaner_ind=random_number_of_conserved_edges_mean(len(G1_ind.nodes),deg_seq_out_ind1,deg_seq_out_ind2,deg_seq_in_ind1,deg_seq_in_ind2)
    #preStddev_ind=random_number_of_conserved_edges_std(len(G1_ind.nodes),deg_seq_out_ind1,deg_seq_out_ind2,deg_seq_in_ind1,deg_seq_in_ind2)
    #nullMean=0.5*(preMeaner/(len(G1_new.edges())+len(G2_ind.edges())-preMeaner)+preMeaner/(len(G2_new.edges())+len(G1_ind.edges())-preMeaner))
    #nullStd=0.5*(preStddev/(len(G1_new.edges())+len(G2_ind.edges())-preStddev)+preStddev/(len(G2_new.edges())+len(G1_ind.edges())-preStddev))
    nullMean=0.5*(preMeaner/len(G1_new.edges())+preMeaner/len(G2_new.edges()))
    nullStd=0.5*(preStddev/len(G1_new.edges())+preStddev/len(G2_new.edges()))
    #nullMean=preMeaner
    #nullStd=preStddev
    tempScore= i[2]
    if nullStd!=0:
      zScore=(tempScore-nullMean)/nullStd
    else:
      zScore=0
    print("paired")
    theNewGraphList.append((i[0],i[1],zScore))
    print(zScore)
    '''
    nullVec=null_distribution_ec_score(G1_temp,G2_temp,resolution=10000)
    nullStd=np.std(nullVec,ddof=1)
    nullMean=np.mean(nullVec)
    tempScore= normalised_ec_score(G1_temp,G2_temp)
    zScore=(tempScore-nullMean)/nullStd
    print("paired")
    theNewGraphList.append((G1_temp,G2_temp,zScore))
    print(zScore)
    '''
    
  graphPair=max(theNewGraphList,key=lambda x:x[2])
  print("the big winner score is",graphPair[2],"and its graphs with",len(graphPair[0].nodes),"nodes")
  return graphPair[0],graphPair[1]
           
def ancestor_finder_without_alignment_gene_family(G1,G2,qMod,qCon,tolerance=0,toleranceEC=0):
  """Returns a pair of networkx graphs G1' and G2' based off of the input graphs G1 and G2
    with nodes merged as to maximise their EC similarity.

    This is a heuristic algorithm that finds pairs of nodes, either in G1 and G2, that are the most likely to have been the most
    recently duplicated pairs and determines which of these pairs, when merged, will increase the
    EC similarity of the two graphs.

    The best pair under these conditions is chosen and is merged, and the process is repeated until the EC score 
    can no longer be improved. That is, a local optimum is found.

    Parameters
    ----------
    G1 : networkx graph
        One of the two graphs
    G2 : networkx graph
        One of the two graphs
    qMod : float
        The probability of edge deletion in the assumed forward DMC model.
        In cases where this is not known it may need to be estimated.
    qCon : float
        The probability of creating an edge between duplicated pairs of vertices in the assumed forward DMC model.
        In cases where this is not known it may need to be estimated.
    tolerance : float, optional (default=0.05)
        A tolerance value used to determine the acceptable likelihood a pair of nodes to have
        been the most recently duplicated pair. A non-zero tolerance allows less likely pairs to be chosen, possibly 
        increasing the EC score more than the best pair.

    Notes
    -----
    This algorithm is not guaranteed to find the global optimum. 

    This algorithm comprises 4 main steps:
    1. The likelihood of all pairs of nodes in both G1 and G2 are calculated.
    2. A list of the 'best' pairs is generated. These pairs are those that have the highest likelihood, 
    depending on the likelihood tolerance value.
    3. From these pairs, the pair(s) that improve(s) the EC score the most is chosen and merged. Most of the time there
    is more than one such pair, in which case a random one is chosen.
    4. The graph in which the most recent merge has occured is updated and the process is repeated until the EC score 
    can no longer be improved.
    
    Issues 
    ------
    1. For the first go of the loop, the maxScore is -1 and so all the scores are greater than it
    2. It is assumed that likelihoods are comparable across the graphs G1 and G2 for different sized graphs. 
    This very well may not be the case and should be followed up.
    3. Many pairs seem to have the give the same EC score. This could be fine, but possibly something is going on here.

    """
  
  #graphPair is a repeatedly updated tuple containing the present state of graphs G1 and G2, the EC score of the current pair,
  # and an int that represents which of G1 and G2 the most recent merge occured in.
  graphPair=(G1,G2,-1,0) 
  #signal is a boolean that is set to False when the EC score of the current pair is no longer improving.
  signal =True 
  
  
  #maxScore records the current best EC score
  maxScore=-1
  n=len(G1.nodes())
  theGraphList=[]
  
  prevScore=conserved_edges(G1,G2)
  countUp=False
  while signal and len(graphPair[0].nodes)>4:
    #step 1: calculate the likelihood of all pairs of nodes in both G1 and G2

    #alignedPairs is a dictionary, keyed by (i,j,k) that records the EC score of graphs resulting from merging 
    # each pair of nodes i,j in graph G_k (k=1 or k=2).
    alignedPairs=[]
    #maxLikelihood is updated each loop and records the likelihood score of the most likely node pair for the current loop
    maxLikelihood = -1
    #pairList is a list of tuples (i,j,k,likelihood) where i and j are the nodes in graph G_k
    # and likelihood is the likelihood of the pair (i,j) being most recently duplicated in graph G_k
    pairList1=[]
    pairList2=[]
    #G1 and G2 are updated to be the most recent pair of graphs
    G1=graphPair[0]
    G2=graphPair[1]
    P1=gene_family_partitioner(G1)
    P2=gene_family_partitioner(G2)
    
    
    #Consider each node pair (i,j) in G1

    for fam in P1:
      for num1,i in enumerate(P1[fam]):
        for num2,j in enumerate(P1[fam]):
          
          if num2>num1:
            #Construct the graph G1_temp resulting from merging (i,j) in G1
            #This pair of if statements ensures that a node merge keeps a label common to both graphs if possible
            
            #check the likelihood function for the pair (i,j)
            tempLikelihood = ancestral_likelihood_dmc(G1,i,j,qMod,qCon)
            
            #Update the pairList and the alignedPairs dictionary with the likelihood of the pair (i,j) and the EC score
            # of the merged node graph G1_temp and G2
            pairList1.append((i,j,1,tempLikelihood))
            
    
    #Consider each node pair (i,j) in G2
    for fam in P2:
      for num1,i in enumerate(P2[fam]):  
        for num2,j in enumerate(P2[fam]):
          if num2>num1:
            #Construct the graph G2_temp resulting from merging (i,j) in G2
            #This pair of if statements ensures that a node merge keeps a label common to both graphs if possible
            
            
            #check the likelihood function for the pair (i,j)
            tempLikelihood = ancestral_likelihood_dmc(G2,i,j,qMod,qCon)
            
            #Update the pairList and the alignedPairs dictionary with the likelihood of the pair (i,j) and the EC score
            # of the merged node graph G2_temp and G1
            pairList2.append((i,j,2,tempLikelihood))
            
            
    #step 2: choose the best pair(s) based on the likelihood function and the tolerance value
    #End algorithm if pairList is empty. Only occurs if both graphs are single node or empty.
    if len(pairList1)==0 or len(pairList2)==0:
      break
    maxLikelihood1=max(pairList1,key=lambda x:x[3])[3]
    maxLikelihood2=max(pairList2,key=lambda x:x[3])[3]
    maxCreen=np.sqrt(maxLikelihood1*maxLikelihood2)
    pairPairList=[]      
    #step 2: choose the best pair(s) based on the likelihood function and the tolerance value
    #End algorithm if pairList is empty. Only occurs if both graphs are single node or empty.
    if pairList1==[] and pairList2==[]:
      signal=False
    #if pairList isn't empty, choose the best pair(s) based on the likelihood function and the tolerance value  
    else:
      print("list begins")
      for i in pairList1:
        for j in pairList2:
            if np.sqrt(i[3]*j[3])>= maxCreen-tolerance*maxCreen:
              pairPairList.append(((i[0],i[1]),(j[0],j[1]),np.sqrt(i[3]*j[3])))
      print("sorting begins")
      pairPairList=sorted(pairPairList, key=lambda x: x[2],reverse=True) #find the maximum likelihood over all pairs
      print("merging begins")
      count=0
      signal=False
      count=0
      for i in pairPairList:
        count=count+1
        if count>=len(pairPairList):
          countUp=True
          break
        
        if i[0][1] not in G2.nodes and i[0][0] in G2.nodes:
            G1_temp=node_merger(G1,i[0][0],i[0][1],self_loops=False)
        else:
            G1_temp=node_merger(G1,i[0][1],i[0][0],self_loops=False)
      
          
        if i[1][1] not in G1.nodes and i[1][0] in G1.nodes:
          G2_temp=node_merger(G2,i[1][0],i[1][1],self_loops=False)
        else:
          G2_temp=node_merger(G2,i[1][1],i[1][0],self_loops=False)
        G1_tempp=copy.deepcopy(G1_temp)
        G2_tempp=copy.deepcopy(G2_temp)
        G_intersect=nx.intersection(G1_tempp,G2_tempp)
        #G1_new=nx.induced_subgraph(G1_tempp,G_intersect.nodes())
        #G2_new=nx.induced_subgraph(G2_tempp,G_intersect.nodes())
        G1_new=G1_tempp
        G2_new=G2_tempp
        tempScore= conserved_edges(G1_new,G2_new)
        tempScore1= conserved_edges(G1,G2_new)
        tempScore2=conserved_edges(G1_new,G2)
        print(i,tempScore, tempScore1,tempScore2)
        if tempScore>=prevScore and tempScore1>=prevScore and tempScore2>=prevScore:
          print("chosen",i)
          graphPair=(G1_temp,G2_temp,tempScore)
          prevScore=tempScore
          signal=True
          break
      
        #if tempScore>=prevScore:
        #  print("chosen",i)
        #  graphPair=(G1_temp,G2_temp,tempScore)
        #  prevScore=tempScore
        #  signal=True
        #  break
        if countUp:
          print("chosen",i)
          #graphPair=(G1_temp,G2_temp,tempScore)
          break
      print("out of the merge")
      theGraphList.append(graphPair)
      print(graphPair,len(graphPair[0].nodes))  
  theNewGraphList=[]
  for i in theGraphList:
    G1_temp=copy.deepcopy(i[0])
    G2_temp=copy.deepcopy(i[1])
    G_intersect=nx.intersection(G1_temp,G2_temp)
    #G1_temp=nx.induced_subgraph(G1_temp,G_intersect.nodes())
    #G2_temp=nx.induced_subgraph(G2_temp,G_intersect.nodes())
    G1_new=G1_temp.copy()
    G2_new=G2_temp.copy()
    outKeys1,deg_seq_out1 = zip(*G1_new.out_degree())
    outKeys2,deg_seq_out2 = zip(*G2_new.out_degree())
    inKeys1,deg_seq_in1 = zip(*G1_new.in_degree())
    inKeys2,deg_seq_in2 = zip(*G2_new.in_degree())
    #G2_ind = nx.induced_subgraph(G2_new,list(G1_new.nodes))
    #G1_ind= nx.induced_subgraph(G1_new,list(G2_new.nodes))
    #outKeys1,deg_seq_out_ind1 = zip(*G1_ind.out_degree())
    #outKeys2,deg_seq_out_ind2 = zip(*G2_ind.out_degree())
    #inKeys1,deg_seq_in_ind1 = zip(*G1_ind.in_degree())
    #inKeys2,deg_seq_in_ind2 = zip(*G2_ind.in_degree())
    
    preMeaner=random_number_of_conserved_edges_mean(len(G1_new.nodes),deg_seq_out1,deg_seq_out2,deg_seq_in1,deg_seq_in2)
    preStddev=random_number_of_conserved_edges_std(len(G1_new.nodes),deg_seq_out1,deg_seq_out2,deg_seq_in1,deg_seq_in2)
    #preMeaner_ind=random_number_of_conserved_edges_mean(len(G1_ind.nodes),deg_seq_out_ind1,deg_seq_out_ind2,deg_seq_in_ind1,deg_seq_in_ind2)
    #preStddev_ind=random_number_of_conserved_edges_std(len(G1_ind.nodes),deg_seq_out_ind1,deg_seq_out_ind2,deg_seq_in_ind1,deg_seq_in_ind2)
    #nullMean=0.5*(preMeaner/(len(G1_new.edges())+len(G2_ind.edges())-preMeaner)+preMeaner/(len(G2_new.edges())+len(G1_ind.edges())-preMeaner))
    #nullStd=0.5*(preStddev/(len(G1_new.edges())+len(G2_ind.edges())-preStddev)+preStddev/(len(G2_new.edges())+len(G1_ind.edges())-preStddev))
    nullMean=0.5*(preMeaner/len(G1_new.edges())+preMeaner/len(G2_new.edges()))
    nullStd=0.5*(preStddev/len(G1_new.edges())+preStddev/len(G2_new.edges()))
    #nullMean=preMeaner
    #nullStd=preStddev
    tempScore= i[2]
    if nullStd!=0:
      #zScore=(tempScore-nullMean)/nullStd
      zScore=tempScore
    else:
      zScore=0
    print("paired")
    theNewGraphList.append((i[0],i[1],zScore))
    print(zScore)
    '''
    nullVec=null_distribution_ec_score(G1_temp,G2_temp,resolution=10000)
    nullStd=np.std(nullVec,ddof=1)
    nullMean=np.mean(nullVec)
    tempScore= normalised_ec_score(G1_temp,G2_temp)
    zScore=(tempScore-nullMean)/nullStd
    print("paired")
    theNewGraphList.append((G1_temp,G2_temp,zScore))
    print(zScore)
    '''
    
  graphPair=max(theNewGraphList,key=lambda x:x[2])
  print("the big winner score is",graphPair[2],"and its graphs with",len(graphPair[0].nodes),"nodes")
  return graphPair[0],graphPair[1]
        
def ancestor_finder_without_alignment_gene_family_separate(G1,G2,qMod,qCon,tolerance=0,toleranceEC=0):
  """Returns a pair of networkx graphs G1' and G2' based off of the input graphs G1 and G2
    with nodes merged as to maximise their EC similarity.

    This is a heuristic algorithm that finds pairs of nodes, either in G1 and G2, that are the most likely to have been the most
    recently duplicated pairs and determines which of these pairs, when merged, will increase the
    EC similarity of the two graphs.

    The best pair under these conditions is chosen and is merged, and the process is repeated until the EC score 
    can no longer be improved. That is, a local optimum is found.

    Parameters
    ----------
    G1 : networkx graph
        One of the two graphs
    G2 : networkx graph
        One of the two graphs
    qMod : float
        The probability of edge deletion in the assumed forward DMC model.
        In cases where this is not known it may need to be estimated.
    qCon : float
        The probability of creating an edge between duplicated pairs of vertices in the assumed forward DMC model.
        In cases where this is not known it may need to be estimated.
    tolerance : float, optional (default=0.05)
        A tolerance value used to determine the acceptable likelihood a pair of nodes to have
        been the most recently duplicated pair. A non-zero tolerance allows less likely pairs to be chosen, possibly 
        increasing the EC score more than the best pair.

    Notes
    -----
    This algorithm is not guaranteed to find the global optimum. 

    This algorithm comprises 4 main steps:
    1. The likelihood of all pairs of nodes in both G1 and G2 are calculated.
    2. A list of the 'best' pairs is generated. These pairs are those that have the highest likelihood, 
    depending on the likelihood tolerance value.
    3. From these pairs, the pair(s) that improve(s) the EC score the most is chosen and merged. Most of the time there
    is more than one such pair, in which case a random one is chosen.
    4. The graph in which the most recent merge has occured is updated and the process is repeated until the EC score 
    can no longer be improved.
    
    Issues 
    ------
    1. For the first go of the loop, the maxScore is -1 and so all the scores are greater than it
    2. It is assumed that likelihoods are comparable across the graphs G1 and G2 for different sized graphs. 
    This very well may not be the case and should be followed up.
    3. Many pairs seem to have the give the same EC score. This could be fine, but possibly something is going on here.

    """
  
  
  #graphPair is a repeatedly updated tuple containing the present state of graphs G1 and G2, the EC score of the current pair,
  # and an int that represents which of G1 and G2 the most recent merge occured in.
  graphPair=(G1,G2,-1,0) 
  #signal is a boolean that is set to False when the EC score of the current pair is no longer improving.
  signal =True 
  
  
  #maxScore records the current best EC score
  maxScore=-1
  n=len(G1.nodes())
  theGraphList=[]
  P1=gene_family_partitioner(G1)
  P2=gene_family_partitioner(G2)
  prevScore=conserved_edges(G1,G2)
  countUp=False
  while signal and len(graphPair[0].nodes)>4:
    #step 1: calculate the likelihood of all pairs of nodes in both G1 and G2

    #alignedPairs is a dictionary, keyed by (i,j,k) that records the EC score of graphs resulting from merging 
    # each pair of nodes i,j in graph G_k (k=1 or k=2).
    alignedPairs=[]
    #maxLikelihood is updated each loop and records the likelihood score of the most likely node pair for the current loop
    maxLikelihood = -1
    #pairList is a list of tuples (i,j,k,likelihood) where i and j are the nodes in graph G_k
    # and likelihood is the likelihood of the pair (i,j) being most recently duplicated in graph G_k
    pairList=[]
    #G1 and G2 are updated to be the most recent pair of graphs
    G1=graphPair[0]
    G2=graphPair[1]
    P1=gene_family_partitioner(G1)
    P2=gene_family_partitioner(G2)
    
    
    #Consider each node pair (i,j) in G1

    for fam in P1:
      for num1,i in enumerate(P1[fam]):
        for num2,j in enumerate(P1[fam]):
          
          if num2>num1:
            #Construct the graph G1_temp resulting from merging (i,j) in G1
            #This pair of if statements ensures that a node merge keeps a label common to both graphs if possible
            
            #check the likelihood function for the pair (i,j)
            tempLikelihood = ancestral_likelihood_dmc(G1,i,j,qMod,qCon)
            
            #Update the pairList and the alignedPairs dictionary with the likelihood of the pair (i,j) and the EC score
            # of the merged node graph G1_temp and G2
            pairList.append((i,j,1,tempLikelihood))
            
    
    #Consider each node pair (i,j) in G2
    for fam in P2:
      for num1,i in enumerate(P2[fam]):  
        for num2,j in enumerate(P2[fam]):
          if num2>num1:
            #Construct the graph G2_temp resulting from merging (i,j) in G2
            #This pair of if statements ensures that a node merge keeps a label common to both graphs if possible
            
            
            #check the likelihood function for the pair (i,j)
            tempLikelihood = ancestral_likelihood_dmc(G2,i,j,qMod,qCon)
            
            #Update the pairList and the alignedPairs dictionary with the likelihood of the pair (i,j) and the EC score
            # of the merged node graph G2_temp and G1
            pairList.append((i,j,2,tempLikelihood))
            
            
    #step 2: choose the best pair(s) based on the likelihood function and the tolerance value
    #End algorithm if pairList is empty. Only occurs if both graphs are single node or empty.
    if len(pairList)==0:
      break
    maxLikelihood=max(pairList,key=lambda x:x[3])[3]
    
    pairPairList=[]      
    #step 2: choose the best pair(s) based on the likelihood function and the tolerance value
    #End algorithm if pairList is empty. Only occurs if both graphs are single node or empty.
    if pairList==[]:
      signal=False
    #if pairList isn't empty, choose the best pair(s) based on the likelihood function and the tolerance value  
    else:
      print("list begins")
      for i in pairList:
        if i[3]>= maxLikelihood-tolerance*maxLikelihood:
              pairPairList.append((i[0],i[1],i[2],i[3]))
      
      print("sorting begins")
      pairPairList=sorted(pairPairList, key=lambda x: x[3],reverse=True) #find the maximum likelihood over all pairs
      print("merging begins")
      count=0
      signal=False
      trigger=False
      chosenLikelihood=0
      count=0
      for i in pairPairList:
        count=count+1
        if count>=len(pairPairList):
          countUp=True
          break
        if trigger and i[3]<chosenLikelihood:
          break
        if i[2]==1:
          if i[1] not in G2.nodes and i[0] in G2.nodes:
            G1_temp=node_merger(G1,i[0],i[1],self_loops=False)
          else:
            G1_temp=node_merger(G1,i[1],i[0],self_loops=False)
          G1_tempp=copy.deepcopy(G1_temp)
          tempScore= conserved_edges(G1_tempp,G2)
          print(i,tempScore)
          if tempScore>=prevScore:
            print("chosen",i)
            graphPair=(G1_temp,G2,tempScore)
            prevScore=tempScore
            trigger=True
            chosenLikelihood=i[3]
            signal=True
            
        if i[2]==2:
          if i[1] not in G1.nodes and i[0] in G1.nodes:
            G2_temp=node_merger(G2,i[0],i[1],self_loops=False)
          else:
            G2_temp=node_merger(G2,i[1],i[0],self_loops=False)
          G2_tempp=copy.deepcopy(G2_temp)
          tempScore= conserved_edges(G1,G2_tempp)
          print(i,tempScore)
          if tempScore>=prevScore:
            print("chosen",i)
            graphPair=(G1,G2_temp,tempScore)
            prevScore=tempScore
            trigger=True
            chosenLikelihood=i[3]
            signal=True
            
      
        if countUp:
          print("chosen",i)
          #graphPair=(G1_temp,G2_temp,tempScore)
          break
      print("out of the merge")
      
      theGraphList.append(graphPair)
      print(graphPair,len(graphPair[0].nodes))  
  #theNewGraphList=[]
  #for i in theGraphList:
    #G1_temp=copy.deepcopy(i[0])
    #G2_temp=copy.deepcopy(i[1])
    #G_intersect=nx.intersection(G1_temp,G2_temp)
    #G1_temp=nx.induced_subgraph(G1_temp,G_intersect.nodes())
    #G2_temp=nx.induced_subgraph(G2_temp,G_intersect.nodes())
    #G1_new=G1_temp.copy()
    #G2_new=G2_temp.copy()
    #outKeys1,deg_seq_out1 = zip(*G1_new.out_degree())
    #outKeys2,deg_seq_out2 = zip(*G2_new.out_degree())
    #inKeys1,deg_seq_in1 = zip(*G1_new.in_degree())
    #inKeys2,deg_seq_in2 = zip(*G2_new.in_degree())
    #G2_ind = nx.induced_subgraph(G2_new,list(G1_new.nodes))
    #G1_ind= nx.induced_subgraph(G1_new,list(G2_new.nodes))
    #outKeys1,deg_seq_out_ind1 = zip(*G1_ind.out_degree())
    #outKeys2,deg_seq_out_ind2 = zip(*G2_ind.out_degree())
    #inKeys1,deg_seq_in_ind1 = zip(*G1_ind.in_degree())
    #inKeys2,deg_seq_in_ind2 = zip(*G2_ind.in_degree())
    
    #preMeaner=random_number_of_conserved_edges_mean(len(G1_new.nodes),deg_seq_out1,deg_seq_out2,deg_seq_in1,deg_seq_in2)
    #preStddev=random_number_of_conserved_edges_std(len(G1_new.nodes),deg_seq_out1,deg_seq_out2,deg_seq_in1,deg_seq_in2)
    #preMeaner_ind=random_number_of_conserved_edges_mean(len(G1_ind.nodes),deg_seq_out_ind1,deg_seq_out_ind2,deg_seq_in_ind1,deg_seq_in_ind2)
    #preStddev_ind=random_number_of_conserved_edges_std(len(G1_ind.nodes),deg_seq_out_ind1,deg_seq_out_ind2,deg_seq_in_ind1,deg_seq_in_ind2)
    #nullMean=0.5*(preMeaner/(len(G1_new.edges())+len(G2_ind.edges())-preMeaner)+preMeaner/(len(G2_new.edges())+len(G1_ind.edges())-preMeaner))
    #nullStd=0.5*(preStddev/(len(G1_new.edges())+len(G2_ind.edges())-preStddev)+preStddev/(len(G2_new.edges())+len(G1_ind.edges())-preStddev))
    #nullMean=0.5*(preMeaner/len(G1_new.edges())+preMeaner/len(G2_new.edges()))
    #nullStd=0.5*(preStddev/len(G1_new.edges())+preStddev/len(G2_new.edges()))
    #nullMean=preMeaner
    #nullStd=preStddev
    #tempScore= i[2]
    #if nullStd!=0:
      #zScore=(tempScore-nullMean)/nullStd
    #  zScore=tempScore
    #else:
    #  zScore=0
    #zScore=tempScore
    #print("paired")
    #theNewGraphList.append((i[0],i[1],zScore))
    #print(zScore)
    '''
    nullVec=null_distribution_ec_score(G1_temp,G2_temp,resolution=10000)
    nullStd=np.std(nullVec,ddof=1)
    nullMean=np.mean(nullVec)
    tempScore= normalised_ec_score(G1_temp,G2_temp)
    zScore=(tempScore-nullMean)/nullStd
    print("paired")
    theNewGraphList.append((G1_temp,G2_temp,zScore))
    print(zScore)
    '''
    
  #graphPair=max(theNewGraphList,key=lambda x:x[2])
  print("the big winner score is",graphPair[2],"and its graphs with",len(graphPair[0].nodes),"nodes")
  return graphPair[0],graphPair[1]      
def ancestor_finder_without_alignment_gene_family_separate_ped_pea(G1,G2,r,q,tolerance=0,toleranceEC=0,true_labels=False):
  """Returns a pair of networkx graphs G1' and G2' based off of the input graphs G1 and G2
    with nodes merged as to maximise their EC similarity.

    This is a heuristic algorithm that finds pairs of nodes, either in G1 and G2, that are the most likely to have been the most
    recently duplicated pairs and determines which of these pairs, when merged, will increase the
    EC similarity of the two graphs.

    The best pair under these conditions is chosen and is merged, and the process is repeated until the EC score 
    can no longer be improved. That is, a local optimum is found.

    Parameters
    ----------
    G1 : networkx graph
        One of the two graphs
    G2 : networkx graph
        One of the two graphs
    qMod : float
        The probability of edge deletion in the assumed forward DMC model.
        In cases where this is not known it may need to be estimated.
    qCon : float
        The probability of creating an edge between duplicated pairs of vertices in the assumed forward DMC model.
        In cases where this is not known it may need to be estimated.
    tolerance : float, optional (default=0.05)
        A tolerance value used to determine the acceptable likelihood a pair of nodes to have
        been the most recently duplicated pair. A non-zero tolerance allows less likely pairs to be chosen, possibly 
        increasing the EC score more than the best pair.

    Notes
    -----
    This algorithm is not guaranteed to find the global optimum. 

    This algorithm comprises 4 main steps:
    1. The likelihood of all pairs of nodes in both G1 and G2 are calculated.
    2. A list of the 'best' pairs is generated. These pairs are those that have the highest likelihood, 
    depending on the likelihood tolerance value.
    3. From these pairs, the pair(s) that improve(s) the EC score the most is chosen and merged. Most of the time there
    is more than one such pair, in which case a random one is chosen.
    4. The graph in which the most recent merge has occured is updated and the process is repeated until the EC score 
    can no longer be improved.
    
    Issues 
    ------
    1. For the first go of the loop, the maxScore is -1 and so all the scores are greater than it
    2. It is assumed that likelihoods are comparable across the graphs G1 and G2 for different sized graphs. 
    This very well may not be the case and should be followed up.
    3. Many pairs seem to have the give the same EC score. This could be fine, but possibly something is going on here.

    """
  
  if true_labels:
    #graphPair is a repeatedly updated tuple containing the present state of graphs G1 and G2, the EC score of the current pair,
    # and an int that represents which of G1 and G2 the most recent merge occured in.
    graphPair=(G1,G2,-1,0) 
    #signal is a boolean that is set to False when the EC score of the current pair is no longer improving.
    signal =True 
    
    
    #maxScore records the current best EC score
    maxScore=-1
    n=len(G1.nodes())
    theGraphList=[]
    P1=gene_family_partitioner(G1)
    P2=gene_family_partitioner(G2)
    prevScore=conserved_edges(G1,G2)
    countUp=False
    while signal and len(graphPair[0].nodes)>4:
      #step 1: calculate the likelihood of all pairs of nodes in both G1 and G2

      #alignedPairs is a dictionary, keyed by (i,j,k) that records the EC score of graphs resulting from merging 
      # each pair of nodes i,j in graph G_k (k=1 or k=2).
      alignedPairs=[]
      #maxLikelihood is updated each loop and records the likelihood score of the most likely node pair for the current loop
      maxLikelihood = -1
      #pairList is a list of tuples (i,j,k,likelihood) where i and j are the nodes in graph G_k
      # and likelihood is the likelihood of the pair (i,j) being most recently duplicated in graph G_k
      pairList=[]
      #G1 and G2 are updated to be the most recent pair of graphs
      G1=graphPair[0]
      G2=graphPair[1]
      P1=gene_family_partitioner(G1)
      P2=gene_family_partitioner(G2)
      
      for fam in P1:
        external_nodes_1=[i for i in P1[fam] if i not in P2[fam]]
      
      #Consider each node pair (i,j) in G1

      for fam in P1:
        for num1,i in enumerate(external_nodes_1):
          for num2,j in enumerate(P1[fam]):
            
            if num2>num1:
              #Construct the graph G1_temp resulting from merging (i,j) in G1
              #This pair of if statements ensures that a node merge keeps a label common to both graphs if possible
              
              #check the likelihood function for the pair (i,j)
              tempLikelihood = ancestral_likelihood_ped_pea(G1,i,j,r,q)
              
              #Update the pairList and the alignedPairs dictionary with the likelihood of the pair (i,j) and the EC score
              # of the merged node graph G1_temp and G2
              pairList.append((i,j,1,tempLikelihood))
              
      
      #Consider each node pair (i,j) in G2
      for fam in P1:
        external_nodes_2=[i for i in P2[fam] if i not in P1[fam]]
      for fam in P2:
        for num1,i in enumerate(external_nodes_2):  
          for num2,j in enumerate(P2[fam]):
            if num2>num1:
              #Construct the graph G2_temp resulting from merging (i,j) in G2
              #This pair of if statements ensures that a node merge keeps a label common to both graphs if possible
              
              
              #check the likelihood function for the pair (i,j)
              tempLikelihood = ancestral_likelihood_ped_pea(G2,i,j,r,q)
              
              #Update the pairList and the alignedPairs dictionary with the likelihood of the pair (i,j) and the EC score
              # of the merged node graph G2_temp and G1
              pairList.append((i,j,2,tempLikelihood))
              
              
      #step 2: choose the best pair(s) based on the likelihood function and the tolerance value
      #End algorithm if pairList is empty. Only occurs if both graphs are single node or empty.
      if len(pairList)==0:
        break
      maxLikelihood=max(pairList,key=lambda x:x[3])[3]
      
      pairPairList=[]      
      #step 2: choose the best pair(s) based on the likelihood function and the tolerance value
      #End algorithm if pairList is empty. Only occurs if both graphs are single node or empty.
      if pairList==[]:
        signal=False
      #if pairList isn't empty, choose the best pair(s) based on the likelihood function and the tolerance value  
      else:
        print("list begins")
        for i in pairList:
          if i[3]>= maxLikelihood-tolerance*maxLikelihood:
                pairPairList.append((i[0],i[1],i[2],i[3]))
        
        print("sorting begins")
        pairPairList=sorted(pairPairList, key=lambda x: x[3],reverse=True) #find the maximum likelihood over all pairs
        print("merging begins")
        count=0
        signal=False
        
        trigger=False
        chosenLikelihood=0
        countUp=False
        for i in pairPairList:
          count=count+1
          if count>=len(pairPairList):
            countUp=True
            break
          if trigger and i[3]<chosenLikelihood:
            print('ay you broke my trigger')
            break
          if i[2]==1:
            if i[1] not in G2.nodes and i[0] in G2.nodes:
              G1_temp=node_merger(G1,i[0],i[1],self_loops=False)
            else:
              G1_temp=node_merger(G1,i[1],i[0],self_loops=False)
            G1_tempp=copy.deepcopy(G1_temp)
            tempScore= conserved_edges(G1_tempp,G2)
            print(i,tempScore)
            if tempScore>=prevScore:
              print("chosen",i)
              graphPair=(G1_temp,G2,tempScore)
              prevScore=tempScore
              trigger=True
              chosenLikelihood=i[3]
              signal=True
              
          if i[2]==2:
            if i[1] not in G1.nodes and i[0] in G1.nodes:
              G2_temp=node_merger(G2,i[0],i[1],self_loops=False)
            else:
              G2_temp=node_merger(G2,i[1],i[0],self_loops=False)
            G2_tempp=copy.deepcopy(G2_temp)
            tempScore= conserved_edges(G1,G2_tempp)
            print(i,tempScore)
            if tempScore>=prevScore:
              print("chosen",i)
              graphPair=(G1,G2_temp,tempScore)
              prevScore=tempScore
              trigger=True
              chosenLikelihood=i[3]
              signal=True
        
          
        print("out of the merge")
        
        theGraphList.append(graphPair)
        print(graphPair,len(graphPair[0].nodes))  
    #theNewGraphList=[]
    #for i in theGraphList:
      #G1_temp=copy.deepcopy(i[0])
      #G2_temp=copy.deepcopy(i[1])
      #G_intersect=nx.intersection(G1_temp,G2_temp)
      #G1_temp=nx.induced_subgraph(G1_temp,G_intersect.nodes())
      #G2_temp=nx.induced_subgraph(G2_temp,G_intersect.nodes())
      #G1_new=G1_temp.copy()
      #G2_new=G2_temp.copy()
      #outKeys1,deg_seq_out1 = zip(*G1_new.out_degree())
      #outKeys2,deg_seq_out2 = zip(*G2_new.out_degree())
      #inKeys1,deg_seq_in1 = zip(*G1_new.in_degree())
      #inKeys2,deg_seq_in2 = zip(*G2_new.in_degree())
      #G2_ind = nx.induced_subgraph(G2_new,list(G1_new.nodes))
      #G1_ind= nx.induced_subgraph(G1_new,list(G2_new.nodes))
      #outKeys1,deg_seq_out_ind1 = zip(*G1_ind.out_degree())
      #outKeys2,deg_seq_out_ind2 = zip(*G2_ind.out_degree())
      #inKeys1,deg_seq_in_ind1 = zip(*G1_ind.in_degree())
      #inKeys2,deg_seq_in_ind2 = zip(*G2_ind.in_degree())
      
      #preMeaner=random_number_of_conserved_edges_mean(len(G1_new.nodes),deg_seq_out1,deg_seq_out2,deg_seq_in1,deg_seq_in2)
      #preStddev=random_number_of_conserved_edges_std(len(G1_new.nodes),deg_seq_out1,deg_seq_out2,deg_seq_in1,deg_seq_in2)
      #preMeaner_ind=random_number_of_conserved_edges_mean(len(G1_ind.nodes),deg_seq_out_ind1,deg_seq_out_ind2,deg_seq_in_ind1,deg_seq_in_ind2)
      #preStddev_ind=random_number_of_conserved_edges_std(len(G1_ind.nodes),deg_seq_out_ind1,deg_seq_out_ind2,deg_seq_in_ind1,deg_seq_in_ind2)
      #nullMean=0.5*(preMeaner/(len(G1_new.edges())+len(G2_ind.edges())-preMeaner)+preMeaner/(len(G2_new.edges())+len(G1_ind.edges())-preMeaner))
      #nullStd=0.5*(preStddev/(len(G1_new.edges())+len(G2_ind.edges())-preStddev)+preStddev/(len(G2_new.edges())+len(G1_ind.edges())-preStddev))
      #nullMean=0.5*(preMeaner/len(G1_new.edges())+preMeaner/len(G2_new.edges()))
      #nullStd=0.5*(preStddev/len(G1_new.edges())+preStddev/len(G2_new.edges()))
      #nullMean=preMeaner
      #nullStd=preStddev
      #tempScore= i[2]
      #if nullStd!=0:
        #zScore=(tempScore-nullMean)/nullStd
      #  zScore=tempScore
      #else:
      #  zScore=0
      #zScore=tempScore
      #print("paired")
      #theNewGraphList.append((i[0],i[1],zScore))
      #print(zScore)
      '''
      nullVec=null_distribution_ec_score(G1_temp,G2_temp,resolution=10000)
      nullStd=np.std(nullVec,ddof=1)
      nullMean=np.mean(nullVec)
      tempScore= normalised_ec_score(G1_temp,G2_temp)
      zScore=(tempScore-nullMean)/nullStd
      print("paired")
      theNewGraphList.append((G1_temp,G2_temp,zScore))
      print(zScore)
      '''
      
    #graphPair=max(theNewGraphList,key=lambda x:x[2])
    print("the big winner score is",graphPair[2],"and its graphs with",len(graphPair[0].nodes),"nodes")
    return graphPair[0],graphPair[1]      
  else:
    #graphPair is a repeatedly updated tuple containing the present state of graphs G1 and G2, the EC score of the current pair,
    # and an int that represents which of G1 and G2 the most recent merge occured in.
    graphPair=(G1,G2,-1,0) 
    #signal is a boolean that is set to False when the EC score of the current pair is no longer improving.
    signal =True 
    
    
    #maxScore records the current best EC score
    maxScore=-1
    n=len(G1.nodes())
    theGraphList=[]
    P1=gene_family_partitioner(G1)
    P2=gene_family_partitioner(G2)
    prevScore=conserved_edges(G1,G2)
    countUp=False
    while signal and len(graphPair[0].nodes)>4:
      #step 1: calculate the likelihood of all pairs of nodes in both G1 and G2

      #alignedPairs is a dictionary, keyed by (i,j,k) that records the EC score of graphs resulting from merging 
      # each pair of nodes i,j in graph G_k (k=1 or k=2).
      alignedPairs=[]
      #maxLikelihood is updated each loop and records the likelihood score of the most likely node pair for the current loop
      maxLikelihood = -1
      #pairList is a list of tuples (i,j,k,likelihood) where i and j are the nodes in graph G_k
      # and likelihood is the likelihood of the pair (i,j) being most recently duplicated in graph G_k
      pairList=[]
      #G1 and G2 are updated to be the most recent pair of graphs
      G1=graphPair[0]
      G2=graphPair[1]
      P1=gene_family_partitioner(G1)
      P2=gene_family_partitioner(G2)
      
      
      #Consider each node pair (i,j) in G1

      for fam in P1:
        for num1,i in enumerate(P1[fam]):
          for num2,j in enumerate(P1[fam]):
            
            if num2>num1:
              #Construct the graph G1_temp resulting from merging (i,j) in G1
              #This pair of if statements ensures that a node merge keeps a label common to both graphs if possible
              
              #check the likelihood function for the pair (i,j)
              tempLikelihood = ancestral_likelihood_ped_pea(G1,i,j,r,q)
              
              #Update the pairList and the alignedPairs dictionary with the likelihood of the pair (i,j) and the EC score
              # of the merged node graph G1_temp and G2
              pairList.append((i,j,1,tempLikelihood))
              
      
      #Consider each node pair (i,j) in G2
      
      for fam in P2:
        for num1,i in enumerate(P2[fam]):  
          for num2,j in enumerate(P2[fam]):
            if num2>num1:
              #Construct the graph G2_temp resulting from merging (i,j) in G2
              #This pair of if statements ensures that a node merge keeps a label common to both graphs if possible
              
              
              #check the likelihood function for the pair (i,j)
              tempLikelihood = ancestral_likelihood_ped_pea(G2,i,j,r,q)
              
              #Update the pairList and the alignedPairs dictionary with the likelihood of the pair (i,j) and the EC score
              # of the merged node graph G2_temp and G1
              pairList.append((i,j,2,tempLikelihood))
              
              
      #step 2: choose the best pair(s) based on the likelihood function and the tolerance value
      #End algorithm if pairList is empty. Only occurs if both graphs are single node or empty.
      if len(pairList)==0:
        break
      maxLikelihood=max(pairList,key=lambda x:x[3])[3]
      
      pairPairList=[]      
      #step 2: choose the best pair(s) based on the likelihood function and the tolerance value
      #End algorithm if pairList is empty. Only occurs if both graphs are single node or empty.
      if pairList==[]:
        signal=False
      #if pairList isn't empty, choose the best pair(s) based on the likelihood function and the tolerance value  
      else:
        print("list begins")
        for i in pairList:
          if i[3]>= maxLikelihood-tolerance*maxLikelihood:
                pairPairList.append((i[0],i[1],i[2],i[3]))
        
        print("sorting begins")
        pairPairList=sorted(pairPairList, key=lambda x: x[3],reverse=True) #find the maximum likelihood over all pairs
        print("merging begins")
        count=0
        signal=False
        
        trigger=False
        chosenLikelihood=0
        countUp=False
        for i in pairPairList:
          count=count+1
          if count>=len(pairPairList):
            countUp=True
            break
          if trigger and i[3]<chosenLikelihood:
            print('ay you broke my trigger')
            break
          if i[2]==1:
            if i[1] not in G2.nodes and i[0] in G2.nodes:
              G1_temp=node_merger(G1,i[0],i[1],self_loops=False)
            else:
              G1_temp=node_merger(G1,i[1],i[0],self_loops=False)
            G1_tempp=copy.deepcopy(G1_temp)
            tempScore= conserved_edges(G1_tempp,G2)
            print(i,tempScore)
            if tempScore>=prevScore:
              print("chosen",i)
              graphPair=(G1_temp,G2,tempScore)
              prevScore=tempScore
              trigger=True
              chosenLikelihood=i[3]
              signal=True
              
          if i[2]==2:
            if i[1] not in G1.nodes and i[0] in G1.nodes:
              G2_temp=node_merger(G2,i[0],i[1],self_loops=False)
            else:
              G2_temp=node_merger(G2,i[1],i[0],self_loops=False)
            G2_tempp=copy.deepcopy(G2_temp)
            tempScore= conserved_edges(G1,G2_tempp)
            print(i,tempScore)
            if tempScore>=prevScore:
              print("chosen",i)
              graphPair=(G1,G2_temp,tempScore)
              prevScore=tempScore
              trigger=True
              chosenLikelihood=i[3]
              signal=True
        
          
        print("out of the merge")
        
        theGraphList.append(graphPair)
        print(graphPair,len(graphPair[0].nodes))  
    #theNewGraphList=[]
    #for i in theGraphList:
      #G1_temp=copy.deepcopy(i[0])
      #G2_temp=copy.deepcopy(i[1])
      #G_intersect=nx.intersection(G1_temp,G2_temp)
      #G1_temp=nx.induced_subgraph(G1_temp,G_intersect.nodes())
      #G2_temp=nx.induced_subgraph(G2_temp,G_intersect.nodes())
      #G1_new=G1_temp.copy()
      #G2_new=G2_temp.copy()
      #outKeys1,deg_seq_out1 = zip(*G1_new.out_degree())
      #outKeys2,deg_seq_out2 = zip(*G2_new.out_degree())
      #inKeys1,deg_seq_in1 = zip(*G1_new.in_degree())
      #inKeys2,deg_seq_in2 = zip(*G2_new.in_degree())
      #G2_ind = nx.induced_subgraph(G2_new,list(G1_new.nodes))
      #G1_ind= nx.induced_subgraph(G1_new,list(G2_new.nodes))
      #outKeys1,deg_seq_out_ind1 = zip(*G1_ind.out_degree())
      #outKeys2,deg_seq_out_ind2 = zip(*G2_ind.out_degree())
      #inKeys1,deg_seq_in_ind1 = zip(*G1_ind.in_degree())
      #inKeys2,deg_seq_in_ind2 = zip(*G2_ind.in_degree())
      
      #preMeaner=random_number_of_conserved_edges_mean(len(G1_new.nodes),deg_seq_out1,deg_seq_out2,deg_seq_in1,deg_seq_in2)
      #preStddev=random_number_of_conserved_edges_std(len(G1_new.nodes),deg_seq_out1,deg_seq_out2,deg_seq_in1,deg_seq_in2)
      #preMeaner_ind=random_number_of_conserved_edges_mean(len(G1_ind.nodes),deg_seq_out_ind1,deg_seq_out_ind2,deg_seq_in_ind1,deg_seq_in_ind2)
      #preStddev_ind=random_number_of_conserved_edges_std(len(G1_ind.nodes),deg_seq_out_ind1,deg_seq_out_ind2,deg_seq_in_ind1,deg_seq_in_ind2)
      #nullMean=0.5*(preMeaner/(len(G1_new.edges())+len(G2_ind.edges())-preMeaner)+preMeaner/(len(G2_new.edges())+len(G1_ind.edges())-preMeaner))
      #nullStd=0.5*(preStddev/(len(G1_new.edges())+len(G2_ind.edges())-preStddev)+preStddev/(len(G2_new.edges())+len(G1_ind.edges())-preStddev))
      #nullMean=0.5*(preMeaner/len(G1_new.edges())+preMeaner/len(G2_new.edges()))
      #nullStd=0.5*(preStddev/len(G1_new.edges())+preStddev/len(G2_new.edges()))
      #nullMean=preMeaner
      #nullStd=preStddev
      #tempScore= i[2]
      #if nullStd!=0:
        #zScore=(tempScore-nullMean)/nullStd
      #  zScore=tempScore
      #else:
      #  zScore=0
      #zScore=tempScore
      #print("paired")
      #theNewGraphList.append((i[0],i[1],zScore))
      #print(zScore)
      '''
      nullVec=null_distribution_ec_score(G1_temp,G2_temp,resolution=10000)
      nullStd=np.std(nullVec,ddof=1)
      nullMean=np.mean(nullVec)
      tempScore= normalised_ec_score(G1_temp,G2_temp)
      zScore=(tempScore-nullMean)/nullStd
      print("paired")
      theNewGraphList.append((G1_temp,G2_temp,zScore))
      print(zScore)
      '''
      
    #graphPair=max(theNewGraphList,key=lambda x:x[2])
    print("the big winner score is",graphPair[2],"and its graphs with",len(graphPair[0].nodes),"nodes")
    return graphPair[0],graphPair[1]     

def ancestor_finder_with_alignment_gene_family_separate_ped_pea(G1,G2,P1,P2,r,q,tolerance=0,toleranceEC=0,true_labels=True):
  """Returns a pair of networkx graphs G1' and G2' based off of the input graphs G1 and G2
    with nodes merged as to maximise their EC similarity.

    This is a heuristic algorithm that finds pairs of nodes, either in G1 and G2, that are the most likely to have been the most
    recently duplicated pairs and determines which of these pairs, when merged, will increase the
    EC similarity of the two graphs.

    The best pair under these conditions is chosen and is merged, and the process is repeated until the EC score 
    can no longer be improved. That is, a local optimum is found.

    Parameters
    ----------
    G1 : networkx graph
        One of the two graphs
    G2 : networkx graph
        One of the two graphs
    qMod : float
        The probability of edge deletion in the assumed forward DMC model.
        In cases where this is not known it may need to be estimated.
    qCon : float
        The probability of creating an edge between duplicated pairs of vertices in the assumed forward DMC model.
        In cases where this is not known it may need to be estimated.
    tolerance : float, optional (default=0.05)
        A tolerance value used to determine the acceptable likelihood a pair of nodes to have
        been the most recently duplicated pair. A non-zero tolerance allows less likely pairs to be chosen, possibly 
        increasing the EC score more than the best pair.

    Notes
    -----
    This algorithm is not guaranteed to find the global optimum. 

    This algorithm comprises 4 main steps:
    1. The likelihood of all pairs of nodes in both G1 and G2 are calculated.
    2. A list of the 'best' pairs is generated. These pairs are those that have the highest likelihood, 
    depending on the likelihood tolerance value.
    3. From these pairs, the pair(s) that improve(s) the EC score the most is chosen and merged. Most of the time there
    is more than one such pair, in which case a random one is chosen.
    4. The graph in which the most recent merge has occured is updated and the process is repeated until the EC score 
    can no longer be improved.
    
    Issues 
    ------
    1. For the first go of the loop, the maxScore is -1 and so all the scores are greater than it
    2. It is assumed that likelihoods are comparable across the graphs G1 and G2 for different sized graphs. 
    This very well may not be the case and should be followed up.
    3. Many pairs seem to have the give the same EC score. This could be fine, but possibly something is going on here.

    """
  
  if true_labels:
    #graphPair is a repeatedly updated tuple containing the present state of graphs G1 and G2, the EC score of the current pair,
    # and an int that represents which of G1 and G2 the most recent merge occured in.
    graphPair=(G1,G2,-1,0) 
    #signal is a boolean that is set to False when the EC score of the current pair is no longer improving.
    signal =True 
    
    
    #maxScore records the current best EC score
    maxScore=-1
    n=len(G1.nodes())
    theGraphList=[]
    
    prevScore=conserved_edges(G1,G2)
    countUp=False
    while signal and len(graphPair[0].nodes)>4:
      #step 1: calculate the likelihood of all pairs of nodes in both G1 and G2

      #alignedPairs is a dictionary, keyed by (i,j,k) that records the EC score of graphs resulting from merging 
      # each pair of nodes i,j in graph G_k (k=1 or k=2).
      alignedPairs=[]
      #maxLikelihood is updated each loop and records the likelihood score of the most likely node pair for the current loop
      maxLikelihood = -1
      #pairList is a list of tuples (i,j,k,likelihood) where i and j are the nodes in graph G_k
      # and likelihood is the likelihood of the pair (i,j) being most recently duplicated in graph G_k
      pairList=[]
      #G1 and G2 are updated to be the most recent pair of graphs
      G1=graphPair[0]
      G2=graphPair[1]
      P1=gene_family_partitioner(G1,orig_label=True)
      P2=gene_family_partitioner(G2,orig_label=True)
      print("p1,p2",P1,P2)
      external_nodes_1=dict()
      external_nodes_2=dict()
      for fam in P1:
        external_nodes_1[fam]=[i for i in P1[fam] if i not in P2[fam]]
      print("external",external_nodes_1)
      #Consider each node pair (i,j) in G1

      for fam in P1:
        for num1,i in enumerate(external_nodes_1[fam]):
          for num2,j in enumerate(P1[fam]):
            
            if i!=j:
              #Construct the graph G1_temp resulting from merging (i,j) in G1
              #This pair of if statements ensures that a node merge keeps a label common to both graphs if possible
              
              #check the likelihood function for the pair (i,j)
              tempLikelihood = ancestral_likelihood_ped_pea(G1,i,j,r,q)
              
              #Update the pairList and the alignedPairs dictionary with the likelihood of the pair (i,j) and the EC score
              # of the merged node graph G1_temp and G2
              pairList.append((i,j,1,tempLikelihood))
              
      
      #Consider each node pair (i,j) in G2
      for fam in P2:
        external_nodes_2[fam]=[i for i in P2[fam] if i not in P1[fam]]
      for fam in P2:
        for num1,i in enumerate(external_nodes_2[fam]):  
          for num2,j in enumerate(P2[fam]):
            if i!=j:
              #Construct the graph G2_temp resulting from merging (i,j) in G2
              #This pair of if statements ensures that a node merge keeps a label common to both graphs if possible
              
              
              #check the likelihood function for the pair (i,j)
              tempLikelihood = ancestral_likelihood_ped_pea(G2,i,j,r,q)
              
              #Update the pairList and the alignedPairs dictionary with the likelihood of the pair (i,j) and the EC score
              # of the merged node graph G2_temp and G1
              pairList.append((i,j,2,tempLikelihood))
              
              
      #step 2: choose the best pair(s) based on the likelihood function and the tolerance value
      #End algorithm if pairList is empty. Only occurs if both graphs are single node or empty.
      if len(pairList)==0:
        break
      maxLikelihood=max(pairList,key=lambda x:x[3])[3]
      
      pairPairList=[]      
      #step 2: choose the best pair(s) based on the likelihood function and the tolerance value
      #End algorithm if pairList is empty. Only occurs if both graphs are single node or empty.
      if pairList==[]:
        signal=False
      #if pairList isn't empty, choose the best pair(s) based on the likelihood function and the tolerance value  
      else:
        print("list begins")
        for i in pairList:
          if i[3]>= maxLikelihood-tolerance*maxLikelihood:
                pairPairList.append((i[0],i[1],i[2],i[3]))
        
        print("sorting begins")
        pairPairList=sorted(pairPairList, key=lambda x: x[3],reverse=True) #find the maximum likelihood over all pairs
        print("merging begins")
        count=0
        signal=False
        
        trigger=False
        chosenLikelihood=0
        countUp=False
        for i in pairPairList:
          count=count+1
          if count>=len(pairPairList):
            countUp=True
            break
          if trigger and i[3]<chosenLikelihood:
            print('ay you broke my trigger')
            break
          if i[2]==1:
            if i[1] not in G2.nodes and i[0] in G2.nodes:
              G1_temp=node_merger(G1,i[0],i[1],self_loops=False)
            else:
              G1_temp=node_merger(G1,i[1],i[0],self_loops=False)
            G1_tempp=copy.deepcopy(G1_temp)
            tempScore= conserved_edges(G1_tempp,G2)
            print(i,tempScore)
            if tempScore>=prevScore:
              print("chosen",i)
              graphPair=(G1_temp,G2,tempScore)
              prevScore=tempScore
              trigger=True
              chosenLikelihood=i[3]
              signal=True
              
          if i[2]==2:
            if i[1] not in G1.nodes and i[0] in G1.nodes:
              G2_temp=node_merger(G2,i[0],i[1],self_loops=False)
            else:
              G2_temp=node_merger(G2,i[1],i[0],self_loops=False)
            G2_tempp=copy.deepcopy(G2_temp)
            tempScore= conserved_edges(G1,G2_tempp)
            print(i,tempScore)
            if tempScore>=prevScore:
              print("chosen",i)
              graphPair=(G1,G2_temp,tempScore)
              prevScore=tempScore
              trigger=True
              chosenLikelihood=i[3]
              signal=True
        
          
        print("out of the merge")
        
        theGraphList.append(graphPair)
        print(graphPair,len(graphPair[0].nodes))  
    #theNewGraphList=[]
    #for i in theGraphList:
      #G1_temp=copy.deepcopy(i[0])
      #G2_temp=copy.deepcopy(i[1])
      #G_intersect=nx.intersection(G1_temp,G2_temp)
      #G1_temp=nx.induced_subgraph(G1_temp,G_intersect.nodes())
      #G2_temp=nx.induced_subgraph(G2_temp,G_intersect.nodes())
      #G1_new=G1_temp.copy()
      #G2_new=G2_temp.copy()
      #outKeys1,deg_seq_out1 = zip(*G1_new.out_degree())
      #outKeys2,deg_seq_out2 = zip(*G2_new.out_degree())
      #inKeys1,deg_seq_in1 = zip(*G1_new.in_degree())
      #inKeys2,deg_seq_in2 = zip(*G2_new.in_degree())
      #G2_ind = nx.induced_subgraph(G2_new,list(G1_new.nodes))
      #G1_ind= nx.induced_subgraph(G1_new,list(G2_new.nodes))
      #outKeys1,deg_seq_out_ind1 = zip(*G1_ind.out_degree())
      #outKeys2,deg_seq_out_ind2 = zip(*G2_ind.out_degree())
      #inKeys1,deg_seq_in_ind1 = zip(*G1_ind.in_degree())
      #inKeys2,deg_seq_in_ind2 = zip(*G2_ind.in_degree())
      
      #preMeaner=random_number_of_conserved_edges_mean(len(G1_new.nodes),deg_seq_out1,deg_seq_out2,deg_seq_in1,deg_seq_in2)
      #preStddev=random_number_of_conserved_edges_std(len(G1_new.nodes),deg_seq_out1,deg_seq_out2,deg_seq_in1,deg_seq_in2)
      #preMeaner_ind=random_number_of_conserved_edges_mean(len(G1_ind.nodes),deg_seq_out_ind1,deg_seq_out_ind2,deg_seq_in_ind1,deg_seq_in_ind2)
      #preStddev_ind=random_number_of_conserved_edges_std(len(G1_ind.nodes),deg_seq_out_ind1,deg_seq_out_ind2,deg_seq_in_ind1,deg_seq_in_ind2)
      #nullMean=0.5*(preMeaner/(len(G1_new.edges())+len(G2_ind.edges())-preMeaner)+preMeaner/(len(G2_new.edges())+len(G1_ind.edges())-preMeaner))
      #nullStd=0.5*(preStddev/(len(G1_new.edges())+len(G2_ind.edges())-preStddev)+preStddev/(len(G2_new.edges())+len(G1_ind.edges())-preStddev))
      #nullMean=0.5*(preMeaner/len(G1_new.edges())+preMeaner/len(G2_new.edges()))
      #nullStd=0.5*(preStddev/len(G1_new.edges())+preStddev/len(G2_new.edges()))
      #nullMean=preMeaner
      #nullStd=preStddev
      #tempScore= i[2]
      #if nullStd!=0:
        #zScore=(tempScore-nullMean)/nullStd
      #  zScore=tempScore
      #else:
      #  zScore=0
      #zScore=tempScore
      #print("paired")
      #theNewGraphList.append((i[0],i[1],zScore))
      #print(zScore)
      '''
      nullVec=null_distribution_ec_score(G1_temp,G2_temp,resolution=10000)
      nullStd=np.std(nullVec,ddof=1)
      nullMean=np.mean(nullVec)
      tempScore= normalised_ec_score(G1_temp,G2_temp)
      zScore=(tempScore-nullMean)/nullStd
      print("paired")
      theNewGraphList.append((G1_temp,G2_temp,zScore))
      print(zScore)
      '''
      
    #graphPair=max(theNewGraphList,key=lambda x:x[2])
    print("the big winner score is",graphPair[2],"and its graphs with",len(graphPair[0].nodes),"nodes")
    return graphPair[0],graphPair[1]      
      

#--------------------------------------------------------------------------------------
#Other functions
def graph_intersection_union(G1,G2):
  G1_temp=copy.deepcopy(G1)
  G2_temp=copy.deepcopy(G2)
  G_intersect=nx.intersection(G1_temp,G2_temp)
  G1_induced=nx.induced_subgraph(G1_temp,G_intersect.nodes())
  G2_induced=nx.induced_subgraph(G2_temp,G_intersect.nodes())
  for i in G1_induced.edges():
    if i not in G_intersect.edges():
      G_intersect.add_edge(i[0],i[1])
  for i in G2_induced.edges():
    if i not in G_intersect.edges():
      G_intersect.add_edge(i[0],i[1])
  return G_intersect
def null_distribution_s3_score(G_random,G_base,resolution=100):
    scoreVec=[]
    edges=len(G_random.edges)
    
    G=copy.deepcopy(G_random)
    i=0
    while len(scoreVec)<resolution:
        
        i=i+1
        
        G=directed_double_edge_swap(G,1)
        if i%edges==0:
            score=s3_score(G,G_base)
            #print(i,"/",resolution*edges,score)
            scoreVec.append(score)
        
    return scoreVec
def null_distribution_ec_score(G_random,G_base,resolution=100000):
    scoreVec=[]
    edges=len(G_random.edges)
    
    G=copy.deepcopy(G_random)
    i=0
    while len(scoreVec)<resolution:
        
        i=i+1
        
        G=directed_double_edge_swap(G,1)
        if isinstance(G,str):
            print("graph too small")
            return scoreVec
        if i%(2*edges)==0:
            score=normalised_ec_score(G,G_base)
            #print(i,"/",resolution*edges,score)
            scoreVec.append(score)
        
    return scoreVec
def random_number_of_conserved_edges_mean(n,deg_seq_out1,deg_seq_out2,deg_seq_in1,deg_seq_in2,self_loops=True):
  summ=0
  m1=sum(deg_seq_in1)
  m2=sum(deg_seq_in2)
  if self_loops:
    for i in range(n):
      for j in range(n):
        summ=summ+deg_seq_out1[i]*deg_seq_out2[i]*deg_seq_in1[j]*deg_seq_in2[j]/(m1*m2)
  else:
    for i in range(n):
      for j in range(n-1):
        summ=summ+deg_seq_out1[i]*deg_seq_out2[i]*deg_seq_in1[j]*deg_seq_in2[j]/((m1-deg_seq_in1[i])*(m2-deg_seq_in2[i]))
  return summ
def random_number_of_conserved_edges_std(n,deg_seq_out1,deg_seq_out2,deg_seq_in1,deg_seq_in2,self_loops=True):
  summ=0
  m1=sum(deg_seq_in1)
  m2=sum(deg_seq_in2)
  if self_loops:
    for i in range(n):
      for j in range(n):
        summ=summ+(deg_seq_out1[i]*deg_seq_out2[i]*deg_seq_in1[j]*deg_seq_in2[j]/(m1*m2))*(1-deg_seq_out1[i]*deg_seq_out2[i]*deg_seq_in1[j]*deg_seq_in2[j]/(m1*m2))
  else:
    for i in range(n):
      for j in range(n-1):
        summ=summ+(deg_seq_out1[i]*deg_seq_out2[i]*deg_seq_in1[j]*deg_seq_in2[j]/((m1-deg_seq_in1[i])*(m2-deg_seq_in2[i])))*(1-deg_seq_out1[i]*deg_seq_out2[i]*deg_seq_in1[j]*deg_seq_in2[j]/((m1-deg_seq_in1[i])*(m2-deg_seq_in2[i])))
  if summ>=0:    
    return np.sqrt(summ)
  else:
    print("error in std")
    return 1
  
def directed_double_edge_swap(G, nswap=1, max_tries=100, seed=None,self_loops=False):
    """Swap two edges in the graph while keeping the node degrees fixed.

    A double-edge swap removes two randomly chosen edges u-v and x-y
    and creates the new edges u-x and v-y::

     u--v            u  v
            becomes  |  |
     x--y            x  y

    If either the edge u-x or v-y already exist no swap is performed
    and another attempt is made to find a suitable edge pair.

    Parameters
    ----------
    G : graph
       An undirected graph

    nswap : integer (optional, default=1)
       Number of double-edge swaps to perform

    max_tries : integer (optional)
       Maximum number of attempts to swap edges

    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.

    Returns
    -------
    G : graph
       The graph after double edge swaps.

    Notes
    -----
    Does not enforce any connectivity constraints.

    The graph G is modified in place.
    """
    
    if nswap > max_tries:
        raise nx.NetworkXError("Number of swaps > number of tries allowed.")
    if len(G) < 4:
        return("graph too small")
    # Instead of choosing uniformly at random from a generated edge list,
    # this algorithm chooses nonuniformly from the set of nodes with
    # probability weighted by degree.
    n = 0
    swapcount = 0
    keys, degrees = zip(*G.degree())  # keys, degree
    cdf = nx.utils.cumulative_distribution(degrees)  # cdf of degree
    discrete_sequence = nx.utils.discrete_sequence
    while swapcount < nswap:
        #        if random.random() < 0.5: continue # trick to avoid periodicities?
        # pick two random edges without creating edge list
        # choose source node indices from discrete distribution
        (ui, xi) = discrete_sequence(2, cdistribution=cdf, seed=seed)
        if ui == xi:
            continue  # same source, skip
        u = keys[ui]  # convert index to label
        x = keys[xi]
        # choose target uniformly from neighbors
        uNeigh=list(G.predecessors(u)) + list(G.successors(u))
        
        xNeigh=list(G.predecessors(x)) + list(G.successors(x))
        

        v = random.choice(uNeigh)
        y = random.choice(xNeigh)
        vNeigh=list(G.predecessors(v)) + list(G.successors(v))
        
        yNeigh=list(G.predecessors(y)) + list(G.successors(y))
        
        if (not self_loops) and (v == y or u == x or y==u or x==v):
            continue  # same target, skip

        if (u,v) in G.edges() and (v,u) not in G.edges():
          if (x,y) in G.edges() and (y,x) not in G.edges():
            if (y not in G.successors(u)) and (v not in G.successors(x)):  # don't create parallel edges
                G.add_edge(u, y)
                G.add_edge(x, v)
                G.remove_edge(u, v)
                G.remove_edge(x, y)
                swapcount += 1
          elif (x,y) not in G.edges() and (y,x) in G.edges():
            if (x not in G.successors(u)) and (v not in G.successors(y)):  # don't create parallel edges
                G.add_edge(u, x)
                G.add_edge(y, v)
                G.remove_edge(u, v)
                G.remove_edge(y, x)
                swapcount += 1
          elif (x,y) in G.edges() and (y,x) in G.edges():
            rand=random.random()
            if rand>0.5:
              if (y not in G.successors(u)) and (v not in G.successors(x)):  # don't create parallel edges
                G.add_edge(u, y)
                G.add_edge(x, v)
                G.remove_edge(u, v)
                G.remove_edge(x, y)
                swapcount += 1
            else:
              if (x not in G.successors(u)) and (v not in G.successors(y)):  # don't create parallel edges
                G.add_edge(u, x)
                G.add_edge(y, v)
                G.remove_edge(u, v)
                G.remove_edge(y, x)
                swapcount += 1
        elif (u,v) not in G.edges() and (v,u) in G.edges():
          if (x,y) in G.edges() and (y,x) not in G.edges():
            if (u not in G.successors(x)) and (y not in G.successors(v)):  # don't create parallel edges
                G.add_edge(x, u)
                G.add_edge(v, y)
                G.remove_edge(v, u)
                G.remove_edge(x, y)
                swapcount += 1
          elif (x,y) not in G.edges() and (y,x) in G.edges():
            if (u not in G.successors(y)) and (x not in G.successors(v)):  # don't create parallel edges
                G.add_edge(y, u)
                G.add_edge(v, x)
                G.remove_edge(v, u)
                G.remove_edge(y, x)
                swapcount += 1
          elif (x,y) in G.edges() and (y,x) in G.edges():
            rand=random.random()
            if rand>0.5:
              if (u not in G.successors(x)) and (y not in G.successors(v)):  # don't create parallel edges
                G.add_edge(x, u)
                G.add_edge(v, y)
                G.remove_edge(v, u)
                G.remove_edge(x, y)
                swapcount += 1
            else:
              if (u not in G.successors(y)) and (x not in G.successors(v)):  # don't create parallel edges
                G.add_edge(y, u)
                G.add_edge(v, x)
                G.remove_edge(v, u)
                G.remove_edge(y, x)
                swapcount += 1
        elif (u,v) in G.edges() and (v,u) in G.edges():
          rand = random.random()
          if rand>0.5:
            if (x,y) in G.edges() and (y,x) not in G.edges():
              if (y not in G.successors(u)) and (v not in G.successors(x)):  # don't create parallel edges
                  G.add_edge(u, y)
                  G.add_edge(x, v)
                  G.remove_edge(u, v)
                  G.remove_edge(x, y)
                  swapcount += 1
            elif (x,y) not in G.edges() and (y,x) in G.edges():
              if (x not in G.successors(u)) and (v not in G.successors(y)):  # don't create parallel edges
                  G.add_edge(u, x)
                  G.add_edge(y, v)
                  G.remove_edge(u, v)
                  G.remove_edge(y, x)
                  swapcount += 1
            elif (x,y) in G.edges() and (y,x) in G.edges():
              rand=random.random()
              if rand>0.5:
                if (y not in G.successors(u)) and (v not in G.successors(x)):  # don't create parallel edges
                  G.add_edge(u, y)
                  G.add_edge(x, v)
                  G.remove_edge(u, v)
                  G.remove_edge(x, y)
                  swapcount += 1
              else:
                if (x not in G.successors(u)) and (v not in G.successors(y)):  # don't create parallel edges
                  G.add_edge(u, x)
                  G.add_edge(y, v)
                  G.remove_edge(u, v)
                  G.remove_edge(y, x)
                  swapcount += 1
          else:
            if (x,y) in G.edges() and (y,x) not in G.edges():
              if (u not in G.successors(x)) and (y not in G.successors(v)):  # don't create parallel edges
                  G.add_edge(x, u)
                  G.add_edge(v, y)
                  G.remove_edge(v, u)
                  G.remove_edge(x, y)
                  swapcount += 1
            elif (x,y) not in G.edges() and (y,x) in G.edges():
              if (u not in G.successors(y)) and (x not in G.successors(v)):  # don't create parallel edges
                  G.add_edge(y, u)
                  G.add_edge(v, x)
                  G.remove_edge(v, u)
                  G.remove_edge(y, x)
                  swapcount += 1
            elif (x,y) in G.edges() and (y,x) in G.edges():
              rand=random.random()
              if rand>0.5:
                if (u not in G.successors(x)) and (y not in G.successors(v)):  # don't create parallel edges
                  G.add_edge(x, u)
                  G.add_edge(v, y)
                  G.remove_edge(v, u)
                  G.remove_edge(x, y)
                  swapcount += 1
              else:
                if (u not in G.successors(y)) and (x not in G.successors(v)):  # don't create parallel edges
                  G.add_edge(y, u)
                  G.add_edge(v, x)
                  G.remove_edge(v, u)
                  G.remove_edge(y, x)
                  swapcount += 1
        if n >= max_tries:
            e = (
                f"Maximum number of swap attempts ({n}) exceeded "
                f"before desired swaps achieved ({nswap})."
            )
            return("graph too small")
        n += 1
    return G
def connected_component_subgraphs(G):
    G=copy.deepcopy(G)
    G=G.to_undirected()
    for c in nx.connected_components(G):
        yield G.subgraph(c)

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
  G.add_edge(4,5)
  G.add_edge(4,6)
  G.add_edge(4,7)
  G.add_edge(4,8)
  G.add_edge(8,0)
  return G
def GRN_seed_graph_dmc(steps,qMod,qCon):
  G=GRN_seed_graph()
  G=dmc_single_lineage(G,steps,qCon,qMod,iteration=0)
  isoNodes=[i for i in list(G.nodes) if len(list(G.predecessors(i)))==0 and len(list(G.successors(i)))==0]
  iteration=steps
  while len(isoNodes)!=0:
    replace=len(isoNodes)
    G.remove_nodes_from(isoNodes)
    iteration=iteration+replace
    G=dmc_single_lineage(G,replace,qCon,qMod,iteration=iteration)
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
    iteration=iteration+replace
    G=ped_pea_single_lineage(G,steps,r,q,iteration=iteration)
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
def gnp_random_graph(n, p, seed=None, directed=False):
    """Returns a $G_{n,p}$ random graph, also known as an Erdős-Rényi graph
    or a binomial graph.

    The $G_{n,p}$ model chooses each of the possible edges with probability $p$.

    Parameters
    ----------
    n : int
        The number of nodes.
    p : float
        Probability for edge creation.
    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.
    directed : bool, optional (default=False)
        If True, this function returns a directed graph.

    See Also
    --------
    fast_gnp_random_graph

    Notes
    -----
    This algorithm [2]_ runs in $O(n^2)$ time.  For sparse graphs (that is, for
    small values of $p$), :func:`fast_gnp_random_graph` is a faster algorithm.

    :func:`binomial_graph` and :func:`erdos_renyi_graph` are
    aliases for :func:`gnp_random_graph`.

    >>> nx.binomial_graph is nx.gnp_random_graph
    True
    >>> nx.erdos_renyi_graph is nx.gnp_random_graph
    True

    References
    ----------
    .. [1] P. Erdős and A. Rényi, On Random Graphs, Publ. Math. 6, 290 (1959).
    .. [2] E. N. Gilbert, Random Graphs, Ann. Math. Stat., 30, 1141 (1959).
    """
    if directed:
        edges = list(itertools.permutations(range(n), 2))
        #for i in range(n):
        #  edges.append((i,i))
          
        G = nx.DiGraph()
    else:
        edges = itertools.combinations(range(n), 2)
        G = nx.Graph()
    G.add_nodes_from(range(n))
    if p <= 0:
        return G
    if p >= 1:
        return complete_graph(n, create_using=G)

    for e in edges:
        rando=np.random.rand(1)
        if rando < p:
            G.add_edge(*e)
    return G

def kperm(k, n, exclude=[], init=None):
	if init==None: 
		init = k
	if k == init-2:
		for i in range(0, n):
			if i not in exclude:
				yield (i,)
	for firstnum in range(0,n):
		if firstnum not in exclude:
			for x in kperm(k-1, n, exclude=exclude+[firstnum], init=init):
				yield tuple((firstnum,) + x) 
def cytoscape_graph(data, attrs=None, name="name", ident="id",value="id"):
    """
    Create a NetworkX graph from a dictionary in cytoscape JSON format.

    Parameters
    ----------
    data : dict
        A dictionary of data conforming to cytoscape JSON format.
    attrs : dict or None (default=None)
        A dictionary containing the keys 'name' and 'ident' which are mapped to
        the 'name' and 'id' node elements in cyjs format. All other keys are
        ignored. Default is `None` which results in the default mapping
        ``dict(name="name", ident="id")``.

        .. deprecated:: 2.6

           The `attrs` keyword argument will be replaced with `name` and
           `ident` in networkx 3.0

    name : string
        A string which is mapped to the 'name' node element in cyjs format.
        Must not have the same value as `ident`.
    ident : string
        A string which is mapped to the 'id' node element in cyjs format.
        Must not have the same value as `name`.

    Returns
    -------
    graph : a NetworkX graph instance
        The `graph` can be an instance of `Graph`, `DiGraph`, `MultiGraph`, or
        `MultiDiGraph` depending on the input data.

    Raises
    ------
    NetworkXError
        If the `name` and `ident` attributes are identical.

    See Also
    --------
    cytoscape_data: convert a NetworkX graph to a dict in cyjs format

    References
    ----------
    .. [1] Cytoscape user's manual:
       http://manual.cytoscape.org/en/stable/index.html

    Examples
    --------
    >>> data_dict = {
    ...     'data': [],
    ...     'directed': False,
    ...     'multigraph': False,
    ...     'elements': {'nodes': [{'data': {'id': '0', 'value': 0, 'name': '0'}},
    ...       {'data': {'id': '1', 'value': 1, 'name': '1'}}],
    ...      'edges': [{'data': {'source': 0, 'target': 1}}]}
    ... }
    >>> G = nx.cytoscape_graph(data_dict)
    >>> G.name
    ''
    >>> G.nodes()
    NodeView((0, 1))
    >>> G.nodes(data=True)[0]
    {'id': '0', 'value': 0, 'name': '0'}
    >>> G.edges(data=True)
    EdgeDataView([(0, 1, {'source': 0, 'target': 1})])
    """
    # ------ TODO: Remove between the lines in 3.0 ----- #
    if attrs is not None:
        import warnings

        msg = (
            "\nThe `attrs` keyword argument of cytoscape_data is deprecated\n"
            "and will be removed in networkx 3.0.\n"
            "It is replaced with explicit `name` and `ident` keyword\n"
            "arguments.\n"
            "To make this warning go away and ensure usage is forward\n"
            "compatible, replace `attrs` with `name` and `ident`,\n"
            "for example:\n\n"
            "   >>> cytoscape_data(G, attrs={'name': 'foo', 'ident': 'bar'})\n\n"
            "should instead be written as\n\n"
            "   >>> cytoscape_data(G, name='foo', ident='bar')\n\n"
            "The default values of 'name' and 'id' will not change."
        )
        warnings.warn(msg, DeprecationWarning, stacklevel=2)

        name = attrs["name"]
        ident = attrs["ident"]
    # -------------------------------------------------- #

    if name == ident:
        raise nx.NetworkXError("name and ident must be different.")

    multigraph = data.get("multigraph")
    directed = data.get("directed")
    if multigraph:
        graph = nx.MultiGraph()
    else:
        graph = nx.Graph()
    if directed:
        graph = graph.to_directed()
    graph.graph = dict(data.get("data"))
    for d in data["elements"]["nodes"]:
        node_data = d["data"].copy()
        node = d["data"][value]

        if d["data"].get(name):
            node_data[name] = d["data"].get(name)
        if d["data"].get(ident):
            node_data[ident] = d["data"].get(ident)

        graph.add_node(node)
        graph.nodes[node].update(node_data)

    for d in data["elements"]["edges"]:
        edge_data = d["data"].copy()
        sour = d["data"]["source"]
        targ = d["data"]["target"]
        if multigraph:
            key = d["data"].get("key", 0)
            graph.add_edge(sour, targ, key=key)
            graph.edges[sour, targ, key].update(edge_data)
        else:
            graph.add_edge(sour, targ)
            graph.edges[sour, targ].update(edge_data)
    return graph
#-----------------------------------------------
#Deprecated functions
'''def NF(G1,G2,alpha=32,beta=0.8):
  aligned=[]
  alignedVert1=[]
  alignedVert2=[]
  maxScore=0
  pairingDictParents=dict()
  pairingDictChildren=dict()
  for x in list(G1.nodes()):
      
      PA=list(G1.predecessors(x))
      
      CA=list(G1.successors(x))
      
      for y in list(G2.nodes()):
        
        PB=list(G2.predecessors(y))
        
        CB=list(G2.successors(y))
        pairingDictParents[(x,y)]=NF_summer_alt(G1,G2,PA,PB)
      
        pairingDictChildren[(x,y)]=NF_summer_alt(G1,G2,CA,CB)
       
        #print(x,y,"prepped")

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
          
          score = NF_scorer_alt(x,y,G1,G2,aligned,alpha,beta,pairingDictParents,pairingDictChildren)
          
          if score>maxScore:
            maxScore=score
            maxX=x
            maxY=y
    aligned.append((maxX,maxY))
    ###print(maxX,maxY)
    alignedVert1.append(maxX)
    alignedVert2.append(maxY)
  return aligned,alignedVert1

def NF_many_to_one(G1,G2,alpha=32,beta=0.8):
  ###print("NF stRT")
  aligned=[]
  alignedVert1=[]
  alignedVert2=[]
  maxScore=0
  ###print("NF while stRT")
  while len(aligned) <min(len(list(G1.nodes())),len(list(G2.nodes()))):
    
    maxScore=0
    score=0
    for x in list(G1.nodes()):
      ###print("find neighbours 1 start")
      PA=list(G1.predecessors(x))
      
      CA=list(G1.successors(x))
      ###print("find neighbours 1 end")
      for y in list(G2.nodes()):
        ###print("find neighbours 2 start")
        PB=list(G2.predecessors(y))
        
        CB=list(G2.successors(y))
        ###print("find neighbours 1 end")
        if (x,y) not in aligned and x not in alignedVert1:
          
          score = NF_scorer(x,y,G1,G2,PA,PB,CA,CB,aligned,alpha,beta)
          #if(x==y):
            ###print("true pair",score,x,y)
          #if score<maxScore:
          #  ##print("lower",score,x,y)
          #if score==maxScore:
            ###print("equal",score,x,y)
          if score>maxScore:
            ###print("greater",score,x,y)
            maxScore=score
            maxX=x
            maxY=y
    aligned.append((maxX,maxY))
    ###print(maxX,maxY)
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
      ###print("find neighbours 1 start")
      PA=list(G1.predecessors(x))
      
      CA=list(G1.successors(x))
      ###print("find neighbours 1 end")
      for y in list(G2.nodes()):
        ###print("find neighbours 2 start")
        PB=list(G2.predecessors(y))
        
        CB=list(G2.successors(y))
        ###print("find neighbours 1 end")
        
        if (x,y) not in aligned:
          
          score = NF_scorer(x,y,G1,G2,PA,PB,CA,CB,aligned,alpha,beta)
          
          if score>maxScore:
            ###print(score)
            maxScore=score
            maxX=x
            maxY=y
    aligned.append((maxX,maxY))
    ###print(maxX,maxY)
    alignedVert1.add(maxX)
    alignedVert2.add(maxY)
    if maxScore <gamma:
      scoreLimit=True
  return aligned,alignedVert1
'''
'''def NF_permlist_defunct(PA,PB):
  
  m=max(len(PA),len(PB))
  n=min(len(PA),len(PB))
  if m ==0 or n ==0:
    return []
  #print("pre list make")
  if len(PA)>len(PB):
    permList=[list(j) for i in itertools.combinations(PA,len(PB)-1) for j in itertools.permutations(list(i))]
    #print("post list make")
    #permList=set(permList)
    ##print("perm",permList)
    return permList
  else:
    permList = [list(j) for i in itertools.combinations(PB,len(PA)-1) for j in itertools.permutations(list(i))]
    #print("post list make")
    #permList=set(permList)
    ##print("perm",permList)
    return permList'''

  
'''def NF_scorer(x,y,G1,G2,PA,PB,CA,CB,aligned,alpha,beta):
  ###print("scorer stRT")
  minListOut=NF_summer_alt(G1,G2,PA,PB)
  minListIn=NF_summer_alt(G1,G2,CA,CB)
  productOut=1
  productIn=1
  
  for i in minListOut:
    lister=list(i)
    productOut = productOut*NF_delta(i[0],i[1],G1,G2,aligned,alpha)
  
 
  for i in minListIn:
    lister=list(i)
    productIn = productIn*NF_delta(i[0],i[1],G1,G2,aligned,alpha)
  ###print("scorer end")
  return beta**(np.abs(G1.out_degree(x)-G2.out_degree(y)))*productOut + beta**(np.abs(G1.in_degree(x)-G2.in_degree(y)))*productIn


'''
'''def NF_summer_alt_alt(G1,G2,PA,PB):
  ##print("tuplelist stRT")
  tempTupleList=[]
  minList=[]
  
  iMinScore=dict()
  iMinMatch=dict()
  scoreDict=dict()
  matchDict=dict()
  alreadyMatched=dict()
  minSummand=10000000
  pMin = min(len(PA),len(PB))
  pMax=max(len(PA),len(PB))
  if len(PA)==pMin:
    minGraph=0
  else:
    minGraph=1
  #print("tuple pre list make")
  permList=NF_permlist(PA,PB)
  if permList==[]:
    return permList
  if minGraph==0:
    for i in list(PA):
      scoreDict[i]=dict()
      matchDict[i]=dict()
      for j in (PB):
        iScore=np.abs(G1.out_degree(lister[0])-G2.out_degree(lister[1]))+np.abs(G1.in_degree(lister[0])-G2.in_degree(lister[1]))
        scoreDict[i][j]=iScore
        matchDict[i][j]=j
        if iScore < iMinScore:
          iMinScore[i] = iScore
          iMinMatch[i]=j
      if iMinMatch[i] not in alreadyMatched:
        alreadyMatched[iMinMatch[i]]= i
      else:
        oldMatcher=alreadyMatched[iMinMatch[i]]
        oldScores = list(scoreDict[oldMatcher].values()).sort()
        newScores= list(scoreDict[i].values()).sort()


        
      
      lister=[i,iMinMatch[i]]
      tempTupleList.append(lister)
      
      for i in range(0,len(l)):
        lister=[PA[i],l[i]]
        tempTupleList.append(lister)
        summand = summand + np.abs(G1.out_degree(lister[0])-G2.out_degree(lister[1]))+np.abs(G1.in_degree(lister[0])-G2.in_degree(lister[1]))
      if summand<minSummand:
        ##print(summand)
        minSummand=summand
        minList=tempTupleList
      if minSummand ==0:
        ##print("hi")
        break
      
  if minGraph==1:
    while True:
      try:
        l= next(permList)
      except StopIteration:
        break
      l=list(l)
      summand=0
      tempTupleList=[]
      for i in range(0,len(l)):
        lister=[l[i],PB[i]]
        tempTupleList.append(lister)
        summand = summand + np.abs(G1.out_degree(lister[0])-G2.out_degree(lister[1]))+np.abs(G1.in_degree(lister[0])-G2.in_degree(lister[1]))
      if summand<minSummand:
        ##print(summand)
        minSummand=summand
        minList=tempTupleList
      if minSummand ==0:
        ##print("hi")
        break
  ##print("tuplelist end")
  return minList

def NF_summer_alt(G1,G2,PA,PB):
  ##print("tuplelist stRT")
  tempTupleList=[]
  minList=[]
  minSummand=10000000
  pMin = min(len(PA),len(PB))
  pMax=max(len(PA),len(PB))
  if len(PA)==pMin:
    minGraph=0
  else:
    minGraph=1
  #print("tuple pre list make")
  permList=NF_permlist(PA,PB)
  if permList==[]:
    return permList
  if minGraph==0:
    while True:
      try:
        l= next(permList)
      except StopIteration:
        break
      l=list(l)
      summand=0
      tempTupleList=[]
      for i in range(0,len(l)):
        lister=[PA[i],l[i]]
        tempTupleList.append(lister)
        summand = summand + np.abs(G1.out_degree(lister[0])-G2.out_degree(lister[1]))+np.abs(G1.in_degree(lister[0])-G2.in_degree(lister[1]))
      if summand<minSummand:
        ##print(summand)
        minSummand=summand
        minList=tempTupleList
      if minSummand ==0:
        ##print("hi")
        break
      
  if minGraph==1:
    while True:
      try:
        l= next(permList)
      except StopIteration:
        break
      l=list(l)
      summand=0
      tempTupleList=[]
      for i in range(0,len(l)):
        lister=[l[i],PB[i]]
        tempTupleList.append(lister)
        summand = summand + np.abs(G1.out_degree(lister[0])-G2.out_degree(lister[1]))+np.abs(G1.in_degree(lister[0])-G2.in_degree(lister[1]))
      if summand<minSummand:
        ##print(summand)
        minSummand=summand
        minList=tempTupleList
      if minSummand ==0:
        ##print("hi")
        break
  ##print("tuplelist end")
  return minList
#Here PA and PB are the sets of parents (children) of the nodes a in graph 1 and b in graph 2 respectively
# G1 and G2 and the two graphs to be aligned
#Returns the 'optimal matching' by finding the set of tuples (a_i,b_j) that minimise the sum given
# in the Node Finger##printing paper
def NF_summer(G1,G2,PA,PB):
  ###print("summer stRT")
  tupleListList=NF_tupleList(PA,PB)
  summand=0
  minList=[]
  minSummand=10000000
  ###print("summer for start")
  ###print("tuplelist len",len(tupleListList))
  for l in tupleListList:
    
    summand=0
    for i in l:
      lister=list(i)
      summand = summand + np.abs(G1.out_degree(lister[0])-G2.out_degree(lister[1]))+np.abs(G1.in_degree(lister[0])-G2.in_degree(lister[1]))
    if summand<minSummand:
      ###print(summand)
      minSummand=summand
      minList=l
    if minSummand ==0:
      ##print("hi")
      break
      
  ###print("summer end")
  return minList'''

'''def ancestor_finder_with_alignment(G1,G2,qMod,qCon,tolerance=0.05,weight=0.5):
  print("ancestor finder")
  #G1=label_conserver(G1)
  #G2=label_conserver(G2)
  G1=nx.convert_node_labels_to_integers(G1)
  G2=nx.convert_node_labels_to_integers(G2)
  #alignVec,mapped=NF(G1,G2,32,0.8)
  #mapping = dict(alignVec)
  #print(mapped)
  #G1_mapped=nx.induced_subgraph(G1,list(mapped))
  #G1_mapped=nx.relabel_nodes(G1_mapped,mapping)
  
  graphPair=(G1,G2,-1,0)
  signal =True
  maxLikelihood = -1
  maxLikelihoodOld=-1
  maxS3=-1
  maxS3Old=-1
  maxScore=-1
  maxScoreOld=-1
  maxScoringGraphList=[]
  maxScoreList=[]
  while signal:
    #print("signal")
    alignedPairs=dict()
    maxLikelihood = -1
    pairList=[]
    G1=graphPair[0]
    G2=graphPair[1]
    G1=nx.convert_node_labels_to_integers(G1)
    G2=nx.convert_node_labels_to_integers(G2)
    for i in list(G1.nodes):
      for j in list(G1.nodes):
        if j!=i:
          #check the S3 score for the mapped graph pair
          G1_temp=node_merger(G1,i,j)
          alignVec,mapped=NF(G1_temp,G2,32,0.8)
          mapping = dict(alignVec)
          G1_mapped=nx.induced_subgraph(G1_temp,list(mapped))
          G1_mapped=nx.relabel_nodes(G1_mapped,mapping)
          #s3Temp= s3_score(G1_mapped,G2)
          s3Temp= normalised_ec_score(G1_mapped,G2)
          
          #check the likelihood function for the pair 
          
          tempLikelihood = ancestral_likelihood_dmc(G1,i,j,qMod,qCon)
          
          
          #if the score is better than the previous best, update the best
          #tempScore=tempLikelihood+s3Temp
          
          tempScore=weight*tempLikelihood+(1-weight)*s3Temp+1-len(G2.nodes)/len(G1.nodes)
          
          if tempScore>=maxScore-tolerance*maxScore:
            
            pairList.append((i,j,1))
            alignedPairs[(i,j,1)]=(G1_mapped,G2,tempScore,1)
            #print(i,j,1,tempScore)

    for i in list(G2.nodes):  
      for j in list(G2.nodes):
        if j!=i:
          #check the S3 score for the mapped graph pair
          G2_temp=node_merger(G2,i,j)
          alignVec,mapped=NF(G2_temp,G1,32,0.8)
          mapping = dict(alignVec)
          G2_mapped=nx.induced_subgraph(G2_temp,list(mapped))
          G2_mapped=nx.relabel_nodes(G2_mapped,mapping)
          #s3Temp= s3_score(G2_mapped,G1)
          s3Temp= normalised_ec_score(G2_mapped,G1)

          #check the likelihood function for the pair 
          tempLikelihood = ancestral_likelihood_dmc(G2,i,j,qMod,qCon)
          
          tempScore=weight*tempLikelihood+(1-weight)*s3Temp+1-len(G1.nodes)/len(G2.nodes)
          
          #if the score is better than the previous best, update the best
          #print(s3Temp,maxS3)
          if tempScore>=maxScore-tolerance*maxScore:
              
              pairList.append((i,j,2))
              alignedPairs[(i,j,2)]=(G1,G2_mapped,tempScore,2)
    highScoreList=[]
    if pairList==[]:
      signal=False
      print("pairlist empty")
      
    else:
      maxScoreOfList=0
      
      print("pairlist length",len(pairList))
      for i in pairList:
        
        graphPair=alignedPairs[i]
        tempScore=graphPair[2]
        if tempScore>=maxScoreOfList:
          maxScoreOfList=tempScore
          
      for i in pairList:
        
        graphPair=alignedPairs[i]
        tempScore=graphPair[2]
        if tempScore>=maxScoreOfList:
          
          highScoreList.append(graphPair)
      rando =np.random.random()
      rando = int(np.round(rando*len(highScoreList)))
      graphPair=highScoreList[rando-1]
      
      maxScore=maxScoreOfList
      print(maxScore)
  return graphPair[0],graphPair[1]
  '''

'''
def ancestor_finder_without_alignment(G1,G2,qMod,qCon,tolerance=0.05,weight=0.5):
  print("ancestor finder")
  
  
  graphPair=(G1,G2,-1,0)
  signal =True
  maxLikelihood = -1
  maxLikelihoodOld=-1
  maxS3=-1
  maxS3Old=-1
  maxScore=-1
  maxScoreOld=-1
  maxScoringGraphList=[]
  maxScoreList=[]
  while signal:
    #print("signal")
    alignedPairs=dict()
    maxLikelihood = -1
    pairList=[]
    G1=graphPair[0]
    G2=graphPair[1]
    
    for i in list(G1.nodes):
      for j in list(G1.nodes):
        if j!=i:
          #check the S3 score for the mapped graph pair
          G1_temp=node_merger(G1,i,j)
          
          s3Temp= normalised_ec_score(G1_temp,G2)
          
          #check the likelihood function for the pair 
          
          tempLikelihood = ancestral_likelihood_dmc(G1,i,j,qMod,qCon)
          
          
          #if the score is better than the previous best, update the best
          #tempScore=tempLikelihood+s3Temp
          
          tempScore=weight*tempLikelihood+(1-weight)*s3Temp+1-len(G2.nodes)/len(G1.nodes)
          
          if tempScore>=maxScore-tolerance*maxScore:
            
            pairList.append((i,j,1))
            alignedPairs[(i,j,1)]=(G1_temp,G2,tempScore,1)
            #print(i,j,1,tempScore)

    for i in list(G2.nodes):  
      for j in list(G2.nodes):
        if j!=i:
          #check the S3 score for the mapped graph pair
          G2_temp=node_merger(G2,i,j)
          
          s3Temp= normalised_ec_score(G2_temp,G1)

          #check the likelihood function for the pair 
          tempLikelihood = ancestral_likelihood_dmc(G2,i,j,qMod,qCon)
          
          tempScore=weight*tempLikelihood+(1-weight)*s3Temp+1-len(G1.nodes)/len(G2.nodes)
          
          #if the score is better than the previous best, update the best
          #print(s3Temp,maxS3)
          if tempScore>=maxScore-tolerance*maxScore:
              
              pairList.append((i,j,2))
              alignedPairs[(i,j,2)]=(G1,G2_temp,tempScore,2)
    highScoreList=[]
    if pairList==[]:
      signal=False
      print("pairlist empty")
      
    else:
      maxScoreOfList=0
      
      print("pairlist length",len(pairList))
      for i in pairList:
        
        graphPair=alignedPairs[i]
        tempScore=graphPair[2]
        if tempScore>=maxScoreOfList:
          maxScoreOfList=tempScore
          
      for i in pairList:
        
        graphPair=alignedPairs[i]
        tempScore=graphPair[2]
        if tempScore>=maxScoreOfList:
          
          highScoreList.append(graphPair)
      rando =np.random.random()
      rando = int(np.round(rando*len(highScoreList)))
      graphPair=highScoreList[rando-1]
      
      maxScore=maxScoreOfList
      print(maxScore)
  return graphPair[0],graphPair[1]
'''