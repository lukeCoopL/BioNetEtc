from re import S
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

def connected_component_subgraphs(G):
    G=copy.deepcopy(G)
    G=G.to_undirected()
    for c in nx.connected_components(G):
        yield G.subgraph(c)
            
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
        G_dup.add_edge(i,str(j)+"_"+str(iteration))
      
      for i in list(G.successors(j)):
         G_dup.add_edge(str(j)+"_"+str(iteration),i)
      
  else:
    for i in list(G.nodes()):
      for j in genes:
        if (i,j) in list(G.edges()):
          
          G_dup.add_edge(i,str(j)+"_c")
        if (j,i) in list(G.edges()):
            G_dup.add_edge(str(j)+"_c",i)
  return G_dup
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
  
#Here PA and PB are the sets of parents (children) of the nodes a in graph 1 and b in graph 2 respectively
#Let n=min(|PA|,|PB|). This function returns a list of all n-permutations of the 
# elements of the larger of PA and PB
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

def NF_summer_alt_alt_alt(G1,G2,PA,PB):
  #still not producing the same result as NF_summer why>????
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

def NF_summer_alt_alt(G1,G2,PA,PB):
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
  return minList
def NF_scorer_alt(x,y,G1,G2,aligned,alpha,beta,pairingDictParents,pairingDictChildren):
  ###print("scorer stRT")
  
  productOut=1
  productIn=1
  
  for i in pairingDictParents[(x,y)]:
    
    productOut = productOut*NF_delta(i[0],i[1],G1,G2,aligned,alpha)
  
 
  for i in pairingDictChildren[(x,y)]:
    
    productIn = productIn*NF_delta(i[0],i[1],G1,G2,aligned,alpha)
  ###print("scorer end")
  return beta**(np.abs(G1.out_degree(x)-G2.out_degree(y)))*productOut + beta**(np.abs(G1.in_degree(x)-G2.in_degree(y)))*productIn

#Returns the score function for a pair of nodes (x,y), x in graph 1 and y in graph 2
def NF_scorer(x,y,G1,G2,PA,PB,CA,CB,aligned,alpha,beta):
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

#Determines the value of the delta function for a pair of nodes (a,b), a in graph 1, b in graph 2
def NF_delta(a,b,G1,G2,aligned,alpha):
  ###print("delta stRT")
  if (a,b) in aligned or max(G1.out_degree(a),G2.out_degree(b))==0 or max(G1.in_degree(a),G2.in_degree(b))==0:
    ###print("delta end")
    return alpha
  else:
    ###print("delta end")
    return min(G1.out_degree(a),G2.out_degree(b))/max(G1.out_degree(a),G2.out_degree(b))+min(G1.in_degree(a),G2.in_degree(b))/max(G1.in_degree(a),G2.in_degree(b))

def NF_alt(G1,G2,alpha=32,beta=0.8):
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
        pairingDictParents[(x,y)]=NF_summer_alt_alt_alt(G1,G2,PA,PB)
        
        pairingDictChildren[(x,y)]=NF_summer_alt_alt_alt(G1,G2,CA,CB)
        
        #print(x,y,"prepped")
 
  while len(aligned) <min(len(list(G1.nodes())),len(list(G2.nodes()))):
    maxScore=0
    score=0
    for x in list(G1.nodes()):
      for y in list(G2.nodes()):
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
  return aligned,alignedVert1,pairingDictParents,pairingDictChildren

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
  
  children=copy.deepcopy(G.successors(dupNode))
  parents=copy.deepcopy(G.predecessors(dupNode))
  for i in parents:
    rando=np.random.rand(1)
    
    if rando <= qMod:
      rando=np.random.rand(1)
      
      if rando >0.5:
        G.remove_edge(i,str(dupNode)+"_"+str(iteration))
      else:
        G.remove_edge(i,dupNode)
  
  for i in children:
    rando=np.random.rand(1)
    if rando <= qMod:
      rando=np.random.rand(1)
      if rando >0.5:
        G.remove_edge(str(dupNode)+"_"+str(iteration),i)
      else:
        G.remove_edge(dupNode,i)
  rando=np.random.rand(1)
  if rando<=qCon:
    rando=np.random.rand(1)
    if rando>0.5:
      G.add_edge(dupNode,str(dupNode)+"_"+str(iteration))
    else:
      G.add_edge(str(dupNode)+"_"+str(iteration),dupNode)
  return G
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

def ancestral_likelihood_dmc(G,i,j,qMod,qCon):
  G=copy.deepcopy(G)
  prod=1
  anchorNeighbours = list(G.predecessors(i))+list(G.successors(i))
  duplicateNeighbours=list(G.predecessors(j))+list(G.successors(j))
  intersectingNeighbours = list(set(anchorNeighbours)&set(duplicateNeighbours))
  symmdiffNeighbours = list(set(anchorNeighbours)^set(duplicateNeighbours))
  for i in intersectingNeighbours:
    prod = prod*(1-qMod)
  for i in symmdiffNeighbours:
    prod = prod*qMod/2
  if (i,j) in list(G.edges):
    prod=prod*qCon/len(list(G.nodes))
  else:
    prod=prod*(1-qCon)/len(list(G.nodes))
  return prod
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
          
          if tempLikelihood>=maxLikelihood:
            
            pairList.append((i,j))
            
            maxLikelihood=tempLikelihood
    if len(pairList)==0:
      return G,arrival,anchor
    rando =np.random.random()
    rando = int(np.round(rando*len(pairList)))
    
    selectedNodes=pairList[rando-1]
    ##print("Merge nodes",selectedNodes)
    arrival[selectedNodes[1]]=len(V)
    anchor[selectedNodes[1]]=selectedNodes[0]
    anchorNeighbours = list(G.predecessors(selectedNodes[0]))+list(G.successors(selectedNodes[0]))
    duplicateNeighbours=list(G.predecessors(selectedNodes[1]))+list(G.successors(selectedNodes[1]))
    s = set(anchorNeighbours)
    anchorNeighbours = [x for x in duplicateNeighbours if x not in s]
    for i in anchorNeighbours:
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
def gene_family_partitioner(G):
  origin=""
  partition = dict()
  for i in list(G.nodes):
    origin=""
    s=str(i)[0]
    j=0
    while s!='_':
      origin=origin+s
      j=j+1
      if str(i)[0]!=str(i)[-1]:
        s=str(i)[j]
      else:
        s='_'
    if origin not in partition:
      partition[origin]=[]
    partition[origin].append(i)
  
  return partition
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

  


  

    
      

    
  