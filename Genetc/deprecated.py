  
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
from utilities import *
from alignment import *
from duplication import *
#-----------------------------------------------
#Deprecated functions
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

def NF_permlist_defunct(PA,PB):
  
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
    return permList

  
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

def ancestor_finder_with_alignment(G1,G2,qMod,qCon,tolerance=0.05,weight=0.5):
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




def NF_gene_family_2(G1,G2,P1,P2,alpha=32,beta=0.8,thresh=5):
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
  zScore=0

  while zScore<0.00001:
    maxScore=0
    score=0
    scoreVec=[]
    for fam in P1:
      for x in P1[fam]:
        for y in P2[fam]:
          if (x,y) not in aligned and x not in alignedVert1 and y not in alignedVert2:
            
            score = NF_scorer(x,y,G1,G2,aligned,alpha,beta,pairingDictParents,pairingDictChildren)
            scoreVec.append(score)
            if score>maxScore:
              maxScore=score
              maxX=x
              maxY=y
    #print(maxX,maxY,maxScore)
    
    
    paramss=ex.fit(scoreVec)
    
    #plt.figure()
    #bins = np.arange(0, 10000, 1) # fixed bin size
    #plt.hist(scoreVec, bins=bins, alpha=0.5)
    scoreMean=ex.mean(loc=paramss[0],scale=paramss[1])
    scoreStd=ex.std(loc=paramss[0],scale=paramss[1])
    zScore=(maxScore-scoreMean)/scoreStd
    #print(zScore)
    zScore=ex.pdf(maxScore,loc=paramss[0],scale=paramss[1])
    print(ex.pdf(maxScore,loc=paramss[0],scale=paramss[1]))
    if zScore<0.00001:
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
 
  while len(aligned) <len(list(G1.nodes())):
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


def NF_many_to_many(G1,G2,alpha=32,beta=0.8):
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
 
  while len(aligned) <len(list(G1.nodes())):
    maxScore=0
    score=0
    for x in list(G1.nodes()):
      for y in list(G2.nodes()):
        if (x,y) not in aligned:
          
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

def GFAF_ped_pea_noalign(G1,G2,r,q,tolerance=0,toleranceEC=0,true_labels=False):
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
        bestScoreVec=[]
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
              G1_temp=node_merger(G1,i[0],i[1],self_loops=True)
            else:
              G1_temp=node_merger(G1,i[1],i[0],self_loops=True)
            G1_tempp=copy.deepcopy(G1_temp)
            tempScore= conserved_edges(G1_tempp,G2)
            print(i,tempScore)
            if tempScore>=prevScore:
              if tempScore>prevScore:
                bestScoreVec=[]
              print("chosen",i)
              graphPair=(G1_temp,G2,tempScore)
              prevScore=tempScore
              trigger=True
              chosenLikelihood=i[3]
              signal=True
              bestScoreVec.append(graphPair)
              
          if i[2]==2:
            if i[1] not in G1.nodes and i[0] in G1.nodes:
              G2_temp=node_merger(G2,i[0],i[1],self_loops=True)
            else:
              G2_temp=node_merger(G2,i[1],i[0],self_loops=True)
            G2_tempp=copy.deepcopy(G2_temp)
            tempScore= conserved_edges(G1,G2_tempp)
            print(i,tempScore)
            if tempScore>=prevScore:
              if tempScore>prevScore:
                bestScoreVec=[]
              print("chosen",i)
              graphPair=(G1,G2_temp,tempScore)
              prevScore=tempScore
              trigger=True
              chosenLikelihood=i[3]
              signal=True
              bestScoreVec.append(graphPair)
        
          
        print("out of the merge")
        if len(bestScoreVec)!=0:
          graphPair=random.choice(bestScoreVec)
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
        bestScoreVec=[]
        for i in pairPairList:
          count=count+1
          if count>len(pairPairList):
            countUp=True
            break
          if trigger and i[3]<chosenLikelihood:
            print('ay you broke my trigger')
            break
          if i[2]==1:
            if i[1] not in G2.nodes and i[0] in G2.nodes:
              G1_temp=node_merger(G1,i[0],i[1],self_loops=True)
            else:
              G1_temp=node_merger(G1,i[1],i[0],self_loops=True)
            G1_tempp=copy.deepcopy(G1_temp)
            tempScore= conserved_edges(G1_tempp,G2)
            print(i,tempScore)
            if tempScore>=prevScore:
              if tempScore>prevScore:
                bestScoreVec=[]
              print("chosen",i)
              graphPair=(G1_temp,G2,tempScore)
              prevScore=tempScore
              trigger=True
              chosenLikelihood=i[3]
              signal=True
              bestScoreVec.append(graphPair)
              
          if i[2]==2:
            if i[1] not in G1.nodes and i[0] in G1.nodes:
              G2_temp=node_merger(G2,i[0],i[1],self_loops=True)
            else:
              G2_temp=node_merger(G2,i[1],i[0],self_loops=True)
            G2_tempp=copy.deepcopy(G2_temp)
            tempScore= conserved_edges(G1,G2_tempp)
            print(i,tempScore)
            if tempScore>=prevScore:
              if tempScore>prevScore:
                bestScoreVec=[]
              print("chosen",i)
              graphPair=(G1,G2_temp,tempScore)
              prevScore=tempScore
              trigger=True
              chosenLikelihood=i[3]
              signal=True
              bestScoreVec.append(graphPair)
        
          
        print("out of the merge")
        if len(bestScoreVec)!=0:
          graphPair=random.choice(bestScoreVec)
        theGraphList.append(graphPair)
        print(graphPair,len(graphPair[0].nodes))  
    
      
    #graphPair=max(theNewGraphList,key=lambda x:x[2])
    print("the big winner score is",graphPair[2],"and its graphs with",len(graphPair[0].nodes),"nodes")
    return graphPair[0],graphPair[1]     
def GFAF_dmc_noalign(G1,G2,qMod,qCon,tolerance=0,toleranceEC=0):
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
        