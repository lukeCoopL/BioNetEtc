
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
from Genetc.utilities import *
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

def NF_gene_family(G1,G2,P1,P2,alpha=32,beta=0.8,thresh=0):
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
    print(maxScore)
    if maxScore>thresh:
      aligned.append((maxX,maxY))
      ###print(maxX,maxY)
      alignedVert1.append(maxX)
      alignedVert2.append(maxY)
  return aligned,alignedVert1

def NF(G1,G2,alpha=32,beta=0.8,NFType="one_to_one"):
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
  if NFType == 'one_to_one':

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
  if NFType=="many_to_one":
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
  if NFType=="many_to_many":
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
