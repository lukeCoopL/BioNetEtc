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
from Genetc.alignment import *
from Genetc.duplication import *

#Utilities
#----------------------------------------------
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
def connected_component_subgraphs(G):
    G=copy.deepcopy(G)
    G=G.to_undirected()
    for c in nx.connected_components(G):
        yield G.subgraph(c)

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
def takeThird(elem):
  return elem[2]
def degree_sequence_plot(G):
  out_deg_dist=dict()
  in_deg_dist=dict()
  for i in G.nodes:
      in_deg_dist[i]=0
      out_deg_dist[i]=0
      for j in G.predecessors(i):
          in_deg_dist[i]=in_deg_dist[i]+1
      for j in G.successors(i):
          out_deg_dist[i]=out_deg_dist[i]+1

  inList=list(in_deg_dist.values())
  outList=list(out_deg_dist.values())
  inList=sorted(inList)
  outList=sorted(outList)
  
  bins = np.arange(0, max(outList), 1) # fixed bin size

  plt.hist(outList, bins=bins, alpha=0.5)

  plt.title('')
  plt.xlabel('Degree')
  plt.ylabel('Count')

  plt.show()
  plt.figure()

  bins = np.arange(0, max(inList), 1) # fixed bin size

  plt.hist(inList, bins=bins, alpha=0.5)

  plt.title('')
  plt.xlabel('Degree')
  plt.ylabel('Count')

  plt.show()
def loglog_degree_sequence_plot(G):
  out_deg_dist=dict()
  in_deg_dist=dict()
  for i in G.nodes:
      in_deg_dist[i]=0
      out_deg_dist[i]=0
      for j in G.predecessors(i):
          in_deg_dist[i]=in_deg_dist[i]+1
      for j in G.successors(i):
          out_deg_dist[i]=out_deg_dist[i]+1

  inList=list(in_deg_dist.values())
  outList=list(out_deg_dist.values())
  inList=sorted(inList)
  outList=sorted(outList)
  bins = np.arange(0, max(outList), 1) # fixed bin size

  hist, bins = np.histogram(outList, bins=bins)
  logbins = np.logspace(np.log10(bins[0]),np.log10(bins[-1]),len(bins))
  plt.hist(outList, bins=bins, alpha=0.5,log=True)
  plt.xscale('log')
  plt.title('')
  plt.xlabel('Degree')
  plt.ylabel('Count')

  plt.show()
  bins = np.arange(0, max(inList), 1) # fixed bin size
  plt.figure()
  hist, bins = np.histogram(inList, bins=bins)
  logbins = np.logspace(np.log10(bins[0]),np.log10(bins[-1]),len(bins))
  plt.hist(inList, bins=bins, alpha=0.5,log=True)
  plt.xscale('log')
  plt.title('')
  plt.xlabel('Degree')
  plt.ylabel('Count')

  plt.show()

def pair_loglog_degree_sequence_plot(G1,G2):
  for G in [G1,G2]:
    out_deg_dist=dict()
    in_deg_dist=dict()
    for i in G.nodes:
        
        in_deg_dist[i]=0.01
        out_deg_dist[i]=0.01
        for j in G.predecessors(i):
            in_deg_dist[i]=in_deg_dist[i]+1
        for j in G.successors(i):
            out_deg_dist[i]=out_deg_dist[i]+1

    inList=list(in_deg_dist.values())
    outList=list(out_deg_dist.values())
    inList=sorted(inList)
    outList=sorted(outList)
    if G==G2:
      print(inList,outList)
    bins = np.arange(0.01, max(outList), 1) # fixed bin size

    hist, bins = np.histogram(outList, bins=bins)
    logbins = np.logspace(np.log10(bins[0]),np.log10(bins[-1]),len(bins))
    plt.hist(outList, bins=bins, alpha=0.5,log=True)
    plt.xscale('log')
    plt.title('')
    plt.xlabel('Degree')
    plt.ylabel('Count')

    
  
  plt.figure()
  for G in [G1,G2]:
    out_deg_dist=dict()
    in_deg_dist=dict()
    for i in G.nodes:
        in_deg_dist[i]=0.01
        out_deg_dist[i]=0.01
        for j in G.predecessors(i):
            in_deg_dist[i]=in_deg_dist[i]+1
        for j in G.successors(i):
            out_deg_dist[i]=out_deg_dist[i]+1

    inList=list(in_deg_dist.values())
    outList=list(out_deg_dist.values())
    inList=sorted(inList)
    outList=sorted(outList)
    bins = np.arange(0.01, max(inList), 1) # fixed bin size
    hist, bins = np.histogram(inList, bins=bins)
    logbins = np.logspace(np.log10(bins[0]),np.log10(bins[-1]),len(bins))
    plt.hist(inList, bins=bins, alpha=0.5,log=True)
    plt.xscale('log')
    plt.title('')
    plt.xlabel('Degree')
    plt.ylabel('Count')

   