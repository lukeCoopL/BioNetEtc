from os import abort
from re import S
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from Genetc import gene_functions as gf
import scipy as sp
import copy
branchLength=50
algType="gene"
print("new")
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
for i in range(7):
    if i in leaves:
        leafGraphs[i]= nx.read_edgelist("test_datasets_ancestral/regulatory_test_ped_pea_r08/anc50_2cherry_branch"+str(branchLength)+"/LEAF"+str(i)+"_two_cherry_branchLength"+str(branchLength)+".txt",create_using=nx.DiGraph)
        #leafGraphs[i]= nx.read_edgelist("test_datasets_ancestral/code_quickrun/LEAF"+str(i)+"_two_cherry_branchLength"+str(branchLength)+".txt",create_using=nx.DiGraph)

    else:
        #internalGraphs[i]= nx.read_edgelist("test_datasets_ancestral/code_quickrun/INTERNAL"+str(i)+"_two_cherry_branchLength"+str(branchLength)+".txt",create_using=nx.DiGraph)

        internalGraphs[i]= nx.read_edgelist("test_datasets_ancestral/regulatory_test_ped_pea_r08/anc50_2cherry_branch"+str(branchLength)+"/INTERNAL"+str(i)+"_two_cherry_branchLength"+str(branchLength)+".txt",create_using=nx.DiGraph)
G_anc_dict=dict()
G_anc_dict[(2,3)]=internalGraphs[1]
G_anc_dict[(3,2)]=internalGraphs[1]
G_anc_dict[(5,6)]=internalGraphs[4]
G_anc_dict[(6,5)]=internalGraphs[4]

G_anc_dict[(2,5)]=internalGraphs[0]
G_anc_dict[(5,2)]=internalGraphs[0]
G_anc_dict[(6,2)]=internalGraphs[0]
G_anc_dict[(2,6)]=internalGraphs[0]
G_anc_dict[(3,6)]=internalGraphs[0]
G_anc_dict[(6,3)]=internalGraphs[0]
G_anc_dict[(3,5)]=internalGraphs[0]
G_anc_dict[(5,3)]=internalGraphs[0]

qMod=0.4
qCon=0.1
r=0.8
q=0.8
iterations=1

iterationVec=[i for i in range(iterations)]

S3_nf_vec=dict()
S3_dmc_vec=dict()
S3_orig_vec=dict()
EC_nf_vec=dict()
EC_dmc_vec=dict()
EC_orig_vec=dict()
ICS_nf_vec=dict()
ICS_dmc_vec=dict()
ICS_orig_vec=dict()
new_metric_nf_vec=dict()
new_metric_dmc_vec=dict()
new_metric_orig_vec=dict()
network_order_dmc=dict()
network_order_nf=dict()
network_order_orig=dict()
EC_find1_vec=dict()
ICS_find1_vec=dict()
S3_find1_vec=dict()
network_order_find1=dict()
EC_find2_vec=dict()
ICS_find2_vec=dict()
S3_find2_vec=dict()
network_order_find2=dict()
EC_find_int_vec=dict()
ICS_find_int_vec=dict()
S3_find_int_vec=dict()
network_order_find_int=dict()
EC_find_mapped_int_vec=dict()
ICS_find_mapped_int_vec=dict()
S3_find_mapped_int_vec=dict()
network_order_int_mapped=dict()
EC_findAl1_vec=dict()
ICS_findAl1_vec=dict()
S3_findAl1_vec=dict()
network_order_findAl1=dict()
EC_findAl2_vec=dict()
ICS_findAl2_vec=dict()
S3_findAl2_vec=dict()
network_order_findAl2=dict()
EC_findAl_int_vec=dict()
ICS_findAl_int_vec=dict()
S3_findAl_int_vec=dict()
network_order_findAl_int=dict()
EC_findAl_mapped_int_vec=dict()
ICS_findAl_mapped_int_vec=dict()
S3_findAl_mapped_int_vec=dict()
network_order_int_mapped=dict()
EC_true_unint_vec=dict()
ICS_true_unint_vec=dict()
S3_true_unint_vec=dict()
network_order_true_unint=dict()
EC_align_unint_vec=dict()
ICS_align_unint_vec=dict()
S3_align_unint_vec=dict()
network_order_align_unint=dict()
EC_unint_vec=dict()
ICS_unint_vec=dict()
S3_unint_vec=dict()
network_order_unint=dict()

conserved_dmc_vec=dict()
conserved_orig_vec=dict()
conserved_unint_vec=dict()
conserved_true_unint_vec=dict()
conserved_find_int_vec=dict()
extra_edges_dmc_vec=dict()
extra_edges_orig_vec=dict()
extra_edges_unint_vec=dict()
extra_edges_true_unint_vec=dict()
extra_edges_find_int_vec=dict()
missed_edges_dmc_vec=dict()
missed_edges_orig_vec=dict()
missed_edges_unint_vec=dict()
missed_edges_true_unint_vec=dict()
missed_edges_find_int_vec=dict()
print(leafGraphs)
for l in leafGraphs:
    for m in leafGraphs:
        if m>l:
            print(l,m)
            G1=leafGraphs[l]
            G2=leafGraphs[m]
            G_anc=G_anc_dict[(l,m)] 
            G_anc=gf.label_conserver(G_anc)
            print("g_anc true edge number",len(G_anc.edges()))  
            G1=gf.label_conserver(G1)
            G2=gf.label_conserver(G2)
            
            for k in range(0,iterations):
                print("qMod:",qMod,k)
                #True label My ancestral algorithm

                G1_orig=copy.deepcopy(G1)
                G2_orig=copy.deepcopy(G2)
                #mapper1=gf.gene_family_relabeller(G1_orig)
                #mapper2=gf.gene_family_relabeller(G2_orig)
                #G1_orig=nx.relabel_nodes(G1_orig,mapper1)
                #G2_orig=nx.relabel_nodes(G2_orig,mapper2)
                #initial alignment
                '''
                G1_labelless=nx.convert_node_labels_to_integers(G1_orig)
                G2_labelless=nx.convert_node_labels_to_integers(G2_orig)
                G1=nx.convert_node_labels_to_integers(G1_orig)
                G2=nx.convert_node_labels_to_integers(G2_orig)
                alignVec,mapped=gf.NF(G1,G2,32,0.8)
                mapping = dict(alignVec)
                
                G1_mapped=nx.induced_subgraph(G1,list(mapped))
                G1_mapped=nx.relabel_nodes(G1_mapped,mapping)
                '''

                t=1
                tEC=0
                if algType=="orig":
                    G_anc_find1,G_anc_find2=gf.ancestor_finder_without_alignment_the_seventh(G1_orig,G2_orig,qMod,qCon,tolerance=t,toleranceEC=tEC)
                if algType=="core":
                    G_anc_find1,G_anc_find2=gf.ancestor_finder_without_alignment_the_fifth(G1_orig,G2_orig,qMod,qCon,tolerance=t,toleranceEC=tEC)
                if algType=="branch":
                    G_anc_find1,G_anc_find2=gf.ancestor_finder_without_alignment_branching(G1_orig,G2_orig,qMod,qCon,tolerance=t,toleranceEC=tEC)
                if algType=="gene":
                    G_anc_find1,G_anc_find2=gf.ancestor_finder_without_alignment_gene_family_separate_ped_pea(G1_orig,G2_orig,r,q,tolerance=t,toleranceEC=tEC)
               
                else:
                    exit
                print("ancestor found")
                G_find_int_anc=nx.intersection(G_anc_find1,G_anc_find2)
                print("my alg edge number int",len(G_find_int_anc.edges()))
                G_true_unint=gf.graph_intersection_union(G_anc_find1,G_anc_find2)
                print("my alg edge number unint",len(G_true_unint.edges()))
                mapping=dict()
                for i in list(G_find_int_anc.nodes):
                    mapping[i]=G_anc_find1.nodes[i]['orig_label']
                for j in list(G_true_unint.nodes()):
                    if (G_true_unint.out_degree(j)==0 and G_true_unint.in_degree(j)==0):
                        G_true_unint.remove_node(j)
                    elif G_true_unint.out_degree(j)==1 and G_true_unint.in_degree(j)==1 and (j,j) in list(G_find_int_anc.edges):
                        G_true_unint.remove_node(j)
                G_true_unint=nx.relabel_nodes(G_true_unint,mapping)    
                EC_true_unint_vec[str((l,m))]=gf.normalised_ec_score(G_true_unint,G_anc)
                ICS_true_unint_vec[str((l,m))]=gf.ics_score(G_true_unint,G_anc)
                S3_true_unint_vec[str((l,m))]=gf.s3_score(G_true_unint,G_anc)
                conservedEdges=gf.conserved_edges(G_true_unint,G_anc)
                conserved_true_unint_vec[str((l,m))]=conservedEdges
                extra_edges_true_unint_vec[str((l,m))]=len(G_true_unint.edges)-conservedEdges
                missed_edges_true_unint_vec[str((l,m))]=len(G_anc.edges)-conservedEdges
                network_order_true_unint[str((l,m))]=len(G_true_unint.nodes)
                for j in list(G_find_int_anc.nodes()):
                    if (G_find_int_anc.out_degree(j)==0 and G_find_int_anc.in_degree(j)==0):
                        G_find_int_anc.remove_node(j)
                    elif G_find_int_anc.out_degree(j)==1 and G_find_int_anc.in_degree(j)==1 and (j,j) in list(G_find_int_anc.edges):
                        G_find_int_anc.remove_node(j)
                
                
                
                G_find_int_anc=nx.relabel_nodes(G_find_int_anc,mapping)

                EC_find_int_vec[str((l,m))]=gf.normalised_ec_score(G_find_int_anc,G_anc)
                ICS_find_int_vec[str((l,m))]=gf.ics_score(G_find_int_anc,G_anc)
                S3_find_int_vec[str((l,m))]=gf.s3_score(G_find_int_anc,G_anc)
                conservedEdges=gf.conserved_edges(G_find_int_anc,G_anc)
                conserved_find_int_vec[str((l,m))]=conservedEdges
                extra_edges_find_int_vec[str((l,m))]=len(G_find_int_anc.edges)-conservedEdges
                missed_edges_find_int_vec[str((l,m))]=len(G_anc.edges)-conservedEdges
                network_order_find_int[str((l,m))]=len(G_find_int_anc.nodes)

                for j in list(G_anc_find1.nodes()):
                    if (G_anc_find1.out_degree(j)==0 and G_anc_find1.in_degree(j)==0):
                        G_anc_find1.remove_node(j)
                    elif G_anc_find1.out_degree(j)==1 and G_anc_find1.in_degree(j)==1 and (j,j) in list(G_anc_find1.edges):
                        G_anc_find1.remove_node(j)

                mapping=dict()
                for i in list(G_anc_find1.nodes):
                    mapping[i]=G_anc_find1.nodes[i]['orig_label']
                G_find1_anc=nx.relabel_nodes(G_anc_find1,mapping)
                
                EC_find1_vec[str((l,m))]=gf.normalised_ec_score(G_find1_anc,G_anc)
                ICS_find1_vec[str((l,m))]=gf.ics_score(G_find1_anc,G_anc)
                S3_find1_vec[str((l,m))]=gf.s3_score(G_find1_anc,G_anc)
                network_order_find1[str((l,m))]=len(G_find1_anc.nodes)
                
                for j in list(G_anc_find2.nodes()):
                    if (G_anc_find2.out_degree(j)==0 and G_anc_find2.in_degree(j)==0):
                        G_anc_find2.remove_node(j)
                    elif G_anc_find2.out_degree(j)==1 and G_anc_find2.in_degree(j)==1 and (j,j) in list(G_anc_find2.edges):
                        G_anc_find2.remove_node(j)
                
                mapping=dict()
                for i in list(G_anc_find2.nodes):
                    mapping[i]=G_anc_find2.nodes[i]['orig_label']
                G_find2_anc=nx.relabel_nodes(G_anc_find2,mapping)

                EC_find2_vec[str((l,m))]=gf.normalised_ec_score(G_find2_anc,G_anc)
                ICS_find2_vec[str((l,m))]=gf.ics_score(G_find2_anc,G_anc)
                S3_find2_vec[str((l,m))]=gf.s3_score(G_find2_anc,G_anc)
                network_order_find2[str((l,m))]=len(G_find2_anc.nodes)

                nx.write_edgelist(G_find_int_anc,"test_datasets_ancestral/regulatory_test_ped_pea_r08/anc50_2cherry_branch"+str(branchLength)+"/myalg_int_anc_graphs_"+str(l)+"_"+str(m)+".txt")
                nx.write_edgelist(G_true_unint,"test_datasets_ancestral/regulatory_test_ped_pea_r08/anc50_2cherry_branch"+str(branchLength)+"/myalg_unint_anc_graphs_"+str(l)+"_"+str(m)+".txt")
                nx.write_edgelist(G_find1_anc,"test_datasets_ancestral/regulatory_test_ped_pea_r08/anc50_2cherry_branch"+str(branchLength)+"/myalgG"+str(l)+"_anc_graphs_"+str(l)+"_"+str(m)+".txt")
                nx.write_edgelist(G_find2_anc,"test_datasets_ancestral/regulatory_test_ped_pea_r08/anc50_2cherry_branch"+str(branchLength)+"/myalgG"+str(m)+"_anc_graphs_"+str(l)+"_"+str(m)+".txt")
                
                #Zhu-Nakleh Reconstruction
                G1_orig=copy.deepcopy(G1)
                G2_orig=copy.deepcopy(G2)
                G_dmc_anc=gf.dmc_anc_rec(G1_orig,G2_orig,qMod,qCon)
                for j in list(G_dmc_anc.nodes()):
                    if (G_dmc_anc.out_degree(j)==0 and G_dmc_anc.in_degree(j)==0):
                        G_dmc_anc.remove_node(j)
                    elif G_dmc_anc.out_degree(j)==1 and G_dmc_anc.in_degree(j)==1 and (j,j) in list(G_dmc_anc.edges):
                        G_dmc_anc.remove_node(j)
                
                EC_dmc_vec[str((l,m))]=gf.normalised_ec_score(G_dmc_anc,G_anc)
                ICS_dmc_vec[str((l,m))]=gf.ics_score(G_dmc_anc,G_anc)
                S3_dmc_vec[str((l,m))]=gf.s3_score(G_dmc_anc,G_anc)
                conservedEdges=gf.conserved_edges(G_dmc_anc,G_anc)
                conserved_dmc_vec[str((l,m))]=conservedEdges
                extra_edges_dmc_vec[str((l,m))]=len(G_dmc_anc.edges)-conservedEdges
                missed_edges_dmc_vec[str((l,m))]=len(G_anc.edges)-conservedEdges
                network_order_dmc[str((l,m))]=len(G_dmc_anc.nodes)
                nx.write_edgelist(G_dmc_anc,"test_datasets_ancestral/regulatory_test_ped_pea_r08/anc50_2cherry_branch"+str(branchLength)+"/dmc_anc_graphs_"+str(l)+"_"+str(m)+".txt")
                '''
                #NF Align then intersect
                G1_orig=copy.deepcopy(G1)
                G2_orig=copy.deepcopy(G2)

                G1_labelless=nx.convert_node_labels_to_integers(G1_orig)
                G2_labelless=nx.convert_node_labels_to_integers(G2_orig)
                alignVec,mapped=gf.NF(G1_labelless,G2_labelless,32,0.8)
                mapping = dict(alignVec)

                G1_mapped=nx.induced_subgraph(G1_labelless,list(mapped))
                G1_mapped=nx.relabel_nodes(G1_mapped,mapping)

                G_intersect=nx.intersection(G1_mapped,G2_labelless)

                for j in list(G_intersect.nodes()):
                    if (G_intersect.out_degree(j)==0 and G_intersect.in_degree(j)==0):
                        G_intersect.remove_node(j)
                    elif G_intersect.out_degree(j)==1 and G_intersect.in_degree(j)==1 and (j,j) in list(G_intersect.edges):
                        G_intersect.remove_node(j)

                mapping=dict()
                for i in list(G_intersect.nodes):
                    mapping[i]=G_intersect.nodes[i]['orig_label']
                G_nf_anc=nx.relabel_nodes(G_intersect,mapping)
                
                EC_nf_vec[str((l,m))]=gf.normalised_ec_score(G_nf_anc,G_anc)
                ICS_nf_vec[str((l,m))]=gf.ics_score(G_nf_anc,G_anc)
                S3_nf_vec[str((l,m))]=gf.s3_score(G_nf_anc,G_anc)
                network_order_nf[str((l,m))]=len(G_nf_anc.nodes)
                nx.write_edgelist(G_nf_anc,"test_datasets_ancestral/anc50_2cherry_branch"+str(branchLength)+"/nf_anc_graphs_"+str(l)+"_"+str(m)+".txt")
                '''
                #True Label Intersection
                G1_orig=copy.deepcopy(G1)
                G2_orig=copy.deepcopy(G2)
                #G1_orig=nx.relabel_nodes(G1_orig,mapper1)
                #G2_orig=nx.relabel_nodes(G2_orig,mapper2)
                
                G_intersect=nx.intersection(G1_orig,G2_orig)
                print("intersect edge number",len(G_intersect.edges()))

                for j in list(G_intersect.nodes()):
                    if (G_intersect.out_degree(j)==0 and G_intersect.in_degree(j)==0):
                        G_intersect.remove_node(j)
                    elif G_intersect.out_degree(j)==1 and G_intersect.in_degree(j)==1 and (j,j) in list(G_intersect.edges):
                        G_intersect.remove_node(j)
                mapping=dict()
                for i in list(G_intersect.nodes):
                    mapping[i]=G1_orig.nodes[i]['orig_label']
                G_intersect=nx.relabel_nodes(G_intersect,mapping)

                G_orig_anc=G_intersect
                
                EC_orig_vec[str((l,m))]=gf.normalised_ec_score(G_orig_anc,G_anc)
                ICS_orig_vec[str((l,m))]=gf.ics_score(G_orig_anc,G_anc)
                S3_orig_vec[str((l,m))]=gf.s3_score(G_orig_anc,G_anc)
                conservedEdges=gf.conserved_edges(G_orig_anc,G_anc)
                conserved_orig_vec[str((l,m))]=conservedEdges
                extra_edges_orig_vec[str((l,m))]=len(G_orig_anc.edges)-conservedEdges
                missed_edges_orig_vec[str((l,m))]=len(G_anc.edges)-conservedEdges
                network_order_orig[str((l,m))]=len(G_orig_anc.nodes)

                G_unint_anc=gf.graph_intersection_union(G1_orig,G2_orig)
                mapping=dict()
                for i in list(G_unint_anc.nodes):
                    mapping[i]=G1_orig.nodes[i]['orig_label']
                for j in list(G_unint_anc.nodes()):
                    if (G_unint_anc.out_degree(j)==0 and G_unint_anc.in_degree(j)==0):
                        G_unint_anc.remove_node(j)
                    elif G_unint_anc.out_degree(j)==1 and G_unint_anc.in_degree(j)==1 and (j,j) in list(G_unint_anc.edges):
                        G_unint_anc.remove_node(j)
                G_unint_anc=nx.relabel_nodes(G_unint_anc,mapping)
                EC_unint_vec[str((l,m))]=gf.normalised_ec_score(G_unint_anc,G_anc)
                ICS_unint_vec[str((l,m))]=gf.ics_score(G_unint_anc,G_anc)
                S3_unint_vec[str((l,m))]=gf.s3_score(G_unint_anc,G_anc)
                conservedEdges=gf.conserved_edges(G_unint_anc,G_anc)
                conserved_unint_vec[str((l,m))]=conservedEdges
                extra_edges_unint_vec[str((l,m))]=len(G_unint_anc.edges)-conservedEdges
                missed_edges_unint_vec[str((l,m))]=len(G_anc.edges)-conservedEdges
                network_order_unint[str((l,m))]=len(G_unint_anc.nodes)
                
                nx.write_edgelist(G_orig_anc,"test_datasets_ancestral/regulatory_test_ped_pea_r08/anc50_2cherry_branch"+str(branchLength)+"/intersect_anc_graphs_"+str(l)+"_"+str(m)+".txt")
                nx.write_edgelist(G_unint_anc,"test_datasets_ancestral/regulatory_test_ped_pea_r08/anc50_2cherry_branch"+str(branchLength)+"/unint_anc_graphs_"+str(l)+"_"+str(m)+".txt")
                print("g_anc true edge number",len(G_anc.edges())) 
                '''
                #Align then apply my algorithm

                G1_orig=copy.deepcopy(G1)
                G2_orig=copy.deepcopy(G2)
                #initial alignment
                
                G1_labelless=nx.convert_node_labels_to_integers(G1_orig)
                G2_labelless=nx.convert_node_labels_to_integers(G2_orig)
                G1_align=nx.convert_node_labels_to_integers(G1_orig)
                G2_align=nx.convert_node_labels_to_integers(G2_orig)
                alignVec,mapped=gf.NF(G1_align,G2_align,32,0.8)
                mapping = dict(alignVec)
                
                G1_mapped=nx.induced_subgraph(G1_align,list(mapped))
                G1_mapped=nx.relabel_nodes(G1_mapped,mapping)
                
                
                tolLike=0
                tolSim=0
                G_anc_find1,G_anc_find2=gf.ancestor_finder_without_alignment_alt_alt_alt_alt(G1_align,G2_align,qMod,qCon,tolerance=tolLike,toleranceEC=tolSim)
                print("ancestor found")
                G_find_int_anc=nx.intersection(G_anc_find1,G_anc_find2)
                print("my alg edge number",len(G_find_int_anc.edges()))
                G_align_unint=gf.graph_intersection_union(G_anc_find1,G_anc_find2)
                print("my alg edge number unint",len(G_align_unint.edges()))
                
                
                for j in list(G_find_int_anc.nodes()):
                    if (G_find_int_anc.out_degree(j)==0 and G_find_int_anc.in_degree(j)==0):
                        G_find_int_anc.remove_node(j)
                    elif G_find_int_anc.out_degree(j)==1 and G_find_int_anc.in_degree(j)==1 and (j,j) in list(G_find_int_anc.edges):
                        G_find_int_anc.remove_node(j)
                
                mapping=dict()
                for i in list(G_find_int_anc.nodes):
                    mapping[i]=G_anc_find1.nodes[i]['orig_label']
                G_find_int_anc=nx.relabel_nodes(G_find_int_anc,mapping)

                EC_findAl_int_vec[str((l,m))]=gf.normalised_ec_score(G_find_int_anc,G_anc)
                ICS_findAl_int_vec[str((l,m))]=gf.ics_score(G_find_int_anc,G_anc)
                S3_findAl_int_vec[str((l,m))]=gf.s3_score(G_find_int_anc,G_anc)
                network_order_findAl_int[str((l,m))]=len(G_find_int_anc.nodes)

                for j in list(G_align_unint.nodes()):
                    if (G_align_unint.out_degree(j)==0 and G_align_unint.in_degree(j)==0):
                        G_align_unint.remove_node(j)
                    elif G_align_unint.out_degree(j)==1 and G_align_unint.in_degree(j)==1 and (j,j) in list(G_align_unint.edges):
                        G_align_unint.remove_node(j)
                
                mapping=dict()
                for i in list(G_align_unint.nodes):
                    mapping[i]=G_anc_find1.nodes[i]['orig_label']
                G_align_unint=nx.relabel_nodes(G_align_unint,mapping)

                EC_align_unint_vec[str((l,m))]=gf.normalised_ec_score(G_align_unint,G_anc)
                ICS_align_unint_vec[str((l,m))]=gf.ics_score(G_align_unint,G_anc)
                S3_align_unint_vec[str((l,m))]=gf.s3_score(G_align_unint,G_anc)
                network_order_align_unint[str((l,m))]=len(G_align_unint.nodes)

                for j in list(G_anc_find1.nodes()):
                    if (G_anc_find1.out_degree(j)==0 and G_anc_find1.in_degree(j)==0):
                        G_anc_find1.remove_node(j)
                    elif G_anc_find1.out_degree(j)==1 and G_anc_find1.in_degree(j)==1 and (j,j) in list(G_anc_find1.edges):
                        G_anc_find1.remove_node(j)

                mapping=dict()
                for i in list(G_anc_find1.nodes):
                    mapping[i]=G_anc_find1.nodes[i]['orig_label']
                G_find1_anc=nx.relabel_nodes(G_anc_find1,mapping)
                
                EC_findAl1_vec[str((l,m))]=gf.normalised_ec_score(G_find1_anc,G_anc)
                ICS_findAl1_vec[str((l,m))]=gf.ics_score(G_find1_anc,G_anc)
                S3_findAl1_vec[str((l,m))]=gf.s3_score(G_find1_anc,G_anc)
                network_order_findAl1[str((l,m))]=len(G_find1_anc.nodes)

                for j in list(G_anc_find2.nodes()):
                    if (G_anc_find2.out_degree(j)==0 and G_anc_find2.in_degree(j)==0):
                        G_anc_find2.remove_node(j)
                    elif G_anc_find2.out_degree(j)==1 and G_anc_find2.in_degree(j)==1 and (j,j) in list(G_anc_find2.edges):
                        G_anc_find2.remove_node(j)
                
                mapping=dict()
                for i in list(G_anc_find2.nodes):
                    mapping[i]=G_anc_find2.nodes[i]['orig_label']
                G_find2_anc=nx.relabel_nodes(G_anc_find2,mapping)

                EC_findAl2_vec[str((l,m))]=gf.normalised_ec_score(G_find2_anc,G_anc)
                ICS_findAl2_vec[str((l,m))]=gf.ics_score(G_find2_anc,G_anc)
                S3_findAl2_vec[str((l,m))]=gf.s3_score(G_find2_anc,G_anc)
                network_order_findAl2[str((l,m))]=len(G_find2_anc.nodes)

                nx.write_edgelist(G_find_int_anc,"test_datasets_ancestral/anc50_2cherry_branch"+str(branchLength)+"/myalg_align_int_anc_graphs_"+str(l)+"_"+str(m)+".txt")
                nx.write_edgelist(G_align_unint,"test_datasets_ancestral/anc50_2cherry_branch"+str(branchLength)+"/myalg_align_unint_anc_graphs_"+str(l)+"_"+str(m)+".txt")
                nx.write_edgelist(G_find1_anc,"test_datasets_ancestral/anc50_2cherry_branch"+str(branchLength)+"/myalgG"+str(l)+"_align_anc_graphs_"+str(l)+"_"+str(m)+".txt")
                nx.write_edgelist(G_find2_anc,"test_datasets_ancestral/anc50_2cherry_branch"+str(branchLength)+"/myalgG"+str(m)+"_align_anc_graphs_"+str(l)+"_"+str(m)+".txt")
                '''
                '''
                G_anc_find1=nx.convert_node_labels_to_integers(G_anc_find1)
                G_anc_find2=nx.convert_node_labels_to_integers(G_anc_find2)
                alignVec,mapped=gf.NF_many_to_one(G_anc_find1,G_anc_find2,32,0.8)
                mapping = dict(alignVec)
                
                G_anc_find1_mapped=nx.induced_subgraph(G_anc_find1,list(mapped))
                G_anc_find1_mapped=nx.relabel_nodes(G_anc_find1_mapped,mapping)

                for j in list(G_anc_find1_mapped.nodes()):
                    if (G_anc_find1_mapped.out_degree(j)==0 and G_anc_find1_mapped.in_degree(j)==0):
                        G_anc_find1_mapped.remove_node(j)
                    elif G_anc_find1_mapped.out_degree(j)==1 and G_anc_find1_mapped.in_degree(j)==1 and (j,j) in list(G_anc_find1_mapped.edges):
                        G_anc_find1_mapped.remove_node(j)
                
                mapping=dict()
                for i in list(G_anc_find1_mapped.nodes):
                    mapping[i]=G_anc_find1_mapped.nodes[i]['orig_label']
                G_anc_find1_mapped=nx.relabel_nodes(G_anc_find1_mapped,mapping)

                EC_find_mapped_int_vec[str((l,m))]=gf.normalised_ec_score(G_anc_find1_mapped,G_anc))
                ICS_find_mapped_int_vec[str((l,m))]=gf.ics_score(G_anc_find1_mapped,G_anc))
                S3_find_mapped_int_vec[str((l,m))]=gf.s3_score(G_anc_find1_mapped,G_anc))
                network_order_int_mapped[str((l,m))]=len(G_anc_find1_mapped.nodes))
                '''
'''                
gf.write_dict_to_file(extra_edges_dmc_vec,"test_datasets_ancestral/anc50_2cherry_branch"+str(branchLength)+"/extra_edges_dmc_vec.txt")
gf.write_dict_to_file(extra_edges_orig_vec,"test_datasets_ancestral/anc50_2cherry_branch"+str(branchLength)+"/extra_edges_orig_vec.txt")
gf.write_dict_to_file(extra_edges_unint_vec,"test_datasets_ancestral/anc50_2cherry_branch"+str(branchLength)+"/extra_edges_unint_vec.txt")
gf.write_dict_to_file(extra_edges_true_unint_vec,"test_datasets_ancestral/anc50_2cherry_branch"+str(branchLength)+"/extra_edges_true_unint_vec.txt")
gf.write_dict_to_file(extra_edges_find_int_vec,"test_datasets_ancestral/anc50_2cherry_branch"+str(branchLength)+"/extra_edges_find_int_vec.txt")
gf.write_dict_to_file(missed_edges_dmc_vec,"test_datasets_ancestral/anc50_2cherry_branch"+str(branchLength)+"/missed_edges_dmc_vec.txt")
gf.write_dict_to_file(missed_edges_orig_vec,"test_datasets_ancestral/anc50_2cherry_branch"+str(branchLength)+"/missed_edges_orig_vec.txt")
gf.write_dict_to_file(missed_edges_unint_vec,"test_datasets_ancestral/anc50_2cherry_branch"+str(branchLength)+"/missed_edges_unint_vec.txt")
gf.write_dict_to_file(missed_edges_true_unint_vec,"test_datasets_ancestral/anc50_2cherry_branch"+str(branchLength)+"/missed_edges_true_unint_vec.txt")
gf.write_dict_to_file(missed_edges_find_int_vec,"test_datasets_ancestral/anc50_2cherry_branch"+str(branchLength)+"/missed_edges_find_int_vec.txt")
gf.write_dict_to_file(conserved_dmc_vec,"test_datasets_ancestral/anc50_2cherry_branch"+str(branchLength)+"/conserved_dmc_vec.txt")
gf.write_dict_to_file(conserved_orig_vec,"test_datasets_ancestral/anc50_2cherry_branch"+str(branchLength)+"/conserved_orig_vec.txt")
gf.write_dict_to_file(conserved_unint_vec,"test_datasets_ancestral/anc50_2cherry_branch"+str(branchLength)+"/conserved_unint_vec.txt")
gf.write_dict_to_file(conserved_true_unint_vec,"test_datasets_ancestral/anc50_2cherry_branch"+str(branchLength)+"/conserved_true_unint_vec.txt")
gf.write_dict_to_file(conserved_find_int_vec,"test_datasets_ancestral/anc50_2cherry_branch"+str(branchLength)+"/conserved_find_int_vec.txt")
gf.write_dict_to_file(S3_nf_vec,"test_datasets_ancestral/anc50_2cherry_branch"+str(branchLength)+"/S3_nf_vec.txt")
gf.write_dict_to_file(S3_dmc_vec,"test_datasets_ancestral/anc50_2cherry_branch"+str(branchLength)+"/S3_dmc_vec.txt")
gf.write_dict_to_file(S3_orig_vec,"test_datasets_ancestral/anc50_2cherry_branch"+str(branchLength)+"/S3_orig_vec.txt")
gf.write_dict_to_file(EC_nf_vec,"test_datasets_ancestral/anc50_2cherry_branch"+str(branchLength)+"/EC_nf_vec.txt")
gf.write_dict_to_file(EC_dmc_vec,"test_datasets_ancestral/anc50_2cherry_branch"+str(branchLength)+"/EC_dmc_vec.txt")
gf.write_dict_to_file(EC_orig_vec,"test_datasets_ancestral/anc50_2cherry_branch"+str(branchLength)+"/EC_orig_vec.txt")
gf.write_dict_to_file(ICS_nf_vec,"test_datasets_ancestral/anc50_2cherry_branch"+str(branchLength)+"/ICS_nf_vec.txt")
gf.write_dict_to_file(ICS_dmc_vec,"test_datasets_ancestral/anc50_2cherry_branch"+str(branchLength)+"/ICS_dmc_vec.txt")
gf.write_dict_to_file(ICS_orig_vec,"test_datasets_ancestral/anc50_2cherry_branch"+str(branchLength)+"/ICS_orig_vec.txt")
gf.write_dict_to_file(network_order_dmc,"test_datasets_ancestral/anc50_2cherry_branch"+str(branchLength)+"/network_order_dmc.txt")
gf.write_dict_to_file(network_order_nf,"test_datasets_ancestral/anc50_2cherry_branch"+str(branchLength)+"/network_order_nf.txt")
gf.write_dict_to_file(network_order_orig,"test_datasets_ancestral/anc50_2cherry_branch"+str(branchLength)+"/network_order_orig.txt")
gf.write_dict_to_file(EC_find1_vec,"test_datasets_ancestral/anc50_2cherry_branch"+str(branchLength)+"/EC_find1_vec.txt")
gf.write_dict_to_file(ICS_find1_vec,"test_datasets_ancestral/anc50_2cherry_branch"+str(branchLength)+"/ICS_find1_vec.txt")
gf.write_dict_to_file(S3_find1_vec,"test_datasets_ancestral/anc50_2cherry_branch"+str(branchLength)+"/S3_find1_vec.txt")
gf.write_dict_to_file(network_order_find1,"test_datasets_ancestral/anc50_2cherry_branch"+str(branchLength)+"/network_order_find1.txt")
gf.write_dict_to_file(EC_find2_vec,"test_datasets_ancestral/anc50_2cherry_branch"+str(branchLength)+"/EC_find2_vec.txt")
gf.write_dict_to_file(ICS_find2_vec,"test_datasets_ancestral/anc50_2cherry_branch"+str(branchLength)+"/ICS_find2_vec.txt")
gf.write_dict_to_file(S3_find2_vec,"test_datasets_ancestral/anc50_2cherry_branch"+str(branchLength)+"/S3_find2_vec.txt")
gf.write_dict_to_file(network_order_find2,"test_datasets_ancestral/anc50_2cherry_branch"+str(branchLength)+"/network_order_find2.txt")
gf.write_dict_to_file(EC_find_int_vec,"test_datasets_ancestral/anc50_2cherry_branch"+str(branchLength)+"/EC_find_int_vec.txt")
gf.write_dict_to_file(ICS_find_int_vec,"test_datasets_ancestral/anc50_2cherry_branch"+str(branchLength)+"/ICS_find_int_vec.txt")
gf.write_dict_to_file(S3_find_int_vec,"test_datasets_ancestral/anc50_2cherry_branch"+str(branchLength)+"/S3_find_int_vec.txt")
gf.write_dict_to_file(network_order_find_int,"test_datasets_ancestral/anc50_2cherry_branch"+str(branchLength)+"/network_order_find_int.txt")
gf.write_dict_to_file(EC_find_mapped_int_vec,"test_datasets_ancestral/anc50_2cherry_branch"+str(branchLength)+"/EC_find_mapped_int_vec.txt")
gf.write_dict_to_file(ICS_find_mapped_int_vec,"test_datasets_ancestral/anc50_2cherry_branch"+str(branchLength)+"/ICS_find_mapped_int_vec.txt")
gf.write_dict_to_file(S3_find_mapped_int_vec,"test_datasets_ancestral/anc50_2cherry_branch"+str(branchLength)+"/S3_find_mapped_int_vec.txt")
gf.write_dict_to_file(network_order_int_mapped,"test_datasets_ancestral/anc50_2cherry_branch"+str(branchLength)+"/network_order_int_mapped.txt")
gf.write_dict_to_file(EC_findAl1_vec,"test_datasets_ancestral/anc50_2cherry_branch"+str(branchLength)+"/EC_findAl1_vec.txt")
gf.write_dict_to_file(ICS_findAl1_vec,"test_datasets_ancestral/anc50_2cherry_branch"+str(branchLength)+"/ICS_findAl1_vec.txt")
gf.write_dict_to_file(S3_findAl1_vec,"test_datasets_ancestral/anc50_2cherry_branch"+str(branchLength)+"/S3_findAl1_vec.txt")
gf.write_dict_to_file(network_order_findAl1,"test_datasets_ancestral/anc50_2cherry_branch"+str(branchLength)+"/network_order_findAl1.txt")
gf.write_dict_to_file(EC_findAl2_vec,"test_datasets_ancestral/anc50_2cherry_branch"+str(branchLength)+"/EC_findAl2_vec.txt")
gf.write_dict_to_file(ICS_findAl2_vec,"test_datasets_ancestral/anc50_2cherry_branch"+str(branchLength)+"/ICS_findAl2_vec.txt")
gf.write_dict_to_file(S3_findAl2_vec,"test_datasets_ancestral/anc50_2cherry_branch"+str(branchLength)+"/S3_findAl2_vec.txt")
gf.write_dict_to_file(network_order_findAl2,"test_datasets_ancestral/anc50_2cherry_branch"+str(branchLength)+"/network_order_findAl2.txt")
gf.write_dict_to_file(EC_findAl_int_vec,"test_datasets_ancestral/anc50_2cherry_branch"+str(branchLength)+"/EC_findAl_int_vec.txt")
gf.write_dict_to_file(ICS_findAl_int_vec,"test_datasets_ancestral/anc50_2cherry_branch"+str(branchLength)+"/ICS_findAl_int_vec.txt")
gf.write_dict_to_file(S3_findAl_int_vec,"test_datasets_ancestral/anc50_2cherry_branch"+str(branchLength)+"/S3_findAl_int_vec.txt")
gf.write_dict_to_file(network_order_findAl_int,"test_datasets_ancestral/anc50_2cherry_branch"+str(branchLength)+"/network_order_findAl_int.txt")
gf.write_dict_to_file(EC_findAl_mapped_int_vec,"test_datasets_ancestral/anc50_2cherry_branch"+str(branchLength)+"/EC_findAl_mapped_int_vec.txt")
gf.write_dict_to_file(ICS_findAl_mapped_int_vec,"test_datasets_ancestral/anc50_2cherry_branch"+str(branchLength)+"/ICS_findAl_mapped_int_vec.txt")
gf.write_dict_to_file(S3_findAl_mapped_int_vec,"test_datasets_ancestral/anc50_2cherry_branch"+str(branchLength)+"/S3_findAl_mapped_int_vec.txt")
gf.write_dict_to_file(network_order_int_mapped,"test_datasets_ancestral/anc50_2cherry_branch"+str(branchLength)+"/network_order_int_mapped.txt")
gf.write_dict_to_file(EC_true_unint_vec,"test_datasets_ancestral/anc50_2cherry_branch"+str(branchLength)+"/EC_true_unint_vec.txt")
gf.write_dict_to_file(ICS_true_unint_vec,"test_datasets_ancestral/anc50_2cherry_branch"+str(branchLength)+"/ICS_true_unint_vec.txt")
gf.write_dict_to_file(S3_true_unint_vec,"test_datasets_ancestral/anc50_2cherry_branch"+str(branchLength)+"/S3_true_unint_vec.txt")
gf.write_dict_to_file(network_order_true_unint,"test_datasets_ancestral/anc50_2cherry_branch"+str(branchLength)+"/network_order_true_unint.txt")
gf.write_dict_to_file(EC_align_unint_vec,"test_datasets_ancestral/anc50_2cherry_branch"+str(branchLength)+"/EC_align_unint_vec.txt")
gf.write_dict_to_file(ICS_align_unint_vec,"test_datasets_ancestral/anc50_2cherry_branch"+str(branchLength)+"/ICS_align_unint_vec.txt")
gf.write_dict_to_file(S3_align_unint_vec,"test_datasets_ancestral/anc50_2cherry_branch"+str(branchLength)+"/S3_align_unint_vec.txt")
gf.write_dict_to_file(network_order_align_unint,"test_datasets_ancestral/anc50_2cherry_branch"+str(branchLength)+"/network_order_align_unint.txt")
gf.write_dict_to_file(EC_unint_vec,"test_datasets_ancestral/anc50_2cherry_branch"+str(branchLength)+"/EC_unint_vec.txt")
gf.write_dict_to_file(ICS_unint_vec,"test_datasets_ancestral/anc50_2cherry_branch"+str(branchLength)+"/ICS_unint_vec.txt")
gf.write_dict_to_file(S3_unint_vec,"test_datasets_ancestral/anc50_2cherry_branch"+str(branchLength)+"/S3_unint_vec.txt")
gf.write_dict_to_file(network_order_unint,"test_datasets_ancestral/anc50_2cherry_branch"+str(branchLength)+"/network_order_unint.txt")
'''
pairs_vec=[]

EC_nf=[]
EC_dmc=[]
EC_orig=[]
EC_unint=[]
EC_find_int=[]
EC_findAl_int=[]
EC_true_unint=[]
EC_align_unint=[]

S3_nf=[]
S3_dmc=[]
S3_orig=[]
S3_unint=[]
S3_find_int=[]
S3_findAl_int=[]
S3_true_unint=[]
S3_align_unint=[]

net_order_nf=[]
net_order_dmc=[]
net_order_orig=[]
net_order_find1=[]
net_order_find2=[]
net_order_find_int=[]
net_order_findAl_int=[]
net_order_true_unint=[]
net_order_align_unint=[]
net_order_unint=[]

conserved_dmc=[]
conserved_orig=[]
conserved_unint=[]
conserved_true_unint=[]
conserved_find_int=[]

extra_edges_dmc=[]
extra_edges_orig=[]
extra_edges_unint=[]
extra_edges_true_unint=[]
extra_edges_find_int=[]

missed_edges_dmc=[]
missed_edges_orig=[]
missed_edges_unint=[]
missed_edges_true_unint=[]
missed_edges_find_int=[]

font = {'family' : 'normal',
        'weight' : 'normal',

        'size'   : 18}


plt.rc('font', **font)
for l in leafGraphs:
    for m in leafGraphs:
        if m>l:
            pairs_vec.append(str((l,m)))
            
            #EC_nf.append(EC_nf_vec[str((l,m))])
            EC_dmc.append(EC_dmc_vec[str((l,m))])
            EC_orig.append(EC_orig_vec[str((l,m))])
            EC_unint.append(EC_unint_vec[str((l,m))])
            EC_find_int.append(EC_find_int_vec[str((l,m))])
            #EC_findAl_int.append(EC_findAl_int_vec[str((l,m))])
            EC_true_unint.append(EC_true_unint_vec[str((l,m))])
            #EC_align_unint.append(EC_align_unint_vec[str((l,m))])
            
            #S3_nf.append(S3_nf_vec[str((l,m))])
            S3_dmc.append(S3_dmc_vec[str((l,m))])
            S3_orig.append(S3_orig_vec[str((l,m))])
            S3_unint.append(S3_unint_vec[str((l,m))])
            S3_find_int.append(S3_find_int_vec[str((l,m))])
            #S3_findAl_int.append(S3_findAl_int_vec[str((l,m))])
            S3_true_unint.append(S3_true_unint_vec[str((l,m))])
            #S3_align_unint.append(S3_align_unint_vec[str((l,m))])

            #net_order_nf.append(network_order_nf[str((l,m))])
            net_order_dmc.append(network_order_dmc[str((l,m))])
            net_order_orig.append(network_order_orig[str((l,m))])
            net_order_find1.append(network_order_find1[str((l,m))])
            net_order_find2.append(network_order_find2[str((l,m))])
            net_order_find_int.append(network_order_find_int[str((l,m))])
            #net_order_findAl_int.append(network_order_findAl_int[str((l,m))])
            net_order_true_unint.append(network_order_true_unint[str((l,m))])
            #net_order_align_unint.append(network_order_align_unint[str((l,m))])
            net_order_unint.append(network_order_unint[str((l,m))])
            
            conserved_dmc.append(conserved_dmc_vec[str((l,m))])
            conserved_orig.append(conserved_orig_vec[str((l,m))])
            conserved_unint.append(conserved_unint_vec[str((l,m))])
            conserved_true_unint.append(conserved_true_unint_vec[str((l,m))])
            conserved_find_int.append(conserved_find_int_vec[str((l,m))])

            
            extra_edges_dmc.append(extra_edges_dmc_vec[str((l,m))])
            extra_edges_orig.append(extra_edges_orig_vec[str((l,m))])
            extra_edges_unint.append(extra_edges_unint_vec[str((l,m))])
            extra_edges_true_unint.append(extra_edges_true_unint_vec[str((l,m))])
            extra_edges_find_int.append(extra_edges_find_int_vec[str((l,m))])

            missed_edges_dmc.append(missed_edges_dmc_vec[str((l,m))])
            missed_edges_orig.append(missed_edges_orig_vec[str((l,m))])
            missed_edges_unint.append(missed_edges_unint_vec[str((l,m))])
            missed_edges_true_unint.append(missed_edges_true_unint_vec[str((l,m))])
            missed_edges_find_int.append(missed_edges_find_int_vec[str((l,m))])
            
            
print(pairs_vec)
plt.figure(figsize =(12, 8))
barWidth = 0.08
fig = plt.subplots(figsize =(12, 8))
# Set position of bar on X axis
br1 = np.arange(len(EC_dmc))
br2 = [x + barWidth for x in br1]
br3 = [x + barWidth for x in br2]
br4 = [x + barWidth for x in br3]
br5 = [x + barWidth for x in br4]
br6 = [x + barWidth for x in br5]
#br7 = [x + barWidth for x in br6]
#br8 = [x + barWidth for x in br7]


# Make the plot
#plt.bar(br1, EC_nf, color ='r', width = barWidth,label ="Align then Intersection")
plt.bar(br1, EC_dmc, color ='g', width = barWidth,label ="Zhu-Nakleh Reconstruction")
plt.bar(br2, EC_orig, color ='b', width = barWidth,label ="True Label Intersection")
plt.bar(br3, EC_unint, color ='c', width = barWidth,label ="True Label Unint")
plt.bar(br4, EC_find_int, color ='y', width = barWidth,label ="My Algorithm (True Labels)")
plt.bar(br5, EC_true_unint, color ='m', width = barWidth,label ="My Algorithm (True unint)")
#plt.bar(br7, EC_findAl_int, color ='k', width = barWidth,label ="My Algorithm (Alignment)")
#plt.bar(br8, EC_align_unint, color ='bisque', width = barWidth,label ="My Algorithm (Align unint)")


# Adding Xticks
plt.xlabel('Leaf Network Pair')
plt.ylabel('EC Score of Predicted and Actual Ancestor')
plt.xticks([r + barWidth+0.16 for r in range(len(EC_dmc))],
        pairs_vec)

# Put a legend to the right of the current axis
plt.legend(bbox_to_anchor=(1.04,0.5), loc="center left", borderaxespad=0)

plt.savefig("test_datasets_ancestral/regulatory_test_ped_pea_r08/anc50_2cherry_branch"+str(branchLength)+"/EC_scores_branchlength"+str(branchLength)+".png",bbox_inches="tight")

plt.figure(figsize =(12, 8))
barWidth = 0.08
fig = plt.subplots(figsize =(12, 8))
# Set position of bar on X axis
br1 = np.arange(len(S3_dmc))
br2 = [x + barWidth for x in br1]
br3 = [x + barWidth for x in br2]
br4 = [x + barWidth for x in br3]
br5 = [x + barWidth for x in br4]
br6 = [x + barWidth for x in br5]
#br7 = [x + barWidth for x in br6]
#br8 = [x + barWidth for x in br7]


# Make the plot
#plt.bar(br1, S3_nf, color ='r', width = barWidth,label ="Align then Intersection")
plt.bar(br1, S3_dmc, color ='g', width = barWidth,label ="Zhu-Nakleh Reconstruction")
plt.bar(br2, S3_orig, color ='b', width = barWidth,label ="True Label Intersection")
plt.bar(br3, S3_unint, color ='c', width = barWidth,label ="True Label Unint")
plt.bar(br4, S3_find_int, color ='y', width = barWidth,label ="My Algorithm (True Labels)")
#plt.bar(br7, S3_findAl_int, color ='k', width = barWidth,label ="My Algorithm (Alignment)")
plt.bar(br5, S3_true_unint, color ='m', width = barWidth,label ="My Algorithm (True unint)")
#plt.bar(br8, S3_align_unint, color ='bisque', width = barWidth,label ="My Algorithm (Align unint)")


# Adding Xticks
plt.xlabel('Leaf Network Pair')
plt.ylabel('S3 Score of Predicted and Actual Ancestor')
plt.xticks([r + barWidth+0.16 for r in range(len(S3_dmc))],
        pairs_vec)

plt.legend(bbox_to_anchor=(1.04,0.5), loc="center left", borderaxespad=0)
plt.savefig("test_datasets_ancestral/regulatory_test_ped_pea_r08/anc50_2cherry_branch"+str(branchLength)+"/S3_scores_branchlength"+str(branchLength)+".png",bbox_inches='tight')

plt.figure(figsize =(12, 8))
barWidth = 0.08
fig = plt.subplots(figsize =(12, 8))
# Set position of bar on X axis
br1 = np.arange(len(net_order_dmc))
br2 = [x + barWidth for x in br1]
br3 = [x + barWidth for x in br2]
br4 = [x + barWidth for x in br3]
br5 = [x + barWidth for x in br4]
br6 = [x + barWidth for x in br5]
br7 = [x + barWidth for x in br6]
br8 = [x + barWidth for x in br7]
br9 = [x + barWidth for x in br8]
br10=[x + barWidth for x in br9]


# Make the plot
#plt.bar(br1, net_order_nf, color ='r', width = barWidth,label ="Align then Intersection")
plt.bar(br1, net_order_dmc, color ='g', width = barWidth,label ="Zhu-Nakleh Reconstruction")
plt.bar(br2, net_order_orig, color ='b', width = barWidth,label ="True Label Intersection")
plt.bar(br3, net_order_unint, color ='c', width = barWidth,label ="True Label Unint")
plt.bar(br4, net_order_find_int, color ='y', width = barWidth,label ="My Algorithm (True Labels)")
#plt.bar(br9, net_order_findAl_int, color ='k', width = barWidth,label ="My Algorithm (Alignment)")
plt.bar(br5, net_order_true_unint, color ='m', width = barWidth,label ="My Algorithm (True unint)")
#plt.bar(br10, net_order_align_unint, color ='bisque', width = barWidth,label ="My Algorithm (Align unint)")
plt.bar(br6, net_order_find1, color ='grey', width = barWidth,label ="My Algorithm G1")
plt.bar(br7, net_order_find2, color ='gold', width = barWidth,label ="My Algorithm G2")

# Adding Xticks
plt.xlabel('Leaf Network Pair')
plt.ylabel('Network Order of Predicted and Actual Ancestor')
plt.xticks([r + barWidth+0.16 for r in range(len(net_order_dmc))],
        pairs_vec)

plt.legend(bbox_to_anchor=(1.04,0.5), loc="center left", borderaxespad=0)
plt.savefig("test_datasets_ancestral/regulatory_test_ped_pea_r08/anc50_2cherry_branch"+str(branchLength)+"/net_order_scores_branchlength"+str(branchLength)+".png",bbox_inches='tight')

plt.figure(figsize =(12, 8))
barWidth = 0.08
fig = plt.subplots(figsize =(12, 8))
# Set position of bar on X axis
br1 = np.arange(len(conserved_dmc))
br2 = [x + barWidth for x in br1]
br3 = [x + barWidth for x in br2]
br4 = [x + barWidth for x in br3]
br5 = [x + barWidth for x in br4]
#br6 = [x + barWidth for x in br5]
#br7 = [x + barWidth for x in br6]
#br8 = [x + barWidth for x in br7]


# Make the plot
#plt.bar(br6, conserved_nf, color ='r', width = barWidth,label ="Align then Intersection")
plt.bar(br1, conserved_dmc, color ='g', width = barWidth,label ="Zhu-Nakleh Reconstruction")
plt.bar(br2, conserved_orig, color ='b', width = barWidth,label ="True Label Intersection")
plt.bar(br3, conserved_unint, color ='c', width = barWidth,label ="True Label Unint")
plt.bar(br4, conserved_find_int, color ='y', width = barWidth,label ="My Algorithm (True Labels)")
plt.bar(br5, conserved_true_unint, color ='m', width = barWidth,label ="My Algorithm (True unint)")
#plt.bar(br7, EC_findAl_int, color ='k', width = barWidth,label ="My Algorithm (Alignment)")
#plt.bar(br8, EC_align_unint, color ='bisque', width = barWidth,label ="My Algorithm (Align unint)")


# Adding Xticks
plt.xlabel('Leaf Network Pair')
plt.ylabel('No. of Correct Edges in Predicted Ancestor')
plt.xticks([r + barWidth+0.16 for r in range(len(conserved_dmc))],
        pairs_vec)

# Put a legend to the right of the current axis
plt.legend(bbox_to_anchor=(1.04,0.5), loc="center left", borderaxespad=0)

plt.savefig("test_datasets_ancestral/regulatory_test_ped_pea_r08/anc50_2cherry_branch"+str(branchLength)+"/conserved_edges_branchlength"+str(branchLength)+".png",bbox_inches="tight")


plt.figure(figsize =(12, 8))

barWidth = 0.08
fig = plt.subplots(figsize =(12, 8))
# Set position of bar on X axis
br1 = np.arange(len(conserved_dmc))
br2 = [x + barWidth for x in br1]
br3 = [x + barWidth for x in br2]
br4 = [x + barWidth for x in br3]
br5 = [x + barWidth for x in br4]
#br6 = [x + barWidth for x in br5]
#br7 = [x + barWidth for x in br6]
#br8 = [x + barWidth for x in br7]


# Make the plot
#plt.bar(br6, conserved_nf, color ='r', width = barWidth,label ="Align then Intersection")
plt.bar(br1, extra_edges_dmc, color ='g', width = barWidth,label ="Zhu-Nakleh Reconstruction")
plt.bar(br2, extra_edges_orig, color ='b', width = barWidth,label ="True Label Intersection")
plt.bar(br3, extra_edges_unint, color ='c', width = barWidth,label ="True Label Unint")
plt.bar(br4, extra_edges_find_int, color ='y', width = barWidth,label ="My Algorithm (True Labels)")
plt.bar(br5, extra_edges_true_unint, color ='m', width = barWidth,label ="My Algorithm (True unint)")
#plt.bar(br7, EC_findAl_int, color ='k', width = barWidth,label ="My Algorithm (Alignment)")
#plt.bar(br8, EC_align_unint, color ='bisque', width = barWidth,label ="My Algorithm (Align unint)")


# Adding Xticks
plt.xlabel('Leaf Network Pair')
plt.ylabel('No. of Extra Edges in Predicted Ancestor')
plt.xticks([r + barWidth+0.16 for r in range(len(conserved_dmc))],
        pairs_vec)

# Put a legend to the right of the current axis
plt.legend(bbox_to_anchor=(1.04,0.5), loc="center left", borderaxespad=0)

plt.savefig("test_datasets_ancestral/regulatory_test_ped_pea_r08/anc50_2cherry_branch"+str(branchLength)+"/extra_edges_branchlength"+str(branchLength)+".png",bbox_inches="tight")


plt.figure(figsize =(12, 8))

barWidth = 0.08
fig = plt.subplots(figsize =(12, 8))
# Set position of bar on X axis
br1 = np.arange(len(conserved_dmc))
br2 = [x + barWidth for x in br1]
br3 = [x + barWidth for x in br2]
br4 = [x + barWidth for x in br3]
br5 = [x + barWidth for x in br4]
#br6 = [x + barWidth for x in br5]
#br7 = [x + barWidth for x in br6]
#br8 = [x + barWidth for x in br7]


# Make the plot
#plt.bar(br6, conserved_nf, color ='r', width = barWidth,label ="Align then Intersection")
plt.bar(br1, missed_edges_dmc, color ='g', width = barWidth,label ="Zhu-Nakleh Reconstruction")
plt.bar(br2, missed_edges_orig, color ='b', width = barWidth,label ="True Label Intersection")
plt.bar(br3, missed_edges_unint, color ='c', width = barWidth,label ="True Label Unint")
plt.bar(br4, missed_edges_find_int, color ='y', width = barWidth,label ="My Algorithm (True Labels)")
plt.bar(br5, missed_edges_true_unint, color ='m', width = barWidth,label ="My Algorithm (True unint)")
#plt.bar(br7, EC_findAl_int, color ='k', width = barWidth,label ="My Algorithm (Alignment)")
#plt.bar(br8, EC_align_unint, color ='bisque', width = barWidth,label ="My Algorithm (Align unint)")


# Adding Xticks
plt.xlabel('Leaf Network Pair')
plt.ylabel('No. Edges in True Ancestor Missed By Predicted Ancestor')
plt.xticks([r + barWidth+0.16 for r in range(len(conserved_dmc))],
        pairs_vec)

# Put a legend to the right of the current axis
plt.legend(bbox_to_anchor=(1.04,0.5), loc="center left", borderaxespad=0)

plt.savefig("test_datasets_ancestral/regulatory_test_ped_pea_r08/anc50_2cherry_branch"+str(branchLength)+"/missed_edges_branchlength"+str(branchLength)+".png",bbox_inches="tight")
gf.write_list_to_file([algType],"test_datasets_ancestral/regulatory_test_ped_pea_r08/anc50_2cherry_branch"+str(branchLength)+"/recent_run")

'''
plt.title("Edge Correctness of Estimated Ancestral Networks compared to True Ancestral Networks")


plt.xlabel("iteration")
plt.ylabel("Edge Correctness")
plt.plot(iterationVec,EC_nf_vec,'r-',label="Align then Intersection")
plt.plot(iterationVec,EC_dmc_vec,'b-',label="Zhu-Nakleh Reconstruction")
plt.plot(iterationVec,EC_orig_vec,'k-',label="True Label Intersection")
plt.plot(iterationVec,EC_unint_vec,'gold',label="True Label Unint")
#plt.plot(iterationVec,EC_find1_vec,'g-',label="MyAlg G1")
#plt.plot(iterationVec,EC_find2_vec,'y-',label="MyAlg G2")
plt.plot(iterationVec,EC_find_int_vec,'c-',label="My Algorithm (True Labels)")
plt.plot(iterationVec,EC_findAl_int_vec,'m-',label="My Algorithm (Alignment)")
plt.plot(iterationVec,EC_true_unint_vec,'g-',label="My Algorithm (True unint)")
plt.plot(iterationVec,EC_align_unint_vec,'y-',label="My Algorithm (Align unint)")
#plt.plot(iterationVec,EC_find_mapped_int_vec,'m-',label="MyAlg Alignment")
plt.legend()

plt.savefig("NFvsDMCAncRec_EC"+str(l)+"_"+str(m)+".png")
plt.show()

plt.figure(figsize=(15,10))
plt.title("Average ICS of Ancestral Networks Estimated by DMC Reconstruction and Net alignment")
plt.xlabel("iteration")
plt.ylabel("Average ICS Score")
plt.plot(iterationVec,ICS_nf_vec,'r-',label="Align then Intersection")
plt.plot(iterationVec,ICS_dmc_vec,'b-',label="DMC Reconstruction")
plt.plot(iterationVec,ICS_orig_vec,'k-',label="Intersection")
plt.plot(iterationVec,ICS_find1_vec,'g-',label="MyAlg G1")
plt.plot(iterationVec,ICS_find2_vec,'y-',label="MyAlg G2")
plt.plot(iterationVec,ICS_find_int_vec,'c-',label="MyAlg Intersection")
#plt.plot(iterationVec,ICS_find_mapped_int_vec,'m-',label="MyAlg Alignment")
plt.legend()

plt.savefig("NFvsDMCAncRec_ICS.png")
plt.show()


plt.figure(figsize=(15,10))
plt.title("S3 of Estimated Ancestral Networks Compared to True Ancestral Networks")
plt.xlabel("iteration")
plt.ylabel("S3")
plt.plot(iterationVec,S3_nf_vec,'r-',label="Align then Intersection")
plt.plot(iterationVec,S3_dmc_vec,'b-',label="Zhu-Naklen Reconstruction")
plt.plot(iterationVec,S3_orig_vec,'k-',label="True Label Intersection")
plt.plot(iterationVec,S3_unint_vec,'gold',label="True Label Unint")
#plt.plot(iterationVec,S3_find1_vec,'g-',label="MyAlg G1")
#plt.plot(iterationVec,S3_find2_vec,'y-',label="MyAlg G2")
plt.plot(iterationVec,S3_find_int_vec,'c-',label="My Algorithm (True Labels)")
plt.plot(iterationVec,S3_findAl_int_vec,'m-',label="My Algorithm (Alignment)")
plt.plot(iterationVec,S3_true_unint_vec,'g-',label="My Algorithm (True unint)")
plt.plot(iterationVec,S3_align_unint_vec,'y-',label="My Algorithm (Align unint)")
#plt.plot(iterationVec,S3_find_mapped_int_vec,'m-',label="MyAlg Alignment")
plt.legend()

plt.savefig("NFvsDMCAncRec_S3"+str(l)+"_"+str(m)+".png")
plt.show()
plt.figure(figsize=(15,10))
plt.title("Network Order of Estimated Ancestral Networks")
plt.xlabel("iteration")
plt.ylabel("Network Order")
plt.plot(iterationVec,network_order_nf,'r-',label="Align then Intersection")
plt.plot(iterationVec,network_order_dmc,'b-',label="Zhu-Nakleh Reconstruction")
plt.plot(iterationVec,network_order_orig,'k-',label="True Label Intersection")
plt.plot(iterationVec,network_order_find1,'g-',label="MyAlg G1")
plt.plot(iterationVec,network_order_find2,'y-',label="MyAlg G2")
plt.plot(iterationVec,network_order_find_int,'c-',label="My Algorithm (True Labels)")
plt.plot(iterationVec,network_order_findAl_int,'m-',label="My Algorithm (Alignment)")
plt.plot(iterationVec,network_order_true_unint,'gold',label="My Algorithm (True unint)")
plt.plot(iterationVec,network_order_align_unint,'gray',label="My Algorithm (Align unint)")
plt.plot(iterationVec,network_order_unint,'bisque',label="True Label unint)")
#plt.plot(iterationVec,network_order_find_mapped_int,'m-',label="MyAlg Alignment")
plt.legend()

plt.savefig("NFvsDMCAncRec_network_order"+str(l)+"_"+str(m)+".png")
'''