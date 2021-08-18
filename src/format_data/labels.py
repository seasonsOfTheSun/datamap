import pandas as pd
import networkx as nx

def max_connected_component(G):
    subs = nx.weakly_connected_component_subgraphs(G)
    max(list(subs), lambda H:len(H))

def descendants_out(G, node):
    """  Recursively get all descendents of the DiGraph G,
       descending down the edges, starting at node.""" 

    immediate = [i for _,i in G.out_edges(node)]
    out = set(immediate)

    for node2 in immediate:
        out |= descendants_out(G, node2)
    
    return out


def descendants_in(G, node):
    """  Recursively get all descendents of the DiGraph G,
       ascending up the edges, starting at node.""" 

    immediate = [i for i,_ in G.in_edges(node)]
    out = set(immediate)

    for node2 in immediate:
        out |= descendants_in(G, node2)
    
    return out

def name_to_terms(G):
    """ Use the term hierarchy G to get the terms annotated to each cid"""
    out = {}
    for term in G.nodes(): 
        try: 
            for namecid in G.nodes()[term]['drugs']: 
                name = namecid.split(":")[0]
                try: 
                    out[name] |= {term} 
                except KeyError: 
                    out[name] = {term} 
        except KeyError: 
             pass
    return out

def filter_n_drug_terms(G, n):
    """ Delete terms in the network if
    annotate less than n drugs."""
    G_copy = G.copy()
    
    
    nodes = G.nodes()
    for node in nodes:
        try:
            if len(G.nodes()[node]['drugs']) < n:
               G_copy.remove_node(node)
        except KeyError:
            pass
    return G_copy


def lowest_in_hierarchy(G, terms):
    """ """
    sub = G.subgraph(terms)
    out = [i for i,v in sub.in_degree() if v == 0]
    return out

import networkx as nx

G_all = nx.read_gml("data/external/annotation_hierarchies/Medical_Subject_Headings__MeSH_.gml")
G = G_all.subgraph(descendants_out(G_all, "Molecular Mechanisms of Pharmacological Action"))

H = G.copy()
H = filter_n_drug_terms(G, 4)    
try:
    H.remove_nodes_from([i for i,_ in G.in_edges('Cytochrome P-450 Enzyme Inhibitors')])
except:
    pass
try:
    H.remove_node('Cytochrome P-450 Enzyme Inhibitors') # not actually relevant to mechanism
except:
    pass
try:
    H.remove_node("Enzyme Inhibitors") # way too general
except:
    pass
try:
    H.remove_node("Neurotransmitter Agents")
except:
    pass
try:
    H.remove_node('Neurotransmitter Uptake Inhibitors')
except:
    pass
H.remove_node("Supplementary Records")

import collections
import pandas as pd

# calcualte frequency of different terms
# across all selected drugs
freq = collections.Counter()
name_to_candidate_terms = {}
for name, terms in name_to_terms(H).items():
    temp = lowest_in_hierarchy(H, terms)
    freq.update(temp)
    name_to_candidate_terms[name] = temp

freq = pd.Series(freq).sort_values()
def get_most_frequent(terms):
    return freq.loc[terms].argmax()

name_to_term = {}
for name, terms in name_to_candidate_terms.items():
    name_to_term[name] =  get_most_frequent(terms)

name_to_term = pd.Series(name_to_term)
name_to_term.name = "MeSH"

folders = ["cell_line", "transcriptional", "morphological"]
for folder in folders:

    drug_names = pd.read_csv(f"data/intermediate/{folder}/drug_names.csv", header = None)
    drug_names.columns = ["Index", "Name"]
    labels = drug_names.join(name_to_term, on = "Name")
    labels.set_index("Index", inplace = True)

    features = pd.read_csv(f"data/intermediate/{folder}/features.csv", index_col = 0)
    labels.reindex(features.index)
    labels.to_csv(f"data/intermediate/{folder}/labels.csv")
