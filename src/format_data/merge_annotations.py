import networkx as nx
import pandas as pd
import json

def merge_into(G1,G2):
    for node in G2.nodes():
        attrs = G2.nodes()[node]
        if node not in G1.nodes():
                G1.add_node(node, **attrs)
    for edge in G2.edges():
        G1.add_edge(edge[0],edge[1])

G_morphology = nx.read_gml("data/external/morphological/MeSH_info_hierarchy.gml")
G_transcriptional = nx.read_gml("data/external/transcriptional/MeSH_info_hierarchy.gml")
G_cell_line = nx.read_gml("data/external/cell_line/MeSH_info_hierarchy.gml")

# 
merge_into(G_morphology,G_cell_line)
merge_into(G_morphology,G_transcriptional)
G = G_morphology
G.nodes()['root']['Name'] = 'The Root' # Make sure every node has a Name attribute to avoid KeyErrors

# Load all the dictionaries that map compound ids to nodes...
x=json.load(open("data/external/morphological/cid_to_MeSH_info.json"))
y=json.load(open("data/external/transcriptional/cid_to_MeSH_info.json"))
z=json.load(open("data/external/cell_line/cid_to_MeSH_info.json"))

# os.
# then merge them all together into a single dictionary
out = {}
for cid in set(x.keys()) | set(y.keys()) | set(z.keys()):
    
    try:
        x_cid = x[cid]
    except KeyError:
        x_cid = []
    try:
        y_cid = y[cid]
    except KeyError:
        y_cid = []
    try:
        z_cid = z[cid]
    except KeyError:
        z_cid = []
    
    out[cid] = x_cid + y_cid + z_cid



# Make a network that has the name of each MeSH attribute as 
# a node with cids allocated and .
G_name = nx.DiGraph()
G.nodes()['root']['Name'] = 'Root'
for cid, nodes in out.items():
    for node in nodes:
        name = G.nodes()[node]['Name']
        if name == 'Supplementary Records':
            continue

        G_name.add_node(name)
        try:
            G_name.nodes()[name]['drugs'] |= {cid}
        except KeyError:
            G_name.nodes()[name]['drugs'] = {cid}

        for node2,_ in G.in_edges(node):
            name2 = G.nodes()[node2]['Name']
            G_name.add_edge(name2, name) 
            
# Load all the dictionaries that map names to compound ids
x=json.load(open("data/external/morphological/names_to_cids.json"))
y=json.load(open("data/external/transcriptional/names_to_cids.json"))
z=json.load(open("data/external/cell_line/names_to_cids.json"))

# os.
# then merge them all together into a single dictionary
pd.Series({**x, **y, **z}).to_csv("cid_to_names.csv")

# 
def extract_singleton(set1):
    if len(set1) != 1:
        raise ValueError('Not a singleton set')
    return set1.__iter__().__next__()
    
for name in G_name.nodes():
    if name == 'Supplementary Records':
        continue
    try:
        desc = {G.nodes()[node]['Description'][0] for node in set(G_name.nodes()[name]['nodes'])}
        print(desc)
        G_name.nodes()[name]['Description'] = extract_singleton(desc)
    except KeyError:
        pass
    
    try:
        url = {G.nodes()[node]['URL'] for node in set(G_name.nodes()[name]['nodes'])}
        G_name.nodes()[name]['URL'] = extract_singleton(url)
    except KeyError:
        pass

    try:
        url = {G.nodes()[node]['HNID'] for node in set(G_name.nodes()[name]['nodes'])}
        G_name.nodes()[name]['HNID'] = extract_singleton(url)
    except KeyError:
        pass

    try:
        G_name.nodes()[name]['drugs']=list(G_name.nodes()[name]['drugs'])
    except KeyError:
        pass




nx.write_gml(G_name, "data/external/All_MeSH_annotations.gml")
