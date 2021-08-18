import numpy as np
import pandas as pd
import re
import sys

import networkx as nx

import urllib.parse as up
import urllib.request as ur

import subprocess as spop
import itertools as it
import json

salt_to_remove = [" maleate", " hydrochloride", " nitrate", 
                  " dihydrochloride", " chloride", " sulfate", 
                  " hydrate", " mesylate", " oxalate", " salt",
                  " from Penicillium brefeldianum", " monohydrate",
                  " trifluoroacetate", " acetate", " isethionate",
                  " hemisulfate", " angular", " sodium", " fumarate",
                  " methanesulfonate", " hemihydrate", " (MW = 374.83)",
                  "\(\+/\-\)-", "\(\+\)-", "\(\-\)-", "S-\(\+\)-", "\(S\)-", "\(±\)-", "D-"]

def remove_salts(cpd):
    try:
        cpd = cpd.lower()
        for s in salt_to_remove:
            cpd = re.sub(s, "", cpd)
        return cpd
    except Exception:
        return np.nan
    
    
def cidFromName(name):
    header = "'Content-Type: application/x-www-form-urlencoded'"
    post_req = 'name='+name
    url = 'https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/cids/txt'

    res = spop.Popen(['curl','--data-urlencode', post_req, url], stderr=spop.PIPE, stdout = spop.PIPE)
    response_string = res.stdout.read().decode('utf-8')

    time.sleep(0.2)
    return response_string.split('\n')[0]

def get_pubchem_info(cid):
    header = "'Content-Type: application/x-www-form-urlencoded'"
    post_req = 'cid='+cid
    url = 'https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/classification/json'
    res = spop.Popen(['curl','--data-urlencode', post_req, url], stdout = spop.PIPE, stderr=spop.PIPE)
    response_string = res.stdout.read().decode('utf-8')
    return json.loads(response_string)








def update_sourcenode2term(sourcenode2term, info_source):
    for nodedict in info_source['Node']:
        try:
            term = nodedict['Information']['Name']
        except KeyError:
            term = nodedict['Information']['Description']
        nodeid = nodedict['NodeID']

        try:
            old_term = sourcenode2term[nodeid]
            assert term == old_term
        except KeyError:
            pass

        sourcenode2term[nodeid] = term

def update_hierarchy(hierarchy, sourcenode2name, info_source):
    for nodedict in info_source['Node']:

        try:
            term = nodedict['Information']['Name']
        except KeyError:
            term = nodedict['Information']['Description']
        
        nodeid = nodedict['NodeID']

        try:
            description = nodedict['Information']['Description']
        except:
            description = ""

        hierarchy.add_node(term, description=description)

        try:
            hierarchy.nodes()[term]['drugs'] |= {":".join((name, cid))}
        except KeyError:
            hierarchy.nodes()[term]['drugs'] = {":".join((name, cid))}
            

        for parent in nodedict['ParentID']:
                parent_term = sourcenode2name[parent]
                hierarchy.add_edge(parent_term, term)

def incorporate_source(hierarchy_dict, node2name, info_source, name, cid):

    source = info_source['SourceName']

    try:
        sourcenode2name = node2name[source]
    except KeyError:
        sourcenode2name = {'root':'Root'}
        node2name[source] = sourcenode2name
        
        
    try:
        hierarchy = hierarchy_dict[source]
    except KeyError:
        hierarchy = nx.DiGraph()
        hierarchy_dict[source] = hierarchy

        
        

    update_sourcenode2term(sourcenode2name, info_source)
    update_hierarchy(hierarchy, sourcenode2name, info_source)
    

def incorporate_response(hierarchy_dict, node2name, pubchem_response, name, cid):
    try:
        info_sources = pubchem_response['Hierarchies']['Hierarchy']
        assert cid != 'Status: 404'
    except Exception:
        return

    for info_source in info_sources:
        source = info_source['SourceName']
        if source == 'Cooperative Patent Classification (CPC)':
            continue # ignore becuase this one is pure nonsense 
        if source == 'KEGG':
            continue # ignore becuase more trouble than it's worth
        if source == 'PubChem':
            continue # ignore becuase just an aggregator for other sources


        incorporate_source(hierarchy_dict, node2name, info_source, name, cid)

all_drugs = set()
for foldername in ["cell_line", "transcriptional", "morphological"]: #sys.argv[2]
    drug_name_file = f"../data/intermediate/{foldername}/drug_names.csv" #sys.argv[1]
    drugs =  pd.read_csv(drug_name_file,index_col =0, header = None)[1]
    unique_drugs = drugs.unique()
    unique_drugs = {i for i in unique_drugs if i == i}
    unique_drugs = {i for i in unique_drugs if i[:4] != 'BRD-'}
    
    all_drugs |= unique_drugs











import time
N = len(all_drugs)
tick = 0

start_time = time.time()

hierarchy_dict = {}
node2name = {}

    
for name in list(all_drugs):
    
    try:
        tick += 1
        current_time = time.time()
        tps = tick / (current_time - start_time)
        print('Estimated completion at:', time.ctime(current_time + (N - tick) / tps), end = '\r')


        cid = cidFromName(name)

        if cid == 'Status: 404':
            continue

        pubchem_response = get_pubchem_info(cid)
        incorporate_response(hierarchy_dict, node2name, pubchem_response, name, cid)
    except Exception:
        pass
        
    
for source in hierarchy_dict.keys():
    for node in hierarchy_dict[source].nodes():
        try:
            drugs =  hierarchy_dict[source].nodes()[node]['drugs']

            hierarchy_dict[source].nodes()[node]['drugs'] = list(drugs)
        except KeyError:
            pass
    filename = "".join([x if x.isalnum() else "_" for x in source])
    nx.write_gml(hierarchy_dict[source], f"{filename}.gml")