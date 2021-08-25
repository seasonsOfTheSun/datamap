import pandas as pd
import numpy as np
import string
import time
import copy
import os
import json

# Misc. utils.
def randstring():return "".join(np.random.choice(list(string.ascii_lowercase), (10,)))

import networkx as nx
import umap
import random
def umap_network(X, nneighbors = 10, metric = 'euclidean'):
    rndstate = np.random.RandomState(nneighbors)
    nodes = list(X.index)
    G,_,_ = umap.umap_.fuzzy_simplicial_set(X/X.std(), 10, rndstate, metric)
    G = nx.from_scipy_sparse_matrix(G)
    return nx.relabel_nodes(G, dict(enumerate(X.index)).get)

# Make clusters 
class SyntheticDataSet:
    """ """
    
    def __init__(self, n_clusters=2,
                       dimension=2, 
                       center_d=1,
                       scale=0.25, 
                       size=10,
                       final_dimension = 100,
                       ellipticity = 0, 
                       scale_range=0, 
                       center_d_range=0, 
                       size_range=0, 
                       transform_dataset = "pass"
                       ):
        self.n_clusters = int(n_clusters)
        self.dimension = int(dimension)
        self.scale = scale
        self.center_d = center_d
        self.size = size
        self.scale_range=scale_range
        self.center_d_range=center_d_range
        self.size_range=size_range
        self.ellipticity=ellipticity
        self.transform_dataset = transform_dataset
        self.network = {}
        self.network_evaluation_time = {}
        self.final_dimension = final_dimension
        
    def vary_clusters(self):
        """
        Make the variables controlling the clusters (e.g. cluster size) 
        vary according to some predefined range.
        
        """
        n_clusters = self.n_clusters
        
        if type(self.scale) != list:
            self.scale_expanded = [self.scale+(i-0.5*n_clusters)*self.scale_range/n_clusters for i in range(n_clusters)]

        if type(self.center_d) != list:
            self.center_d_expanded = [self.center_d+(i-0.5*n_clusters)*self.center_d_range/n_clusters for i in range(n_clusters)]
        
        if type(self.size) != list:
            self.size_expanded = [int(self.size+(i-0.5*n_clusters)*self.size_range/n_clusters) for i in range(n_clusters)]
    
    def make_dataset(self):
        """ Create the features from the parameters given, then """
        
        self.vary_clusters()
        
        # Make cluster centers
        randdirs = np.random.randn(self.n_clusters, self.dimension)
        randdirs = randdirs / np.sqrt((randdirs**2).sum(axis = 1)).reshape((self.n_clusters,1))
        self.centers = np.array([[i] for i in self.center_d_expanded]) * randdirs
        
        out = []# Make the data points within each cluster and scale and distribute accordingly
        for i,n in enumerate(self.size_expanded):
            temp = self.scale_expanded[i]*np.random.randn(n, self.dimension)
            
            # Add variation to the clusters along different axes so they are less spherical
            temp = (1 + self.ellipticity*np.random.rand(self.dimension)).reshape(-1,self.dimension) * temp
            temp = pd.DataFrame(temp)
            temp = temp + self.centers[i,:]
            out.append(temp)


        # Join into dataframes
        self.data= pd.concat(out)
        self.labels = np.concatenate([[i]*v for i,v in enumerate(self.size_expanded)])

    
        # Consistent names for columns and indices
        self.elementnames = [randstring() for i in range(len(self.data.index))]
        self.data.index = self.elementnames
        self.original_features = [randstring() for i in range(len(self.data.columns))]


        self.data.columns = self.original_features
        self.labels = pd.Series(self.labels, index = self.elementnames)
                
        self.data = self.data.sample(frac=1) # re-order the datapoints so that nothing 
                                             # can be accidentally inferred form their ordering.
            
        exec(self.transform_dataset) # apply a nonlinear transform to creat a new set of features

        metric = 'euclidean'
        nneighbors = 10
                
        start_time = time.time()
        self.network[(metric, nneighbors)] = umap_network(self.data, nneighbors = nneighbors, metric = metric)
        end_time = time.time()
        self.network_evaluation_time[(metric, nneighbors)] = end_time - start_time


    def parameter_summary(self):

        parameterdict = {'n_clusters':self.n_clusters,
        'dimension':self.dimension,
        'center_d':self.center_d,
        'scale':self.scale,
        'size':self.size,
        'ellipticity':self.ellipticity,
        'size_range':self.size_range,
        'scale_range':self.scale_range,
        'center_d_range':self.center_d_range,
        'size_range':self.size_range,
        'transform_dataset':self.transform_dataset}
        
        return parameterdict


    def save(self, foldername = "scratch",  save_tabular_data = True):

        os.makedirs(foldername, exist_ok = True)

        parameterdict = {'n_clusters':self.n_clusters,
        'dimension':self.dimension,
        'center_d':self.center_d,
        'scale':self.scale,
        'size':self.size,
        'ellipticity':self.ellipticity,
        'size_range':self.size_range,
        'scale_range':self.scale_range,
        'center_d_range':self.center_d_range,
        'size_range':self.size_range,
        'transform_dataset':self.transform_dataset}

        fp = open(f"{foldername}/parameters.json",'w')
        json.dump(parameterdict, fp)
        fp.close()

        self.data.to_csv(f"{foldername}/features.csv")
        self.labels.to_csv(f"{foldername}/labels.csv")

        for parameters, network in self.network.items():
            
            for edge in network.edges():
                network.edges()[edge]['weight'] = str(network.edges()[edge]['weight'])

            metric  = parameters[0]
            nneighbors =  parameters[1]

            nx.write_gml(network,f"{foldername}/metric_{metric}_nneighbors_{nneighbors}.gml")
            fp = open(f"{foldername}/evaluation_time_metric_{metric}_nneighbors_{nneighbors}", 'w')
            fp.write(str(self.network_evaluation_time[parameters]))
            fp.close()

import re
def load(foldername):

    parameterdict = json.load(open(f"{foldername}/parameters.json"))
    dataset = SyntheticDataSet(parameterdict['n_clusters'],
                                     parameterdict['dimension'],
                                     parameterdict['center_d'],
                                     parameterdict['scale'],
                                     parameterdict['size'],
                                     parameterdict['ellipticity'],
                                     parameterdict['scale_range'],
                                     parameterdict['center_d_range'],
                                     parameterdict['size_range'],
                                     transform_dataset = parameterdict['transform_dataset'])



    dataset.data = pd.read_csv(f"{foldername}/features.csv", index_col = 0)
    dataset.labels = pd.read_csv(f"{foldername}/labels.csv", index_col = 0, header = None)[1]

    for filename in [i for i in os.listdir(foldername) if i.split(".")[-1] == "gml"]:

        g = re.match("metric_(?P<metric>.*?)_nneighbors_(?P<nneighbors>.*?).gml", filename)
        parameters = g.groupdict()
        metric = parameters['metric']
        nneighbors = int(parameters['nneighbors'])

        dataset.network[(metric, nneighbors)] = nx.read_gml(f"{foldername}/metric_{metric}_nneighbors_{nneighbors}.gml")
        fp = open(f"{foldername}/evaluation_time_metric_{metric}_nneighbors_{nneighbors}")
        dataset.network_evaluation_time[(metric, nneighbors)]  = float(fp.read().strip())

    return dataset



class SyntheticDataSetSeries:

    def __init__(self, start_dataset, attr, value_range):
        
        self.start_dataset = start_dataset
        self.attr = attr
        self.value_range = value_range

    def make_series(self):
        """Make the series by changin the value of one of the parameters,
        or the mean value if the parameter varies by cluster within the dataset.
        """ 
        out = []
        for i in self.value_range:
            copied_dataset = copy.deepcopy(self.start_dataset)
            copied_dataset.__setattr__(self.attr, i)
            copied_dataset.vary_clusters()
            copied_dataset.make_dataset()
            out.append(copied_dataset)
        self.datasets = out

    def save(self, foldername = "data/synthetic/scratch",  save_tabular_data = True):
        
        os.makedirs(foldername, exist_ok = True)
        for i, dataset in enumerate(self.datasets):
            dataset.save(foldername + f"/dataset_{i}/")

if __name__ == "__main__":
    

    base_dataset = SyntheticDataSet(
        n_clusters=10,
        dimension=5, 
        center_d=10,
        scale=10, 
        size=10,
        
        transform_dataset = """
amplitude = 1
period = 10
n = 100

for i in range(self.final_dimension):
    col = np.random.choice(self.original_features, n)
    randmat = np.random.randn(n)
    bart = (randmat.reshape((1,n)) * self.data[col]).sum(axis = 1)

    self.data[f"Transformed_{i}"] = list(map(lambda x : x**2, bart))

for col in self.original_features:
    del self.data[col]

self.data = self.data/self.data.var()
true_dimension = len(self.data.columns)"""
    )

    scale_dataset = SyntheticDataSetSeries(
        base_dataset,
        'scale',
        list(np.linspace(0.0, 20.0, 11))
    )
    scale_dataset.make_series()
    scale_dataset.save("data/synthetic/scale")

    size_dataset = SyntheticDataSetSeries(
        base_dataset,
        'size',
        [int(i) for i in np.logspace(0,5,10)]
    )
    size_dataset.make_series()
    size_dataset.save("data/synthetic/size")

    dimension_dataset = SyntheticDataSetSeries(
        base_dataset,
        'dimension',
        [int(i) for i in np.logspace(0.2,2.5,10)]
    )
    dimension_dataset.make_series()
    dimension_dataset.save("data/synthetic/dimension")

    final_dimension_dataset = SyntheticDataSetSeries(
        base_dataset,
        'final_dimension',
        [int(i) for i in np.logspace(0.2,3,10)]
    )
    final_dimension_dataset.make_series()
    final_dimension_dataset.save("data/synthetic/final_dimension"s)

