import numpy
import pandas
import time
import memory_profiler

class ClusteringMethodType:
    
    def __init__(self, name, network_method = False):
        self.name = name
        self.network_method = network_method

    def summary(self):
        return {"type_name":self.name, "is_network_method":str(self.network_method)}

import sklearn.metrics

class ClusteringMethod:

    def apply(self, dataset,  scoring_method = sklearn.metrics.adjusted_rand_score):
        
        memory,_ = memory_profiler.memory_usage((self.cluster, (dataset,)), max_usage = True, retval = True)
        self.labels = self.labels.reindex(dataset.labels.index)
        masked = self.labels.isna() | dataset.labels.isna()
        score = scoring_method(self.labels[~masked], dataset.labels[~masked])   
        return score, self.evaluation_time, sum(masked), memory

    def write(self, dataset, filename, **kwargs):
        parameterdict = dataset.parameter_summary()
        result_summary = dict(zip(["score", "time", "omitted", "max_memory"],self.apply(dataset)))
        temp = pandas.Series({**parameterdict, **self.summary(), **self.methodtype.summary(), **result_summary, **kwargs})
        temp.to_csv(filename, header = None)


class DataClusteringMethod(ClusteringMethod):
    
    def __init__(self, methodtype, name, function, parameters = {}):
        self.methodtype = methodtype
        self.name = name
        self.function = function
        self.parameters = parameters

    def cluster(self, dataset):
        start_time = time.time()
        self.labels = self.function(dataset.data, **self.parameters)
        end_time = time.time()
        self.evaluation_time = end_time - start_time

    def summary(self):
        return {"name":self.name, **self.parameters}

class NetworkClusteringMethod(ClusteringMethod):

    def __init__(self, methodtype, name, function, parameters = {}, network_parameters = ('euclidean', 10)):
        self.methodtype = methodtype
        self.name = name
        self.function = function
        self.network_parameters = network_parameters
        self.parameters = parameters


    def cluster(self, dataset):

        network = dataset.network[self.network_parameters]
        start_time = time.time()
        self.labels = self.function(network, **self.parameters)
        end_time = time.time()
        self.evaluation_time = end_time - start_time + dataset.network_evaluation_time[self.network_parameters] 

    def summary(self):
        return {"name":self.name,
                "network_metric":self.network_parameters[0],
                "network_nneighbors":self.network_parameters[1],
                **self.parameters}

import sklearn.cluster
# k-Means clustering for baseline comparison
def kmeans(X, n_clusters = 10):
    km = sklearn.cluster.KMeans(n_clusters = n_clusters)
    km.fit(X)
    return pandas.Series(km.labels_, index = X.index)
kmeans_type = ClusteringMethodType("k-Means")

# OPTICS (Ordering Points To Infer Cluster Structure)
def optics(X):
    model = sklearn.cluster.OPTICS(min_samples=10, eps = 1000)
    clusters = model.fit_predict(X)
    clusters = pandas.Series(clusters, index = X.index)
    return clusters.replace(-1, numpy.nan)
optics_type = ClusteringMethodType("Optics")

# Spectral Clustering
def spectral_cluster(X, i):
    model = sklearn.cluster.SpectralClustering(n_clusters=i)
    clusters = model.fit_predict(X)
    return pandas.Series(clusters, index = X.index)

sc_type = ClusteringMethodType("Spectral Clustering")

# Data-Net approach
import networkx as nx
import umap
import random

def umap_network(X):
    rndstate = numpy.random.RandomState(10)
    nodes = list(X.index)
    G,_,_ = umap.umap_.fuzzy_simplicial_set(X/X.std(), 10, rndstate, 'euclidean')
    G = nx.from_scipy_sparse_matrix(G)
    return nx.relabel_nodes(G, dict(enumerate(X.index)).get)

def greedyModularity(G):
    nodes = list(G.nodes())
    node2id = {v:i for i,v in enumerate(nodes)}
    H = nx.relabel_nodes(G, node2id)
    clusters = nx.community.modularity_max.greedy_modularity_communities(H)
    df = pandas.DataFrame([[i in a for a in clusters] for i in range(len(nodes))])
    df.index = nodes
    return df.idxmax(axis = 1)
        
gmtype = ClusteringMethodType('GreedyModularity', network_method = True)

import community.community_louvain
def louvain(G):
    return pandas.Series(community.community_louvain.best_partition(G))

lvtype = ClusteringMethodType('Louvain', network_method = True)

import scipy.sparse
def network_spectral(G, n_evectors=30, n_clusters=10):

    nodelist = np.array(G.nodes())

    # rescale rows and columns by degree
    normalized_adjacency = scipy.sparse.eye(G.order()) - nx.normalized_laplacian_matrix(G)
    #normalized_adjacency = nx.adjacency_matrix(G)
    # get eigenvalues and eigenvectors of the matrix
    e,evecs = scipy.sparse.linalg.eigsh(normalized_adjacency, k = n_evectors)

    # reverse the order of eigenvalues and eigenvectors
    e = e[::-1]; evecs = evecs[:,::-1]
    
    # cluster the normalized vectors
    import sklearn.cluster
    m = sklearn.cluster.KMeans(n_clusters = n_clusters)
    m.fit(evecs/np.sum(evecs, axis = 1).reshape(-1,1))
    return pd.Series(m.labels_, index=nodelist)

nstype = ClusteringMethodType('network_spectral', network_method = True)



# Autoencoder
import tensorflow.keras as keras
from tensorflow.keras import layers

def autoencode(df, encoding_dim = 2, validation_split = 0.1):
    n = len(df.columns)
    df = (df - df.min()) / (df.max() - df.min())
    input_img = keras.Input(shape=(n,))
    encoded = layers.Dense(encoding_dim, activation='relu')(input_img)
    decoded = layers.Dense(n, activation='sigmoid')(encoded)
    autoencoder = keras.Model(input_img, decoded)
    encoder = keras.Model(input_img, encoded)
    encoded_input = keras.Input(shape=(encoding_dim,))
    decoder_layer = autoencoder.layers[-1]
    decoder = keras.Model(encoded_input, decoder_layer(encoded_input))
    
    autoencoder.compile(optimizer='adam', loss='mse')
    autoencoder.fit(df.values, df.values,
                    epochs=1000,
                    batch_size=256,
                    shuffle=True,
                    verbose=0,
                    validation_split = validation_split)
    
    codes = encoder.predict(df)
    return codes


import sklearn.cluster
def autencoded_clustering(df, encoding_dim = 2,n_clusters =10 ):
    codes = autoencode(df,encoding_dim=encoding_dim, validation_split = 0.0)
    km = sklearn.cluster.KMeans(n_clusters=n_clusters)
    km.fit(codes)
    return  pandas.Series(km.labels_, index = df.index)

autoencode_type = ClusteringMethodType("Autoencode")

    
clustering_methods = []

for i in [3,4,5]:
    clustering_methods.append(DataClusteringMethod(autoencode_type, 
                                               f"{i}_dimensional_autoencoder",
                                                lambda X:autencoded_clustering(X, encoding_dim = i))
                             )

for i in [9,10,11]:
    clustering_methods.append(DataClusteringMethod(sc_type, 
                                               f"spectral_clustering_{i}_dimensions",
                                                lambda X:spectral_cluster(X, i))
                             )

for i in [9,10,11]:
    clustering_methods.append(DataClusteringMethod(kmeans_type,
                                               f"k_means_{i}",
                                               lambda X:kmeans(X, n_clusters=i))
                             )

clustering_methods.append(DataClusteringMethod(optics_type, 
                                           "optics",
                                            optics)
                         )

clustering_methods.append(NetworkClusteringMethod(lvtype,
                                           'louvain',
                                           louvain)
                         )

clustering_methods.append(NetworkClusteringMethod(nstype,
                                           'network_spectral',
                                           network_spectral)
                         )

clustering_methods.append(NetworkClusteringMethod(gmtype,
                                            'greedy_modularity',
                                            greedyModularity
                                            )
                         )

clustering_method_dict = {c.name:c for c in clustering_methods}

class DataSet:

    
    def __init__(self, **kwargs):
        """
        Makes a dataset 
        """
        self.parameters = kwargs

    def parameter_summary(self):
        return self.parameters

import json
import pandas as pd
import re
def load(foldername):

    parameterdict = json.load(open(f"{foldername}/parameters.json"))
    dataset = DataSet(**parameterdict)

    dataset.data = pd.read_csv(f"{foldername}/features.csv", index_col = 0)
    dataset.labels = pd.read_csv(f"{foldername}/labels.csv", index_col = 0)["labels"]

    dataset.network = {}
    dataset.network_evaluation_time = {}
    for filename in [i for i in os.listdir(foldername) if i.split(".")[-1] == "gml"]:

        name = re.match("(?P<name>.*?).gml", filename).groupdict()['name']

        dataset.network[name] = nx.read_gml(f"{foldername}/{name}.gml")
        fp = open(f"{foldername}/evaluation_time_{name}")
        dataset.network_evaluation_time[name]  = float(fp.read().strip())

    return dataset


if __name__ == "__main__":

     import os
     import sys
     import re

     clustering_method_name = sys.argv[1]
     dataset_filename = sys.argv[2]
    
     try:
         seed = int(sys.argv[3])
     except IndexError:
         seed = 108

     numpy.random.seed(seed)
     random.seed(seed)

     output_filename = clustering_method_name + "_" + re.sub("/", "_", dataset_filename) + "_" + str(seed) + ".csv"

     method = clustering_method_dict[clustering_method_name]
     dataset = load(dataset_filename)

     method.write(dataset,
                  "data/processed/clustering/"+output_filename)