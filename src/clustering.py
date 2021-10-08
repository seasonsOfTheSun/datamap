import numpy
import pandas
import time
import memory_profiler

class ClusteringMethodType:
    
    def __init__(self, name, color, network_method = False):
        self.name = name
        self.color = color
        self.network_method = network_method

    def summary(self):
        return {"name":self.name, "color":self.color, "is_network_method":str(self.network_method)}

import sklearn.metrics

class ClusteringMethod:

    def apply(self, dataset,  scoring_method = sklearn.metrics.adjusted_rand_score):
        
        memory,_ = memory_profiler.memory_usage((self.cluster, (dataset,)), max_usage = True, retval = True)
        self.labels = self.labels.reindex(dataset.labels.index)
        masked = self.labels.isna() | dataset.labels.isna()
        score = scoring_method(self.labels[~masked], dataset.labels[~masked])   
        return score, self.evaluation_time, sum(masked), memory[0]

    def write(self, dataset, filename, dataset_name = "default"):
        parameterdict = dataset.parameter_summary()
        result_summary = dict(zip(["score", "time", "omitted", "max_memory"],self.apply(dataset)))
        temp = pandas.Series({"dataset_name":dataset_name, **parameterdict, **self.summary(), **self.methodtype.summary(), **result_summary})
        temp.to_csv(filename, header = None)


class DataClusteringMethod(ClusteringMethod):
    
    def __init__(self, methodtype, name_specific, function, parameters = {}):
        self.methodtype = methodtype
        self.name_specific = name_specific
        self.function = function
        self.parameters = parameters

    def cluster(self, dataset):
        start_time = time.time()
        self.labels = self.function(dataset.data, **self.parameters)
        end_time = time.time()
        self.evaluation_time = end_time - start_time

    def summary(self):
        return {"specific_name":self.name_specific, **self.parameters}

class NetworkClusteringMethod(ClusteringMethod):

    def __init__(self, methodtype, name_specific, function, parameters = {}, network_parameters = ('euclidean', 10)):
        self.methodtype = methodtype
        self.name_specific = name_specific
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
        return {"specific_name":self.name_specific,
                "network_metric":self.network_parameters[0],
                "network_nneighbors":self.network_parameters[1],
                **self.parameters}

cmap = {"black":"#000000",
"princeton-orange": "#ee8434ff",
"cobalt-blue": "#1446a0ff",
"razzmatazz": "#db3069ff",
"maximum-green": "#698f3fff",
"medium-purple": "#a682ffff",
"turquoise": "#42d9c8ff",
"mindaro": "#ddfc74ff",
"cyan-process": "#01baefff", 
"dark-pastel-green": "#20bf55ff", 
"orchid-pink": "#f6c0d0ff"}

import sklearn.cluster
# k-Means clustering for baseline comparison
def kmeans(X, n_clusters = 10):
    km = sklearn.cluster.KMeans(n_clusters = n_clusters)
    km.fit(X)
    return pandas.Series(km.labels_, index = X.index)
kmeans_type = ClusteringMethodType("k-Means", "#000000")

# OPTICS (Ordering Points To Infer Cluster Structure)
def optics(X):
    model = sklearn.cluster.OPTICS(min_samples=10, eps = 1000)
    clusters = model.fit_predict(X)
    clusters = pandas.Series(clusters, index = X.index)
    return clusters.replace(-1, numpy.nan)
optics_type = ClusteringMethodType("Optics",  cmap["turquoise"])

# Spectral Clustering
def spectral_cluster(X, i):
    model = sklearn.cluster.SpectralClustering(n_clusters=i)
    clusters = model.fit_predict(X)
    return pandas.Series(clusters, index = X.index)

sc_type = ClusteringMethodType("Spectral Clustering",  cmap["turquoise"])

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
        
gmtype = ClusteringMethodType('GreedyModularity', cmap["medium-purple"], network_method = True)

import community.community_louvain
def louvain(G):
    return pandas.Series(community.community_louvain.best_partition(G))

lvtype = ClusteringMethodType('Louvain', cmap["razzmatazz"], network_method = True)

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
def autencoded_clustering(df, encoding_dim = 2, validation_split = 0.0):
    codes = autoencode(df,encoding_dim=encoding_dim, validation_split =validation_split)
    km = sklearn.cluster.KMeans(n_clusters =10)
    km.fit(codes)
    return  pandas.Series(km.labels_, index = df.index)

autoencode_type = ClusteringMethodType("Autoencode",  cmap["princeton-orange"])

    
clustering_methods = []

for i in range(2,3):
    clustering_methods.append(DataClusteringMethod(autoencode_type, 
                                               f"{i}-Dimensional Autoencoder",
                                                lambda X:autencoded_clustering(X, encoding_dim = i))
                             )

for i in range(8,10):
    clustering_methods.append(DataClusteringMethod(sc_type, 
                                               f"Spectral Clustering {i} Dimensions",
                                                lambda X:spectral_cluster(X, i))
                             )

for i in range(1,20):
    clustering_methods.append(DataClusteringMethod(kmeans_type,
                                               f"k-Means {i}",
                                               lambda X:kmeans(X, n_clusters=i))
                             )

clustering_methods.append(DataClusteringMethod(optics_type, 
                                           "Optics",
                                            optics)
                         )

clustering_methods.append(NetworkClusteringMethod(lvtype,
                                           'Louvain',
                                           louvain)
                         )

clustering_methods.append(NetworkClusteringMethod(gmtype,
                                            'GreedyModularity',
                                            greedyModularity
                                            )
                         )

clustering_method_dict = {c.name_specific:c for c in clustering_methods}

if __name__ == "__main__":

     import os
     import sys
     import synthetic_data
     import uuid

     clustering_method_name = sys.argv[1]
     dataset_filename = sys.argv[2]
     output_folder = sys.argv[3]
     if output_folder[-1] != "/":
         output_folder += "/"
     output_filename = output_folder + str(uuid.uuid4()) + ".csv"
     method = clustering_method_dict[clustering_method_name]
     dataset = synthetic_data.load(dataset_filename)

     method.write(dataset, output_filename)
