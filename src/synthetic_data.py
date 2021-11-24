import numpy as np
import pandas as pd

import string
import os
import json
import time
import copy

import networkx as nx
import umap
import random
def umap_network(X, nneighbors = 10, metric = 'euclidean'):
    rndstate = np.random.RandomState(nneighbors)
    nodes = list(X.index)
    G,_,_ = umap.umap_.fuzzy_simplicial_set(X/X.std(), 10, rndstate, metric)
    G = nx.from_scipy_sparse_matrix(G)
    return nx.relabel_nodes(G, dict(enumerate(X.index)).get)


def randstring():return "".join(np.random.choice(list(string.ascii_lowercase), (10,)))

class NoNormalError(Exception):
    
    def __init__(self):
        pass

def make_normal(span, seed_vector = None):
    """
    Makes a unit normal vector to a subspace.
    
    Produces a unit vector orthogonal to all columns of `span`.
    
    Parameters
    ----------
    span : an ndarray for whose column-span a normal is desired.
    seed_vector : a vector
    
    Returns
    -------
    normal : a 1-dimensionl unit vector
    
    Raises
    ------
    NoNormalError if the seed vector was in the hyperplane
    (likely if the hyperplane was the entire space)
    
    """
    
    # get a random seed vector if none provided
    if seed_vector == None:
        seed_vector = np.random.randn(span.shape[0])
        
    # find the nearest point to seed_vectoor inside the plane using 
    # np.lstsq, and then find the 
    a,_,_, _ = np.linalg.lstsq(span, seed_vector)
    normal = seed_vector - np.matmul(span, a)
    
    if np.linalg.norm(normal) < 1e-10: # if the normal is really small i.e. was in the span
        raise NoNormalError # raise an error
        
    return normal / np.linalg.norm(normal)# o/w normalize

def build_simplex(space_dimension, simplex_dimension = None, center_origin = False):
    """
    make a n-simplex with unit length edges embedded in high dimensional space
    
    where n is `simplex_dimension`, 
    
    Parameters
    ----------
    space_dimension : Integer, dimension of space the simplex is embedded into.
    simplex_dimension : Integer, number of points in the simplex.
    center_origin : bool, if true recenter the simplex to the origin
    
    Returns
    -------
    simplex : an ndarray with shape (space dimension, simplex_dimension) 
    """
    if simplex_dimension == None:
        simplex_dimension = space_dimension
        
    # initialise a random point exactly distance 1 from  the origin
    point = np.random.randn(space_dimension)
    point = point/np.linalg.norm(point)

    # make the origin and this point into a 1-simplex
    simplex = np.vstack([np.array([0]*space_dimension), point]).T
    
    # build up the simplex point by point, dimension by dimension
    for i in range(simplex_dimension-2):
        
        # get the central point of the simplex
        avg =  np.mean(simplex, axis = 1).reshape(-1,1)
        base_radii = np.linalg.norm(simplex - avg, axis = 0)# distance of the centre to the points of the simplex
        base_radius = base_radii[0]# these are all equal so pick the 0th entry indifferently
        assert all(base_radius - base_radii < 1e-9)# check they're equal
        
        # get a unit vector normal to the simplex
        normal = make_normal(simplex)
        new_point = avg + np.sqrt(1 - base_radius**2)*normal.reshape(-1,1)# use pyhtogoaras to project it to a new point opf the simplex
        assert all(np.linalg.norm(simplex - new_point, axis = 0) - 1 < 1e-9)
        
        simplex = np.hstack([simplex, new_point])
        
    if center_origin:
        # recenter the simplex to the origin
        avg =  np.mean(simplex, axis = 1).reshape(-1,1)
        simplex = simplex-avg
        
        
    return simplex

def make_ctime(pretransformed_dimension, d = 0, std = 1):
    """ Makes a temporary (linear) feature of the data that can be transformed into a nonlinear feature
    the degree of Spearman correlation between these features can be tweaked via the parameters d and std
    
    Parameters
    ----------
    d : the offset of the cloud that the directions are renormalized from
    
    std : the standard deviation of the cloud that the directions are renormalized from"""
    
    # draw from the cloud
    mean = np.array([d]+[0]*(pretransformed_dimension-1))
    to_normalize = std * np.random.randn(pretransformed_dimension) + mean 

    # and normalize
    return to_normalize/np.linalg.norm(to_normalize)

def fold_data(folding_value, height, transition_width):
    """
        # divide points into lower sheet, upper sheet, and hinge region
    # bases on which side of the hinge they are    # calculate  position for points in the hinge, will lie in the arc
    # of a circle connectiong the sheets    # calculate the position of the upper sheet,a mirror image of their original postion, 
    #  shifted up into the newly created dimension
    
    Parameters
    ----------
    folding_value : the values along which we will fold the data
    
"""

    # divide points into lower sheet, upper sheet, and hinge region
    # bases on which side of the hinge they are
    lower_sheet =         folding_value  < -transition_width/2
    hinge_region = np.abs(folding_value) <= transition_width/2
    upper_sheet =         folding_value  > transition_width/2

    # calculate  position for points in the hinge, will lie in the arc
    # of a circle connectiong the sheets
    hinge_new_axis=height * (np.sin(np.pi * (folding_value/transition_width))+1)/2
    hinge_replacement_axis=np.cos(np.pi * (folding_value/transition_width)) - transition_width/2


    # calculate the position of the upper sheet,a mirror image of their original postion, 
    #  shifted up into the newly created dimension
    upper_new_axis=height
    upper_replacement_axis=-folding_value

    
    # output values
    replacement_axis = (upper_replacement_axis*upper_sheet + 
                       hinge_replacement_axis*hinge_region +
                       lower_sheet*folding_value)

    new_axis = (upper_new_axis*upper_sheet + 
                       hinge_new_axis*hinge_region +
                       lower_sheet*0)
    
    return replacement_axis,new_axis  

import numpy as np
def square_wave(t, t_period, xy_period, amplitude, rescale):

    #       _____      _____      _____
    #      |     |    |     |    |     |
    #      |     |    |     |    |     |
    #  ____|     |____|     |____|     |___
    #

    # number of complete periods already elapsed
    n = t//t_period
    
    # fraction of current period elapsed
    f = t%t_period

    x = n*xy_period
    y = 0
    

    if 0 <= f < 0.25: # low horizontal
        x += 2*f*xy_period
        y += -amplitude

    if 0.25 <= f < 0.5: # ascending vertical
        x += 0.5*xy_period
        y += (8*amplitude*(f-0.25) - amplitude)

    if 0.5 <= f < 0.75: # high horizontal
        x += 2*(f-0.5)*xy_period + 0.5*xy_period
        y += amplitude

    if 0.75 <= f <= 1.0:
        x += xy_period
        y += -8*amplitude*(f-0.75) + amplitude

    return x,y

class SyntheticDataSet:

    
    def __init__(self, **kwargs):
        """
        Makes a dataset 


        Parameters
        ----------
        size : int
        n_clusters : int
        pretransformed_dimension : int
        posttransformed_dimension : int
        pretransformed_noise : float
        posttransformed_noise : float

        """
        for i,v in kwargs.items():
            if i in ["value_range", "attr"]:
                pass
            self.__setattr__(i, v)
        
        
            
    def make_pretransformed(self):
        
        centers = build_simplex(self.pretransformed_dimension,
                                self.n_clusters)
        
        cluster_size = self.size // self.n_clusters
        
        out = []
        self.labels = []
        for i in range(self.n_clusters):
            
            # 
            temp = self.pretransformed_noise*np.random.randn(cluster_size,self.pretransformed_dimension)
            temp = temp + centers[:,i]
            
            
            out.append(temp)# 
            self.labels.extend([i]*cluster_size)

        self.pretransformed_data = np.vstack(out)
        
        
    def transform(self):
        """
        get a direction normal to plane, 
        get another direction normal to 
        plane and prev. selected normal
        """
        plane_injection = np.random.randn(self.posttransformed_dimension,
                                          self.pretransformed_dimension)
        previously_spanned = plane_injection

        out = []
        for i in range(self.posttransformed_dimension):

            # 
            ctime = make_ctime(self.pretransformed_dimension, d = 1, std = self.correlation_parameter)
            value = np.matmul(ctime.reshape((1,-1)), self.pretransformed_data.T).flatten()

            # get transformation method and apply it to `value`
            transformation_name = self.__getattribute__("transformation")
            transformation = self.__getattribute__(transformation_name)
            transformed_values = transformation(value)
            
            # 
            out.append(transformed_values.T)
            
        self.data = np.hstack(out)[:,:self.posttransformed_dimension]
        
    def square_wave(self, value):
        """ self.t_period, self.xy_period, self.amplitude, self.rescale"""
        xy = [square_wave(i, self.t_period, self.xy_period, self.amplitude, self.rescale) for i in value]
        replacement_axis = [i[0] for i in xy]
        new_axis = [i[1] for i in xy]
        return np.hstack([replacement_axis, new_axis]).reshape((2,-1))
    
    def fold(self, value):
        x,y = fold_data(value, self.height, self.transition_width)
        return np.hstack([x,y]).reshape((2,-1))
    
    def none(self, value):
        return value
             
    def add_more_noise(self):
        """ Add more noise to the transformed data,
        obscuring manifold shape.
        """
        self.data += self.posttransformed_noise * np.random.randn(self.data.shape[0], self.data.shape[1])

    def add_metadata(self):
        """ """
        # save data as a dataframe and create names for rows and columns
        self.data = pd.DataFrame(self.data)
        self.elementnames = [randstring() for i in range(self.size)]
        self.data.index = self.elementnames
        self.original_features = [randstring() for i in range(self.posttransformed_dimension)]

        # make sure the labels are given the same element names
        self.data.columns = self.original_features
        self.labels = pd.Series(self.labels, index = self.elementnames, name = "labels")

        # scramble the orders the elements are in so cluster can't be learned form the order
        self.data = self.data.sample(frac=1)
        
    def make_network(self):
        
        self.network = {}
        self.network_evaluation_time = {}
        network_name = "nneighbors_10_metric_euclidean"
        
        start_time = time.time()
        self.network[network_name] = umap_network(self.data, nneighbors = 10, metric = 'euclidean')
        end_time = time.time()
        self.network_evaluation_time[network_name] = end_time - start_time
        
        
        
    def parameter_summary(self):
        unnecessary_keys = ['labels',
                    'pretransformed_data',
                    'data',
                    'original_features',
                    'elementnames',
                    'network',
                    'network_evaluation_time']
            
        return {k:v for k,v in self.__dict__.items() if k not in unnecessary_keys}

    def make(self):
        self.make_pretransformed()
        self.transform()
        self.add_more_noise()
        self.add_metadata()
        self.make_network()
        
    def save(self, foldername = "scratch",  save_tabular_data = True):
        """# Save the parameters used to create the dataset
            # potentially save the datasets, save the different networks created using those features
            """
        os.makedirs(foldername, exist_ok = True)
        
        # Save the parameters used to create the dataset
        parameterdict = self.parameter_summary()
        fp = open(f"{foldername}/parameters.json",'w')
        json.dump(parameterdict, fp)
        fp.close()
        
        # potentially save the datasets
        if save_tabular_data:
            self.data.to_csv(f"{foldername}/features.csv")
            self.labels.to_csv(f"{foldername}/labels.csv")
        
        # save the different networks created using those features
        for network_name, network in self.network.items():
            
            for edge in network.edges():# 
                network.edges()[edge]['weight'] = str(network.edges()[edge]['weight'])
            
            nx.write_gml(network,f"{foldername}/{network_name}.gml")
            fp = open(f"{foldername}/evaluation_time_{network_name}", 'w')
            fp.write(str(self.network_evaluation_time[network_name]))
            fp.close()        

if __name__ == "__main__":
    import sys
    name = sys.argv[1]                
    if  name == "amplitude":
        x = {"size" : 1000,
        "n_clusters" : 2,
        "pretransformed_dimension" : 2,
        "posttransformed_dimension" : 7,
        "pretransformed_noise" : 0.5,
        "posttransformed_noise" : 0.3,
        "transformation" : "square_wave",
        "t_period" : 1,
        "xy_period" : 1,
        "amplitude" : 1,
        "rescale" : 1,
        "correlation_parameter" : 2}

        for i in np.linspace(1,101,11):
            x["amplitude"] = int(i)
            self = SyntheticDataSet(**x)
            self.make()
            self.save("data/synthetic/amplitude_"+str(round(i,2)))
            
    elif name == "final_dimension":
        x = {"size" : 1000,
        "n_clusters" : 2,
        "pretransformed_dimension" : 2,
        "posttransformed_dimension" : 7,
        "pretransformed_noise" : 0.5,
        "posttransformed_noise" : 0.3,
        "transformation" : "fold",
        "height" : 3,
        "transition_width":1,
        "correlation_parameter" : 2}

        for i in np.linspace(1,101,11):
            x["posttransformed_dimension"] = int(i)
            self = SyntheticDataSet(**x)
            self.make()
            self.save("data/synthetic/"+name+"_"+str(int(i)))