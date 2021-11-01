#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import string
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
            temp = self.pretransformed_noise*np.random.randn(cluster_size,
                                                             self.pretransformed_dimension)
            temp = temp + centers[:,i]
            
            
            out.append(temp)
            self.labels.extend([i]*cluster_size)

        self.pretransformed_data = np.vstack(out)#pd.concat(out)
        
        
    def sine_transform(self, period = 0.5, amplitude = 1):
        """ """
        plane_injection = np.random.randn(self.posttransformed_dimension,
                                          self.pretransformed_dimension)
        
        self.baseline = np.matmul(plane_injection, self.pretransformed_data.T).T
        print(self.baseline.shape)
        previously_spanned = plane_injection

        out = self.baseline
        for i in range(self.posttransformed_dimension-self.pretransformed_dimension):
            print(i)
            time = self.baseline[:,i]# select a direction within high-D space as to oscillate
            normal = make_normal(previously_spanned)# get a direction normal to plane


            z = amplitude * np.sin(time / period)
            deviation = np.matmul(z.reshape(-1, 1),normal.reshape(1,-1))
            out = out + deviation

            previously_spanned = np.hstack([previously_spanned, normal.reshape(-1,1)])

        self.baseline
        self.data = out
        
    def add_more_noise(self):
        self.data += self.posttransformed_noise * np.random.randn(self.data.shape[0], self.posttransformed_dimension)
        
        self.data = pd.DataFrame(self.data)
        self.elementnames = [randstring() for i in range(self.size)]
        self.data.index = self.elementnames
        self.original_features = [randstring() for i in range(self.posttransformed_dimension)]


        self.data.columns = self.original_features
        self.labels = pd.Series(self.labels, index = self.elementnames)                
        self.data = self.data.sample(frac=1)
        
    def parameter_summary(self):
        unnecessary_keys = ['labels',
                    'pretransformed_data',
                    'data',
                    'original_features',
                    'elementnames'
                   ]
            
        return {k:v for k,v in self.__dict__.items() if k not in unnecessary_keys}

    def save(self, foldername = "scratch",  save_tabular_data = True):

        os.makedirs(foldername, exist_ok = True)
        
        # Save the parameters used to create the dataset
        parameterdict, = self.parameter_summary()
        fp = open(f"{foldername}/parameters.json",'w')
        json.dump(parameterdict, fp)
        fp.close()
        
        # potentially save the datasets
        if save_tabular_data:
            self.data.to_csv(f"{foldername}/features.csv")
            self.labels.to_csv(f"{foldername}/labels.csv")
        
        # save the different networks created using those features
        for parameters, network in self.network.items():
            
            for edge in network.edges():# 
                network.edges()[edge]['weight'] = str(network.edges()[edge]['weight'])

            metric  = parameters[0]
            nneighbors =  parameters[1]

            nx.write_gml(network,f"{foldername}/metric_{metric}_nneighbors_{nneighbors}.gml")
            fp = open(f"{foldername}/evaluation_time_metric_{metric}_nneighbors_{nneighbors}", 'w')
            fp.write(str(self.network_evaluation_time[parameters]))
            fp.close()        
        


# $$f(\bar{x}) = a\sum_i x_i\bar{v}_i + b\sum_j \bar{w}_j\sin(\bar{c_j} \cdot \bar{x})$$
# 
# where we have $ w_j \perp w_i  $, $ v_j \perp v_i  $ , for all $i\neq j$ and and$\forall i,j$, $ w_j \perp v_i $

# 

# $$\frac{\partial f}{\partial x_i} = a \bar{v_i} + b\sum_j [\bar{c_{j}}]_i \bar{w}_j\cos(\bar{c_j} \cdot \bar{x})$$
# 
# $$||f(\bar{x}) - f(\bar{y})||^2 = a^2\sum_i |x_i - y_i|^2 + b^2\sum_j |\sin(\bar{c_j} \cdot \bar{x}) -\sin(\bar{c_j} \cdot \bar{y})|^2 $$
# 

# $$||f(\bar{x}+\bar{y}) - f(\bar{x})||^2 = a^2\sum_i |y_i|^2 + b^2\sum_j |\sin(\bar{c_j} \cdot \bar{x}+ \bar{c_j} \cdot \bar{y}) -\sin(\bar{c_j} \cdot \bar{x})|^2 $$

# $$||f(\bar{x}+\bar{y}) - f(\bar{x})||^2 = a^2\sum_i y_i^2 + b^2\sum_j (\sin(\bar{c_j} \cdot \bar{x})(\cos(\bar{c_j} \cdot \bar{y})-1) + \cos(\bar{c_j} \cdot \bar{x})\sin(\bar{c_j} \cdot \bar{y}))^2 $$

# $$\frac{\partial}{\partial y_i}||f(\bar{x}+\bar{y}) - f(\bar{x})||^2 = 2a^2 y_i $$
# $$+ 2b^2\sum_j c_{ij} \cos(\bar{c_j} \cdot (\bar{x} + \bar{y}))[\sin(\bar{c_j} \cdot (\bar{x}+ \bar{y})) -\sin(\bar{c_j} \cdot \bar{x})] $$

# $$ \sin(\bar{c_j} \cdot \bar{x}+ \bar{c_j} \cdot \bar{y}) $$
# $$ \sin(a+ b) = \sin(a)\cos(b) + \cos(a)\sin(b)$$
# $$ \sin(a+ b) - \sin(a) = \sin(a)(\cos(b)-1) + \cos(a)\sin(b)$$

# In[2]:


import pandas as pd
x = {"size" : 1000,
"n_clusters" : 2,
"pretransformed_dimension" : 2,
"posttransformed_dimension" : 300,
"pretransformed_noise" : 0.5,
"posttransformed_noise" : 0.3}

self = SyntheticDataSet(**x)
self.make_pretransformed()
self.sine_transform()
self.add_more_noise()


# In[143]:


def fold_data(X, axis , height, transition_width):

    folding_value=X[:,axis]#the values along axis along which we will fold the data

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

    
    # new
    replacement_axis = (upper_replacement_axis*upper_sheet + 
                       hinge_replacement_axis*hinge_region +
                       lower_sheet*folding_value)

    new_axis = (upper_new_axis*upper_sheet + 
                       hinge_new_axis*hinge_region +
                       lower_sheet*0)
    
    # creat the output dataset with the new 
    out = np.zeros((X.shape[0], X.shape[1]+1))
    out[:,:-1] = X
    out[:,-1] = new_axis
    out[:,axis] = replacement_axis
    
    return out


# In[278]:


import numpy as np
@np.vectorize
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
        #print(x,8*amplitude*(f-0.25) - amplitude,y)

    if 0.5 <= f < 0.75: # high horizontal
        x += 2*(f-0.5)*xy_period + 0.5*xy_period
        y += amplitude

    if 0.75 <= f <= 1.0:
        #print(t, 8*amplitude*(f-0.25) - amplitude)
        x += xy_period
        y += -8*amplitude*(f-0.75) + amplitude

    return x,y

def squarify(X, axis,t_period= 1, xy_period = 1.0, amplitude = 1):
    xy = [square_wave(i, t_period, xy_period, amplitude, rescale) for i in X[:,axis]]
    replacement_axis = [i[0] for i in xy]
    new_axis = [i[1] for i in xy]
    
    # creat the output dataset with the new 
    out = np.zeros((X.shape[0], X.shape[1]+1))
    out[:,:-1] = X
    out[:,-1] = new_axis
    out[:,axis] = replacement_axis
    return out


# In[203]:


import scipy.sparse


# In[205]:


m = scipy.sparse.eye(3,4)
m = m.tolil()
m.toarray()
m[2,3] = -5.0
m = m.tocoo()


# $$f(\bar{x}) = \sum_i f_i(\bar{c_i}\cdot\bar{x})\bar{v_i}$$

# $$||f(\bar{y}) - f(\bar{x})||^2 = \sum_i ||f_i(\bar{c_i}\cdot\bar{y})-f_i(\bar{c_i}\cdot\bar{x})||^2$$

# In[206]:


m.shape


# In[207]:


m.toarray()


# In[289]:


import numpy as np
@np.vectorize
def square_wave(t, t_period, xy_period, amplitude):

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
        #print(x,8*amplitude*(f-0.25) - amplitude,y)

    if 0.5 <= f < 0.75: # high horizontal
        x += 2*(f-0.5)*xy_period + 0.5*xy_period
        y += amplitude

    if 0.75 <= f <= 1.0:
        #print(t, 8*amplitude*(f-0.25) - amplitude)
        x += xy_period
        y += -8*amplitude*(f-0.75) + amplitude

    return x,y

def squarify(X, axis,t_period= 1, xy_period = 1.0, amplitude = 1):
    xy = [square_wave(i, t_period, xy_period, amplitude) for i in X[:,axis]]
    replacement_axis = [i[0] for i in xy]
    new_axis = [i[1] for i in xy]
    
    # creat the output dataset with the new 
    out = np.zeros((X.shape[0], X.shape[1]+1))
    out[:,:-1] = X
    out[:,-1] = new_axis
    out[:,axis] = replacement_axis
    return out


import numpy as np
X_old = np.random.randn(1000, 2)
height = 1.0
transition_width = 1.0

axis = 0
X = squarify(X_old, axis)

G = umap_network(pd.DataFrame(X))

nodelist = G.nodes() # 
A = scipy.sparse.eye(X.shape[0])  - nx.normalized_laplacian_matrix(G, nodelist=nodelist)
e,evec = scipy.sparse.linalg.eigsh(A)
fig = px.scatter_3d(
                    x=X[:,0],
                    y=X[:,1],
                    z=X[:,2],#X[:,2],
                    color = evec[:,-5],
                   )
config = {
  'toImageButtonOptions': {
    'format': 'png', # one of png, svg, jpeg, webp
    'filename': 'custom_image',
     'colorbar':{'title':'Eigenvector 3'},
    'height': 1000,
    'width': 1400,
    'scale': 5 # Multiply title/legend/axis/canvas sizes by this factor
  }
}

fig.show(config=config)


# In[291]:


import numpy as np
X_old = np.random.randn(1000, 2)
height = 1.0
transition_width = 1.0
X = fold_data(X_old, 1, height, transition_width)
X = fold_data(X, 0, height, transition_width)

import plotly.express as px

G = umap_network(pd.DataFrame(X))

nodelist = G.nodes() # 
A = scipy.sparse.eye(X.shape[0])  - nx.normalized_laplacian_matrix(G, nodelist=nodelist)
e,evec = scipy.sparse.linalg.eigsh(A)
fig = px.scatter_3d(
                    x=X[:,0],
                    y=X[:,1],
                    z=2*X[:,3] - X[:,2],
                    color = evec[:,-2],
                   )
config = {
  'toImageButtonOptions': {
    'format': 'png', # one of png, svg, jpeg, webp
    'filename': 'custom_image',
     'colorbar':{'title':'Eigenvector 3'},
    'height': 1000,
    'width': 1400,
    'scale': 5 # Multiply title/legend/axis/canvas sizes by this factor
  }
}

fig.show(config=config)


# In[214]:


import sklearn.decomposition


# In[266]:


import numpy as np
X = np.random.randn(1000, 2)
height = 1.0
transition_width = 1.0
X = fold_data(X, 1, height, transition_width)
X = fold_data(X, 0, height, transition_width)

import plotly.express as px

G = umap_network(pd.DataFrame(X))

nodelist = G.nodes() # 
A = scipy.sparse.eye(X.shape[0])  - nx.normalized_laplacian_matrix(G, nodelist=nodelist)
e,evec = scipy.sparse.linalg.eigsh(A)
fig = px.scatter_3d(
                    x=X[:,0],
                    y=X[:,1],
                    z=2*(X[:,2]-height/2) + X[:,3]-height/2,
                    color = evec[:,-3],
                   )
config = {
  'toImageButtonOptions': {
    'format': 'png', # one of png, svg, jpeg, webp
    'filename': 'custom_image',
     'colorbar':{'title':'Eigenvector 3'},
    'height': 1000,
    'width': 1400,
    'scale': 5 # Multiply title/legend/axis/canvas sizes by this factor
  }
}

fig.show(config=config)


# In[ ]:


fig.


# In[225]:


import matplotlib.cm as cm
import matplotlib.pyplot as plt


# In[133]:


a = cm.seismic_r(np.linspace(0,1,100))
plt.plot(a[:,0], c = 'r')
plt.plot(a[:,1], c = 'g')
plt.plot(a[:,2], c = 'b')
plt.plot(0.2126*R + 0.7152*G + 0.0722*B, c='k')


# In[58]:


a = cm.viridis_r(np.linspace(0,1,100))
plt.plot(a[:,0], c = 'r')
plt.plot(a[:,1], c = 'g')
plt.plot(a[:,2], c = 'b')
plt.plot(0.2126*R + 0.7152*G + 0.0722*B, c='k')


# In[54]:


a = cm.bone_r(np.linspace(0,1,100))
R = a[:,0]
G = a[:,1]
B = a[:,2]
plt.plot(R, c = 'r')
plt.plot(G, c = 'g')
plt.plot(B, c = 'b')
plt.plot(0.2126*R + 0.7152*G + 0.0722*B, c='k')


# In[59]:


a = cm.hot_r(np.linspace(0,1,100))
R = a[:,0]
G = a[:,1]
B = a[:,2]
plt.plot(R, c = 'r')
plt.plot(G, c = 'g')
plt.plot(B, c = 'b')
plt.plot(0.2126*R + 0.7152*G + 0.0722*B, c='k')


# In[8]:


np.matmul(self.data.values.T,self.data.values)


# In[ ]:





# In[237]:


pd.DataFrame(self.data)


# In[238]:


u.shape, e.shape, v.shape


# In[62]:


u,e,v = np.linalg.svd(self.data.values)
print("True dimension",sum([i > 1e-4 for i in e]))


# In[63]:


e


# In[15]:


import plotly.express as px


# In[17]:


px.scatter_3d(x=self.final[:,0],
             y=self.final[:,1],
             z=self.final[:,2])
#             c=self.labels)


# In[18]:


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
                       ellipticity = 1, 
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
        centers = np.random.randn(self.n_clusters,  self.dimension)
        out = []

        for i,n in enumerate(self.size_expanded):
            temp = self.scale_expanded[i]*np.random.randn(n, self.dimension)

            # Add variation to the clusters along different axes so they are less spherical
            temp = (1+(self.ellipticity-1)*np.random.rand(self.dimension)).reshape(-1,self.dimension) * temp
            temp = pd.DataFrame(temp)
            temp = temp + centers[i,:]
            out.append(temp)

        self.pretransformed = pd.concat(out).values
        exec(self.transform_dataset) # apply a nonlinear transform to creat a new set of features
        self.data = pd.DataFrame(self.data)

        # make labels
        self.labels = np.concatenate([[i]*v for i,v in enumerate(self.size_expanded)])

    
        # Consistent names for columns and indices
        self.elementnames = [randstring() for i in range(len(self.data.index))]
        self.data.index = self.elementnames
        self.original_features = [randstring() for i in range(len(self.data.columns))]


        self.data.columns = self.original_features
        self.labels = pd.Series(self.labels, index = self.elementnames)                
        self.data = self.data.sample(frac=1) # re-order the datapoints so that nothing 
                                             # can be accidentally inferred form their ordering.
            
        

        metric = 'euclidean'
        nneighbors = 10
                
        start_time = time.time()
        self.network[(metric, nneighbors)] = umap_network(self.data, nneighbors = nneighbors, metric = metric)
        end_time = time.time()
        self.network_evaluation_time[(metric, nneighbors)] = end_time - start_time

    def parameter_summary(self):
        parameterdict = {'n_clusters':self.n_clusters,
        'dimension':self.dimension,
        'final_dimension':self.final_dimension,                         
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
        'final_dimension':self.final_dimension,
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


# In[19]:


import pandas as pd


# In[20]:


pd.DataFrame(pretransformed)


# In[ ]:


G = umap_network(self.final)


# In[ ]:


def network_spectral(G, n_evectors=100, n_clusters=100):

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
mooo = network_spectral(G)


# In[ ]:





# In[ ]:


#np.matmul(plane_injection, a)


# In[ ]:


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
    
    if any(normal < 1e-10): # if the normal is really small i.e. was in the span
        raise NoNormalError # raise an error
        
    return normal / np.linalg.norm(normal)# o/w normalize

def build_simplex(space_dimension, simplex_dimension = None):
    """
    make a n-simplex with unit length edges embedded in high dimensional space
    
    where n is `simplex_dimension`, 
    
    Parameters
    ----------
    space_dimension : Integer, dimension of space the simplex is embedded into.
    simplex_dimension : Integer, number of points in the simplex.
    
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
    for i in range(simplex_dimension-1):
        
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
        
    return simplex


# In[ ]:





# In[ ]:


np.linalg.lstsq(span, [1]*span.shape[0])


# In[ ]:


x = """\ndef make_normal(span, seed_vector = None):\n    if seed_vector == None:\n        seed_vector = np.random.randn(span.shape[0])\n    a,_,_, _ = np.linalg.lstsq(span, seed_vector)\n    normal = seed_vector - np.matmul(span, a)\n    return normal / np.linalg.norm(normal)\n\n\nplane_injection = np.random.randn(self.final_dimension, self.dimension)\nbaseline = np.matmul(plane_injection, self.pretransformed.T).T\nout = baseline\n\npreviously_spanned = plane_injection\n\nfor i in range(self.final_dimension):\n    time = baseline[:,i]\n    normal = make_normal(previously_spanned)\n    \n    period = 4\n    amplitude = 100\n    z = amplitude * np.sin(time / period)\n    deviation = np.matmul(z.reshape(-1, 1),normal.reshape(1,-1))\n    out = out + deviation\n    \n    previously_spanned = np.hstack([previously_spanned, normal.reshape(-1,1)])\n    \nself.final  = out"""


# In[ ]:





# In[ ]:


span.shape


# In[27]:


plane_injection = np.random.randn(final_dimension, dimension)
baseline = np.matmul(plane_injection, pretransformed.T).T
out = baseline

previously_spanned = plane_injection

for i in range(final_dimension):
    time = baseline[:,i]
    normal = make_normal(previously_spanned)
    
    period = 4
    amplitude = 100
    z = amplitude * np.sin(time / period)
    deviation = np.matmul(z.reshape(-1, 1),normal.reshape(1,-1))
    out = out + deviation
    
    previously_spanned = np.hstack([previously_spanned, normal.reshape(-1,1)])
    
final  = out


# In[28]:


x = """
def make_normal(span, seed_vector = None):
    if seed_vector == None:
        seed_vector = np.random.randn(span.shape[0])
    a,_,_, _ = np.linalg.lstsq(span, seed_vector)
    normal = seed_vector - np.matmul(span, a)
    return normal / np.linalg.norm(normal)


plane_injection = np.random.randn(self.final_dimension, self.dimension)
baseline = np.matmul(plane_injection, self.pretransformed.T).T
out = baseline

previously_spanned = plane_injection

for i in range(self.final_dimension):
    time = baseline[:,i]
    normal = make_normal(previously_spanned)
    
    period = 4
    amplitude = 100
    z = amplitude * np.sin(time / period)
    deviation = np.matmul(z.reshape(-1, 1),normal.reshape(1,-1))
    out = out + deviation
    
    previously_spanned = np.hstack([previously_spanned, normal.reshape(-1,1)])
    
self.final  = out"""


# In[29]:


x


# In[30]:


normal.shape


# In[31]:


normal


# In[32]:





# In[33]:





# In[34]:


import sklearn.cluster


# In[35]:


sklearn.cluster.k_means


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[36]:


deviation


# In[37]:


pd.DataFrame(final)


# In[ ]:





# In[38]:


final.shape


# In[12]:


import umap


# In[39]:


m,_,_ = umap.umap_.fuzzy_simplicial_set(final, 10, np.random.RandomState(10),"euclidean")


# In[40]:


import networkx as nx


# In[41]:


G = nx.from_scipy_sparse_matrix(m)


# In[42]:


nx.draw_networkx(G, node_size = 0)


# In[22]:


import matplotlib.pyplot as plt


# In[23]:


final.shape


# In[ ]:





# In[24]:


np.concatenate(plane_injection)


# In[33]:


foo.parameter_summary()


# In[26]:


orthbase_and_normal, _ = np.linalg.qr(np.hstack([plane_injection, np.array([[1]]*plane_injection.shape[0])]))
orthbase = orthbase_and_normal[:, :-1]


# In[ ]:





# In[47]:


foo.pretransformed


# In[109]:


transform ="""plane_injection = np.random.randn(self.final_dimension, self.dimension)
a,_,_, _ = np.linalg.lstsq(plane_injection, [1]*plane_injection.shape[0])
self.normal = [1]*plane_injection.shape[0] - np.matmul(plane_injection, a)
baseline = np.matmul(plane_injection, self.pretransformed.T).T
time = baseline[:,1]

period = 1.3 
amplitude = 8
self.z = amplitude * np.sin(time / period)
self.deviation = np.matmul(self.z.reshape(-1, 1),self.normal.reshape(1,-1))
self.data = baseline + self.deviation"""

foo = SyntheticDataSet(dimension=2,  center_d = 1.0, size= 50, scale = 0.5, final_dimension = 3, n_clusters = 10, transform_dataset = transform)
foo.vary_clusters()
foo.make_dataset()
#foo.transform_data()

import plotly.express as px

fig = px.scatter_3d(foo.data,
                    x=foo.data.columns[0],
                    y=foo.data.columns[1],
                    z=foo.data.columns[2],
                    color=foo.labels.reindex(foo.data.index)
                   )


fig.show()


# In[30]:


nx.draw_networkx(foo.network[('euclidean', 10)], nodelist = foo.labels.index, 
                 with_labels = False, cmap = 'tab10', 
                 node_size = 20, node_color = foo.labels
                )


# In[ ]:





# In[31]:


foo.transform_data()
import plotly.express as px
fig = px.scatter_3d(foo.data,
                    x=foo.data.columns[0],
                    y=foo.data.columns[1],
                    z=foo.data.columns[2],
                    color=foo.labels.reindex(foo.data.index)
                   )

fig.show()


# In[11]:


plt.scatter(foo.pretransformed[:,0], foo.pretransformed[:,1],c=foo.labels)


# In[ ]:





# In[12]:


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(foo.data.iloc[:,0].values,foo.data.iloc[:,1].values, foo.data.iloc[:,2].values, c=foo.labels)


# In[160]:


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


transform_dataset ="""plane_injection = np.random.randn(self.final_dimension, self.dimension)
a,_,_, _ = np.linalg.lstsq(plane_injection, [1]*plane_injection.shape[0])
normal = [1]*plane_injection.shape[0] - np.matmul(plane_injection, a)
baseline = np.matmul(plane_injection, pretransformed.T).T
time = baseline[:,1]

period = 4
amplitude = 20
z = amplitude * np.sin(time / period)
deviation = np.matmul(z.reshape(-1, 1),normal.reshape(1,-1))
self.data = baseline + deviation"""
# Make clusters

x = {"n_clusters":2,
"dimension":2,
"center_d":1,
"scale":0.25,
"size":10}


# In[ ]:


final_dimension = 100
ellipticity = 0
scale_range=0
center_d_range=0,
size_range=0, 
transform_dataset = transform_dataset 


# In[225]:





# In[226]:


self.make_pretransformed()


# In[223]:


self.pretransformed_data


# In[198]:



space_dimension = self.pretransformed_dimension
space_dimension = self.n_clusters

if simplex_dimension == None:
    simplex_dimension = space_dimension

# initialise a random point exactly distance 1 from  the origin
point = np.random.randn(space_dimension)
point = point/np.linalg.norm(point)

# make the origin and this point into a 1-simplex
simplex = np.vstack([np.array([0]*space_dimension), point]).T

# build up the simplex point by point, dimension by dimension
for i in range(simplex_dimension-1):
    print(i)
    # get the central point of the simplex
    avg =  np.mean(simplex, axis = 1).reshape(-1,1)
    base_radii = np.linalg.norm(simplex - avg, axis = 0)# distance of the centre to the points of the simplex
    base_radius = base_radii[0]# these are all equal so pick the 0th entry indifferently
    assert all(np.abs(base_radius - base_radii) < 1e-9)# check they're equal

    # get a unit vector normal to the simplex
    normal = make_normal(simplex)
    new_point = avg + np.sqrt(1 - base_radius**2)*normal.reshape(-1,1)# use pyhtogoaras to project it to a new point opf the simplex
    assert all(np.linalg.norm(simplex - new_point, axis = 0) - 1 < 1e-9)


    simplex = np.hstack([simplex, new_point])

if center_origin:
    # recenter the simplex to the origin
    avg =  np.mean(simplex, axis = 1).reshape(-1,1)
    simplex = simplex-avg


# In[199]:


simplex


# In[184]:


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
    
    randdirs = np.random.randn(self.n_clusters,  self.dimension)
    randdirs = randdirs / np.sqrt((randdirs**2).sum(axis = 1)).reshape((self.n_clusters,1))


    centers = np.array([[i] for i in  self.center_d_expanded]) * randdirs
    out = []

    for i,n in enumerate(self.size_expanded):
        temp = self.scale_expanded[i]*np.random.randn(n, self.dimension)

        # Add variation to the clusters along different axes so they are less spherical
        #temp = * temp
        temp = pd.DataFrame(temp)
        temp = temp + centers[i,:]
        out.append(temp)

    self.pretransformed = pd.concat(out).values
    exec(self.transform_dataset) # apply a nonlinear transform to creat a new set of features
    self.data = pd.DataFrame(self.data)


    
    # make labels
    self.labels = np.concatenate([[i]*v for i,v in enumerate(self.size_expanded)])


    # Consistent names for columns and indices
    self.elementnames = [randstring() for i in range(len(self.data.index))]
    self.data.index = self.elementnames
    self.original_features = [randstring() for i in range(len(self.data.columns))]


    self.data.columns = self.original_features
    # make labels and index them 
    self.labels = np.concatenate([[i]*v for i,v in enumerate(self.size_expanded)])
    self.labels = pd.Series(self.labels, index = self.elementnames)
    
    self.data = self.data.sample(frac=1) # re-order the datapoints so that nothing 
                                         # can be accidentally inferred form their ordering.
        
    

    metric = 'euclidean'
    nneighbors = 10
            
    start_time = time.time()
    self.network[(metric, nneighbors)] = umap_network(self.data, nneighbors = nneighbors, metric = metric)
    end_time = time.time()
    self.network_evaluation_time[(metric, nneighbors)] = end_time - start_time


def parameter_summary(self):

    parameterdict = {'n_clusters':self.n_clusters,
    'dimension':self.dimension,
    'final_dimension':self.final_dimension,                         
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


# In[ ]:





# In[ ]:





# In[ ]:


import plotly.express as pxp
import plotly.graph_objs as go

fig = go.Figure()

x = [1, 4, 2, 3, 3, 4]
y = [0, 0, 1, 1, 2, 2]

fig.add_trace(
    go.Scatter(x=x, y=y))
    
fig.show()


# In[50]:





# In[58]:


import scipy.sparse
nodelist = list(G.nodes())
a = scipy.sparse.eye(G.order()) - nx.normalized_laplacian_matrix(G, nodelist =nodelist)


# In[59]:


e,evec = scipy.sparse.linalg.eigsh(a, k= 5)


# In[60]:


evec = pd.DataFrame(evec, index = nodelist)


# In[63]:


import networkx as nx
import umap
import random
def umap_network(X, nneighbors = 10, metric = 'euclidean'):
    rndstate = np.random.RandomState(nneighbors)
    nodes = list(X.index)
    G,_,_ = umap.umap_.fuzzy_simplicial_set(X/X.std(), 10, rndstate, metric)
    G = nx.from_scipy_sparse_matrix(G)
    return nx.relabel_nodes(G, dict(enumerate(X.index)).get)

H = umap_network(evec)


# In[66]:


nx.draw_networkx(H, node_size=1, with_labels=False, alpha=0.1)


# In[10]:





# In[52]:





# In[53]:


np.linalg.norm(point)


# In[54]:


simplex


# In[153]:


get_ipython().run_line_magic('pinfo', 'np.linalg.qr')


# In[147]:


get_ipython().run_line_magic('pinfo', 'build_simplex')


# In[146]:





# In[124]:


i = 10
j = 55
np.linalg.norm(simplex[:,i] - simplex[:,j])


# In[113]:


base_radius


# In[118]:





# In[92]:


foo.rehspe() - simplex[:,0] == foo


# In[89]:


np.linalg.norm(foo)


# In[30]:



$$\frac{\partial}{\partial x_i}||f(\bar{x}) - f(\bar{y})||^2 = 2a\sum_i (x_i - y_i) + 2b \sum_j [\bar{c_{j}}]_i \cos(\bar{c_j} \cdot \bar{x})|\sin(\bar{c_j} \cdot \bar{x}) -\sin(\bar{c_j} \cdot \bar{y})|$$

$$\frac{\partial}{\partial y_i}||f(\bar{x} + \bar{y}) - f(\bar{x})||^2 = a\sum_i y_i + b \sum_j [\bar{c_{j}}]_i \cos(\bar{c_j} \cdot \bar{y}) ( \cos(\bar{c_j} \cdot \bar{x})\sin(\bar{c_j} \cdot \bar{y}) + (\cos(\bar{c_j} \cdot \bar{y})-1)\sin(\bar{c_j} \cdot \bar{x})$$


# In[ ]:




