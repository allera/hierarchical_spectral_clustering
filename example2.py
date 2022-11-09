#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 18:58:48 2021

@author: alblle
"""
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import SpectralClustering
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as shc
from scipy.linalg import eigh
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler

from scipy.spatial.distance import cosine
import sys


#-->load hc module
toolbox_path='/Users/alblle/Dropbox/POSTDOC/EU_aims/stratifications/paper_code/hierarchical_clustering_module_git' 
sys.path.append(toolbox_path) 
#from hc import hierarchical_spectral_clustering
#from hc import fancy_dendrogram
#from hc import get_distances
import hc_v1 as hc
#<--load hc module




mycolors=['b','g','r','c','m','y']



#-->load datasets
####################################
#https://scikit-learn.org/stable/auto_examples/cluster/plot_cluster_comparison.html#sphx-glr-auto-examples-cluster-plot-cluster-comparison-py

from sklearn import datasets

np.random.seed(0)

# ============
# Generate datasets. We choose the size big enough to see the scalability
# of the algorithms, but not too big to avoid too long running times
# ============
n_samples = 100
noisy_circles = datasets.make_circles(n_samples=n_samples, factor=.5,
                                      noise=.05)
noisy_moons = datasets.make_moons(n_samples=n_samples, noise=.05)
blobs = datasets.make_blobs(n_samples=n_samples, random_state=8)
no_structure = np.random.rand(n_samples, 2), None

# Anisotropicly distributed data
random_state = 170
X, y = datasets.make_blobs(n_samples=n_samples, random_state=random_state)
transformation = [[0.6, -0.6], [-0.4, 0.8]]
X_aniso = np.dot(X, transformation)
aniso = (X_aniso, y)

# blobs with varied variances
varied = datasets.make_blobs(n_samples=n_samples,
                             cluster_std=[1.0, 2.5, 0.5],
                             random_state=random_state)
####################################
#<--load datasets







#-->run our hierarchical spectral clustering
####################################
datasets=['noisy_circles','noisy_moons','blobs']
for dataset in datasets:
    if dataset =='noisy_circles':
        data=noisy_circles[0]
    elif dataset =='noisy_moons':
        data=noisy_moons[0]
    elif dataset=='blobs':
        data=blobs[0]
        
    # normalize dataset for easier parameter selection
    data = StandardScaler().fit_transform(data)
    plt.scatter(data[:,0],data[:,1])


    print('Running spectral clusterin on ', dataset)
    S=hc.compute_similarity(X=data,Similarity='euclid')
    L,D,W=hc.compute_graph_Laplacian(S=S)
    yy,path_length=hc.compute_spectral_decomposition_graph_Laplacian(L=L,D=D,dim=1)
    print('plotting results')
    link_mat,minbic,labels = hc.spectral_clustering(yy,data)  
    
    