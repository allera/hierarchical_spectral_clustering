#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 16:36:46 2022

@author: alblle
"""

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



#-->load dataset
####################################
#https://scikit-learn.org/stable/auto_examples/cluster/plot_cluster_comparison.html#sphx-glr-auto-examples-cluster-plot-cluster-comparison-py
from sklearn import datasets
np.random.seed(0)
n_samples = 100
noisy_circles = datasets.make_circles(n_samples=n_samples, factor=.5,noise=.05)
data=noisy_circles[0]
# normalize dataset for easier parameter selection
data = StandardScaler().fit_transform(data)
plt.scatter(data[:,0],data[:,1])
####################################
#<--load datasets



#-->run hierarchical spectral clustering

#1) compute similarity matrix
S=hc.compute_similarity(X=data,Similarity='euclid')

#2) compute graph Laplacian
L,D,W=hc.compute_graph_Laplacian(S=S)

#3) compute spectral decomposition 
yy,path_length=hc.compute_spectral_decomposition_graph_Laplacian(L=L,D=D,dim=1)
print('plotting results')

#4 plot hierachical spectral clustering dendogram and select a model order based on BIC
link_mat,minbic,labels = hc.spectral_clustering(yy,data)  


#<--run hierarchical spectral clustering

