#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 18:57:02 2021

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
from scipy.spatial.distance import cosine
from sklearn.preprocessing import StandardScaler




def norm(X):
    
    from scipy.spatial.distance import pdist
    from scipy.spatial.distance import squareform

    return squareform(pdist(X))

def adjacency(X):

    from networkx import is_connected
    from networkx import from_numpy_matrix
    
    emin = 0
    emax = np.max(X)
    tol = 0.0001
    maxiter = 1000
    cntr = 0
    
    done = False
    while not done:        
        e = (emin + emax) / 2
        A = (X < e) - np.eye(X.shape[0])
        G = from_numpy_matrix(A)
        if is_connected(G):
            emax = e
            if (emax - emin) < tol:
                done = True
        else:
            emin = e       
        cntr += 1
        if cntr == maxiter:
            done = True      
    
    return A


def compute_similarity(X=None,Similarity='corr'):
#X matrix
#Similarity = 'corr','euclid','dot',cosine'
    print('Computing Similarity matrix ...')
    #S=similarity matrix
    if Similarity=='corr': #congrads
        S=np.corrcoef(X)  
        #S[S<0]=0
    elif Similarity=='euclid':
        S=pairwise_distances(X)#, metric="cosine")  
    elif Similarity =='dot':
        S=np.dot(X,X.T)
    elif Similarity =='cosine':
       S=pairwise_distances(X, metric="cosine")  
    else:
        print('Simlarity type not define, inpu Similarity = corr,euclid,dot,or cosine')
    print('...done')
    return S

def compute_graph_Laplacian(S=None):
#S= Similarity matrix output of function compute_similarity
    print('Computing Graph Laplacian')
    dist = norm(S)**2
    W = np.multiply(adjacency(dist),S)
    D = np.diag(np.sum(W,0))
    L = np.subtract(D,W) # L =  D-W, D is diagonal, eig(L,D)=eig(D-W,W);if lets say c1=D-W, c2=2W-D -> c1+c2=W
    print('...done')
    return L,D,W

def compute_spectral_decomposition_graph_Laplacian(L,D,dim):
#L y D first two outputs of function compute_graph_Laplacian
#dim dimensionaliy of decomposition
    print('Computing spectral decomposition of Graph Laplacian')
    # Solve generalised eigenvalue problem Ly = lDy
    print('Computing the dominant ' + str(dim) + ' Laplacian eigemaps ...')   
    lll,yy = eigh(L,D,subset_by_index= [1,dim]) #eigvals=(0,nmaps)) # I remove the first eigvec since eigval is zero
    
    yy = StandardScaler().fit_transform(yy)
    print('...done')
    
    if dim==1:
        path_length=np.sum(np.ediff1d(np.sort(yy), to_end=None, to_begin=None))
    else:
        path_length=np.zeros(dim)
        for i in range(dim):
            path_length[i]=np.sum(np.ediff1d(np.sort(yy[:,i]), to_end=None, to_begin=None))
            
    return yy,path_length



def spectral_clustering(yy=None,data=np.zeros([1,3])):
#yy is output of function compute_spectral_decomposition_graph_Laplacian
#data can be passed if data is 2d to make some nice plot otherwise can be ignored
    mycolors=['b','g','r','c','m','y']

    if data.shape[1]==2: # if data is two dimensional
            fig, ax = plt.subplots(1,4,figsize=(2*7,7))
    else:
            fig, ax = plt.subplots(1,2,figsize=(2*7,7))

    
    #make dendogram
    # setting distance_threshold=0 ensures we compute the full tree.
    model = AgglomerativeClustering(distance_threshold=0, n_clusters=None)
    model = model.fit(yy)
    from scipy.cluster.hierarchy import dendrogram
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_,
                                      counts]).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, p=10, ax=ax[0], truncate_mode='level',no_labels=True)#, color_threshold=None, get_leaves=True, orientation='top', labels=None, count_sort=False, distance_sort=False, show_leaf_counts=True, no_plot=False, no_labels=False, leaf_font_size=None, leaf_rotation=None, leaf_label_func=None, show_contracted=False, link_color_func=None, above_threshold_color='C0')
    ax[0].set_title('Hierarchical Clustering Dendrogram')

    # plot the top three levels of the dendrogram
    #plot_dendrogram(model, truncate_mode='level', p=10)
    #plt.xlabel("Number of points in node (or index of point if no parenthesis).")
    #plt.show()
    
    
    from sklearn.cluster import KMeans
    Nc = range(1, 10)
    kmeans = [KMeans(n_clusters=i,n_init=20) for i in Nc]
    score = [- kmeans[i].fit(yy).score(yy) for i in range(len(kmeans))]
    aic_score = [ i - kmeans[i].fit(yy).score(yy) for i in range(len(kmeans))]
    bic_score= [ (i*np.log(yy.shape[0]/2)) - kmeans[i].fit(yy).score(yy) for i in range(len(kmeans))]
    #score = [kmeans[i].inertia_ for i in range(len(kmeans))]
    #ax[1].plot(Nc,score)
    ax[1].plot(Nc,aic_score)
    ax[1].plot(Nc,bic_score)
    ax[1].set_xlabel('Number of Clusters')
    ax[1].set_ylabel('Score')
    ax[1].set_title('Model order,AIC,BIC')
    ax[1].set_xticks([2,4,6,8,10])#,12,14,16,18,20])
    ax[1].grid('on')
    minbic=np.argwhere(bic_score==np.min(bic_score))+1
    
    kmeans = KMeans(n_clusters=minbic[0][0], random_state=0, n_init=50).fit(yy)
    
    labels=kmeans.labels_
    
    if data.shape[1]==2:
        for k in range(int(minbic)):
            ax[2].scatter(data[labels==k,0],data[labels==k,1],c=mycolors[k])
        
        ax[2].set_title('clustering solution my model')
    
    
        im=ax[3].scatter(data[:,0],data[:,1], marker='+', s=150, linewidths=4, c=yy, cmap=plt.cm.coolwarm)
        fig.colorbar(im,ax=ax[3])
        
    plt.show()
    
    return linkage_matrix, minbic, kmeans.labels_


