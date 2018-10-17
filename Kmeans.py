#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 24 18:32:19 2017

@author: user
"""

import numpy as np

#from keras import models
#from keras import layers
#
#from sklearn.model_selection import train_test_split

import scipy.io as sio

feats=sio.loadmat('combined_lbptop_20_segmented_43_with_labels.mat')
X=feats['combined_lbptop_20_segments_43']
Y=feats['final_lables_segments_43']
#%%  
from sklearn.cluster import KMeans, MiniBatchKMeans

X = np.array(X)
X=np.nan_to_num(X)
Y=Y.astype(float)
n_clusters=10
#%%
kmeans = KMeans(n_clusters=n_clusters,verbose=1, random_state=0).fit(X)

#%%
Minibatch= MiniBatchKMeans (n_clusters=n_clusters, verbose=1, random_state=0).fit(X)
#%%
kmeans.predict(X);
labels=kmeans.labels_
print(labels)
#%%
Minibatch.predict(X);
labels1=Minibatch.labels_
print(labels1)
#%%


#%%
for i in xrange(n_clusters):
    a=np.where(labels==i)
    z = Y[a]
    
    values,counts =np.unique(z, return_counts= True)
    #counts[0,1,2,3] gives the number of occurances of 0,1,2,3
    total_counts= z.shape[0]
    final= counts[1]+2*counts[2]+3*counts[3];
    final= final.astype(float)
    final=final/total_counts
    Y[a]=final;

#%%
np.save('X_50000_hog_au_Feats_5_clusters.npy', X)
np.save('Y_50000_hog_au_relabeled_5_clusters.npy', Y)

#f = h5py.File('X_Y_50000_hog_au_Feats_Labels_5_clusters.h5', "w")
## point to the default data to be plotted
#f.attrs[u'X']          = X
## give the HDF5 root some more attributes
#f.attrs[u'Y']        = Y
#f.close() 

#import scipy.io as sio
#
#sio.savemat('X_Y_50000_hog_au_Feats_Labels_5_clusters.mat', mdict={'X_50000': X, 'Y_50000': Y})

