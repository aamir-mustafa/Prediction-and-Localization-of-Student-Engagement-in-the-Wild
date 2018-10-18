#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  6 21:36:44 2018

@author: user
"""

"""
Created on Fri Dec 29 18:34:44 2017

@author: Aamir
"""

# Load libraries
import numpy as np
#from keras.preprocessing.text import Tokenizer
from keras import models
from keras import layers
#from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
#from sklearn import preprocessing
import scipy.io as sio
from sklearn.metrics import mean_squared_error

feats=sio.loadmat('Normalized_relabeled_lbptop_20_segments_with_labels_taking_max1_150_per_video.mat')

X=feats['combined_lbptop_20_segments_53'] #Features: 150 per video 
Y=feats['labels']                           #Relabeled labels

feats_original=sio.loadmat('lbptop_feats_combined_150_seq_video_normalized.mat')    
Y_original=feats_original['final_labels_150_seg_per_video']             #Manual labels of each segment (in total 7950)
Y_original= Y_original.astype(float)
Y_original=Y_original/3
#%%
#Z=Y.astype(float) 
Y=Y/3   
# Set random seed
np.random.seed(0)

X_train=X[0:5250,:]
Y_train=Y[0:5250,:]
X_test=X[5250:7950,:]
Y_test=Y_original[5250:7950,:]


# Divide our data into training and test sets
#X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=0)
#%%
# Start neural network
network = models.Sequential()

# Add fully connected layer with a ReLU activation function
network.add(layers.Dense(units=200, activation='relu', input_dim=177))

# Add fully connected layer with a ReLU activation function
network.add(layers.Dense(units=500, activation='relu'))

network.add(layers.Dense(units=500, activation='relu'))

# Add fully connected layer with no activation function
network.add(layers.Dense(units=1, activation='sigmoid'))

# Compile neural network
network.compile(loss='mse', optimizer='RMSprop', metrics=['mse']) 

# Train neural network
history = network.fit(X_train, Y_train, epochs=100, verbose=1,
                      batch_size=5000, validation_data=(X_test, Y_test))     

#%%

pred=network.predict(X_test, batch_size=100, verbose=1)
p=pred*3
pp=Y_test*3
#pred=int(round(pred))

sio.savemat('kmeans_pred_nn_last18.mat', {'pred':pred, 'original':Y_original_test})    #mse on actual manual labels vs the predicted labels

import sklearn.metrics
err=sklearn.metrics.mean_squared_error(pred, Y_test)