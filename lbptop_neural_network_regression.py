# -*- coding: utf-8 -*-
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
from keras.optimizers import RMSprop
feats=sio.loadmat('Normalized_relabeled_lbptop_20_segments_with_labels_taking_mean_150_per_video.mat')
X=feats['combined_lbptop_20_segments_53']
Y=feats['labels']
#%%
Z=Y.astype(float) 
Y=Z/3   
# Set random seed
np.random.seed(0)

X_train = X[0:5250,:]
Y_train = Y[0:5250,:]
#
X_test= X[5250:7950,:]
Y_test= Y[5250:7950,:]


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
learning_rate=1e-5
rms=RMSprop(lr=learning_rate)
network.compile(loss='mse', optimizer=rms, metrics=['mse']) 
print network.summary()
# Train neural network
history = network.fit(X_train, Y_train, epochs=100, verbose=1,
                      batch_size=2000, validation_data=(X_test, Y_test))                

pred=network.predict(X_test)

original=sio.loadmat('labels_53_by_Shreyank.mat')
original=original['labels_53']
original=original.astype(float)
original=original/3;
original=original[35:53,:]

sio.savemat('pred_original_18_nn_mean.mat', {'original': original, 'pred': pred})










