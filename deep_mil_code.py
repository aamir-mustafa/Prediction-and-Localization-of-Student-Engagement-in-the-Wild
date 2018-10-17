# -*- coding: utf-8 -*-
"""
Created on Mon Dec 25 15:10:52 2017

@author: sony
"""

from __future__ import print_function
import keras.backend as K
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization, SpatialDropout2D
from keras.layers import advanced_activations
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils
import numpy as np
#from keras.callbacks import ModelCheckpoint
from keras.regularizers import l1_l2
#import theano.gpuarray
#theano.gpuarray.use("cuda")

import h5py as h5
from keras.utils.generic_utils import get_custom_objects

def custom_activation(x):
    return (K.sigmoid(x) * x)

get_custom_objects().update({'custom_activation': Activation(custom_activation)})

#from keras.layers.core import  Lambda
#from keras.layers import  Merge
import sklearn
from keras.layers.convolutional import Convolution2D
from keras import backend as K
from keras.regularizers import l1_l2, Regularizer
from keras.engine import Layer
from theano import function, shared, printing
from keras.engine import InputSpec
from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Dropout, Reshape, Permute, Activation,     Input, merge
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
import numpy as np
from scipy.misc import imread, imresize, imsave
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
import theano.tensor as T
import scipy.io as sio
#%%
from keras.utils.generic_utils import get_custom_objects
#



batch_size = 20
SEQUENCES_PER_VIDEO=150;
FEATURES_PER_SEQUENCE=177;

class SortLayer(Layer):
    # Rerank is difficult. It is equal to the number of points (>0.5) to be fixed number.
    def __init__(self,k=1,label=1,**kwargs):
        # k is the factor we force to be 1
        self.k = k*1.0
        self.label = label
        
        self.input_spec = [InputSpec(ndim=4)]
        
        super(SortLayer, self).__init__(**kwargs)

    def build(self,input_shape):
        self.input_spec = [InputSpec(shape=input_shape)]

    def call(self, x,mask=None):
        
        
      
        #btch_size= batch_size
#    
        x=K.reshape(x, (-1, SEQUENCES_PER_VIDEO))
        #T.tensor.as_tensor_variable(np.asarray(x))
#        print(np.shape(x))
#        
#        first = K.max(x, axis=1)
#        m1 = K.argmax(x, axis=1)
#        m1 = K.one_hot(m1, 150)
#        x = x * (1 - m1)
#        
#        second = K.max(x, axis=1)
#        m1 = K.argmax(x, axis=1)
#        m1 = K.one_hot(m1, 150)
#        x = x * (1 - m1)
#        
#        third = K.max(x, axis=1)
#        m1 = K.argmax(x, axis=1)
#        m1 = K.one_hot(m1, 150)
#        x = x * (1 - m1)
#        
#        fourth = K.max(x, axis=1)
#        m1 = K.argmax(x, axis=1)
#        m1 = K.one_hot(m1, 150)
#        x = x * (1 - m1)
#        
#        fifth = K.max(x, axis=1)
#        
#        out = K.stack([first, second, third, fourth, fifth])
#        out = K.stack([fifth])
#        #print(out)
#        #out.eval()
#        response = K.mean(K.transpose(out), axis=1)
        #print(K.eval(x))
        response = K.mean(x, axis=-1, keepdims=True)
        ##print(btch_size)
        ##tmp=np.zeros(btch_size *  (FRAMES_PER_VIDEO-1))
        ##tmp=np.reshape(tmp, (btch_size, FRAMES_PER_VIDEO-1))
        ##tmp_th=shared(tmp.astype("float64"))
        #D=T.concatenate([response, 1-response], axis=1)
        #response=response[:,0]
        #response=K.reshape(response, (batch_size, 1))
        #return response
        # Show output
        #return out.T
        
       
        
        
        
        
        
        #D=K.stack([response, 1-response])
        #return D.T
        #response = K.mean(x, axis=-1, keepdims=True)
        ##print(btch_size)
        ##tmp=np.zeros(btch_size *  (FRAMES_PER_VIDEO-1))
        ##tmp=np.reshape(tmp, (btch_size, FRAMES_PER_VIDEO-1))
        ##tmp_th=shared(tmp.astype("float64"))
#        a = K.variable(0)
#        b = K.variable(0)
#        c = K.variable(0)
#        d = K.variable(0)
#        e = K.variable(0)
#        y = K.variable(1)
#        #keras.backend.switch(condition, then_expression, else_expression)
##        K.switch(response>=0 and response <=0.333, c=0 , e=0)
##        K.switch(response>=0 and response <=0.333, d=0 , e=0)
#        
#        b=K.switch(response>=0 and response <=0.333, lambda: K.update_sub(response, 0)/0.333, lambda: K.update_sub(e, 2))
#        response = K.mean(x, axis=-1, keepdims=True)
#        
#        a=K.switch(response>=0 and response <=0.333,  lambda: K.update_sub(y, b) , lambda: K.update_sub(e, 2))
#        response = K.mean(x, axis=-1, keepdims=True)
#        
#        #K.switch(response >0.333 and response <=0.666, a=0, e=0)
#        #K.switch(response >0.333 and response <=0.666, d=0 , e=0)
#        c=K.switch(response >0.333 and response <=0.666, lambda: K.update_sub(response, 0.333)/0.333 ,lambda: K.update_sub(e, 2))
#        response = K.mean(x, axis=-1, keepdims=True)
#        b=K.switch(response >0.333 and response <=0.666, lambda: K.update_sub(y, c) ,lambda:  K.update_sub(e, 2))
#        response = K.mean(x, axis=-1, keepdims=True)
#        #K.switch(response>0.666 and response <=1, a=0 , e=0)
#       # K.switch(response>0.666 and response <=1, b=0 , e=0)
#        d=K.switch(response>0.666 and response <=1, lambda: K.update_sub(response-0.666)/0.333,lambda: K.update_sub(e, 2))
#        response = K.mean(x, axis=-1, keepdims=True)
#        c=K.switch(response>0.666 and response <=1,  lambda: K.update_sub(y, d) ,lambda: K.update_sub(e, 2))
#        response = K.mean(x, axis=-1, keepdims=True)
##        if (response>=0 and response <=0.333):
#            c=0; d=0;
#            b=(response-0)/0.333;
#            a=1-b;
#        elif( response >0.333 and response <=0.666):
#            a=0; d=0;
#            c=(response-0.333)/0.333;
#            b=1-c;
#        else:
#            a=0; b=0;
#            d=(response-0.666)/0.333;
#            c=1-d;
#            
        
        
        #D=T.concatenate([1-response, response], axis=1)
        #D=T.concatenate([a,b,c,d], axis=1)
        #response=response[:,0]
        #response=K.reshape(response, (batch_size, 1))
        #return response
        #return D
        #return response
        #l=newx.eval()
        #print(np.shape(l))
        #newx = newx[::-1]
        #result[0,:]=newx[0]
        #resultVec=theano.shared(np.array(result).astype("float32"))
        #result[0,:]=newx[0,:]
        #print(result)

        #result=shared(result)
        

 
        #response = K.reverse(newx, axes=1)
        #response = K.sum(x> 0.5, axis=1) / self.k
        #response = K.reshape(newx,[-1,1])
        #return K.concatenate([1-response, response], axis=self.label)
        #response = K.reshape(x[:,self.axis], (-1,1))
        #return K.concatenate([1-response, response], axis=self.axis)
        #e = K.exp(x - K.max(x, axis=self.axis, keepdims=True))
        #s = K.sum(e, axis=self.axis, keepdims=True)
        return response

    def get_output_shape_for(self, input_shape):
        return tuple([input_shape[0], 1])
        #return input_shape

    def compute_output_shape(self, input_shape):
        print(input_shape)
        return tuple([input_shape[0], 1])


# In[46]:


import theano
import theano.tensor as T

#epsilon = 1.0e-9
#def custom_objective(y_true, y_pred):
#    '''Just another crossentropy'''
#    y_pred = T.clip(y_pred, epsilon, 1.0 - epsilon)
#    y_pred /= y_pred.sum(axis=-1, keepdims=True)
#    cce = T.nnet.categorical_crossentropy(y_pred, y_true)
#    return cce

#model creation
l2factor = 1e-5
l1factor = 2e-7
outdim=200
outdim2=500

def createModel():
    inputs = Input(shape=(SEQUENCES_PER_VIDEO, FEATURES_PER_SEQUENCE))
    dense_1 = Dense(1024,name='dense_1')(inputs)
    act1 = Activation(custom_activation, name="relu1")(dense_1)
    dense_2 = Dense(512,name='dense_2')(act1)
    act2 = Activation(custom_activation, name="relu2")(dense_2)
    
    dense_3 = Dense(128,name='dense_3')(act2)
    act3 = Activation(custom_activation, name="relu3")(dense_3)
    
    dense_4 = Dense(1,name='dense_4')(act3)
    sigmoid = Activation("sigmoid",name="sigmoid1")(dense_4)
    
    prediction = Flatten(name='flatten')(sigmoid)
    #Returns a 10 dim matrix
    prediction = SortLayer(k=1, label=1, name='output')(prediction)
    model = Model(inputs=inputs, outputs=prediction)
    return model

def createModel2():
    inputs = Input(shape=(SEQUENCES_PER_VIDEO, FEATURES_PER_SEQUENCE))
   
    dense_1 = Dense(outdim,name='dense_1',kernel_regularizer=l1_l2(l1=l1factor, l2=l2factor))(inputs)
    act1 = Activation("relu",name="relu1")(dense_1)
    dense_2 = Dense(outdim2,name='dense_2',kernel_regularizer=l1_l2(l1=l1factor, l2=l2factor))(act1)
    act2 = Activation("relu",name="relu2")(dense_2)

    dense_3 = Dense(1,name='dense_3',kernel_regularizer=l1_l2(l1=l1factor, l2=l2factor))(act2)
    sigmoid = Activation("sigmoid",name="sigmoid1")(dense_3)
    prediction = Flatten(name='flatten')(sigmoid)
    prediction = SortLayer(k=1, label=1, name='output')(prediction)
    model = Model(inputs=inputs, outputs=prediction)

    return model

def createModel3():
    inputs = Input(shape=(SEQUENCES_PER_VIDEO, FEATURES_PER_SEQUENCE))
   

    dense_3 = Dense(1,name='dense_3',kernel_regularizer=l1_l2(l1=l1factor, l2=l2factor))(inputs)
    sigmoid = Activation("sigmoid",name="sigmoid1")(dense_3)
    prediction = Flatten(name='flatten')(sigmoid)
    prediction = SortLayer(k=1, label=1, name='output')(prediction)
    model = Model(inputs=inputs, outputs=prediction)
    return model


# In[47]:


# In[48]:
feats=sio.loadmat('Norm_bagwise_lbptop_feats_53_with_relabeled_KMeans_max1.mat');
X=feats['lbptop_feats_bagwise'];
Y=feats['labels_bagwise'];
Y=Y/3;

#with h5.File('segmented_hog_au_50000_with_labels.h5')as hf:
#        X=hf['segments'][:].transpose()
#        Y=hf['label'][:].transpose()
##a=np.where(Y==0)        
##c=np.where(Y==2)
##d=np.where(Y==3)
##Y[c]=1;
##Y[d]=1;
#
#Y = np_utils.to_categorical(Y, 2)     
#X=X.astype('float32')
#Y=Y.astype('float32')   
#np.shape(newFeatures)
#newFeatures=np.reshape(newFeatures, (np.shape(newLabelsOneHot)[0], FRAMES_PER_VIDEO, FEATURES_PER_FRAME))
##np.shape(newFeatures)
#
##np.shape(newFeaturesVal)
#newFeaturesVal=np.reshape(newFeaturesVal, (np.shape(newLabelsOneHotVal)[0], FRAMES_PER_VIDEO, FEATURES_PER_FRAME))
##np.shape(newFeaturesVal)
#

# In[49]:

#
#newFeatures=newFeatures.astype('float32')
#newLabelsOneHot=newLabelsOneHot.astype('float32')
#newFeaturesVal=newFeaturesVal.astype('float32')
#newLabelsOneHotVal=newLabelsOneHotVal.astype('float32')


# In[50]:


learning_rate=6e-5
sgd = Adam(lr=learning_rate) #SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)


#nb_epoch = 10
#batch_size = 256


# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

# define 10-fold cross validation test harness
#kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
#cvscores = []

#X=newFeatures
#Y=newLabelsOneHot

# split into 67% for train and 33% for test
#X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=seed)
#x_train = X[0:3500,:,:]
#y_train = Y[0:3500,:]
#
#x_test= X[3500:4999,:,:]
#y_test= Y[3500:4999,:]

Y_Ori=sio.loadmat("labels_53_by_Shreyank.mat");
Y_Ori=Y_Ori['labels_53']
Y_Ori=Y_Ori.astype(float)
Y_Ori=Y_Ori/3;

final_score=[];
for i in xrange(8):
    i=i+1;
    print(i)
    
    if i==1:
        X_train=X[0:48,:,:]
        y_train=Y[0:48,:]
        X_test=X[48:53,:,:]
        y_test=Y[48:53,:]
        
        Y_o=Y_Ori[48:53,:]
        Y_oo=Y_Ori[0:48,:]
        
    elif i==8:
        X_train=X[7:53,:,:]
        y_train=Y[7:53,:]
        X_test=X[0:7,:,:]
        y_test=Y[0:7,:]
        
        Y_o=Y_Ori[0:7,:]
        Y_oo=Y_Ori[7:53,:]
        
    elif i==2:
        X_train=np.concatenate(X[0:7,:,:],X[14:53,:,:])
        y_train=np.concatenate(Y[0:7,:],Y[14:53,:])
        X_test=X[7:14,:,:]
        y_test=Y[7:14,:]
        
        Y_o=Y_Ori[7:14,:]
        Y_oo=np.concatenate(Y_Ori[0:7,:],Y_Ori[14:53,:])
        
    elif i==3:
        X_train=np.concatenate(X[0:14,:,:],X[21:53,:,:])
        y_train=np.concatenate(Y[0:14,:],Y[21:53,:])
        X_test=X[14:21,:,:]
        y_test=Y[14:21,:]
        
        Y_o=Y_Ori[14:21,:]
        Y_oo=np.concatenate(Y_Ori[0:14,:],Y_Ori[21:53,:])
    elif i==4:
        X_train=np.concatenate(X[0:21,:,:],X[28:53,:,:])
        y_train=np.concatenate(Y[0:21,:],Y[28:53,:])
        X_test=X[21:28,:,:]
        y_test=Y[21:28,:]
        
        Y_o=Y_Ori[21:28,:]
        Y_oo=np.concatenate(Y_Ori[0:21,:],Y_Ori[28:53,:])
    elif i==5:
        
        X_train=np.concatenate(X[0:28,:,:],X[35:53,:,:])
        y_train=np.concatenate(Y[0:28,:],Y[35:53,:])
        X_test=X[28:35,:,:]
        y_test=Y[28:35,:]
        
        Y_o=Y_Ori[28:35,:]
        Y_oo=np.concatenate(Y_Ori[0:28,:],Y_Ori[35:53,:])
    elif i==6:
        X_train=np.concatenate(X[0:35,:,:],X[42:53,:,:])
        y_train=np.concatenate(Y[0:35,:],Y[42:53,:])
        X_test=X[35:42,:,:]
        y_test=Y[35:42,:]
        
        Y_o=Y_Ori[35:42,:]
        Y_oo=np.concatenate(Y_Ori[0:35,:],Y_Ori[42:53,:])
    elif i==7:
        X_train=np.concatenate(X[0:42,:,:],X[48:53,:,:])
        y_train=np.concatenate(Y[0:42,:],Y[48:53,:])
        X_test=X[42:48,:,:]
        y_test=Y[42:48,:]
        
        Y_o=Y_Ori[42:48,:]
        Y_oo=np.concatenate(Y_Ori[0:42,:],Y_Ori[48:53,:])     

#X_train=X[0:40,:,:]
#y_train=Y[0:40,:]
#X_test=X[40:53,:,:]
#y_test=Y[40:53,:]
#
#
#Y_o=Y_Ori[40:53,:]
#Y_oo=Y_Ori[0:40,:]


#X_train_shape=np.shape(X_train)[0]
#X_train_shape_rem=X_train_shape%batch_size
#X_train=X_train[0:-X_train_shape_rem, :]
#y_train=y_train[0:-X_train_shape_rem, :]
#
######
#X_train=np.concatenate((X_train,newFeaturesVal), axis=0);
#y_train=np.concatenate((y_train,newLabelsOneHotVal), axis=0);

#####

#print(np.shape(X_train))
#print(np.shape(y_train))
#print(np.shape(X_val))
#print(np.shape(y_val))
#create model
    model=createModel()
    print(model.summary())
    # Compile model
    #model.compile(loss='mean_squared_error',
    #            optimizer=sgd,
    #            metrics=['accuracy'])
    #learning_rate = 2e-4
    #rms=RMSprop(lr=learning_rate)
    model.compile(loss='mse', optimizer=sgd, metrics=['mse']) 
    # Fit the model
    
    #model.compile(loss=custom_objective, optimizer='adadelta')
    #model.fit(X_train, y_train,
    #              batch_size=batch_size,
    #              epochs=nb_epoch,
    #              shuffle=True)
    from keras import callbacks
    m_name='DeepMIL_MaxPooling_swish'+str(i)+'.h5'
    
    checkpoint = callbacks.ModelCheckpoint(m_name, monitor='val_mean_squared_error',
                                           save_best_only=True, save_weights_only=True, verbose=1,mode='min')
    early_stopping=callbacks.EarlyStopping(monitor='val_mean_squared_error', patience=200, verbose=0, mode='min')
    
    model.fit(X_train, y_train, epochs=1000,batch_size=1,verbose=1,
                  validation_data=(X_test, Y_o),callbacks=[checkpoint,early_stopping])
    
    model.load_weights(m_name)
    score = model.evaluate(X_test, Y_o,  verbose=0)
    #model.save('DeepMILMaxPooling_swish.h5')
    
    prediction = model.predict(X_test, batch_size=1)
    np.shape(prediction)
    
    
    error1=sklearn.metrics.mean_squared_error(Y_o, prediction)
    print(error1)
    #sio.savemat('prediction_aamir_deep_mil_lbptop.mat', {'prediction' : prediction})
    final_score.append(error1)
    #all_pred= model.predict(X, batch_size=1)
    #score = model.evaluate(X, Y_Ori,  verbose=0)
    #sio.savemat('prediction_deepmil_swish_BestModel.mat', {'prediction' : all_pred, 'Original_labels': Y_Ori})
    
#sio.savemat('prediction_deepmil_swish_MaxPooling_only13.mat', {'prediction' : prediction, 'Original_labels': Y_o})
print (final_score)