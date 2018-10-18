#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 16:25:44 2018

@author: user
"""

from keras.utils.generic_utils import get_custom_objects


    
from keras.utils import CustomObjectScope

with CustomObjectScope({'SortLayer': SortLayer}):
    model = load_model('DeepMIL_BestModel_swish.h5')
    
    
layer_name='flatten'; #name of the layer to pick

from keras.models import Model

  
intermediate_layer_model = Model(inputs=model.input,outputs=model.get_layer(layer_name).output)


intermediate_output = intermediate_layer_model.predict(X, batch_size=1, verbose=1)
    
    
import  scipy.io

scipy.io.savemat('Localization_53_subjects.mat', mdict={'values':intermediate_output })    