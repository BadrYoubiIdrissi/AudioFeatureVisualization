#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 08:27:40 2018

@author: badrdr
"""
import numpy as np
from model import loadTruncatedModel, getLayerDict
from optimization import getIterateFunction, gradientAscent

import time



if __name__ == '__main__':
    
    #We load the trained VGG model

    pathToModel = "model/model_VGG16_11_12_2018.h5"
    
    #Corresponds to the size of the training mel spectrogram
    
    imgWidth, imgHeight = 128, 129
    
    model = loadTruncatedModel(pathToModel)
    layerDict = getLayerDict(model)
    
    ##Filter Visualization on multiple filters and keeping the useful ones
    
    keptFilters = []
    
    inputTensor = model.input
    layerName = 'block1_conv1'
    
    for filterIndex in range(64):
        print('Processing filter %d' % filterIndex)
        startTime = time.time()
        
        iterate = getIterateFunction(layerDict, layerName, filterIndex, inputTensor)
        
        inputImgData = (np.random.random((1, imgWidth, imgHeight, 3)) - 0.5) * 20 + 128
        
        step = 1.
        
        lossValue = gradientAscent(iterate, inputImgData, step)
        
        print('Current loss value:', lossValue)
        
        if lossValue > 0.:
            keptFilters.append((inputImgData[0,:,:,0], lossValue))
        endTime = time.time()
        print('Filter %d processed in %ds' % (filterIndex, endTime - startTime))
    