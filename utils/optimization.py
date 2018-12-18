#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 11:32:40 2018

@author: badrdr
"""

import numpy as np

from keras import backend as K


def getIterateFunction(layerDict, layerName, filterIndex, inputTensor):
    """ Returns a function that gives the loss and grads given the input picture"""
    # build a loss function that maximizes the activation
    # of the nth filter of the layer considered
    layerOutput = layerDict[layerName].output
    loss = K.mean(layerOutput[:, :, filterIndex])
    
    # compute the gradient of the input picture wrt this loss
    grads = K.gradients(loss, inputTensor)[0]
    
    # normalization trick: we normalize the gradient
    grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)
    
    iterate = K.function([inputTensor], [loss, grads])
    
    return iterate

def gradientAscent(iterate, inputImgData, step, threshold=0.):
    """Simple gradient ascent given an iterate function and an initial input"""
    for i in range(100):
        lossValue, gradsValue = iterate([inputImgData])
        inputImgData += gradsValue * step
        if lossValue <= threshold:
            # some filters get stuck to 0, we can skip them
            print("failed")
            break
    return lossValue