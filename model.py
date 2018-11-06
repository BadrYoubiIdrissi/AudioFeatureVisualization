#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 11:31:58 2018

@author: badrdr
"""

from keras.models import load_model
from keras import applications

def loadTruncatedModel(pathToModel):
    """
    Return a truncated model.
    Takes a path to an h5 model
    We remove the last fully connected layer to get rid of the fixed input size
    The resulting CNN is fuly convolutional and hence --theoratically-- doesn't need a fixed input size
    But to get rid of the fixed input size in practice we need to load the weights into a topless vgg
    """
    trainedModel = load_model(pathToModel)
    trainedModel.layers.pop()
    model = applications.VGG16(include_top=False)
    model.set_weights(trainedModel.get_weights())
    return model


def getLayerDict(model):
    """Return a dictionnary with layer_names as keys and the corresponding layers as values"""
    return dict([(layer.name, layer) for layer in model.layers])