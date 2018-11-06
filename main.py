#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 08:27:40 2018

@author: badrdr
"""
import numpy as np

from keras import backend as K
from keras.models import load_model
from keras import applications

import time

#We load the trained VGG model


path_to_model = "../gtzan.keras/models/GtzanVGG20181016.h5"

trainedModel = load_model(path_to_model)

img_width, img_height = 128, 129


"""
We remove the last fully connected layer to get rid of the fixed input size
The resulting CNN is fuly convolutional and hence --theoratically-- doesn't need a fixed input size
But to get rid of the fixed input size in practice we need to load the weights into a topless vgg
"""

trainedModel.layers.pop()

model = applications.VGG16(include_top=False)

model.set_weights(trainedModel.get_weights())

#We get the layer names and associate them to said layer in a dictionnary

layer_dict = dict([(layer.name, layer) for layer in model.layers])

##FilterVisualization

kept_filters = []

input_img = model.input
layer_name = 'block1_conv1'

for filter_index in range(64):
    print('Processing filter %d' % filter_index)
    start_time = time.time()
    # build a loss function that maximizes the activation
    # of the nth filter of the layer considered
    layer_output = layer_dict[layer_name].output
    loss = K.mean(layer_output[:, :, :, filter_index])
    
    # compute the gradient of the input picture wrt this loss
    grads = K.gradients(loss, input_img)[0]
    
    # normalization trick: we normalize the gradient
    grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)
    
    # this function returns the loss and grads given the input picture
    iterate = K.function([input_img], [loss, grads])
    
    input_img_data = (np.random.random((1, img_width, img_height, 3)) - 0.5) * 20 + 128
    
    step = 1.
    
    for i in range(100):
        loss_value, grads_value = iterate([input_img_data])
        input_img_data += grads_value * step
        if loss_value <= 0.:
            # some filters get stuck to 0, we can skip them
            break
    print('Current loss value:', loss_value)
    if loss_value > 0:
        kept_filters.append((input_img_data[0,:,:,0], loss_value))
    end_time = time.time()
    print('Filter %d processed in %ds' % (filter_index, end_time - start_time))
    
# we will stich the best 64 filters on a 8 x 8 grid.
n = 6

# the filters that have the highest loss are assumed to be better-looking.
# we will only keep the top 64 filters.
kept_filters.sort(key=lambda x: x[1], reverse=True)
kept_filters = kept_filters[:n * n]

# build a black picture with enough space for
# our 8 x 8 filters of size 128 x 128, with a 5px margin in between
margin = 5
width = n * img_width + (n - 1) * margin
height = n * img_height + (n - 1) * margin
stitched_filters = np.zeros((width, height))

# fill the picture with our saved filters
for i in range(n):
    for j in range(n):
        img, loss = kept_filters[i * n + j]
        width_margin = (img_width + margin) * i
        height_margin = (img_height + margin) * j
        stitched_filters[width_margin: width_margin + img_width,
            height_margin: height_margin + img_height] = img
