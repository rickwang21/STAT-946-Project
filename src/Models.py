import helpers
import pandas as pd
import numpy as np
import theano
import theano.tensor as T
import lasagne

def Create_simple_LSTM(N_HIDDEN=1000,GRAD_CLIP=100,vocab=80000,layer=4):
    network = lasagne.layers.InputLayer(shape=(None, 1, 300))

    for _ in range(layer):
        network = lasagne.layers.LSTMLayer(
            network, N_HIDDEN, grad_clipping=GRAD_CLIP,
            nonlinearity=lasagne.nonlinearities.tanh)

    network = lasagne.layers.SliceLayer(network, -1, 1)
    softmax = lasagne.nonlinearities.softmax

    network = lasagne.layers.DenseLayer(network, num_units=vocab, W = lasagne.init.Normal(), nonlinearity=softmax)

    return network

