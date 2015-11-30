import helpers
import pandas as pd
import numpy as np
import theano
import theano.tensor as T
import lasagne

def Create_simple_LSTM(N_HIDDEN=1000,GRAD_CLIP=100,vocab=80000,layer=4,LEARNING_RATE=0.01):
    l_in = lasagne.layers.InputLayer(shape=(None, None, 300))
    network = lasagne.layers.LSTMLayer(
            l_in, N_HIDDEN, grad_clipping=GRAD_CLIP,
            nonlinearity=lasagne.nonlinearities.tanh)

    for _ in range(layer-1):
        network = lasagne.layers.LSTMLayer(
            network, N_HIDDEN, grad_clipping=GRAD_CLIP,
            nonlinearity=lasagne.nonlinearities.tanh)

    network = lasagne.layers.SliceLayer(network, -1, 1)
    softmax = lasagne.nonlinearities.softmax

    l_out = lasagne.layers.DenseLayer(network, num_units=vocab, W = lasagne.init.Normal(), nonlinearity=softmax)

    target_values = T.ivector('target_out')
    network_output = lasagne.layers.get_output(l_out)
    cost = T.nnet.categorical_crossentropy(network_output,target_values).mean()
    all_params = lasagne.layers.get_all_params(l_out)
    updates = lasagne.updates.adagrad(cost, all_params, LEARNING_RATE)
    train = theano.function([l_in.input_var, target_values], cost, updates=updates, allow_input_downcast=True)
    compute_cost = theano.function([l_in.input_var, target_values], cost, allow_input_downcast=True)
    probs = theano.function([l_in.input_var],network_output,allow_input_downcast=True)
    return l_out, train, probs

