import helpers
import pandas as pd
import numpy as np
import theano
import theano.tensor as T
import lasagne

def Create_simple_LSTM(N_HIDDEN=1000,GRAD_CLIP=100,vocab=80002,layer=4,LEARNING_RATE=0.01,input_var=None):
    target_var = T.ivector('targets')
    l_in = lasagne.layers.InputLayer(shape=(None, None, 300),input_var=input_var)
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
    test_acc = T.mean(T.eq(T.argmax(network_output, axis=1), target_values),dtype=theano.config.floatX)
    all_params = lasagne.layers.get_all_params(l_out)
    updates = lasagne.updates.nesterov_momentum(
            cost, all_params, learning_rate=LEARNING_RATE, momentum=0.9)
    train = theano.function([input_var, target_values], cost, updates=updates, allow_input_downcast=True)
    compute_cost = theano.function([input_var, target_values], cost, allow_input_downcast=True)
    probs = theano.function([input_var],network_output,allow_input_downcast=True)
    val_fn = theano.function([input_var, target_values], [cost, test_acc])
    return l_out, train, val_fn,network_output

