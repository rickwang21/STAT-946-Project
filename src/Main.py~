import helpers
import pandas as pd
import numpy as np
import theano
import theano.tensor as T
import lasagne
import Models
import os

english_vectors = pd.read_csv('../data/small_english.txt',header=0,delimiter=';',dtype=np.str)

french_vocab = pd.read_csv('../data/small_french.txt',header=0,delimiter=';',dtype=np.str)

lasagne.random.set_rng(np.random.RandomState(1))

#parameters
# Number of units in the (LSTM) layers
N_HIDDEN = 1000

# Optimization learning rate
LEARNING_RATE = .01

# All gradients above this will be clipped
GRAD_CLIP = 100

# How often should we check the output?
PRINT_FREQ = 1000

# Number of epochs to train the net
NUM_EPOCHS = 50

# Batch Size
BATCH_SIZE = 128

#Vocab (target)
vocab = 80000

def main():
    network = Models.Create_simple_LSTM()
    target_values = T.ivector('target_output')
    network_output = lasagne.layers.get_output(network)
    cost = T.nnet.categorical_crossentropy(network_output,target_values).mean()
    all_params = lasagne.layers.get_all_params(network)

    updates = lasagne.updates.adagrad(cost, all_params, LEARNING_RATE)

