import helpers
import pandas as pd
import numpy as np
import theano
import theano.tensor as T
import lasagne
import Models
import os
import math
import pickle

#parameters
# Number of units in the (LSTM) layers
N_HIDDEN = 999

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

#Vocab (target), 1 more for end of line token
vocab = 80001

lasagne.random.set_rng(np.random.RandomState(1))

english_vocab = pd.read_csv('../data/small_english.txt',header=0,delimiter=';',dtype=np.str)

french_vocab = pd.read_csv('../data/small_french.txt',header=0,delimiter=';',dtype=np.str)


def create_training_set(sentences,min_size):
    processed = []
    for i in range(3,100):
        if(len(sentences[sentences['fr_length']==i]['french']) >= min_size):
            processed.extend(np.array_split(sentences[sentences['fr_length']==i],math.floor(len(sentences[sentences['fr_length']==i]['french'])/min_size)))
    return processed


def main():
    network,train,probs = Models.Create_simple_LSTM()
    if(os.path.isfile('stored_processed.p')!=True):
        english_set = pd.read_csv('../data/processed_en_reduced',header=0,delimiter='!')
        french_set = pd.read_csv('../data/processed_fr_reduced',header=0,delimiter='!')
        combined_set = pd.concat([english_set,french_set],axis=1)
        processed = create_training_set(combined_set,1)
        pickle.dump(processed,open('stored_processed.p','wb'))
    else:
        processed = pickle.load(open('stored_processed.p','rb'))
    french_proccessed,english_processed = helpers.convert_to_vector(processed[1],french_vocab,english_vocab)
    print(french_proccessed.shape)


main()
