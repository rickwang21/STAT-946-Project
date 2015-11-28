import pandas as pd
import numpy as np
import theano
import theano.tensor as T

def find_vector(word,vectors):
    if(len(vectors[vectors['word'] == word])==1):
        return np.array([vectors[vectors['word'] == word]['vector'].str.split()])
    else:
        return 'NULL'
