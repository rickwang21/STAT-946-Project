import pandas as pd
import numpy as np
import theano
import theano.tensor as T
from random import shuffle

def find_vector(word,vectors):
    if(len(vectors[vectors['word'] == word])==1):
        temp = vectors[vectors['word'] == word]['vector'].str.split().as_matrix()
        return temp
    else:
        return 'NULL'

def convert_to_vector(sentences,french_set,english_set):
    french = sentences['french'].str.split(' ')
    length = french.shape[0]
    english = sentences['english'].str.split(' ')
    french = french.apply(lambda list:list_to_word_vectors(list,french_set)).values.tolist()
    english = english.apply(lambda list:list_to_word_vectors(list,english_set)).values.tolist()
    french_mat = np.ones([length,french[0].shape[0],300])
    i = 0
    for sentence in french:
        if(sentence.shape[0]==french[0].shape[0]):
            french_mat[i] = sentence
        i += 1
    return french_mat,english

def list_to_word_vectors(list_of_strings,dictionary):
    temp = np.ones((300,len(list_of_strings)),dtype=np.float32)
    i = 0
    for word in list_of_strings:
        temp_result = find_vector(word,dictionary)
        if(temp_result!='NULL'):
            temp[:,i] = temp_result[0]
        i += 1
    temp = temp.transpose()
    return temp

