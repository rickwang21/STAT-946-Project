import pandas as pd
import numpy as np
import theano
import theano.tensor as T
from random import shuffle

def find_vector(word,vectors):
    if word in vectors:
        temp = vectors[word]
        return True, temp
    else:
        #print(word)
	#with open('missing','ab') as f: f.write(word+'\n')
        return False, 'NULL'

def convert_to_vector(sentences,french_set,english_set):
    #print(sentences)
    #print(sentences.shape)
    french = sentences['french'].str.split()
    length = french.shape[0]
    english = sentences['english'].str.split()
    length_en = english.shape[0]
    french = french.apply(lambda list:list_to_word_vectors(list,french_set)).values.tolist()
    english = english.apply(lambda list:word_to_idx(list,english_set)).values.tolist()
    french_mat = np.ones([length,french[0].shape[0],300])
    i = 0
    for sentence in french:
        french_mat[i] = sentence
        i += 1
    i = 0
    english_mat = np.ones([length_en,english[0].shape[1]])
    for sentence in english:
        english_mat[i] = sentence
        i += 1
    #print(english_mat.shape)
    #print(french_mat.shape)
    return [french_mat.astype('float32'),english_mat.astype('int')]

def list_to_word_vectors(list_of_strings,dictionary):
    temp = np.ones((300,len(list_of_strings)),dtype=np.float32)
    i = 0
    for word in list_of_strings:
        tr,temp_result = find_vector(word,dictionary)
        if(tr):
            temp[:,i] = temp_result
        i += 1
    temp = temp.transpose()
    return temp

def word_to_idx(list_of_strings,dictionary):
    temp = np.ones((1,len(list_of_strings)+1),dtype=np.int)*10001
    i = 0
    for word in list_of_strings:
        if word in dictionary:
            temp[0,i] = dictionary[word]
        i += 1
    temp[0,len(list_of_strings)] = 10002
    return temp

def shift_to_input(batch,word,ix_to_vector):
   temp_eng = np.copy(batch[1])
   temp_fr = np.copy(batch[0])
   new_fr = np.ones((temp_eng.shape[0],temp_fr.shape[1]+1,300))
   #print(new_fr.shape," ",temp_eng.shape)
   for i in range(temp_eng.shape[0]):
       if temp_eng[i,0] in ix_to_vector:
            new_fr[i,temp_fr.shape[1]] = ix_to_vector[temp_eng[i,0]]
       #else:
            #print(temp_eng[i,0])
   new_fr[:,:temp_fr.shape[1],:] = temp_fr[:,:,:]
   return new_fr.astype('float32'),temp_eng[:,1:]

def special_shift(batch,word,ix_to_vector):
   temp_eng = np.copy(batch[1])
   temp_fr = np.copy(batch[0])
   new_fr = np.ones((temp_eng.shape[0],temp_fr.shape[1]+1,300))
   #print(new_fr.shape," ",temp_eng.shape)
   for i in range(temp_eng.shape[0]):
       if temp_eng[i,0] in ix_to_vector:
            new_fr[i,temp_fr.shape[1]] = ix_to_vector[temp_eng[i,0]]
       #else:
            #print(temp_eng[i,0])
   new_fr[:,:temp_fr.shape[1],:] = temp_fr[:,:,:]
   return new_fr.astype('float32'),temp_eng[:,:]
        
