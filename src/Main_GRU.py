import helpers
import params
import pandas as pd
import numpy as np
import theano
import theano.tensor as T
import lasagne
import Models
import os
import math
import pickle
import multiprocessing as mp
import gc
import time
from random import shuffle

#parameters

#Epochs
epoch = 5

# Number of units in the (LSTM) layers
N_HIDDEN = 500

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

#Vocab (target), 1 for end of line token, 1 for not found word
vocab = 80002

#Weights file name
weights = 'small_gru'

lasagne.random.set_rng(np.random.RandomState(1))



char_to_ix = {}
ix_to_char = {}
ix_to_vector = {}
i=0
chars = []
with open('../data/topVocab') as f:
    for ch in f:
        line = ch.rstrip()
        if line:
            char_to_ix[line] = i
            ix_to_char[i] = line
            i += 1

english_vocab = {}
with open('../data/new_english.txt') as f:
    for ch in f:
        words = ch.split(";")
        english_vocab[words[0]] = np.squeeze(np.array(words[1].split()))
        if words[0] in char_to_ix:
            ix_to_vector[char_to_ix[words[0]]] = np.squeeze(np.array(words[1].split()))
        #print(english_vocab[words[0]].shape)

french_vocab = {}
with open('../data/new_french.txt') as f:
    for ch in f:
        words = ch.split(";")
        french_vocab[words[0]] = np.squeeze(np.array(words[1].split()))


#english_vocab = pd.read_csv('../data/reduced_english.txt',header=0,delimiter=';',dtype=np.str)
#french_vocab = pd.read_csv('../data/reduced_french.txt',header=0,delimiter=';',dtype=np.str)

def create_training_set(sentences,min_size,batch_size):
    processed = []
    temp2 = 0
    for i in range(3,50):
        for j in range(i-2,i+2):
            temp = sentences[(sentences['fr_length']==i) & (sentences['en_length']==j)]
            if(len(temp['english']) >= min_size):
                #print(len(temp['english']))
                #temp2 += math.ceil(len(temp['english'])/float(batch_size))
                #print(temp2)
                processed.extend(np.array_split(temp,math.ceil(len(temp['english'])/float(batch_size))))
    print(len(processed))
    return processed


def lack_ram():   
    processed_test = []
    processed_val = []
    input_var = T.ftensor3('inputs')
    network,train_fn,val_fn,output = Models.Create_simple_GRU(input_var=input_var,N_HIDDEN=N_HIDDEN,layer=4,vocab=vocab)
    processed = []
    if(os.path.isfile(weights+'.params')):
        print("loading Weights")
        params.read_model_data(network, weights)
    if(os.path.isfile('stored_batch.p')!=True):
        if(os.path.isfile('stored_processed.p')!=True):
            print('Creating processed sentences file')
            print('Loading english and french data files')
            english_set = pd.read_csv('../data/processed_en',header=None,delimiter=',',names=['english','en_length'])
            french_set = pd.read_csv('../data/processed_fr',header=None,delimiter=',',names=['french','fr_length'])
            print('Combining the files')
            combined_set = pd.concat([english_set,french_set],axis=1)
            print('Removing Duplicates')
            print(len(combined_set['french']))
            combined_set = combined_set.drop_duplicates()
            print(len(combined_set['french'])) 
            print('Grouping sentences together by input and output sentence length')
            processed = create_training_set(combined_set,BATCH_SIZE,BATCH_SIZE)
            print('Store batches in a pickle file')
            pickle.dump(processed,open('stored_processed.p','wb'))
            gc.collect()
        else:
            print('Loading grouped sentences')
            processed = pickle.load(open('stored_processed.p','rb'))
            print('number of grouped sentences',len(processed))
        #print('Creating matrix file for grouped sentences')
        gc.collect()
        #pool = mp.Pool(processes=2)
        #processed_batch = [pool.apply_async(helpers.convert_to_vector,args=(batch,french_vocab,char_to_ix)) for batch in processed]
        #processed_batch = [p.get() for p in processed_batch]
        #for batch in processed:
        #    processed_batch.append(helpers.convert_to_vector(batch,french_vocab,char_to_ix))
        #print(len(processed_batch))
        #print('Dumping matrix data to file')
        #pickle.dump(processed_batch,open('stored_batch.p','wb'))
    else:
        print('Loading input and output matrix file')
        processed_batch = pickle.load(open('stored_batch.p','rb'))
    #print(ix_to_char)
    print("Shuffle and set validation set")
    processed_test = processed[:len(processed)-50]
    processed_val = processed[len(processed)-50:]
    #processed_test = processed[:1]
    #processed_val = processed[501:502]
    for i in range(epoch):
    	shuffle(processed_test) #Shuffle Batches     
        train_main_b = 0
        train_err = 0
        train_batches = 0
        start_time = time.time()
        for batch in processed_test:           
            curr_batch = helpers.convert_to_vector(batch,french_vocab,char_to_ix)
            fr,eng = helpers.shift_to_input(curr_batch,0,ix_to_vector)
            train_err += train_fn(fr,eng[:,0])
            train_batches += 1
            train_main_b += 1
            print("new batch ",train_main_b,len(processed_test))
            if(train_main_b % 2000 == 0):
                print("saving model",train_main_b)
                params.write_model_data(network, weights)
            for word in range(1,curr_batch[1].shape[1]-1):
                #print(word)
                #print(T.argmax(lasagne.layers.get_output(network,fr,allow_input_downcast=True),axis=1).eval())
                eng[:,0] = T.argmax(lasagne.layers.get_output(network,fr,allow_input_downcast=True),axis=1).eval().transpose()
                fr,eng = helpers.shift_to_input([fr,eng],word,ix_to_vector)
                train_err += train_fn(fr,eng[:,0])
                train_batches += 1

        #params.write_model_data(network, weights)
        # And a full pass over the validation data:
        val_err = 0
        val_acc = 0
        val_batches = 0
        for batch in processed_val:
            curr_batch = helpers.convert_to_vector(batch,french_vocab,char_to_ix)
            fr,eng = helpers.shift_to_input(curr_batch,0,ix_to_vector)
            error,acc = val_fn(fr,eng[:,0])
            val_err += error
            val_acc += acc
            val_batches += 1
            for word in range(1,curr_batch[1].shape[1]-1):
                eng[:,0] = T.argmax(lasagne.layers.get_output(network,fr,allow_input_downcast=True),axis=1).eval().transpose()
                fr,eng = helpers.shift_to_input([fr,eng],word,ix_to_vector)
                error,acc = val_fn(fr,eng[:,0])
                val_err += error
                val_acc += acc
                val_batches += 1

        # Then we print the results for this epoch:
        print("Epoch {} of {} took {:.3f}s".format(
            i, epoch, time.time() - start_time))
        print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
        print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
        print("  validation accuracy:\t\t{:.2f} %".format(
            val_acc / val_batches * 100))
        params.write_model_data(network, weights)
        #print(french_proccessed.shape)
        #print(english_processed)

def lots_of_ram():
    processed_test = []
    processed_val = []
    input_var = T.ftensor3('inputs')
    network,train_fn,val_fn,output = Models.Create_simple_LSTM(input_var=input_var,N_HIDDEN=N_HIDDEN,layer=4,vocab=vocab)
    processed = []
    if(os.path.isfile(weights+'.params')):
        print("loading Weights")
        params.read_model_data(network, weights)
    if(os.path.isfile('stored_batch.p')!=True):
        if(os.path.isfile('stored_processed.p')!=True):
            print('Creating processed sentences file')
            print('Loading english and french data files')
            english_set = pd.read_csv('../data/processed_en',header=None,delimiter=',',names=['english','en_length'])
            french_set = pd.read_csv('../data/processed_fr',header=None,delimiter=',',names=['french','fr_length'])
            print('Combining the files')
            combined_set = pd.concat([english_set,french_set],axis=1)
            print('Removing Duplicates')
            print(len(combined_set['french']))
            combined_set = combined_set.drop_duplicates()
            print(len(combined_set['french'])) 
            print('Grouping sentences together by input and output sentence length')
            processed = create_training_set(combined_set,3,100)
            print('Store batches in a pickle file')
            pickle.dump(processed,open('stored_processed.p','wb'))
            gc.collect()
        else:
            print('Loading grouped sentences')
            processed = pickle.load(open('stored_processed.p','rb'))
            print('number of grouped sentences',len(processed))
        #print('Creating matrix file for grouped sentences')
        gc.collect()
        #pool = mp.Pool(processes=2)
        #processed_batch = [pool.apply_async(helpers.convert_to_vector,args=(batch,french_vocab,char_to_ix)) for batch in processed]
        #processed_batch = [p.get() for p in processed_batch]
        #for batch in processed:
        #    processed_batch.append(helpers.convert_to_vector(batch,french_vocab,char_to_ix))
        #print(len(processed_batch))
        #print('Dumping matrix data to file')
        #pickle.dump(processed_batch,open('stored_batch.p','wb'))
    else:
        print('Loading input and output matrix file')
        processed_batch = pickle.load(open('stored_batch.p','rb'))
    #print(ix_to_char)
    print("Shuffle and set validation set")
    shuffle(processed) #Shuffle Batches
    processed_test = processed[:len(processed)-500]
    processed_val = processed[len(processed)-500:]
    #processed_test = processed[:20]
    #processed_val = processed[501:510]    
    p_test = [];
    for batch in processed_test:  
        p_test.append(helpers.convert_to_vector(batch,french_vocab,char_to_ix))
    p_val = [];
    for batch in processed_val:
        p_val.append(helpers.convert_to_vector(batch,french_vocab,char_to_ix))
    for i in range(epoch):
        train_main_b = 0
        train_err = 0
        train_batches = 0
        start_time = time.time()
        for curr_batch in p_test:           
            #curr_batch = helpers.convert_to_vector(batch,french_vocab,char_to_ix)
            fr,eng = helpers.shift_to_input(curr_batch,0,ix_to_vector)
            train_err += train_fn(fr,eng[:,0])
            train_batches += 1
            train_main_b += 1
            #print("new batch ",train_main_b,len(processed_test))
            if(train_main_b % 2000 == 0):
                print("saving model",train_main_b)
                params.write_model_data(network, weights)
            for word in range(1,curr_batch[1].shape[1]-1):
                #eng[:,0] = T.argmax(lasagne.layers.get_output(network,fr,allow_input_downcast=True),axis=1).eval().transpose()
                #print(word)
                #print(T.argmax(lasagne.layers.get_output(network,fr,allow_input_downcast=True),axis=1).eval())
                fr,eng = helpers.shift_to_input([fr,eng],word,ix_to_vector)
                train_err += train_fn(fr,eng[:,0])
                train_batches += 1

        params.write_model_data(network, weights)
        # And a full pass over the validation data:
        val_err = 0
        val_acc = 0
        val_batches = 0
        for curr_batch in p_val:
            #curr_batch = helpers.convert_to_vector(batch,french_vocab,char_to_ix)
            fr,eng = helpers.shift_to_input(curr_batch,0,ix_to_vector)
            error,acc = val_fn(fr,eng[:,0])
            val_err += error
            val_acc += acc
            val_batches += 1
            for word in range(1,curr_batch[1].shape[1]-1):
                eng[:,0] = T.argmax(lasagne.layers.get_output(network,fr,allow_input_downcast=True),axis=1).eval().transpose()
                fr,eng = helpers.shift_to_input([fr,eng],word,ix_to_vector)
                error,acc = val_fn(fr,eng[:,0])
                val_err += error
                val_acc += acc
                val_batches += 1
        # Then we print the results for this epoch:
        print("Epoch {} of {} took {:.3f}s".format(
            i, epoch, time.time() - start_time))
        print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
        print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
        print("  validation accuracy:\t\t{:.2f} %".format(
            val_acc / val_batches * 100))
        params.write_model_data(network, weights)
        #print(french_proccessed.shape)
        #print(english_processed)

lack_ram()
