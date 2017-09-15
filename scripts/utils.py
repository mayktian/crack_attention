import json
import numpy as np
import spacy

def create_id2vec(word2id,word2vec,dim_of_vector):
    unk_vec = word2vec['unk']
    dim_of_vector = len(unk_vec)
    num_of_tokens = len(word2id)
    id2vec = np.zeros((num_of_tokens+1,dim_of_vector),dtype=np.float32)
    for word,t_id in word2id.items():
        if word in word2vec:
            
            id2vec[t_id,:] = word2vec[word]
            
        elif word.lower() in word2vec:
            
            id2vec[t_id,:] = word2vec[word.lower()]
        
        else:
            
            id2vec[t_id,:] =  unk_vec
    return id2vec



def load_word2vec(file_path):
    word2vec = {}
    dim = 0
    with open(file_path) as lines:
        for line in lines:
            split = line.split()
            word = split[0]
            vector_strings = split[1:]
            vector = [float(num) for num in vector_strings]
            word2vec[word] = np.array(vector)
            dim  = len(vector)
    return word2vec,dim