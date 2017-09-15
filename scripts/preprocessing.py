from sklearn.externals import joblib
import sys
def read_dictionary(fn,omit_first=False,key_func=str,val_func=int,delim='|'):
    with open(fn) as dicts:
        d = {}
        if omit_first:
            dicts.readline()
        for ln in dicts:
            tup =  ln.rstrip().split(delim)
            d[key_func(tup[0])] = val_func(tup[1])
    return d

def add_word(w,token2id,id2token):
    if w not in token2id:
        token2id[w] = len(token2id)
        id2token[len(token2id)-1] = w
    return token2id[w]
data_dir = sys.argv[1]#'/home/mayk/crack_attention/data/raw/stanfordSentimentTreebank'
phrase_dictionary = read_dictionary(data_dir + '/dictionary.txt')
senti_dictionary = read_dictionary(data_dir + '/sentiment_labels.txt',omit_first=True,key_func=int,val_func=float)
split_dictionary = read_dictionary(data_dir + '/datasetSplit.txt',omit_first=True,key_func=int,val_func=int,delim=',')
with open(data_dir + '/SOStr.txt') as sents:
    d = {}
    cnt = 0
    binary_labels =  []
    fine_labels = []
    words =[]
    word2id = dict()
    id2word = dict()
    data = [[[],[]] for _ in range(3)]
    bin_data = [[[],[]] for _ in range(3)]

    senid = 0
    for ln in sents:
        sent =  ln.rstrip().split('|')
        senid+=1
        data_split  = split_dictionary[senid]
        label = senti_dictionary[phrase_dictionary[' '.join(sent)]]
        sent = [ add_word(w,word2id,id2word) for w in sent]
        if label<=0.4 or label>0.6:
            bin_data[data_split-1][0].append(sent)
            bin_data[data_split-1][1].append(0 if label<=0.4 else 1)
        
        if label<=0.2:
            fine_label = 0
        elif label <=0.4:
            fine_label = 1
        elif label <= 0.6:
            fine_label = 2
        elif label <=0.8:
            fine_label = 3
        else:
            fine_label = 4
        data[data_split-1][0].append(sent)
        data[data_split-1][1].append(fine_label)
train,test,dev = data
bin_train,bin_test,bin_dev = bin_data

joblib.dump({'data':[train,dev,test],'dicts':{'word2id':word2id,'id2word':id2word}},'data/preprocessed/fine.pkl')
joblib.dump({'data':[bin_train,bin_dev,bin_test],'dicts':{'word2id':word2id,'id2word':id2word}},'data/preprocessed/binary.pkl')