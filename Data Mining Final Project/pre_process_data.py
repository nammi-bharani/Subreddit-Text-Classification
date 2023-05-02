import pandas as pd
import numpy as np
import tensorflow as tf
from collections import Counter
from torchvision import transforms

import torch
from torch import optim
from torch import nn
from torch import optim
import torch.nn.functional as F

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from sklearn.model_selection import train_test_split

from sklearn.model_selection import KFold
import random
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader,TensorDataset,random_split,SubsetRandomSampler
from sklearn.preprocessing import LabelEncoder
from itertools import product
torch.manual_seed(0)

from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
import re

def pre_process_remove_star(text):
    if text[-1] == "*":
        text = text[:-1]
    else:
        text

    return text

def index_encoding(text, word2index):
    #print(text)
    encoded = [word2index[ch] for ch in text.split(' ')]
    encoded.insert(0, 1)

    encoded.append(2)
    return encoded


#####################################################################################
def remove_stopwords(text, stop_words):
    return " ".join([word for word in str(text).split() if word not in stop_words])

def stem_words(text, stemmer):
    return " ".join([stemmer.stem(word) for word in text.split()])

def lemmatize_words(text, lemmatizer):
    return " ".join([lemmatizer.lemmatize(word) for word in text.split()])

#https://www.analyticsvidhya.com/blog/2021/06/must-known-techniques-for-text-preprocessing-in-nlp/
#####################################################################################
def preProcess(file):
    '''return preprocessed data word2idx and idx2word'''
    df = pd.read_excel(file)
    
    seqs = df['post_title']
    data = df[['post_title','subreddit']]
    labelencoder = LabelEncoder()
    data['subreddit_encoded'] = labelencoder.fit_transform(data['subreddit']
                                                          )
    #contraction_dict = {"ain't": "are not","'s":" is","aren't": "are not"}
    #contractions_re=re.compile('(%s)' % '|'.join(contraction_dict.keys()))
   # data['post_title']=data['post_title'].apply(lambda x:expand_contractions(x, contractions_re))
    
    #data['post_title'] = data['post_title'].lower()
    
    stop_words = set(stopwords.words('english'))
    stop_words.add('[Project]')
    stop_words.add('[D]')
    stop_words.add('[P]')
    stop_words.add('[R]')
    stop_words.add('[N]')
    stop_words.add('[d]')
    stop_words.add('[p]')
    stop_words.add('[r]')
    
    data['post_title'] = data['post_title'].apply(lambda x: remove_stopwords(x, stop_words))
    stemmer = PorterStemmer()
    
    data["post_title"] = data["post_title"].apply(lambda x: stem_words(x, stemmer))
    
    
    lemmatizer = WordNetLemmatizer()
    data["post_title"] = data["post_title"].apply(lambda text: lemmatize_words(text, lemmatizer))
    
    data["post_title"] = data["post_title"].apply(lambda text: re.sub(' +', ' ', text))
    
    data['post_title'] = data['post_title'].apply(lambda x: re.sub('[%s]' % re.escape(string.punctuation), '' , x))
    data['post_title'] = data['post_title'].apply(lambda x: re.sub('W*dw*','',x))
    
    
    return data


def buildVocab(data):
    total_counts = Counter()
    for i in range(0, len(data)):
        for word in data.iloc[i][0].split(' '):
            total_counts[word] += 1
            
    vocab = set(total_counts.keys())
    return sorted(vocab)


def FinalData(file):
    df = preProcess(file)
    vocab = buildVocab(df)
    
    
    maxlen = max([len(df['post_title'][i]) for i in range(len(df))])

    word2index = {}
    index2word = {}
    for i, word in enumerate(vocab):
        if(word != '*'):
            word2index[word] = i+3
            index2word[i+3] = word        
    index2word[2] = "*"
    word2index["*"] = 2

    word2index['<start>'] = 1
    index2word[1] = '<start>'

    word2index['p'] = 0
    index2word[0] = 'p'

    encoded_seq = []
    label = []
        
    for i in range(len(df)):
        encoded_seq.append(index_encoding(df['post_title'].iloc[i], word2index))
        #label.append(label_encoding(df['subreddit'].iloc[i]))

    df['idx_encoded'] = encoded_seq
    #df['label_encoding'] = label_encoding
    
    source_vocab_max_size = max([len(encoded_seq[i]) for i in range(len(encoded_seq))])
    
    return df, word2index, index2word, source_vocab_max_size, vocab


