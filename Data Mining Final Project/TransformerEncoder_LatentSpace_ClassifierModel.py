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

#from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from sklearn.model_selection import train_test_split

from sklearn.model_selection import KFold
import random

import matplotlib.pyplot as plt

import seaborn as sns

from torch.utils.data import Dataset, DataLoader,TensorDataset,random_split,SubsetRandomSampler
torch.manual_seed(0)
##############################################################################################################
#took this transformer code from
#Just the encoder part of the transformer
#https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/more_advanced/transformer_from_scratch/transformer_from_scratch.py

class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads
        
        self.values = nn.Linear(self.head_dim, self.head_dim, bias = False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias = False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias = False)
        
        self.fc_out = nn.Linear(heads*self.head_dim, embed_size)
        self.attention = None
        self.after_attention_layer = None
        self.after_linear_layer = None
        
    def forward(self, values, keys, query):
        N = query.shape[0]  
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]
        
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys  = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, query_len, self.heads, self.head_dim)
        
        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)
        e = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        
        # Attention(Q, K, V)
        attention = torch.softmax(e / (self.embed_size**(1/2)), dim = 3)
        self.attention = attention
        
        #print(attention.shape)
        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(N, query_len,self.heads*self.head_dim)
        #print(out.shape)
        self.after_attention_layer = out
        out = self.fc_out(out)
        self.after_linear_layer = out
        #print(out.shape)
        return out
    
class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        
        self.attention = SelfAttention(embed_size, heads)    
        self.norm1 = nn.LayerNorm(embed_size) 
        self.norm2 = nn.LayerNorm(embed_size)
        
        self.feed_forward = nn.Sequential(nn.Linear(embed_size, forward_expansion*embed_size),
                                          nn.ReLU(),
                                          nn.Linear(forward_expansion*embed_size, embed_size))
        self.dropout = nn.Dropout(dropout)
            
    def forward(self, values, keys, query):        
        attention = self.attention(values, keys, query)  
        x = self.dropout(self.norm1(attention + query))
                
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
               
        return out
            
class Encoder(nn.Module):
                                                                                                #max sentence length
    def __init__(self, vocab_size, embed_size, num_layers, heads, device, forward_expansion, dropout, max_lenght, batch_size):
        super(Encoder, self).__init__()
        self.batch_size = batch_size
        self.embed_size = embed_size
        self.device = device
        self.word_embedding = nn.Embedding(vocab_size, embed_size, padding_idx = 0)
        self.position_embedding = nn.Embedding(max_lenght, embed_size, padding_idx = 0)
        
        self.layers = nn.ModuleList([TransformerBlock(embed_size, heads, dropout, forward_expansion) for _ in range(num_layers)])
        
        self.dropout = nn.Dropout(dropout)
        ###########################################################################################################################
        #I added these lines for the classification head
        self.fc_out1 = nn.Linear(embed_size,512)
        
        self.relu = nn.PReLU()
        
        self.fc_out2 = nn.Linear(512,3)
        
        self.result = None
        #self.fc_out3 = nn.Linear(64,1)
        
        self.softmax = nn.Softmax(dim=1)
        ###########################################################################################################################
    def forward(self, x):
        #print(x.shape)
        #fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(15, 15))
        N , seq_lenght = x.shape
   
        positions = torch.arange(0, seq_lenght).expand(N, seq_lenght).to(self.device)
        
        out = self.dropout(self.word_embedding(x) + self.position_embedding(positions))
        
        for layer in self.layers:
            out = layer(out, out, out)

        ###########################################################################################################################
        #I added these lines for the classification head    
        out = self.fc_out1(out.mean(dim=1))
        self.result = self.fc_out2(self.relu(out))
        out = self.fc_out2(self.relu(out))
        ###########################################################################################################################
        return self.softmax(out)

#######################################################################################################################################################
