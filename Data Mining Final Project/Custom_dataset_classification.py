from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import torch

from torch import optim
from torch import nn
from torch import optim
import torch.nn.functional as F

class MyDataset(Dataset):
    def __init__(self, data, source_vocab_max_size):
        self.data = data
        self.source_vocab_max_size = source_vocab_max_size

        #Machine Learning = 1
        #DataScience = 2
        #artificial = 3
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):

        y = self.data.iloc[idx][2]
            
        #y = np.expand_dims(y, axis=0)
        x = self.data.iloc[idx][3]

        z = self.data.iloc[idx][0]
        
        
        if len(x) != self.source_vocab_max_size:
            diff = self.source_vocab_max_size-len(x)
            x = np.pad(x, (0, diff), 'constant')
        else:
            x
        #### returning one Hot for genarator model ---- change this or remove when dealing with classification
        return torch.tensor(x).to(torch.int64), F.one_hot(torch.tensor(y).to(torch.int64), num_classes=3), z

def createSamplesOfData(data, frac):
    df = data.copy().groupby('subreddit', group_keys=False).apply(lambda x: x.sample(frac=frac))
    train_df = data[~data.apply(tuple,1).isin(df.apply(tuple,1))]
    return train_df, df
