# dataset.py

import torch
from torchtext.legacy import data

class textDataset:
    '''
    Define a dataset loader.
    
    Attributes:
        data_path: Path to the data file
        filename: Filename of the dataset
    '''
    def __init__(self, texts, labels):
        """
        parameters:
        texts: a numpy array
        labels: a vector, numpy array
        """
        self.texts = texts
        self.labels = labels
        
        LABEL = data.LabelField(use_vocab=True)
        TEXT = data.Field(sequential=True, tokenize=lambda x:x.split(), lower=True)

        if self.split == True:
            dataset  = data.TabularDataset(path=fname, 
                                           format='csv', 
                                           fields=[('text', TEXT),('label', LABEL)], 
                                           skip_header=True)
            # split the dataset, 8:2
            train_dataset, valid_dataset = dataset.split(split_ratio=[0.8,0.2], random_state=random.getstate())
        
        
            
    
    def __len__(self):
        # return length of the dataset
        return len(self.texts)
    
    def __getitem__(self,item):
        # for any given item, which is an int, return text and label as torch tensor
        # item is the index of the item in concern

        text = self.texts[item,:]
        label = self.label[item]

        return {
            "text":torch.tensor(text,dtype=torch.long),
            "label":torch.tensor(label,dtype=torch.float)
        }
