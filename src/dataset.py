# dataset.py

import torch

class myDataset:
    def __init__(self,texts,labels):
        """
        parameters:
        texts: a numpy array
        labels: a vector, numpy array
        """
        self.texts = texts
        self.label = labels
    
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
