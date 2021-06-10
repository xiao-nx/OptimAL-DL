# dataset.py

import config
import torch

class BertDataset:
    def __init__(self,text,label):
        """
        parameters:
        texts: list or numpy array of strings
        labels: list or numpy array which is binary
        """
        self.text = text
        self.label = label
        # we fetch max len and tokenizer from config.py
        self.tokenizer = config.TOKENIZER
        self.max_len = config.MAX_LEN
    
    def __len__(self):
        # return length of the dataset
        return len(self.text)
    
    def __getitem__(self,item):
        # for any given item index, return a dictionary of inputs
        text = str(self.text[item])
        text = " ".join(text.split())

        # encode_plus comes from hugginface's transformers and exists for all tokenizers they offer it 
        # can be used to convert a given string to ids, mask and token type ids which are needed for models
        # like BERT here, the texts is a string.

        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
        )

        # print("inputs:",inputs)
        # ids are ids of tokens generated after tokenizing texts
        ids = inputs["input_ids"]
        # mask is 1 where we have input and 0 where we have padding
        mask = inputs["attention_mask"]
        # token type ids behave the same way as mask in this specific case in case of two sentences, this is 0
        # for first sentence and 1 for second sentence.
        token_type_ids = inputs["token_type_ids"]
        
        # now we return everything note that ids,mask and token_type_ids are all long datatypes and targets is float
        return {
            "ids": torch.tensor(ids, dtype=torch.long),
            "mask": torch.tensor(mask,dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_ids,dtype=torch.long),
            "labels": torch.tensor(self.label[item],dtype=torch.float)
        }