# config.py

import transformers
import os 

# this is the maximum number of tokens in the sentence
MAX_LEN = 512

# this is the maximum number of tokens in the sentence
TRAIN_BATCH_SIZE = 16  # 1
VALIDATION_BATCH_SIZE = 1

# maximum train epochs
EPOCHS = 10

# define path to BERT model files
# BERT_PATH = "../inputs/bert_base_uncased"
BERT_PATH = "../inputs/biobert-v1.1"

# file to save the model
MODEL_PATH = "pytorch_model.bin"


# train dataset file
DATASET_PATH = '../inputs/datasets/'
TRAIN_DATASET_FNAME = 'train_cleaned_dataset.csv'
VALIDATION_DATASET_FNAME = 'crowd_all_cleaned_dataset.csv'
TEST_DATASET_FNAME = 'expert_dataset.csv'
TRAIN_DATASET_PATH = os.path.join(DATASET_PATH, TRAIN_DATASET_FNAME)
VALIDATION_DATASET_PATH = os.path.join(DATASET_PATH, VALIDATION_DATASET_FNAME)
TEST_DATASET_PATH = os.path.join(DATASET_PATH, TEST_DATASET_FNAME)

# TOKENIZER = transformers.BertTokenizer.from_pretrained('bert-base-uncased')

# BioBert
TOKENIZER = transformers.BertTokenizer.from_pretrained('dmis-lab/biobert-v1.1', do_lower_case=True)

# BioBert
# TOKENIZER = BertTokenizer(vocab_file='biobert_v1.1_pubmed/vocab.txt', do_lower_case=False)

# from pytorch_pretrained_bert import BertTokenizer, BertConfig

# TOKENIZER = BertTokenizer.from_pretrained('biobert_v1.1_pubmed', do_lower_case=True)
