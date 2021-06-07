# config.py

import transformers

# this is the maximum number of tokens in the sentence
MAX_LEN = 512

# this is the maximum number of tokens in the sentence
TRAIN_BATCH_SIZE = 1 
VALIDATION_BATCH_SIZE = 1

# maximum train epochs
EPOCHS = 10

# define path to BERT model files
# BERT_PATH = "../inputs/bert_base_uncased"
BERT_PATH = "../inputs/biobert-v1.1"

# file to save the model
MODEL_PATH = "pytorch_model.bin"


# train dataset file
TRAIN_DATASET_FNAME = '../inputs/train_cleaned_dataset.csv'
VALIDATION_DATASET_FNAME = '../inputs/crowd_all_cleaned_dataset.csv'
TEST_DATASET_FNAME = '../inputs/expert_dataset.csv'

# define the tokenizer, tokernizer and model form huggingface'transformers
# TOKENIZER = transformers.BertTokenizer.from_pretrained(
#     BERT_PATH,
#     do_lower_case = True
# )

# TOKENIZER = transformers.BertTokenizer.from_pretrained('bert-base-uncased')

# BioBert
TOKENIZER = transformers.BertTokenizer.from_pretrained('dmis-lab/biobert-v1.1', do_lower_case=True)

# BioBert
# TOKENIZER = BertTokenizer(vocab_file='biobert_v1.1_pubmed/vocab.txt', do_lower_case=False)

# from pytorch_pretrained_bert import BertTokenizer, BertConfig

# TOKENIZER = BertTokenizer.from_pretrained('biobert_v1.1_pubmed', do_lower_case=True)
