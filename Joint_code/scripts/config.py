# config.py
# define all the configuration here

MAX_LEN = 128

POS_DIS_LIMIT = 300
POS_EMB_DIM = 60

TRAIN_BATCH_SIZE = 64
VAL_BATCH_SIZE = 4
EPOCHS = 30
LEARNING_RATE = 1e-3
TRAIN_DATASET_FNAME = '../inputs/datasets/train_datasets.csv'
TEST_DATASET_FNAME = '../inputs/datasets/expert_datasets.csv'

# NETWORK = 'lstm'
EMBEDDING_FNAME = '../inputs/pretrained_embeddings/crawl-300d-2M.vec'

NETWORK = 'CNN' # LSTM BiLSTM
