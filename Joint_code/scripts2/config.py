# config.py
# define all the configuration here

TRAIN_BATCH_SIZE = 64
VAL_BATCH_SIZE = 4
EPOCHS = 30
LEARNING_RATE = 1e-3
TRAIN_DATASET_FNAME = '../inputs/datasets/train_cleaned_dataset.csv'
TEST_DATASET_FNAME = '../inputs/datasets/expert_dataset.csv'

# NETWORK = 'lstm'
EMBEDDING_FNAME = '../inputs/pretrained_embeddings/crawl-300d-2M.vec'

NETWORK = 'CNN' # LSTM BiLSTM
