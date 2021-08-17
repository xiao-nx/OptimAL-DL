# config.py
# define all the configuration here

TRAIN_BATCH_SIZE = 64
VAL_BATCH_SIZE = 4
TEST_BATCH_SIZE = 4

# 
RANDOM_SEED = 1234

# max epoch
EPOCHS = 100

# loss
ALPHA = 3 # 1-5

# learning rate
LEARNING_RATE = 1e-3

# the parameter use to select model
SELECTED = 'loss'  # F1score/loss

# dataset file path
TRAIN_DATASET_FNAME = '../inputs/datasets/dataset_sentences/train_datasets_two.csv'
TEST_DATASET_FNAME = '../inputs/datasets/dataset_sentences/expert_datasets_two.csv'

# embedding
EMBEDDING_FNAME = '../inputs/pretrained_embeddings/crawl-300d-2M.vec'

# type of network
NETWORK = 'TextCNN' # TextCNN textRNN textRNN_Att DPCNN fastText TextRCNN Transformer

# 
MAX_LENGTH = 512



