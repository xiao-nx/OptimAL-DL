# model prediction
import engine
from torchtext.legacy import data
import config
import torch
from torchtext import vocab
from importlib import import_module

def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))

def load_vectors(fname):
    """
    load pretrained word vector
    parameters:
    fname: the path of pretrined vector
    """
    # taken from: https://fasttext.cc/docs/en/english-vectors.html
    vectors_data = vocab.Vectors(name=fname)

    return vectors_data


def predicte(model_path):
    
    LABEL = data.LabelField(use_vocab=True)
    TEXT = data.Field(sequential=True, tokenize=lambda x:x.split(), lower=True, fix_length=512)

    train_dataset  = data.TabularDataset(path=config.TRAIN_DATASET_FNAME, 
                                    format='csv', 
                                    fields=[('text', TEXT),('label', LABEL)], 
                                    skip_header=True)
    
    test_data  = data.TabularDataset(path=config.TEST_DATASET_FNAME,
                                    format='csv', 
                                    fields=[('text', TEXT)], 
                                    skip_header=True)
    # load embeddings
    vectors_data = load_vectors(config.EMBEDDING_FNAME)
    

    TEXT.build_vocab(train_dataset, vectors=vectors_data)
    embedding_pretrained_matrix = TEXT.vocab.vectors
    vocab_size = len(TEXT.vocab)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    
    test_it = data.BucketIterator(test_data, 
                                  batch_size=config.TEST_BATCH_SIZE, 
                                  sort_key=lambda x: len(x.text), 
                                  shuffle=False,
                                  device=device)
    # selecte network  
    x = import_module('networks.'+config.NETWORK)
    model = x.Model(vocab_size,embedding_pretrained=embedding_pretrained_matrix)
    
    # load trained model
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    start_time = time.time()
    
    test_outputs, test_labels, _ = engine.evaluate_fn(test_it, model, device)
    test_outputs = torch.Tensor(test_outputs)
    _, test_predicted = torch.max(test_outputs, dim=1)
    print('test_predicted: ',test_predicted)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)


    
if __name__=='__main__':
    
    model_path = './outputs_1/checkpoint.pt'
    predicte(model_path)
    
