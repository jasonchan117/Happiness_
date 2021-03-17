from utils import *
import torch
from torchtext import data
import torch.nn as nn
import time
import torch.optim as optim
import argparse
import random

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--traindata', default='data/TRAIN/labeled_10k.csv',
                    help='The path to training data.')
parser.add_argument('--testdata', default='data/TEST/labeled_17k.csv',
                    help='The path to test data.')

# Training settings
parser.add_argument('--epochs', default=10, type=int,
                    help='epochs to train')
parser.add_argument('--dropout', default=0.5,
                    help='Dropout rate of RNN.')
parser.add_argument('--lr', default=1e-4, type=float,
                    help='learning rate')
parser.add_argument('--batch', default=64)
parser.add_argument('--weight_decay', default=0, type=float,
                    help='factor for L2 regularization')
parser.add_argument('--seed', default=594277, type=int,
                    help='manual seed')
parser.add_argument('--cuda', action='store_true',
                    help='Use cuda or not')
# RNN settings
parser.add_argument('--embed_dim', default=100,
                    help='The dimension of embedding.')
parser.add_argument('--hidden_dim', default=256,
                    help='The dimension of hidden layer.')
parser.add_argument('--layer',default=2,
                    help='The number of RNN layers.')
parser.add_argument('--bid',action='store_true',
                    help='RNN is bidirectional or not.')
args = parser.parse_args()

raw_x, raw_y = get_raw_data(args.traindata)
test_x, test_y = get_raw_data(args.testdata)



tokenize = lambda x: x.split()
TEXT = data.Field(sequential=True, tokenize=tokenize, lower=True, fix_length=200)
LABEL = data.Field(sequential=False, use_vocab=False,dtype=torch.float)

train_examples, train_fields = get_dataset(raw_x,raw_y, TEXT, LABEL, data=data)
test_examples, test_fields = get_dataset(test_x,test_y, TEXT, LABEL, data=data, test=True)

# Build training and validation datasets
train = data.Dataset(train_examples, train_fields)
test = data.Dataset(test_examples, test_fields)

train, valid = train.split(random_state=random.seed(args.seed))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(args.seed)
if args.cuda == True:
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
# Load word embedding
TEXT.build_vocab(train, max_size=25000, vectors="glove.6B.100d", unk_init=torch.Tensor.normal_)
LABEL.build_vocab(train)

train_iterator,valid_iterator, valid_iterator = data.BucketIterator.splits(
    (train, valid, test),
    batch_size = args.batch,
    sort_key= lambda x : len(x.text),
    sort_within_batch=False,
    device = device
)


INPUT_DIM = len(TEXT.vocab)  # 25002
EMBEDDING_DIM = args.embed_dim
HIDDEN_DIM = args.hidden_dim
OUTPUT_DIM = 1
N_LAYERS = args.layer
BIDIRECTIONAL = args.bid
DROPOUT = args.dropout
# The index of token 'pad'
PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]
# Initialize the model
