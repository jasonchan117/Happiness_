from utils import *
import torch
from torchtext import data
import torch.nn as nn
import time
import torch.optim as optim
import argparse
import random
from model import *

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
parser.add_argument('--seed', default=1234, type=int,
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

train, valid = train.split(random_state=random.seed(args.seed), split_ratio=0.8)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(args.seed)
if args.cuda == True:
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
# Load word embedding
TEXT.build_vocab(train, max_size=25000, vectors="glove.6B.100d", unk_init=torch.Tensor.normal_)
LABEL.build_vocab(train)

train_iterator,valid_iterator, test_iterator = data.BucketIterator.splits(
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
model_a = RNN(INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM, N_LAYERS, BIDIRECTIONAL, DROPOUT, PAD_IDX)
model_s = RNN(INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM, N_LAYERS, BIDIRECTIONAL, DROPOUT, PAD_IDX)

pretrained_embeddings = TEXT.vocab.vectors
model_a.embedding.weight.data.copy_(pretrained_embeddings)
model_s.embedding.weight.data.copy_(pretrained_embeddings)

optimizer_a = optim.Adam(model_a.parameters(),lr=args.lr, weight_decay=args.weight_decay)
optimizer_s = optim.Adam(model_s.parameters(),lr=args.lr, weight_decay=args.weight_decay)

#optimizer_ = optim.Adam(model.parameters(),lr=args.lr, weight_decay=args.weight_decay)
model_a = model_a.to(device)
model_s = model_s.to(device)
criterion = nn.BCEWithLogitsLoss()
criterion = criterion.to(device)
best_valid_loss = float('inf')
for epoch in range(args.epochs):
    start_time = time.time()
    train_loss_a, train_acc_a = training(model_a, train_iterator, optimizer_a,  criterion)
    train_loss_s, train_acc_s = training(model_s, train_iterator, optimizer_s,  criterion, label_name= 'social')

    train_loss = train_loss_a + train_loss_s

    # Derives the metric values including accuracy, precision, recall and f1.
    valid_loss, valid_acc_a, valid_prec_a, valid_recall_a, valid_f1_a, valid_acc_s, valid_prec_s, valid_recall_s, valid_f1_s = evaluate(model_a, model_s, valid_iterator, criterion)
    end_time = time.time()
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)

    # Save the model.
    # if valid_loss < best_valid_loss:
    #     best_valid_loss = valid_loss
    #     torch.save(model.state_dict(), 'wordavg-model.pt')

    ###Printing results
    print(f'Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.6f} | Train Acc of Agency: {train_acc_a * 100:.5f} | Train Acc of Social: {train_acc_s * 100:.5f}%')
    print(f'\t Val. Loss: {valid_loss:.6f} | Val. Acc of Agency: {valid_acc_a * 100:.5f}% | Val. Acc of Social: {valid_acc_s * 100:.5f}%')
    #print(f'\t Val. Loss: {valid_loss:.3f} | Val. Acc of Agency: {valid_acc_a * 100:.2f}% | Val. Prec of Agency: {valid_prec_a * 100:.2f}% | Val. Recall of Agency: {valid_recall_a * 100:.2f}% | Val. F1 of Agency: {valid_f1_a * 100:.2f} | Val. Acc of Social: {valid_acc_s * 100:.2f}% | Val. Prec of Social: {valid_prec_s * 100:.2f}% | Val. Recall of Social: {valid_recall_s * 100:.2f}% | Val. F1 of Social: {valid_f1_s * 100:.2f}%')

