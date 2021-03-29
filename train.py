from sklearn.metrics import classification_report
from utils import *
import torch
from torchtext import data
import torch.nn as nn
import time
import datetime
import torch.optim as optim
import argparse
import random
import os
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
parser.add_argument('--pretrain',action='store_true',
                    help='Use pretrain models or not.')
parser.add_argument('--bid',action='store_true',
                    help='RNN is bidirectional or not.')
parser.add_argument('--test',action='store_true',
                    help='Test or not.')
parser.add_argument('--output',default='../output')
args = parser.parse_args()


os.makedirs(args.output, exist_ok=True)
time_str = datetime.datetime.now().isoformat()
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

train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
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
best_valid_loss_a = float('inf')
best_valid_loss_s = float('inf')

if args.pretrain == True:
  print('Loading models')
  state_dict_a = torch.load('./RNN_model_a.pt')
  state_dict_s = torch.load('./RNN_model_s.pt')
  
  model_a.load_state_dict(state_dict_a)
  model_s.load_state_dict(state_dict_s)

if args.test == True and args.pretrain == True:
  print('Testing')
  flag = 1
  for batch in test_iterator:
      if flag == 1:
        lab_s = batch.agency 
        lab_a = batch.social
        # Social
        predictions_s = model_s(batch.text).squeeze(1)
        # Agency
        predictions_a = model_a(batch.text).squeeze(1)
        flag = 0
      else:
        predictions_s = torch.cat([predictions_s, model_s(batch.text).squeeze(1)], 0)
        predictions_a = torch.cat([predictions_a, model_a(batch.text).squeeze(1)], 0)
        lab_a = torch.cat([lab_a, batch.agency], 0)
        lab_s = torch.cat([lab_s, batch.social], 0)


  print('Agency:')
  print(classification_report(lab_a.cpu(), torch.round(predictions_a).cpu()))
  print('Social:')
  print(classification_report(lab_s.cpu(), torch.round(predictions_s).cpu()))

  sys.exit()


# Start training
for epoch in range(args.epochs):
    start_time = time.time()
    train_loss, train_acc = training(model_s, train_iterator, optimizer_s, criterion, label_name = 'social')
    valid_loss, valid_acc ,valid_prec, valid_recall, valid_f1= evaluate(model_s, valid_iterator, criterion, label_name='social')

    train_loss_a, train_acc_a = training(model_a, train_iterator, optimizer_a, criterion)
    valid_loss_a, valid_acc_a , valid_prec_a, valid_recall_a, valid_f1_a = evaluate(model_a, valid_iterator, criterion)

    end_time = time.time()
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)



    print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
    #Save the model.
    if valid_loss < best_valid_loss_s:
        best_valid_loss_s = valid_loss
        torch.save(model_s.state_dict(), 'RNN_model_s.pt')
    
    if valid_loss_a < best_valid_loss_a:
        best_valid_loss_a = valid_loss_a
        torch.save(model_a.state_dict(), 'RNN_model_a.pt')



    print('Agency:')
    print(f'\tTrain Loss: {train_loss_a:.6f} | Train Acc: {train_acc_a*100:.5f}%')
    print(f'\t Val. Loss: {valid_loss_a:.6f} |  Val. Acc: {valid_acc_a*100:.5f} |  Val. Prec: {valid_prec_a*100:.5f} |  Val. Recall: {valid_recall_a*100:.5f}|  Val. F1: {valid_f1_a*100:.5f}%')

    print('Social:')
    print(f'\tTrain Loss: {train_loss:.6f} | Train Acc: {train_acc*100:.5f}%')
    print(f'\t Val. Loss: {valid_loss:.6f} |  Val. Acc: {valid_acc*100:.5f}|  Val. Prec: {valid_prec*100:.5f} |  Val. Recall: {valid_recall*100:.5f}|  Val. F1: {valid_f1*100:.5f}%')
# Testing


