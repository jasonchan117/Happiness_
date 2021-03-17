import pandas as pd
from tqdm import tqdm
import torch
from sklearn.metrics import precision_score, recall_score, f1_score,accuracy_score
def get_raw_data(path):
    data = pd.read_csv(path)
    x = list(data.moment)
    y_agency = list(data.agency == 'yes')
    y_social = list(data.social == 'yes')
    y = []
    for i in range(len(y_social)):
        y_social[i] = 1 if y_social[i] else 0
        y_agency[i] = 1 if y_agency[i] else 0
        y.append([y_agency[i], y_social[i]])
    return x, y


# Construct example and field objects.
def get_dataset(data_x, label, text_field, label_field, data,test=False):

    fields = [("id", None),("text", text_field), ("agency", label_field), ('social', label_field)]
    examples = []

    if test:
        for text in tqdm(data_x):
            examples.append(data.Example.fromlist([None, text, None, None], fields))
    else:
        print(label[0])
        for i in range(len(data_x)):
            examples.append(data.Example.fromlist([None, data_x[i], label[i][0], label[i][1]], fields))
    return examples, fields


def training(model, iterator, optimizer,criterion):
    epoch_loss = 0
    epoch_acc = 0
    total_len = 0

    model.train()

    for batch in iterator:

        optimizer.zero_grad()
        predictions = model(batch.text).squeeze(1)
        loss = criterion(batch.label,predictions)
        acc = binary_accuracy(predictions, batch.label)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * len(batch.label)
        epoch_acc += acc.item() * len(batch.label)
        total_len += len(batch.label)

    return epoch_loss / total_len, epoch_acc / total_len

def evaluate(model, iterator, criterion):
    epoch_loss = 0
    total_len = 0

    model.eval()

    with torch.no_grad():

        ind=0
        sum_prec=0.
        sum_recall=0.
        sum_f1=0.
        sum_acc=0.
        for batch in iterator:

            predictions = model(batch.text).squeeze(1)
            loss = criterion(predictions, batch.label)
            acc = accuracy_score(batch.label.cpu(), torch.round(torch.sigmoid(predictions)).cpu())
            prec = precision_score(batch.label.cpu(), torch.round(torch.sigmoid(predictions)).cpu())
            recall = recall_score(batch.label.cpu(), torch.round(torch.sigmoid(predictions)).cpu())
            f1= f1_score(batch.label.cpu(), torch.round(torch.sigmoid(predictions)).cpu())
            sum_prec+=prec
            sum_f1+=f1
            sum_recall+=recall
            sum_acc+=acc

            epoch_loss += loss.item() * len(batch.label)
            total_len += len(batch.label)

            ind+=1
    model.train()

    return epoch_loss / total_len, sum_acc/ind , sum_prec / ind, sum_recall / ind , sum_f1 / ind

def binary_accuracy(preds, y):

    rounded_preds = torch.round(torch.sigmoid(preds))

    correct = (rounded_preds == y).float()
    acc = correct.sum() / len(correct)

    return acc
