import pandas as pd
from tqdm import tqdm
import torch
from sklearn.metrics import precision_score, recall_score, f1_score,accuracy_score
def get_raw_data(path):
    data = pd.read_csv(path, encoding="unicode_escape")
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
    epoch_loss = 0.
    epoch_acc_a = 0.
    total_len = 0
    epoch_acc_s = 0.
    model.train()
    for batch in iterator:
        optimizer.zero_grad()
        predictions=[]
        prediction = model(batch.text)
        predictions.append(prediction[0].squeeze(1))
        predictions.append(prediction[1].squeeze(1))
        loss_a = criterion(batch.agency, predictions[0])
        loss_s = criterion(batch.social, predictions[1])
        loss = loss_a + loss_s
        acc_a = binary_accuracy(predictions[0], batch.agency).item()
        acc_s = binary_accuracy(predictions[1], batch.social).item()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        epoch_acc_a += acc_a
        epoch_acc_s += acc_s
        total_len += 1

    return epoch_loss/total_len, epoch_acc_a/total_len, epoch_acc_s/total_len

def evaluate(model, iterator, criterion):
    epoch_loss = 0.
    model.eval()
    with torch.no_grad():
        ind=0
        sum_prec_a=0.
        sum_recall_a=0.
        sum_f1_a=0.
        sum_acc_a =0.
        sum_prec_s=0.
        sum_recall_s=0.
        sum_f1_s=0.
        sum_acc_s =0.
        for batch in iterator:

            predictions = model(batch.text).squeeze(1)

            loss1 = criterion(predictions[0], batch.agency)
            loss2 = criterion(predictions[0], batch.agency)
            loss = loss1 + loss2
            # agency
            acc_a = accuracy_score(batch.agency.cpu(), torch.round(torch.sigmoid(predictions[0])).cpu())
            prec_a = precision_score(batch.agency.cpu(), torch.round(torch.sigmoid(predictions[0])).cpu())
            recall_a = recall_score(batch.agency.cpu(), torch.round(torch.sigmoid(predictions[0])).cpu())
            f1_a = f1_score(batch.agency.cpu(), torch.round(torch.sigmoid(predictions[0])).cpu())
            sum_prec_a+=prec_a
            sum_f1_a+=f1_a
            sum_recall_a+=recall_a
            sum_acc_a+=acc_a
            # social
            acc_s = accuracy_score(batch.social.cpu(), torch.round(torch.sigmoid(predictions[1])).cpu())
            prec_s = precision_score(batch.social.cpu(), torch.round(torch.sigmoid(predictions[1])).cpu())
            recall_s = recall_score(batch.social.cpu(), torch.round(torch.sigmoid(predictions[1])).cpu())
            f1_s = f1_score(batch.social.cpu(), torch.round(torch.sigmoid(predictions[1])).cpu())
            sum_prec_s+=prec_s
            sum_f1_s+=f1_s
            sum_recall_s+=recall_s
            sum_acc_s+=acc_s

            epoch_loss += loss.item()
            ind+=1
    model.train()

    return epoch_loss / ind, sum_acc_a/ind , sum_prec_a / ind, sum_recall_a / ind , sum_f1_a / ind , sum_acc_s/ind , sum_prec_s / ind, sum_recall_s / ind , sum_f1_s / ind
def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def binary_accuracy(preds, y):

    rounded_preds = torch.round(torch.sigmoid(preds))

    correct = (rounded_preds == y).float()
    acc = correct.sum() / len(correct)

    return acc
