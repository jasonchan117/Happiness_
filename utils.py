import pandas as pd
from tqdm import tqdm
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

