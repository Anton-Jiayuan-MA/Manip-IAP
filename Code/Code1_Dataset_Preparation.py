# Setup
import pandas as pd
import csv
from sklearn.model_selection import train_test_split

# Constructor: load dataset, process dataset
def import_data(file_name):
    with open(file_name, 'r', newline='', encoding='utf-8') as infile:
        content = csv.reader(infile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
        data = []
        columns = None
        for idx, row in enumerate(content):
            if idx == 0:
                columns = row
            else:
                data.append(row)
    dataframe = pd.DataFrame(data, columns=columns)
    # drop certain columns；
    if 'ID' in dataframe.columns:
        dataframe = dataframe.drop(['ID'], axis=1)
    if 'Technique' in dataframe.columns:
        dataframe = dataframe.drop(['Technique'], axis=1)
    if 'Vulnerability' in dataframe.columns:
        dataframe = dataframe.drop(['Vulnerability'], axis=1)
    # switch row of 'Manipulative' into type of category
    if 'Manipulative' in dataframe.columns:
        dataframe['Manipulative'] = dataframe['Manipulative'].astype('category')
    return dataframe

# Constructor: split dataset
def split_train_test(dataframe, train_ratio, test_ratio, random_state=17):
    # shuffle dataset at random
    df_shuffled = dataframe.sample(frac=1, random_state=random_state).reset_index(drop=True)
    # split dataset proportionally
    train, test = train_test_split(
        df_shuffled,
        train_size=train_ratio,
        test_size=test_ratio,
        stratify=df_shuffled['Manipulative'],
        random_state=random_state
    )
    # shuffle train set and test set at random
    train = train.sample(frac=1, random_state=random_state).reset_index(drop=True)
    test = test.sample(frac=1, random_state=random_state).reset_index(drop=True)
    return train, test

# Load dataset, process dataset
dataframe = import_data('/Dataset/mentalmanip_con.csv')
total_counts = dataframe['Manipulative'].value_counts()
print("------Total counts in dataset------")
print(total_counts)

# Split dataset
train, test = split_train_test(dataframe, 0.7, 0.3)
print(f"Train samples = {len(train)}, Test samples = {len(test)}")

# Check number of 1 and 0 in train set
train_counts = train['Manipulative'].value_counts()
print("\n------Counts in the training set------")
print(train_counts)

# Check number of 1 and 0 in test set
test_counts = test['Manipulative'].value_counts()
print("\n------Counts in the test set------")
print(test_counts)

# save the train set and test set as CSV
train.to_csv('Dataset/train.csv', index=False)
test.to_csv('Dataset/test.csv', index=False)