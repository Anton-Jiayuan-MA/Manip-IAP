# Setup
import pandas as pd
import numpy as np
# Import Dataset
train = pd.read_csv('Dataset/train.csv') # Use your own path

# Constructor: Example Selection
def select_examples(data, k_manip, k_nonmanip):
    manip_list = data[data['Manipulative'] == 1].index.tolist()
    # randomly select k_manip examples from manip_list
    selected_manip_idx = np.random.choice(manip_list, k_manip, replace=False)
    nonmanip_list = data[data['Manipulative'] == 0].index.tolist()
    # randomly select k_nonmanip examples from nonmanip_list
    selected_nonmanip_idx = np.random.choice(nonmanip_list, k_nonmanip, replace=False)
    manip_examples = data.loc[selected_manip_idx]
    nonmanip_examples = data.loc[selected_nonmanip_idx]
    # Combine and export to CSV
    fewshot_examples = pd.concat([manip_examples, nonmanip_examples])
    fewshot_examples.to_csv('Dataset/fewshot_examples.csv', index=False) # Use your own path
    return manip_examples, nonmanip_examples

# Select examples
manip_examples, nonmanip_examples = select_examples(train, k_manip=1, k_nonmanip=2)
print("------Fewshot Selected Examples------")
print(manip_examples)
print(nonmanip_examples)
