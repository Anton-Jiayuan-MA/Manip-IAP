# Setup
import pandas as pd
import csv
import os
from openai import OpenAI
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
import numpy as np

# Constructor: Zeroshot Prompting
def zeroshot_prompting(dialogue, client, gpt_model):
    system_prompt = """
    I will provide you with a dialogue. \
    Please determine if it contains elements of mental manipulation. \
    Just answer with 'Yes' or 'No', and don't add anything else.\n
    """
    response = client.chat.completions.create(
        model=gpt_model,
        temperature=0.1,
        top_p=0.5,
        frequency_penalty=0.0,
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": dialogue,
            }
        ]
    )
    res = response.choices[0].message.content
    if 'yes' in res.lower():
        return 1
    elif 'no' in res.lower():
        return 0

# Constructor: Zeroshot Prediction
def zeroshot_prediction(test_data, client, gpt_model):
    client=client
    model=gpt_model
    targets = [int(v) for v in test_data['Manipulative'].values]
    preds = []
    for idx, row in test_data.iterrows():
        dialogue = row['Dialogue']
        pred = zeroshot_prompting(dialogue)
        preds.append(pred)
        test_data.at[idx, 'Prediction'] = pred
    test_data.to_csv('/Users/anton.j.ma/Manip-IAP/zeroshot_prediction.csv', index=False)
    # Performance Indicators
    accuracy = accuracy_score(targets, preds)
    precision = precision_score(targets, preds, zero_division=0)
    recall = recall_score(targets, preds, zero_division=0)
    weighted_f1 = f1_score(targets, preds, average='weighted', zero_division=0)
    macro_f1 = f1_score(targets, preds, average='macro', zero_division=0)
    conf_matrix = confusion_matrix(targets, preds)
    FP = conf_matrix[0][1]  # False Positives: predicted 1, actual 0
    FN = conf_matrix[1][0]  # False Negatives: predicted 0, actual 1
    # Print Results
    print(f"- Accuracy = {accuracy:.3f}")
    print(f"- Precision = {precision:.3f}")
    print(f"- Recall = {recall:.3f}")
    print(f"- Weighted F1-Score = {weighted_f1:.3f}")
    print(f"- Macro F1-Score = {macro_f1:.3f}")
    print(f"- Confusion Matrix = \n{conf_matrix}")
    print(f"- False Positives (FP) = {FP}")
    print(f"- False Negatives (FN) = {FN}")

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
    fewshot_examples.to_csv('/content/drive/MyDrive/COLING2025 MetanlManip Intent/dataset/fewshot_examples.csv', index=False)
    return manip_examples, nonmanip_examples

# Constructor: Fewshot Prompting
def fewshot_prompting(manip_examples, nonmanip_examples, dialogue):
    example_list = []
    total_example_num = len(manip_examples) + len(nonmanip_examples)
    count_example = 0
    # Add examples with manipulation
    for idx, row in manip_examples.iterrows():
        count_example += 1
        example = [
            {"role": "user", "content": f"Example {count_example}:\n{row['Dialogue']}"},
            {"role": "assistant", "content": "Yes"},
        ]
        example_list.extend(example)
    # Add examples without manipulation
    for idx, row in nonmanip_examples.iterrows():
        count_example += 1
        example = [
            {"role": "user", "content": f"Example {count_example}:\n{row['Dialogue']}"},
            {"role": "assistant", "content": "No"},
        ]
        example_list.extend(example)
    # System Prompt
    system_prompt = f"""I will provide you with a dialogue. Please determine if it contains
    elements of mental manipulation. Just answer with 'Yes' or 'No', and don't add anything else.
    Here are {total_example_num} examples:\n"""
    messages = [{"role": "system",
                 "content": system_prompt}]
    messages += example_list
    messages.append({"role": "user",
                     "content": dialogue})
    # API
    response = client.chat.completions.create(
        model=gpt_model,
        temperature=temperature,
        top_p=top_p,
        frequency_penalty=penal,
        messages=messages
    )
    res = response.choices[0].message.content
    if 'yes' in res.lower():
        return 1
    elif 'no' in res.lower():
        return 0

# Constructor: Fewshot Prediction
def fewshot_prediction(test_data, manip_examples, nonmanip_examples):
    targets = [int(v) for v in test_data['Manipulative'].values]
    preds = []
    for idx, row in test_data.iterrows():
        dialogue = row['Dialogue']
        pred = fewshot_prompting(manip_examples, nonmanip_examples, dialogue)
        preds.append(pred)
        test_data.at[idx, 'Prediction'] = pred
    test_data.to_csv('/content/drive/MyDrive/COLING2025 MetanlManip Intent/dataset/fewshot_prediction.csv', index=False)
    # Performance Indicators
    accuracy = accuracy_score(targets, preds)
    precision = precision_score(targets, preds, zero_division=0)
    recall = recall_score(targets, preds, zero_division=0)
    weighted_f1 = f1_score(targets, preds, average='weighted', zero_division=0)
    macro_f1 = f1_score(targets, preds, average='macro', zero_division=0)
    conf_matrix = confusion_matrix(targets, preds)
    FP = conf_matrix[0][1]  # False Positives: predicted 1, actual 0
    FN = conf_matrix[1][0]  # False Negatives: predicted 0, actual 1
    # Print Results
    print(f"- Accuracy = {accuracy:.3f}")
    print(f"- Precision = {precision:.3f}")
    print(f"- Recall = {recall:.3f}")
    print(f"- Weighted F1-Score = {weighted_f1:.3f}")
    print(f"- Macro F1-Score = {macro_f1:.3f}")
    print(f"- Confusion Matrix = \n{conf_matrix}")
    print(f"- False Positives (FP) = {FP}")
    print(f"- False Negatives (FN) = {FN}")


# Constructor: CoT Prompting
def cot_prompting(dialogue):
    system_prompt = """
    I will provide you with a dialogue. \
    Please determine if it contains elements of mental manipulation. \
    Just answer with 'Yes' or 'No', and don't add anything else. \
    Let's think step by step. \n
    """
    response = client.chat.completions.create(
        model=gpt_model,
        temperature=temperature,
        top_p=top_p,
        frequency_penalty=penal,
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": dialogue,
            }
        ]
    )
    res = response.choices[0].message.content
    if 'yes' in res.lower():
        return 1
    elif 'no' in res.lower():
        return 0

# Constructor: CoT Prediction
def cot_prediction(test_data):
    targets = [int(v) for v in test_data['Manipulative'].values]
    preds = []
    for idx, row in test_data.iterrows():
        dialogue = row['Dialogue']
        pred = cot_prompting(dialogue)
        preds.append(pred)
        test_data.at[idx, 'Prediction'] = pred
    test_data.to_csv('/content/drive/MyDrive/COLING2025 MetanlManip Intent/dataset/cot_prediction.csv', index=False)
    # Performance Indicators
    accuracy = accuracy_score(targets, preds)
    precision = precision_score(targets, preds, zero_division=0)
    recall = recall_score(targets, preds, zero_division=0)
    weighted_f1 = f1_score(targets, preds, average='weighted', zero_division=0)
    macro_f1 = f1_score(targets, preds, average='macro', zero_division=0)
    conf_matrix = confusion_matrix(targets, preds)
    FP = conf_matrix[0][1]  # False Positives: predicted 1, actual 0
    FN = conf_matrix[1][0]  # False Negatives: predicted 0, actual 1
    # Print Results
    print(f"- Accuracy = {accuracy:.3f}")
    print(f"- Precision = {precision:.3f}")
    print(f"- Recall = {recall:.3f}")
    print(f"- Weighted F1-Score = {weighted_f1:.3f}")
    print(f"- Macro F1-Score = {macro_f1:.3f}")
    print(f"- Confusion Matrix = \n{conf_matrix}")
    print(f"- False Positives (FP) = {FP}")
    print(f"- False Negatives (FN) = {FN}")

# Constructor: Person1 Intent
def intent_p1(data):
    system_prompt = """
    I will provide you with a dialogue. \
    Please summarize the intent \
    of the statement made by Person1 \
    in one sentence. \n
    """
    def analyze_dialogue(dialogue):
        response = client.chat.completions.create(
            model=gpt_model,
            temperature=temperature,
            top_p=top_p,
            frequency_penalty=penal,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": dialogue,
                }
            ]
        )
        return response.choices[0].message.content
    data['Intent_p1'] = data['Dialogue'].apply(analyze_dialogue)
    return data

# Constructor: Person2 Intent
def intent_p2(data):
    system_prompt = """
    I will provide you with a dialogue. \
    Please summarize the intent \
    of the statement made by Person2 \
    in one sentence. \n
    """
    def analyze_dialogue(dialogue):
        response = client.chat.completions.create(
            model=gpt_model,
            temperature=temperature,
            top_p=top_p,
            frequency_penalty=penal,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": dialogue,
                }
            ]
        )
        return response.choices[0].message.content
    data['Intent_p2'] = data['Dialogue'].apply(analyze_dialogue)
    return data

# Constructor: Two-intent Prompting
def twointent_prompting(dialogue, intent_p1, intent_p2):
    # system prompt
    system_prompt = """
    I will provide you with a dialogue \
    and intent of person1, \
    and intent of person2. \
    Please carefully analyze the dialogue and intents, \
    and determine if it contains elements of mental manipulation. \
    Just answer with 'Yes' or 'No', \
    and don't add anything else. \n
    """
    # user prompt
    user_input = f"{dialogue} {intent_p1} {intent_p2}"
    response = client.chat.completions.create(
        model=gpt_model,
        temperature=temperature,
        top_p=top_p,
        frequency_penalty=penal,
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_input
            }
        ]
    )
    res = response.choices[0].message.content
    if 'yes' in res.lower():
        return 1
    elif 'no' in res.lower():
        return 0

# Constructor: Two-intent Prediction
def twointent_prediction(test_data):
    targets = [int(v) for v in test_data['Manipulative'].values]
    preds = []
    for idx, row in test_data.iterrows():
        intent_p1 = row['Intent_p1']
        intent_p2 = row['Intent_p2']
        dialogue = row['Dialogue']
        pred = twointent_prompting(dialogue, intent_p1, intent_p2)
        preds.append(pred)
        test_data.at[idx, 'Prediction'] = pred
    test_data.to_csv('/content/drive/MyDrive/COLING2025 MetanlManip Intent/dataset/twointent_prediction.csv', index=False)
    # Performance Indicators
    accuracy = accuracy_score(targets, preds)
    precision = precision_score(targets, preds, zero_division=0)
    recall = recall_score(targets, preds, zero_division=0)
    weighted_f1 = f1_score(targets, preds, average='weighted', zero_division=0)
    macro_f1 = f1_score(targets, preds, average='macro', zero_division=0)
    conf_matrix = confusion_matrix(targets, preds)
    FP = conf_matrix[0][1]  # False Positives: predicted 1, actual 0
    FN = conf_matrix[1][0]  # False Negatives: predicted 0, actual 1
    # Print Results
    print(f"- Accuracy = {accuracy:.3f}")
    print(f"- Precision = {precision:.3f}")
    print(f"- Recall = {recall:.3f}")
    print(f"- Weighted F1-Score = {weighted_f1:.3f}")
    print(f"- Macro F1-Score = {macro_f1:.3f}")
    print(f"- Confusion Matrix = \n{conf_matrix}")
    print(f"- False Positives (FP) = {FP}")
    print(f"- False Negatives (FN) = {FN}")



