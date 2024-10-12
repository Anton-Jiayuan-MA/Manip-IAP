# Setup
import pandas as pd
from openai import OpenAI
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
# API Key
api_key = "sk-4GGGT4r0yD_5SZjcqLCzC9ONRDRG5CKFEb_q3G0GTAT3BlbkFJTTv5eIShVMiBP-ad9EuKW_mze2s3HptlKRPfJ_hgMA" # Use your own api key
# Model Parameters
client = OpenAI(api_key=api_key)
# Import Dataset
test = pd.read_csv('/Users/anton.j.ma/Manip-IAP/test.csv') # Use your own path
examples = pd.read_csv('/Users/anton.j.ma/Manip-IAP/fewshot_examples.csv') # Use your own path
manip_examples = examples[examples['Manipulative'] == 1]
nonmanip_examples = examples[examples['Manipulative'] == 0]

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
        temperature=0.1,
        top_p=0.5,
        frequency_penalty=0.0,
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
    # Edit filename below using 'gpt-4-1106-preview', 'gpt-3.5-turbo-0125'
    test_data.to_csv('/Users/anton.j.ma/Manip-IAP/fewshot_prediction_gpt-3.5-turbo-0125.csv', index=False) # Use your own path
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

# Fewshot Prompting
gpt_model = "gpt-3.5-turbo-0125" # Raplace it using 'gpt-4-1106-preview', 'gpt-3.5-turbo-0125'
print("------Baseline 2: Fewshot Prompting Using gpt-3.5-turbo-0125------")
fewshot_prediction(test, manip_examples, nonmanip_examples)