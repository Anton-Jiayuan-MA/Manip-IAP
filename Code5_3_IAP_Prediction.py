# Setup
import pandas as pd
from openai import OpenAI
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
# API Key
api_key = "sk-hBeAQRKWMQ2SDP89FKl07H0TrJpR9eAzv5JOiDRzbbT3BlbkFJLLIvw8y4m2N2i3F_H3vSpa3rBk58Yj9UdUmflWT5gA" # Use your own api key
# Model Parameters
client = OpenAI(api_key=api_key)
# Import Dataset
test = pd.read_csv('/Users/anton.j.ma/Manip-IAP/test.csv') # Use your own path
intent1 = pd.read_csv('/Users/anton.j.ma/Manip-IAP/intent1_gpt-4-1106-preview.csv') # Use your own path
intent2 = pd.read_csv('/Users/anton.j.ma/Manip-IAP/intent2_gpt-4-1106-preview.csv') # Use your own path
# Prepare Dataset
test['Intent_p1'] = intent1['Intent_p1']
test['Intent_p2'] = intent2['Intent_p2']
print(test.head())

# Constructor: Two-intent Prompting
def iap_prompting(dialogue, intent_p1, intent_p2):
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

# Constructor: IAP Prediction
def iap_prediction(test_data):
    targets = [int(v) for v in test_data['Manipulative'].values]
    preds = []
    for idx, row in test_data.iterrows():
        intent_p1 = row['Intent_p1']
        intent_p2 = row['Intent_p2']
        dialogue = row['Dialogue']
        pred = iap_prompting(dialogue, intent_p1, intent_p2)
        preds.append(pred)
        test_data.at[idx, 'Prediction'] = pred
    test_data.to_csv('/Users/anton.j.ma/Manip-IAP/twointent_prediction.csv', index=False)
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

# IAP
gpt_model = "gpt-4-1106-preview" # Raplace it using 'gpt-4-1106-preview', 'gpt-4', 'gpt-4-turbo'
print("------Experiment: IAP Using gpt-4-1106-preview------")
iap_prediction(test)