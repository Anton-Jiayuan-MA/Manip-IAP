# Setup
import pandas as pd
from openai import OpenAI
# API Key
api_key = "xxxxxx" # Use your own api key
# Model Parameters
client = OpenAI(api_key=api_key)
# Import Dataset
test = pd.read_csv('Dataset/test.csv') # Use your own path

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
        return response.choices[0].message.content
    data['Intent_p2'] = data['Dialogue'].apply(analyze_dialogue)
    data.to_csv('Dataset/intent2_gpt-4-1106-preview.csv', index=False)  # Use your own path
    return data

# Intent 2
gpt_model = "gpt-4-1106-preview"
print("------Person2 Intent------")
intent2 = intent_p2(test)
print(intent2)