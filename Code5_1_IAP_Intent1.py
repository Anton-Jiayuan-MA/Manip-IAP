# Setup
import pandas as pd
from openai import OpenAI
# API Key
api_key = "sk-hBeAQRKWMQ2SDP89FKl07H0TrJpR9eAzv5JOiDRzbbT3BlbkFJLLIvw8y4m2N2i3F_H3vSpa3rBk58Yj9UdUmflWT5gA" # Use your own api key
# Model Parameters
client = OpenAI(api_key=api_key)
# Import Dataset
test = pd.read_csv('/Users/anton.j.ma/Manip-IAP/test.csv') # Use your own path

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
    data['Intent_p1'] = data['Dialogue'].apply(analyze_dialogue)
    data.to_csv('/Users/anton.j.ma/Manip-IAP/intent1_gpt-4-1106-preview.csv', index=False)  # Use your own path
    return data

# Intent 1
gpt_model = "gpt-4-1106-preview" # Raplace it using 'gpt-4-1106-preview', 'gpt-4', 'gpt-4-turbo'
print("------Person1 Intent Using gpt-4-1106-preview------")
intent1 = intent_p1(test)
print(intent1)