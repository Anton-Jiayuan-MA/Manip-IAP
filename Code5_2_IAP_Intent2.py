# Setup
import pandas as pd
from openai import OpenAI
# API Key
api_key = "sk-4GGGT4r0yD_5SZjcqLCzC9ONRDRG5CKFEb_q3G0GTAT3BlbkFJTTv5eIShVMiBP-ad9EuKW_mze2s3HptlKRPfJ_hgMA" # Use your own api key
# Model Parameters
client = OpenAI(api_key=api_key)
# Import Dataset
test = pd.read_csv('/Users/anton.j.ma/Manip-IAP/test.csv') # Use your own path

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
    # Edit filename below using 'gpt-4-1106-preview', 'gpt-3.5-turbo-0125'
    data.to_csv('/Users/anton.j.ma/Manip-IAP/intent2_gpt-3.5-turbo-0125.csv', index=False)  # Use your own path
    return data

# Intent 2
gpt_model = "gpt-3.5-turbo-0125" # Raplace it using 'gpt-4-1106-preview', 'gpt-3.5-turbo-0125'
print("------Person2 Intent Using gpt-3.5-turbo-0125------")
intent2 = intent_p2(test)
print(intent2)