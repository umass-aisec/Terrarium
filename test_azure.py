# save this as test_azure.py in your Terrarium folder
from openai import AzureOpenAI
import os
from dotenv import load_dotenv

load_dotenv()

client = AzureOpenAI(
    api_key=os.getenv("AI_FOUNDRY_API_KEY"),
    azure_endpoint="https://inference-experiments.openai.azure.com/openai/v1",
    api_version="2025-01-01-preview"
)

response = client.chat.completions.create(
    model="gpt-4.1-nano",
    messages=[{"role": "user", "content": "Say hi."}],
    max_tokens=5
)
print(response.choices[0].message.content)