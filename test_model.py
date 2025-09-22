from models import client
from models_enum import AzureModels

response = client.chat.completions.create(
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of Iran?"},
    ],
    max_completion_tokens=1000,
    model=AzureModels.GPT_5_MINI.value,
)

print(response.choices[0].message.content)
