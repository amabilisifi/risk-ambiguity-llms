import os

from dotenv import load_dotenv
from openai import AsyncAzureOpenAI, AzureOpenAI

load_dotenv()

client = AzureOpenAI(
    api_version="2024-12-01-preview",
    azure_endpoint="https://zanistagpteastus2.openai.azure.com/",
    api_key=os.getenv("API_KEY"),
)


def get_client() -> AzureOpenAI:
    return client


def get_async_client() -> AsyncAzureOpenAI:
    return AsyncAzureOpenAI(
        api_version="2024-12-01-preview",
        azure_endpoint="https://zanistagpteastus2.openai.azure.com/",
        api_key=os.getenv("API_KEY"),
    )
