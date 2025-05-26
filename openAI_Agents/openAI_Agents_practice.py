from openai import AsyncOpenAI
from agents import OpenAIChatCompletionsModel,Agent,Runner,set_default_openai_client
from agents.model_settings import ModelSettings

# 由于Agents SDK默认支持的模型是OpenAI的GPT系列，因此在修改底层模型的时候，需要将external_client 设置为：set_default_openai_client(external_client)

def external2openai_client(model:str,  base_url:str, API_KEY:str, API_KEY,)
external_client = AsyncOpenAI(base_url = base_url, api_key=API_KEY,)