import os
import asyncio
from openai import AsyncOpenAI
from openai.types.responses import ResponseTextDeltaEvent
from agents import OpenAIChatCompletionsModel, Agent, Runner, set_default_openai_client
from agents.model_settings import ModelSettings
from rich import print
from rich.markdown import Markdown
import platform
from typing import Literal


# 由于Agents SDK默认支持的模型是OpenAI的GPT系列，因此在修改底层模型的时候，需要将external_client 设置为：set_default_openai_client(external_client)

def external2default_openai_model(model: str, base_url: str, api_key: str, ):
    external_client = AsyncOpenAI(base_url=base_url, api_key=api_key)
    set_default_openai_client(external_client)
    default_openai_model = OpenAIChatCompletionsModel(model=model, openai_client=external_client)
    return default_openai_model


async def agents_chat_once(agent: Agent, prompt: str,
                           runner_mode:Literal['async','sync','stream'] = 'async'):
    input_items = [{"role": "user", "content": prompt}]
    result = None
    if runner_mode == 'async':
        result = await Runner.run(agent, input_items)
        print(Markdown(result.final_output))
        return result.to_input_list()
    elif runner_mode == 'sync':
        result = Runner.run_sync(agent, input_items)
        print(result.final_output)
        return result
    elif runner_mode == 'stream':
        result = Runner.run_streamed(agent, input_items)
        async for event in result.stream_events():
            if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
                print(Markdown(event.data.delta), end="", flush=True)


async def agents_chat_continuous(agent: Agent, runner_mode:Literal['async','sync','stream'] = 'async'):
    input_items = []
    while True:
        user_input = input("💬 请输入你的消息(输入quit退出):")
        if user_input.lower() in ['exit', 'quit']:
            print("✅ 对话已结束")
            break

        await agents_chat_once(agent=agent, prompt=user_input, runner_mode=runner_mode)



if __name__ == '__main__':
    # model = 'qwen-plus'
    model = 'qwen-turbo-latest'
    base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    default_OpenAIModel = external2default_openai_model(model=model,
                                                        base_url=base_url,
                                                        api_key=os.getenv("DASHSCOPE_API_KEY"),
                                                        )
    agent = Agent(name="my_assistant", instructions="你是一名助人为乐的助手。",
                  model=default_OpenAIModel,
                  model_settings=ModelSettings(
                      tool_choice=None,
                      parallel_tool_calls=False),
                  )
    # 设置事件循环策略
    if platform.system() == 'Windows':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    # 运行主协程
    # asyncio.run(agents_chat_continuous(agent), debug=False)
    prompt = "hi"
    # result_items = asyncio.run(agents_chat_once(agent, prompt=prompt, runner_mode='sync'), debug=False)
    result_items = agents_chat_once(agent, prompt=prompt, runner_mode='sync')
    print(type(result_items))
