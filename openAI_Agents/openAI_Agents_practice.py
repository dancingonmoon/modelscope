import os
import asyncio
from openai import AsyncOpenAI
from openai.types.responses import ResponseTextDeltaEvent
from agents import OpenAIChatCompletionsModel, Agent, Runner, set_default_openai_client, set_tracing_disabled, WebSearchTool
from agents.model_settings import ModelSettings
from rich import print
from rich.markdown import Markdown
import platform
from typing import Literal


# 由于Agents SDK默认支持的模型是OpenAI的GPT系列，因此在修改底层模型的时候，需要将external_client 设置为：set_default_openai_client(external_client)

def custom2default_openai_model(model: str, base_url: str, api_key: str, ):
    custom_client = AsyncOpenAI(base_url=base_url, api_key=api_key)
    set_default_openai_client(custom_client)
    # we disable tracing under the assumption that you don't have an API key
    # from platform.openai.com. If you do have one, you can either set the `OPENAI_API_KEY` env var
    # or call set_tracing_export_api_key() to set a tracing specific key
    set_tracing_disabled(disabled=True)  # 不在platform.openai.com上trace
    default_openai_model = OpenAIChatCompletionsModel(model=model, openai_client=custom_client)
    return default_openai_model


async def agents_async_chat_once(agent: Agent, input_items: list[dict],
                                 runner_mode: Literal['async', 'stream'] = 'async'):
    """
    输入[{"role": "user", "content": prompt}]格式prompt,输出agent的result.to_input_list(),即全部历史记录；
    :param agent:
    :param input_items: list[dict],表示输入的prompt格式列表，例如: [{"role": "user", "content": prompt}]
    :param runner_mode:
    :return:
    """
    result = None
    if runner_mode == 'async':
        result = await Runner.run(agent, input_items)
        print(Markdown(result.final_output))
        return result.to_input_list()
    elif runner_mode == 'stream':
        result = Runner.run_streamed(agent, input_items)
        async for event in result.stream_events():
            if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
                print(event.data.delta, end="", flush=True)
        return result.to_input_list()


async def agents_chat_continuous(agent: Agent, runner_mode: Literal['async', 'stream'] = 'async'):
    result_input_list = []
    while True:
        user_input = input("\n💬 请输入你的消息(输入quit退出):")
        if user_input.lower() in ['exit', 'quit']:
            print("✅ 对话已结束")
            break
        result_input_list.append({"role": "user", "content": user_input})
        result_input_list = await agents_async_chat_once(agent=agent, input_items=result_input_list, runner_mode=runner_mode)
        result_input_list.append({"role": "user", "content": user_input})



if __name__ == '__main__':
    # model = 'qwen-plus'
    model = 'qwen-turbo-latest'
    base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    default_OpenAIModel = custom2default_openai_model(model=model,
                                                      base_url=base_url,
                                                      api_key=os.getenv("DASHSCOPE_API_KEY"),
                                                      )
    agent = Agent(name="my_assistant", instructions="你是一名助人为乐的助手。",
                  model=default_OpenAIModel,
                  model_settings=ModelSettings(
                      tool_choice=None,
                      parallel_tool_calls=False),
                  # tools=[WebSearchTool(user_location={"type": "approximate", "city": "New York"})], # 目前只支持openAI的模型

                  )
    # 设置事件循环策略
    if platform.system() == 'Windows':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    # 运行主协程
    asyncio.run(agents_chat_continuous(agent,  runner_mode='stream'), debug=False)
    # prompt = "hi"
    # result_items = asyncio.run(agents_async_chat_once(agent, prompt=prompt, runner_mode='async'), debug=False)
    # print(type(result_items))
