import os
import asyncio
from openai import AsyncOpenAI
from openai.types.responses import ResponseTextDeltaEvent
from agents import OpenAIChatCompletionsModel, Agent, Runner, set_default_openai_client, set_tracing_disabled, \
    function_tool
from agents.model_settings import ModelSettings
from rich import print
from rich.markdown import Markdown
from typing import Literal
import base64
import pathlib


# 由于Agents SDK默认支持的模型是OpenAI的GPT系列，因此在修改底层模型的时候，需要将custom_client 设置为：set_default_openai_client(external_client)

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
    输入[{"role": "user", "content": prompt}]格式prompt,输出agent的result类，可以通过result.new_items属性来查看全部的事件；
    result.new_items[0].raw_item，可以看具体的回复内容；to_input_list()方法，可以直接将用户的输入和本次输出结果拼接成一个消息列表
    :param agent:
    :param input_items: list[dict],表示输入的prompt格式列表，例如: [{"role": "user", "content": prompt}]
    :param runner_mode:
    :return:
    """
    result = None
    if runner_mode == 'async':
        result = await Runner.run(agent, input_items)
        print(Markdown(result.final_output))
    elif runner_mode == 'stream':
        result = Runner.run_streamed(agent, input_items)
        async for event in result.stream_events():
            if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
                print(event.data.delta, end="", flush=True)
    return result


async def agents_chat_continuous(agent: Agent, runner_mode: Literal['async', 'stream'] = 'async'):
    input_item = []
    while True:
        user_input = input("\n💬 请输入你的消息(输入quit退出):")
        if user_input.lower() in ['exit', 'quit']:
            print("✅ 对话已结束")
            break
        input_item.append({"role": "user", "content": user_input})
        result = await agents_async_chat_once(agent=agent, input_items=input_item, runner_mode=runner_mode)
        input_item = result.to_input_list()


@function_tool
def folder_search(query: str, folder_path: str):
    """
    搜索指定文件夹下的所有文件，并输出文件列表
    :param query:
    :param folder_path:
    :return:
    """
    files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if
             os.path.isfile(os.path.join(folder_path, file))]
    return files


def base64_image(image_path):
    """
    读取本地文件，并编码为 Base64 格式
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


@function_tool
def load_img(image_path):
    """
    1) 根据路径，加载一张图片文档，获取图片文件后缀;
    2) 对不支持的文件后缀，错误退出，并返回不支持的图像文档;
    3) 对支持的图片文档，进行base64编码;
    4) 按照图片的后缀，输出Qwen-VL模型input_item格式：{
                    "type": "image_url",
                    # 需要注意，传入Base64，图像格式（即image/{format}）需要与支持的图片列表中的Content Type保持一致。"f"是字符串格式化的方法。
                    # PNG图像：  f"data:image/png;base64,{base64_image}"
                    # JPEG图像： f"data:image/jpeg;base64,{base64_image}"
                    # WEBP图像： f"data:image/webp;base64,{base64_image}"
                    "image_url": {"url": f"data:image/png;base64,{base64_image}"}
    :param image_path: 单张图片，本地文件路径;支持的后缀：.bmp,.png,.jpe, .jpeg, .jpg,.tif,.tiff,.webp,.heic;
    :return: 返回Qwen-VL要求的本地图片文件上传格式;
    """
    supported_img = [".bmp", ".png", ".jpe", ".jpeg", ".jpg", ".tif", ".tiff", ".webp", ".heic"]
    jpg_variant = ['.jpe', '.jpeg', '.jpg']
    tif_variant = ['.tif', '.tiff']
    img_format = pathlib.Path(image_path).suffix
    if img_format not in supported_img:
        print(f"不支持的图片格式：{img_format}")
        return None
    if not pathlib.Path.exists(image_path):
        print(f"文件不存在：{image_path}")
        return None
    base64_img = base64_image(image_path)
    if img_format in jpg_variant:
        img_format = "jpeg"
    elif img_format in tif_variant:
        img_format = "tiff"
    input_item = {
        "type": "image_url",
        "image_url": {"url": f"data:image/{img_format};base64,{base64_img}"}
    }
    return input_item


# 通义千问VL：qwen-vl-plus-latest，模型可以根据您传入的图片来进行回答 输入:0.0015;输出:0.0045
# 图像问答：描述图像中的内容或者对其进行分类打标，如识别人物、地点、花鸟鱼虫等。
# 数学题目解答：解答图像中的数学问题，适用于中小学、大学以及成人教育阶段。
# 视频理解：分析视频内容，如对具体事件进行定位并获取时间戳，或生成关键时间段的摘要。
# 物体定位：定位图像中的物体，返回外边界矩形框的左上角、右下角坐标或者中心点坐标。
# 文档解析：将图像类的文档（如扫描件/图片PDF）解析为 QwenVL HTML格式，该格式不仅能精准识别文本，还能获取图像、表格等元素的位置信息。
# 文字识别与信息抽取：识别图像中的文字、公式，或者抽取票据、证件、表单中的信息，支持格式化输出文本；可识别的语言有中文、英语、日语、韩语、阿拉伯语、越南语、法语、德语、意大利语、西班牙语和俄语。

VL_agent = Agent(
    name="VL",
    instructions='''
    你是一个助人为乐的助手，可以根据您传入的图片来进行:
    1)图像问答：描述图像中的内容或者对其进行分类打标，如识别人物、地点、花鸟鱼虫等。
    2)数学题目解答：解答图像中的数学问题，适用于中小学、大学以及成人教育阶段。
    3)视频理解：分析视频内容，如对具体事件进行定位并获取时间戳，或生成关键时间段的摘要。
    4)物体定位：定位图像中的物体，返回外边界矩形框的左上角、右下角坐标或者中心点坐标。
    5)文档解析：将图像类的文档（如扫描件/图片PDF）解析为 QwenVL HTML格式，该格式不仅能精准识别文本，还能获取图像、表格等元素的位置信息。
    6)文字识别与信息抽取：识别图像中的文字、公式，或者抽取票据、证件、表单中的信息，支持格式化输出文本；可识别的语言有中文、英语、日语、韩语、阿拉伯语、越南语、法语、德语、意大利语、西班牙语和俄语。
    ''',
    model=custom2default_openai_model(model="qwen-vl-plus-latest",
                                      base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
                                      api_key=os.getenv("DASHSCOPE_API_KEY"),
                                      ),
    model_settings=ModelSettings(
                      tool_choice='auto',
                      parallel_tool_calls=False,
                      extra_body={
                          # "enable_thinking": True, # only support stream call
                          "enable_search": True,
                          'search_options': {
                              "forced_search": False,  # 强制开启联网搜索
                              "enable_source": False,  # 使返回结果包含搜索来源的信息，OpenAI 兼容方式暂不支持返回
                              "enable_citation": True,  # 开启角标标注功能
                              "citation_format": "[ref_<number>]",  # 角标形式为[ref_i]
                              "search_strategy": "pro"  # "pro"时,模型将搜索10条互联网信息
                          }
                      }
                  ),
                  tools=[load_img]

)

# 通义千问OCR：qwen-vl-ocr-latest，（输入输出：0.005），是文字提取专有模型，专注于文档、表格、试题、手写体文字等类型图像的文字提取能力。它能够识别多种文字，目前支持的语言有：汉语、英语、阿拉伯语、法语、德语、意大利语、日语、韩语、葡萄牙语、俄语、西班牙语、越南语。
# 支持在文字提取前，对图像进行旋转矫正，适合图像倾斜的场景。#
# 新增六种内置的OCR任务，分别是通用文字识别、信息抽取、文档解析、表格解析、公式识别、多语言识别。#
# 未设置内置任务时，支持用户输入Prompt进行指引；如设置了内置任务时，为保证识别效果，模型内部会使用任务指定的Prompt。
# 仅DashScope SDK支持对图像进行旋转矫正和设置内置任务。如需使用OpenAI SDK进行内置的OCR任务，需要手动填写任务指定的Prompt进行引导。


# Qwen2.5-VL模型支持将图像类的文档（如扫描件/图片PDF）解析为 QwenVL HTML格式，该格式不仅能精准识别文本，还能获取图像、表格等元素的位置信息。
# Prompt技巧：您需要在提示词中引导模型输出QwenVL HTML，否则将解析为不带位置信息的HTML格式的文本
if __name__ == '__main__':
    # model = 'qwen-plus'
    model = 'qwen-turbo-latest'
    base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    instruction = instructions = "你是一名助人为乐的助手，并且搜索并输出指定文件夹下的文件"
    default_OpenAIModel = custom2default_openai_model(model=model,
                                                      base_url=base_url,
                                                      api_key=os.getenv("DASHSCOPE_API_KEY"),
                                                      )
    agent = Agent(name="my_assistant", instructions=instruction,
                  model=default_OpenAIModel,
                  model_settings=ModelSettings(
                      tool_choice='auto',
                      parallel_tool_calls=False,
                      extra_body={
                          # "enable_thinking": True, # only support stream call
                          "enable_search": True,
                          'search_options': {
                              "forced_search": False,  # 强制开启联网搜索
                              "enable_source": False,  # 使返回结果包含搜索来源的信息，OpenAI 兼容方式暂不支持返回
                              "enable_citation": True,  # 开启角标标注功能
                              "citation_format": "[ref_<number>]",  # 角标形式为[ref_i]
                              "search_strategy": "pro"  # "pro"时,模型将搜索10条互联网信息
                          }
                      }
                  ),
                  # tools=[WebSearchTool(user_location={"type": "approximate", "city": "New York"})], # 目前只支持openAI的模型
                  tools=[folder_search]
                  )
    # 运行主协程
    asyncio.run(agents_chat_continuous(agent, runner_mode='stream'), debug=False)
