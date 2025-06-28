import os
import asyncio
from contextlib import AsyncExitStack
from openai import AsyncOpenAI, OpenAI
from openai.types.responses import ResponseTextDeltaEvent
from agents import OpenAIChatCompletionsModel, Agent, Runner, set_default_openai_client, set_tracing_disabled, \
    function_tool, TResponseInputItem, ItemHelpers
from agents.model_settings import ModelSettings
from agents.mcp import MCPServer, MCPServerStdio, MCPServerSse, MCPServerStreamableHttp

from rich import print
from rich.markdown import Markdown
from typing import Literal
import base64
import pathlib
from pydantic import BaseModel


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


async def agents_async_chat_once(agent: Agent, input_items: list[TResponseInputItem] | TResponseInputItem,
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
            elif event.type == "agent_updated_stream_event":
                print(f"Agent updated: {event.new_agent.name}")
                continue
            elif event.type == "run_item_stream_event":
                if event.item.type == "tool_call_item":
                    print("-- Tool was called")
                elif event.item.type == "tool_call_output_item":
                    print(f"-- Tool output: {event.item.output}")
                # elif event.item.type == "message_output_item": # 如果完成后一次性输出
                #     print(f"-- Message output:\n {ItemHelpers.text_message_output(event.item)}")
                else:
                    pass  # Ignore other event types
    return result


async def agents_chat_continuous(agent: Agent, runner_mode: Literal['async', 'stream'] = 'async',
                                 enable_fileloading: bool = False):
    """
    输入用户输入，输出agent的result类，可以通过result.new_items属性来查看全部的事件；
    result.new_items[0].raw_item，可以看具体的回复内容；to_input_list()方法，可以直接将用户的输入和本次输出结果拼接成一个消息列表
    :param agent:
    :param runner_mode:
    :param enable_fileloading: 某些模型需要文件上传，当不需要文件上传时，可以避免每次input()文件路径
    :return:
    """
    input_item = []
    result = None
    while True:
        contents = []
        msg_input = input("\n💬 请输入你的消息(输入quit退出):")
        if msg_input.lower() in ['exit', 'quit']:
            print("✅ 对话已结束")
            break
        if enable_fileloading:
            file_input = input("\n📁 请输入图片或者文档路径(输入quit退出):")
            file_input = file_input.strip("'\"")  # 文件路径去除首位引号，否则会pathlib.Path认为字符串
            file_path = pathlib.Path(file_input)
            if file_input not in ['cancel', 'no_file', 'quit']:
                if file_path.exists() and file_path.is_file():
                    if file_path.suffix.lower() in ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp',
                                                    '.heic']:
                        img_item = load_img(file_input)
                        contents.append(img_item)
                        input_item.append({"role": "user", "content": contents})
                else:
                    print(f"✅ 对话已结束,{file_path.suffix.lower()}图片格式不支持")
                    break
            else:
                print("✅ 对话已结束, 文档不是文件或者不存在")
        input_item.append({"role": "user", "content": msg_input})
        result = await agents_async_chat_once(agent=agent, input_items=input_item, runner_mode=runner_mode)
        input_item = result.to_input_list()
    return result


@function_tool
def folder_search(folder_path: str):
    """
    搜索指定文件夹下的所有文件，并输出文件列表
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


def load_img(image_path: str | pathlib.Path):
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
    img_path_obj = pathlib.Path(image_path)
    img_format = img_path_obj.suffix
    if img_format not in supported_img:
        print(f"不支持的图片格式：{img_format}")
        return None
    if not pathlib.Path.exists(img_path_obj):
        print(f"文件不存在：{image_path}")
        return None
    base64_img = base64_image(image_path)
    if img_format in jpg_variant:
        img_format = "jpeg"
    elif img_format in tif_variant:
        img_format = "tiff"
    input_item = {
        # "type": "image_url", # qwen的OpenAI格式,与openai-agent不同
        # "image_url": {"url": f"data:image/{img_format};base64,{base64_img}"} # qwen的OpenAI格式,与openai-agent不同
        "type": "input_image",
        "detail": "auto",
        "image_url": f"data:image/{img_format};base64,{base64_img}"}  # openAI-Aents格式
    return input_item


class mcp_stdio(BaseModel):
    command: str
    args: list[str]


class mcp_sse(BaseModel):
    url: str


class openAI_Agents_create:
    """
    创建openAI-Agents,可选工具,例如搜索,自定义function_tool, 流式输出;
    """

    def __init__(self, agent_name: str, instruction: str, model: str, base_url: str = None, api_key: str = None,
                 handoffs: list[Agent] = None, handoff_description: str = None, enable_thinking: bool = False,
                 enable_search: bool = True, force_search: bool = False, enable_source: bool = True,
                 enable_citation: bool = True, citation_format: bool = "[ref_<number>]", search_strategy="pro",
                 tool_choice: str = None, parallel_tool_calls: bool = False, tools: list = None,
                 custom_extra_body: dict = None,
                 ):
        """
        OpenAI-Agents初始化
        :param model: 譬如: 'model': 'qwen-turbo-latest',   # 输入0.0003元;思考模式0.006元;非思考模式0.0006元
                            'model': 'qwq-plus-latest',   # 输入0.0016元;   输出0.004元
                            'model': 'qwen-max-latest',   # 输入0.0024元;   输出0.0096元
                            'model': 'qwen-plus-latest',  # 输入0.0008元;思考模式0.016元;非思考模式0.002元
                            'model': 'qwen-vl-plus-latest',输入:0.0015;输出:0.0045
        :param base_url: base_url, 譬如:'http://localhost:8000/v1'
        :param api_key:  模型api-key
        :param handoffs: 分诊agent列表
        :param handoff_description: A description of the agent. This is used when the agent is used as a handoff, so that an
    LLM knows what it does and when to invoke it
        :param enable_thinking: 对于Qwen模型，仅在stream打开时，使用；
        :param enable_search: # 开启联网搜索的参数
        :param force_search: # 强制开启联网搜索
        :param enable_source: # 使返回结果包含搜索来源的信息，OpenAI 兼容方式暂不支持返回
        :param enable_citation: # 开启角标标注功能
        :param citation_format: # 角标形式为[ref_i]
        :param search_strategy: "pro"时,模型将搜索10条互联网信息
        :param instruction: 例如: "你是一个乐于助人的助理，按照用户需求，你先画图，再运行代码...."
        :param tools: 列表,包含自定义function_tool,或其它工具
        :param tool_choice: None, 'auto' 等
        :param parallel_tool_calls: bool
        :param custom_extra_body: dict 当custom_body != None时，将自定义extra_body,

        """
        if api_key is None:
            api_key = os.getenv("DASHSCOPE_API_KEY")
        if base_url is None:
            base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"

        if custom_extra_body is None:
            extra_body = {
                "enable_thinking": enable_thinking,  # only support stream call
                "enable_search": enable_search,
                'search_options': {
                    "forced_search": force_search,  # 强制开启联网搜索
                    "enable_source": enable_source,  # 使返回结果包含搜索来源的信息，OpenAI 兼容方式暂不支持返回
                    "enable_citation": enable_citation,  # 开启角标标注功能
                    "citation_format": citation_format,  # 角标形式为[ref_i]
                    "search_strategy": search_strategy  # "pro"时,模型将搜索10条互联网信息
                }
            }
        else:
            extra_body = custom_extra_body

        model_settings = ModelSettings(
            tool_choice=tool_choice,
            parallel_tool_calls=parallel_tool_calls,
            extra_body=extra_body, )

        self.agent_params = {
            'name': agent_name,
            'instructions': instruction,
            'model_settings': model_settings,
        }

        default_OpenAIModel = custom2default_openai_model(model=model,
                                                          base_url=base_url,
                                                          api_key=api_key,
                                                          )
        self.agent_params['model'] = default_OpenAIModel

        if tools is not None:
            self.agent_params['tools'] = tools
            # tools=[WebSearchTool(user_location={"type": "approximate", "city": "New York"})], # 目前只支持openAI的模型

        if handoffs is not None:
            self.agent_params['handoffs'] = handoffs
        if handoff_description is not None:
            self.agent_params['handoff_description'] = handoff_description

        self.agent = Agent(**self.agent_params)
        self.instruction = instruction

    async def mcp_server_initialize(self, mcp_names: list[str] = None,
                                    mcp_params: list[dict] = None,
                                    mcp_io_methods: list[
                                        Literal["MCPServerStdio", "MCPServerSse", "MCPServerStreamableHttp"]] = None,
                                    mcp_added_instructions: list[str] = None):
        """
        初始化mcp_server,并初始化包含mcp_servers的agent
        :param mcp_names: [mcp_name] ,当有mcp server时，配置mcp name; mcp_names/mcp_params列表中一一对应;
        :param mcp_params: [mcp_parm],当有mcp server时，配置mcp params: 支持， stdio, sse, streamableHttp;mcp_names/mcp_params列表中一一对应
        :param mcp_io_methods: 对应每一个mcp server, stdio, sse, streamableHttp 三种io传输方式选一
        :param mcp_added_instructions: [mcp_added_instruction], 列表
        :return:
        """
        # 处理mcp_server的参数
        if mcp_added_instructions is not None:
            self.agent_params['instructions'] = self.instruction.join(mcp_added_instructions)
        if mcp_names is not None and mcp_params is not None and mcp_io_methods is not None:
            self.agent_params['mcp_servers'] = []
            # 使用 AsyncExitStack 自动管理多个上下文退出
            stack = AsyncExitStack()
            for mcp_name, mcp_param, mcp_io_method in zip(mcp_names, mcp_params, mcp_io_methods):
                if mcp_io_method == "MCPServerStdio":
                    # 手动创建并启动server:
                    mcp_server = MCPServerStdio(name=mcp_name,
                                                cache_tools_list=True,
                                                params=mcp_param)
                elif mcp_io_method == "MCPServerSse":
                    # 手动创建并启动server:
                    mcp_server = MCPServerSse(name=mcp_name,
                                              cache_tools_list=True,
                                              params=mcp_param)
                elif mcp_io_method == "MCPServerStreamableHttp":
                    # 手动创建并启动server:
                    mcp_server = MCPServerStreamableHttp(name=mcp_name,
                                                         cache_tools_list=True,
                                                         params=mcp_param)
                else:
                    mcp_server = None
                # 启动server
                await mcp_server.connect()
                # 创建并进入所有 server 上下文
                stacked_mcp_server = await stack.enter_async_context(mcp_server)

                self.agent_params['mcp_servers'].append(stacked_mcp_server)
            self.agent = Agent(**self.agent_params)

    async def mcp_server_cleanup(self, ):
        mcp_servers = self.agent_params.get('mcp_servers', None)
        if mcp_servers is not None:
            for mcp_server in mcp_servers:
                try:
                    await mcp_server.cleanup()
                except Exception as e:
                    print(f"[WARNING] 清理 MCP Server 时发生异常: {e}")
                del self.agent_params['mcp_servers']

    async def async_chat_once(self, input_items: list[TResponseInputItem] | TResponseInputItem,
                              runner_mode: Literal['async', 'stream'] = 'async'):
        """
        输入[{"role": "user", "content": prompt}]格式prompt,输出agent的result类，可以通过result.new_items属性来查看全部的事件；
        result.new_items[0].raw_item，可以看具体的回复内容；to_input_list()方法，可以直接将用户的输入和本次输出结果拼接成一个消息列表
        :param input_items: list[dict],表示输入的prompt格式列表，例如: [{"role": "user", "content": prompt}]
        :param runner_mode:
        :return:
        """
        result = await agents_async_chat_once(agent=self.agent,
                                              input_items=input_items,
                                              runner_mode=runner_mode)
        return result

    async def chat_continuous(self, runner_mode: Literal['async', 'stream'] = 'async',
                              enable_fileloading: bool = False):
        result = await agents_chat_continuous(agent=self.agent, runner_mode=runner_mode,
                                              enable_fileloading=enable_fileloading)
        return result

    async def multi_mcp_chat_continuous(self, mcp_names: list[str] = None, mcp_params: list[dict] = None,
                                        mcp_io_methods: list[Literal[
                                            "MCPServerStdio", "MCPServerSse", "MCPServerStreamableHttp"]] = None,
                                        mcp_added_instructions: list[str] = None,
                                        runner_mode: Literal['async', 'stream'] = 'async',
                                        enable_fileloading: bool = False, ):
        # 处理mcp_server的参数
        if mcp_added_instructions is not None:
            self.agent_params['instructions'] = self.instruction.join(mcp_added_instructions)
        if mcp_names is not None and mcp_params is not None and mcp_io_methods is not None:
            # 使用 AsyncExitStack 自动管理多个上下文退出
            async with AsyncExitStack() as stack:
                self.agent_params['mcp_servers'] = []
                for mcp_name, mcp_param, mcp_io_method in zip(mcp_names, mcp_params, mcp_io_methods):
                    # 创建并进入所有 mcp_server 上下文
                    if mcp_io_method == "MCPServerStdio":
                        mcp_server = MCPServerStdio(name=mcp_name,
                                                    cache_tools_list=True,
                                                    params=mcp_param)
                    elif mcp_io_method == "MCPServerSse":
                        mcp_server = MCPServerSse(name=mcp_name,
                                                  cache_tools_list=True,
                                                  params=mcp_param)
                    elif mcp_io_method == "MCPServerStreamableHttp":
                        mcp_server = MCPServerStreamableHttp(name=mcp_name,
                                                             cache_tools_list=True,
                                                             params=mcp_param)
                    else:
                        mcp_server = None

                    stacked_mcp_server = await stack.enter_async_context(mcp_server)
                    self.agent_params['mcp_servers'].append(stacked_mcp_server)
                self.agent = Agent(**self.agent_params)
                result = await agents_chat_continuous(agent=self.agent, runner_mode=runner_mode,
                                                      enable_fileloading=enable_fileloading)
                return result


# 2
# 通义千问VL：qwen-vl-plus-latest，模型可以根据您传入的图片来进行回答 输入:0.0015;输出:0.0045
# 图像问答：描述图像中的内容或者对其进行分类打标，如识别人物、地点、花鸟鱼虫等。
# 数学题目解答：解答图像中的数学问题，适用于中小学、大学以及成人教育阶段。
# 视频理解：分析视频内容，如对具体事件进行定位并获取时间戳，或生成关键时间段的摘要。
# 物体定位：定位图像中的物体，返回外边界矩形框的左上角、右下角坐标或者中心点坐标。
# 文档解析：将图像类的文档（如扫描件/图片PDF）解析为 QwenVL HTML格式，该格式不仅能精准识别文本，还能获取图像、表格等元素的位置信息。
# 文字识别与信息抽取：识别图像中的文字、公式，或者抽取票据、证件、表单中的信息，支持格式化输出文本；可识别的语言有中文、英语、日语、韩语、阿拉伯语、越南语、法语、德语、意大利语、西班牙语和俄语。
# 3
# 通义千问OCR：qwen-vl-ocr-latest，（输入输出：0.005），是文字提取专有模型，专注于文档、表格、试题、手写体文字等类型图像的文字提取能力。它能够识别多种文字，目前支持的语言有：汉语、英语、阿拉伯语、法语、德语、意大利语、日语、韩语、葡萄牙语、俄语、西班牙语、越南语。
# 支持在文字提取前，对图像进行旋转矫正，适合图像倾斜的场景。#
# 新增六种内置的OCR任务，分别是通用文字识别、信息抽取、文档解析、表格解析、公式识别、多语言识别。#
# 未设置内置任务时，支持用户输入Prompt进行指引；如设置了内置任务时，为保证识别效果，模型内部会使用任务指定的Prompt。
# 仅DashScope SDK支持对图像进行旋转矫正和设置内置任务。如需使用OpenAI SDK进行内置的OCR任务，需要手动填写任务指定的Prompt进行引导。

QwenVL_model = 'qwen-vl-plus-latest'
base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
QwenVL_agent_instruction = '''
    您是一个助人为乐的助手，可以根据传入的图片来进行:
    1)图像问答：描述图像中的内容或者对其进行分类打标，如识别人物、地点、花鸟鱼虫等。
    2)数学题目解答：解答图像中的数学问题，适用于中小学、大学以及成人教育阶段。
    3)视频理解：分析视频内容，如对具体事件进行定位并获取时间戳，或生成关键时间段的摘要。
    4)物体定位：定位图像中的物体，返回外边界矩形框的左上角、右下角坐标或者中心点坐标。
    5)文档解析：将图像类的文档（如扫描件/图片PDF）解析为 QwenVL HTML格式，该格式不仅能精准识别文本，还能获取图像、表格等元素的位置信息。
    6)文字识别与信息抽取：识别图像中的文字、公式，或者抽取票据、证件、表单中的信息，支持格式化输出文本；可识别的语言有中文、英语、日语、韩语、阿拉伯语、越南语、法语、德语、意大利语、西班牙语和俄语。
    你只对带有图片的prompt，做出响应。
    '''


# Qwen-MT模型是基于通义千问模型优化的机器翻译大语言模型，擅长中英互译、中文与小语种互译、英文与小语种互译
# qwen-mt-plus  0.015元/0.045元;
# qwen-mt-turbo 0.001元/0.003元
# 不支持指定 System Message，也不支持多轮对话；messages 数组中有且仅有一个 User Message，用于指定需要翻译的语句。
# 如果您希望翻译的风格更符合某个领域的特性，如法律、政务领域翻译用语应当严肃正式，社交领域用语应当口语化，可以用一段自然语言文本描述您的领域，将其提供给大模型作为提示。# 领域提示语句暂时只支持英文。

class Term(BaseModel):
    source: str
    target: str


def Qwen_MT_func(prompt: str, model: str = 'qwen-mt-turbo', api_key: str = None, source_lang: str = 'auto',
                 target_lang: str = 'English', terms: list[Term] = None, tm_list: list[Term] = None,
                 domains: str = None):
    """
    Qwen-MT模型是基于通义千问模型优化的机器翻译大语言模型，擅长中英互译、中文与小语种互译、英文与小语种互译;在多语言互译的基础上，提供术语干预、领域提示、记忆库等能力，提升模型在复杂应用场景下的翻译效果。
    :param prompt: str, 输入的prompt
    :param model: str, 您对翻译质量有较高要求，建议选择qwen-mt-plus模型；如果您希望翻译速度更快或成本更低，建议选择qwen-mt-turbo模型
    :param api_key: str, 阿里云百炼API Key
    :param source_lang: str, 源语言
    :param target_lang: str, 目标语言
    :param terms: list[dict], 技术术语可以提前翻译，并将其提供给Qwen-MT模型作为参考；每个术语是一个JSON对象，包含术语和翻译过的术语信息，格式如下：{"source": "术语", "target": "提前翻译好的术语"}
    :param tm_list: list[dict], 如果您已经有标准的双语句对并且希望大模型在后续翻译时能参考这些标准译文给出结果，可以使用翻译记忆功能；每个JSON对象包含源语句与对应的已翻译的语句，格式如下：{"source": "源语句","target": "已翻译的语句"}
    :param domains: str, 如果您希望翻译的风格更符合某个领域的特性，可以用一段自然语言文本描述您的领域(暂时只支持英文)
    :return: str, 翻译结果
    """
    if api_key is None:
        api_key = os.getenv("DASHSCOPE_API_KEY")

    client = OpenAI(
        # 若没有配置环境变量，请用阿里云百炼API Key将下行替换为：api_key="sk-xxx",
        api_key=api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    messages = [{"role": "user", "content": prompt}]

    translation_options = {
        "source_lang": source_lang,
        "target_lang": target_lang
    }
    if terms is not None:
        translation_options['terms'] = terms
    if tm_list is not None:
        translation_options['tm_list'] = tm_list
    if domains is not None:
        translation_options['domains'] = domains

    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        extra_body={
            "translation_options": translation_options
        }
    )
    # print(completion.choices[0].gradio_message.content)
    return completion.choices[0].message.content


@function_tool
def _Qwen_MT_func(prompt: str, model: str = 'qwen-mt-turbo', api_key: str = None, source_lang: str = 'auto',
                  target_lang: str = 'English', terms: list[Term] = None, tm_list: list[Term] = None,
                  domains: str = None):
    """
    Qwen-MT模型是基于通义千问模型优化的机器翻译大语言模型，擅长中英互译、中文与小语种互译、英文与小语种互译;在多语言互译的基础上，提供术语干预、领域提示、记忆库等能力，提升模型在复杂应用场景下的翻译效果。
    :param prompt: str, 输入的prompt
    :param model: str, 您对翻译质量有较高要求，建议选择qwen-mt-plus模型；如果您希望翻译速度更快或成本更低，建议选择qwen-mt-turbo模型
    :param api_key: str, 阿里云百炼API Key
    :param source_lang: str, 源语言
    :param target_lang: str, 目标语言
    :param terms: list[dict], 技术术语可以提前翻译，并将其提供给Qwen-MT模型作为参考；每个术语是一个JSON对象，包含术语和翻译过的术语信息，格式如下：{"source": "术语", "target": "提前翻译好的术语"}
    :param tm_list: list[dict], 如果您已经有标准的双语句对并且希望大模型在后续翻译时能参考这些标准译文给出结果，可以使用翻译记忆功能；每个JSON对象包含源语句与对应的已翻译的语句，格式如下：{"source": "源语句","target": "已翻译的语句"}
    :param domains: str, 如果您希望翻译的风格更符合某个领域的特性，可以用一段自然语言文本描述您的领域(暂时只支持英文)
    :return: str, 翻译结果
    """
    result = Qwen_MT_func(prompt, model, api_key, source_lang, target_lang, terms, tm_list, domains)
    return result


@function_tool
def save2file(file_path: pathlib.Path, content):
    """
    用于将LLM的输出，依照一定的格式写入本地文件file_name
    :param file_path:
    :param content:
    :return:
    """
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)


async def main():  # 便于异步上下文管理，建议多语句放入异步函数中，一起执行

    mcp_names = ['file_system']
    mcp_params = [{
        "command": "npx",
        "args": [
            "-y",
            "@modelcontextprotocol/server-filesystem",
            ".",
        ]}]
    mcp_io_methods = ["MCPServerStdio"]

    QwenVL_agent = openAI_Agents_create(agent_name='通义千问视觉理解智能体',
                                        instruction=QwenVL_agent_instruction,
                                        model=QwenVL_model,
                                        base_url=None,
                                        api_key=None,
                                        tools=[save2file],
                                        handoff_description="当prompt有图片时,使用QwenVL模型进行视觉推理,并且必要时，按要求将约定的内容存入本地文件"
                                        )

    Qwen_model = 'qwen-turbo-latest'
    Qwen_model_instruction = """
        你是一名助人为乐的助手,
        1)当prompt中有文件时，请handoff至视觉推理模型;
        2)否则，就直接回答问题;
        3) 必要时，可以将约定的内容存入本地文件。
        """
    handoff_description = """
        本模型仅仅处理不带有文件的prompt;当prompt图片文件时，请handoff至视觉推理模型，并给出结果。
        """
    Qwen3_agent = openAI_Agents_create(agent_name='通义千问智能体(general)',
                                       instruction=Qwen_model_instruction,
                                       model=Qwen_model,
                                       base_url=None,
                                       api_key=None,
                                       tools=[save2file],
                                       handoffs=[QwenVL_agent.agent],
                                       handoff_description=handoff_description

                                       )
    # 运行主协程
    await Qwen3_agent.chat_continuous(runner_mode='stream', enable_fileloading=True)
    # await QwenVL_agent.multi_mcp_chat_continuous(runner_mode='async', enable_fileloading=False,
    #                                                    mcp_names=mcp_names,
    #                                                    mcp_params=mcp_params,
    #                                                    mcp_io_methods=mcp_io_methods)


if __name__ == '__main__':
    # Windows 上推荐使用 ProactorEventLoop，但需确保事件循环正确关闭
    policy = asyncio.WindowsProactorEventLoopPolicy()
    asyncio.set_event_loop_policy(policy)

    asyncio.run(main())
