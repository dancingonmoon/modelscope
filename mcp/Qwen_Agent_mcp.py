import os
import pathlib
from qwen_agent.agents import Assistant
from qwen_agent.gui import WebUI
from qwen_agent.utils.output_beautify import typewriter_print
# `typewriter_print` prints streaming messages in a non-overlapping manner.

llm_cfg = {
    'model': 'qwen-turbo-latest',   # 输入0.0003元;思考模式0.006元;非思考模式0.0006元
    # 'model': 'qwq-plus-latest',   # 输入0.0016元;   输出0.004元
    # 'model': 'qwen-max-latest',   # 输入0.0024元;   输出0.0096元
    # 'model': 'qwen-plus-latest',  # 输入0.0008元;思考模式0.016元;非思考模式0.002元
    'model_server': 'dashscope',
    # 'api_key': ''  # **fill your api key here**
    'generate_cfg': {
            # When using the Dash Scope API, pass the parameter of whether to enable thinking mode in this way
            'enable_thinking': False,
            'enable_search': True, # 开启联网搜索的参数
            'search_options': {
                "forced_search": True, # 强制开启联网搜索
                "enable_source": True, # 使返回结果包含搜索来源的信息，OpenAI 兼容方式暂不支持返回
                "enable_citation": True, # 开启角标标注功能
                "citation_format": "[ref_<number>]", # 角标形式为[ref_i]
                "search_strategy": "pro" # 模型将搜索10条互联网信息
            },

        },

    # Use a model service compatible with the OpenAI API, such as vLLM or Ollama:
    # 'model': 'Qwen3-8B',
    # 'model_server': 'http://localhost:8000/v1',  # base_url, also known as api_base
    # 'api_key': 'EMPTY'
}

tools = [{
    "mcpServers": {
        "filesystem": {
            "command": "npx",
            "args": [
                "-y",
                "@modelcontextprotocol/server-filesystem",
                '.',
            ]
        },
        "mindmap": {
            "command": "uvx",
            "args": ["mindmap-mcp-server", "--return-type", "filePath"]
        },
        "time": {
            "command": "uvx",
            "args": ["mcp-server-time","--local-timezone=Asia/Shanghai"]
                }
                }
        },
    # 'code_interpreter',  # Built-in tools
]

class Qwen_Agent_mcp:
    """
    初始化Qwen-Agent,可选工具,例如搜索,code_interpreter, mcp, 流式输出;
    :return:
    """
    def __init__(self, model:str, model_server:str='dashscope', api_key:str=None,enable_thinking:bool=False,
                 enable_search:bool=True, force_search:bool=True, enable_source:bool=True,
                 enable_citation:bool=True, citation_format:bool="[ref_<number>]", search_strategy="pro",
                 code_interpreter:bool=False, mcp:dict|bool=None,
                 system_message:str='', description:str='', files:list[str]=None):
        """

        :param model:
        :param model_server: 'dashscope',或者url_base, 譬如:'http://localhost:8000/v1'
        :param api_key: 如环境变量中设定,则此处为None
        :param enable_thinking:
        :param enable_search: # 开启联网搜索的参数
        :param force_search: # 强制开启联网搜索
        :param enable_source: # 使返回结果包含搜索来源的信息，OpenAI 兼容方式暂不支持返回
        :param enable_citation: # 开启角标标注功能
        :param citation_format: # 角标形式为[ref_i]
        :param search_strategy: "pro"时,模型将搜索10条互联网信息
        :param code_interpreter:
        :param mcp:
        :param system_message: "按照用户需求，你先画图，再运行代码...."
        :param description: '使用RAG检索并回答，支持文件类型：PDF/Word/PPT/TXT/HTML。'
        :param files: list,譬如: [os.path.join('.', 'doc.pdf')]
        :parm
        """
        tools  = []
        if mcp is None and mcp is not False:
            mcp = {
                "mcpServers": {
                    "filesystem": {
                        "command": "npx",
                        "args": [
                            "-y",
                            "@modelcontextprotocol/server-filesystem",
                            '.',
                        ]
                    },
                    "time": {
                        "command": "uvx",
                        "args": ["mcp-server-time", "--local-timezone=Asia/Shanghai"]
                    }
                }
            }
            tools.append(mcp)

        if code_interpreter:
            tools.append('code_interpreter')

        if model_server == 'dashscope':
            llm_config = {
            'model': model,
            'model_server': model_server,
            # 'api_key':  #  如果环境变量中已经设定,则该项可以不填''  # **fill your api key here**
            'generate_cfg': {
                # When using the Dash Scope API, pass the parameter of whether to enable thinking mode in this way
                'enable_thinking': enable_thinking,
                'enable_search': enable_search, # 开启联网搜索的参数
                'search_options': {
                    "forced_search": force_search, # 强制开启联网搜索
                    "enable_source": enable_source, # 使返回结果包含搜索来源的信息，OpenAI 兼容方式暂不支持返回
                    "enable_citation": enable_citation, # 开启角标标注功能
                    "citation_format": citation_format, # 角标形式为[ref_i]
                    "search_strategy": search_strategy # "pro"时,模型将搜索10条互联网信息
                },

            },

        }
        else:
            llm_config = {
                # Use a model service compatible with the OpenAI API, such as vLLM or Ollama:
                'model': model,
                'model_server': model_server,  # base_url, also known as api_base
                'api_key': api_key
            }
        self.agent = Assistant(
            llm=llm_config,
            function_list=tools,
            name='my Assistant',
            system_message= system_message,
            description= description,
            files = files
            )

    def chat_once(self, query: str, file_path):
        """

        :param query:
        :param file_path: str|os.path对象,文件路径
        :return:
        """
        # messages = [{'role': 'user', 'content': [{'text': '介绍图一'},
        #             {'file': 'https://arxiv.org/pdf/1706.03762.pdf'}]}]

        messages = []  # 这里储存聊天历史。
        # 例如，输入请求 "绘制一只狗并将其旋转 90 度"。
        # 将用户请求添加到聊天历史。
        if file_path is None:
            messages.append({'role': 'user', 'content': query})
        else:
            file_content = pathlib.Path(file_path).read_bytes()
            messages.append({'role': 'user', 'content': [{'text': query},
                          {'file': file_content}]})
        response_plain_text = ''
        print('机器人回应:')
        for response in agent.run(messages=messages):
            # 流式输出。
            response_plain_text = typewriter_print(response, response_plain_text)





if __name__ == '__main__':
    agent = Assistant(
        llm=llm_cfg,
        function_list=tools,
        name='my Assistant',
        # system_message="按照用户需求，你先画图，再运行代码...."
        # description='使用RAG检索并回答，支持文件类型：PDF/Word/PPT/TXT/HTML。'
        # files = [os.path.join('.', 'doc.pdf')]
    )
    WebUI(agent).run()

    # messages = []  # 这里储存聊天历史。
    # while True:
    #     # 例如，输入请求 "绘制一只狗并将其旋转 90 度"。
    #     query = input('\n用户请求: ')
    #     if query.lower() == "exit":
    #         break
    #     # 将用户请求添加到聊天历史。
    #     messages.append({'role': 'user', 'content': query})
    #     response = []
    #     response_plain_text = ''
    #     print('机器人回应:')
    #     for response in agent.run(messages=messages):
    #         # 流式输出。
    #         response_plain_text = typewriter_print(response, response_plain_text)
    #     # 将机器人的回应添加到聊天历史。
    #     messages.extend(response)

