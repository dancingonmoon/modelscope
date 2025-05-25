import pathlib
import ast
from qwen_agent.agents import Assistant
from qwen_agent.gui import WebUI
from qwen_agent.utils.output_beautify import typewriter_print
# `typewriter_print` prints streaming messages in a non-overlapping manner.

import qwen_agent
class Qwen_Agent_mcp:
    """
    初始化Qwen-Agent,可选工具,例如搜索,code_interpreter, mcp, 流式输出;
    :return:
    """
    def __init__(self, model:str, model_server:str='dashscope', api_key:str=None,enable_thinking:bool=False,
                 enable_search:bool=True, force_search:bool=False, enable_source:bool=True,
                 enable_citation:bool=True, citation_format:bool="[ref_<number>]", search_strategy="pro",
                 code_interpreter:bool=False, mcp:dict|bool=None,
                 system_message:str='', description:str='', files:list[str]=None):
        """

        :param model: 譬如: 'model': 'qwen-turbo-latest',   # 输入0.0003元;思考模式0.006元;非思考模式0.0006元
                            'model': 'qwq-plus-latest',   # 输入0.0016元;   输出0.004元
                            'model': 'qwen-max-latest',   # 输入0.0024元;   输出0.0096元
                            'model': 'qwen-plus-latest',  # 输入0.0008元;思考模式0.016元;非思考模式0.002元
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
        :param mcp: {}|Bool: None时，为缺省加载mcp:文件系统，time; False时，不加载mcp
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
                            './mcp_docs/'
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





    def chat_once(self, query: str, query_file_path:str=None, messages_history:list=None):
        """
        单次对话。此处输入query_file_path，接受url或者文件路径；采用的是openAI兼容的File格式, 例如：{messages = [{'role': 'user', 'content': [{'text': '介绍图一'},
        {'file': 'https://arxiv.org/pdf/1706.03762.pdf'}]}]}； 事实上，由于Qwen模型API本身并不支持File格式，这是由Qwen-Agent SDK做了转换后实现。
        :param query:
        :param query_file_path: str;文件路径(支持docx,pdf,不支持jpg)，或者url
        :param messages_history: list,
        :return:
        """
        # messages = [{'role': 'user', 'content': [{'text': '介绍图一'},
        #             {'file': 'https://arxiv.org/pdf/1706.03762.pdf'}]}]

        if messages_history is None:
            messages_history = []
        # 例如，输入请求 "绘制一只狗并将其旋转 90 度"。
        # 将用户请求添加到聊天历史。
        if query_file_path is None:
            messages_history.append({'role': 'user', 'content': query})
        else:
            # 文本文档的后缀列表,openAI格式也支持docx,doc,pdf：
            text_extensions = ['.txt', '.md', '.csv', '.json', '.py', '.html', '.htm', '.xml', '.yaml', '.yml','.docx','.doc','.pdf']
            text_file = pathlib.Path(query_file_path)
            if text_file.exists() and text_file.suffix.lower() in text_extensions: # 文本文档
                messages_history.append({'role': 'user', 'content': [{'text': query},
                                                                         {'file': query_file_path}]})
            elif query_file_path.startswith('http:') or query_file_path.startswith('https:'): # url
                messages_history.append({'role': 'user', 'content': [{'text': query},
                                                                     {'file': query_file_path}]})
            else:
                print("不合法的文件路径，或者不支持的非图像文档后缀!")
                return messages_history

        response = []
        response_plain_text = ''
        print('Agent 回应:')
        for response in self.agent.run(messages=messages_history):
            # 流式输出。
            response_plain_text = typewriter_print(response, response_plain_text)
        messages_history.append(response)
        return messages_history

    def chat_continuous(self, ):
        """
        连续对话。input()输入的问题，可以是一个字符串;或者列表，列表第一个元素为问题，第二个元素为文件路径或者url;或者字典, 例如： {'text': '介绍图一'},
                    {'file': 'https://arxiv.org/pdf/1706.03762.pdf'}
        :return:
        """
        message_history = []
        while True:
            query = ''
            file_path = None
            message = input('\n💬 请输入你的消息(输入exit或quit退出):')
            if message.lower() in ['exit','quit']:
                print("✅ 对话已结束")
                break
            message = ast.literal_eval(message) #  安全解析字符串为 Python 字面量
            if isinstance(message, str):
                query = message
            if isinstance(message,list):
                query = message[0]
                file_path = message[1]
            if isinstance(message,dict):
                query  = message['text']
                file_path = message['file']

            message_dict = self.chat_once(query, query_file_path=file_path)
            message_history.append(message_dict)

        return message_history



    def webUI(self,user_name:str=None,user_avatar:str=None,
              input_placeholder:str=None,prompt_suggestions:list=None,
              share:bool=False,server_port:int=None,enable_emotion:bool=True):
        """
        gradio UI
        :param user_name:
        :param user_avatar: 图像路径
        :param input_placeholder:
        :param prompt_suggestions:
        :return:
        """
        if prompt_suggestions is None:
            prompt_suggestions = []
        if input_placeholder is None:
            input_placeholder = ''
        chatbot_config = {
            'input.placeholder': input_placeholder,
            'prompt.suggestions': prompt_suggestions
        }
        if user_name is not None:
            chatbot_config['user.name'] = user_name
        if user_avatar is not None:
            chatbot_config['user.avatar'] = user_avatar
        WebUI(self.agent,chatbot_config=chatbot_config).run(
            share = share,
            server_port = server_port,
            enable_mention = enable_emotion)



if __name__ == '__main__':

    qwen_agent  = Qwen_Agent_mcp(model='qwen-turbo-latest', mcp=False)
    # message_history = qwen_agent.chat_once("请总结今天的新闻10条")
    message_history = qwen_agent.chat_continuous()
    # print(message_history)
    # qwen_agent.webUI(user_name=None,)






