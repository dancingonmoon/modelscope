import os
from typing import Literal, Union

import typing_extensions
from typing_extensions import TypedDict
from typing import Annotated
from pydantic import BaseModel

from tempfile import TemporaryDirectory

from langchain_community import chat_models
from langchain_community.document_loaders import WebBaseLoader, Docx2txtLoader
from langchain_community.agent_toolkits import FileManagementToolkit
from langchain.chat_models import init_chat_model
from langchain_qwq import ChatQwen, ChatQwQ
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.messages import AnyMessage
from langchain_tavily import TavilySearch

from langgraph.graph.message import add_messages
from langgraph.graph import MessageGraph, MessagesState, StateGraph, START, END
from langgraph.types import Command

from langgraph.prebuilt import create_react_agent
from langgraph.prebuilt.chat_agent_executor import AgentState
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver

import asyncio


async def web_txtLoader(url: str | list[str] = '',
                        verify_ssl: bool = True):
    #  https://python.langchain.com/docs/how_to/document_loader_web/
    headers = {'User-Agent': 'Mozilla/5.0'}  # 设置请求头,防止网站反爬虫机制

    loader = WebBaseLoader(web_path=url, verify_ssl=verify_ssl)
    docs = []
    async for doc in loader.alazy_load():
        docs.append(doc)
    return docs


async def docx_txtLoader(file_path: str | list[str] = None, ):
    #  https://python.langchain.com/docs/integrations/document_loaders/microsoft_word/
    loader = Docx2txtLoader(file_path=file_path, )
    docs = []
    async for doc in loader.alazy_load():
        docs.append(doc)
    return docs


class langchain_qwen_llm:
    def __init__(self,
                 model: str = 'qwen-turbo-latest',
                 base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",  # 阿里云国内站点(默认为国际站点),
                 streaming: bool = True,
                 enable_thinking: bool = False,
                 thinking_budget: int = 100,
                 extra_body: dict = None,
                 tools: list = None,
                 structure_output: dict[str, typing_extensions.Any] | BaseModel | type | None = None,
                 system_instruction: str | list[AnyMessage] = "You are a helpful assistant."
                 ):
        """
        langchain-qwq库中的ChatQwQ与ChatQwen针对Qwen3进行了优化；然而，其缺省的base_url却是阿里云的国际站点；国内使用需要更改base_url为国内站点
        :param model: str
        :param base_url: str
        :param streaming: bool
        :param enable_thinking: bool; Qwen3 model only
        :param thinking_budget: int
        :param extra_body: dict; 缺省{"enable_search": True}
        :param tools: list
        :param structure_output: TypedDict
        :param system_instruction: str
        """
        if extra_body is None:
            extra_body = {
                "enable_search": True
            }
        self.model = ChatQwen(
            model=model,
            base_url=base_url,
            streaming=streaming,
            enable_thinking=enable_thinking,
            thinking_budget=thinking_budget,
            extra_body=extra_body
        )
        if tools is not None:
            self.model.bind_tools(tools)

        if structure_output is not None:
            self.model.with_structured_output(structure_output)

    async def astreamPrint(self, prompt,
                           thread_id: str = None):
        """
        异步流式打印Agent response
        response json: {'agent':{'messages':[AIMessage(content='',
                                             additional_kwargs={'reasoning_content': '正在思考中...'},
                                             response_metadata={'finish_reason','model_name'},
                                             id='',
                                             usage_metadata={},
                                             output_token_details={})]}}
        :param prompt:
        :param thread_id: Short-term memory (thread-level persistence) enables agents to track multi-turn conversations
        :return:
        """
        config = {"configurable": {"thread_id": thread_id}}
        response = self.model.astream(input=prompt, config=config, )

        is_first = True
        is_end = False
        async for msg in response:
            # print(f'response,msg: {msg}')
            if hasattr(msg, 'additional_kwargs') and "reasoning_content" in msg.additional_kwargs:
                if is_first:
                    print("Starting to think...")
                    is_first = False
                    is_end = True
                print(msg.additional_kwargs["reasoning_content"], end="", flush=True)
            if hasattr(msg, 'content') and msg.content:
                if is_end:
                    print("\nThinking ended")
                    is_end = False
                print(msg.content, end="", flush=True)

    async def multi_turn_conversation(self, thread_id: str = None):
        while True:
            try:
                user_input = input("\nUser: ")
                if user_input.lower() in ["quit", "exit", "q"]:
                    print("Goodbye!")
                    break

                await self.astreamPrint(user_input, thread_id=thread_id)
            except Exception as e:
                print(f"发生错误: {str(e)}")
                break
            except KeyboardInterrupt:
                print("\n用户中断，退出程序")
                break


class langgraph_agent:
    def __init__(self,
                 model: Union[str, chat_models] = 'qwen-turbo-latest',
                 tools: list = None,
                 structure_output: dict[str, typing_extensions.Any] | BaseModel | type | None = None,
                 system_instruction: str | list[AnyMessage] = "You are a helpful assistant."
                 ):
        """
        :param model: str
        :param tools: list
        :param structure_output: TypedDict
        :param system_instruction: str
        """
        checkpointer = InMemorySaver()
        params = dict(
            model=model,
            response_format=structure_output,
            prompt=system_instruction,
            checkpointer=checkpointer)
        if tools is not None:
            params['tools'] = tools
        else:
            params['tools'] = []
        if structure_output is not None:
            params['response_format'] = structure_output

        self.agent = create_react_agent(**params)

    async def astreamPrint(self, prompt,
                           stream_mode: Literal['values', 'updates', 'custom', 'messages', 'debug'] = 'updates',
                           thread_id: str = None):
        """
        异步流式打印Agent response
        response json: {'agent':{'messages':[AIMessage(content='',
                                             additional_kwargs={'reasoning_content': '正在思考中...'},
                                             response_metadata={'finish_reason','model_name'},
                                             id='',
                                             usage_metadata={},
                                             output_token_details={})]}}
        :param prompt:
        :param stream_mode: str
        :param thread_id: Short-term memory (thread-level persistence) enables agents to track multi-turn conversations
        :return:
        """
        message = {"messages": prompt}
        config = {"configurable": {"thread_id": thread_id}}
        response = self.agent.astream(input=message, config=config, stream_mode=stream_mode)
        # response json: {'agent':{'messages':[AIMessage(content='',
        #                                      additional_kwargs={'reasoning_content': '正在思考中...'},
        #                                      response_metadata={'finish_reason','model_name'},
        #                                      id='',
        #                                      usage_metadata={},
        #                                      output_token_details={})
        is_first = True
        is_end = False
        async for msg in response:
            # print(f'response: {msg}')
            agent_msgs = {}
            tool_msgs = {}
            # 处理agent消息
            if 'agent' in msg:
                if 'messages' in msg['agent']:
                    agent_msgs = msg['agent']["messages"]  # 消息列表
            for agent_msg in agent_msgs:
                if hasattr(agent_msg, 'additional_kwargs') and "reasoning_content" in agent_msg.additional_kwargs:
                    if is_first:
                        print("Starting to think...")
                        is_first = False
                        is_end = True
                    print(agent_msg.additional_kwargs["reasoning_content"], end="", flush=True)
                if hasattr(agent_msg, 'content') and agent_msg.content:
                    if is_end:
                        print("\nThinking ended")
                        is_end = False
                    print(agent_msg.content, end="", flush=True)
            # 处理tools消息
            if 'tools' in msg:
                if 'messages' in msg['tools']:
                    tool_msgs = msg['tools']["messages"]  # 消息列表
            for tool_msg in tool_msgs:
                if hasattr(tool_msg, 'name'):
                    print(f"tool: {tool_msg.name}, invoked ")
                if hasattr(tool_msg, 'content') and tool_msg.content:
                    print(tool_msg.content, end="", flush=True)

    async def multi_turn_conversation(self, stream_mode: Literal[
        'values', 'updates', 'custom', 'messages', 'debug'] = 'updates',
                                      thread_id: str | None = None):
        """
        多轮对话(似乎不设置thread_id时，也是具备会话的记忆)
        :param stream_mode: Literal['values', 'updates', 'custom', 'messages', 'debug']
        :param thread_id: Short-term memory (thread-level persistence) enables agents to track multi-turn conversations
        :return:
        """
        while True:
            try:
                user_input = input("\nUser: ")
                if user_input.lower() in ["quit", "exit", "q"]:
                    print("Goodbye!")
                    break

                await self.astreamPrint(user_input, stream_mode=stream_mode,
                                        thread_id=thread_id)
            except Exception as e:
                print(f"发生错误: {str(e)}")
                break
            except KeyboardInterrupt:
                print("\n用户中断，退出程序")
                break


class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


class QwenML_translationoptions(TypedDict):
    text: str  # 待翻译的文本
    source_lang: str  # "Chinese"
    target_lang: str  # "English"
    domains: str  # 翻译的风格具备某领域的特性，自然语言(英文)描述
    end: bool  # 表明输入是否是关于语言翻译的请求，如True，则结束对话


# graph_builder = StateGraph(State)

# Initialize Tavily Search Tool
# https://python.langchain.com/docs/integrations/tools/tavily_search/
tavily_search_tool = TavilySearch(
    max_results=5,
    topic="general",
    # include_answer=False,
    # include_raw_content=False,
    # include_images=False,
    # include_image_descriptions=False,
    # search_depth="basic",
    # time_range="day",
    # include_domains=None,
    # exclude_domains=None
)
# result = tavily_search_tool.invoke(input="今日国际新闻3条")
# result.keys:dict_keys(['query', 'follow_up_questions', 'answer', 'images', 'results', 'response_time'])
# result.results[0].keys:dict_keys(['url', 'title', 'content', 'score', 'raw_content'])

# We'll make a temporary directory to avoid clutter
working_directory = TemporaryDirectory(dir='.')
LocalFileSystem = FileManagementToolkit(
    root_dir=str(working_directory.name),  # pass the temporary directory in as a root directory as a workspace
    # # [CopyFileTool, DeleteFileTool, FileSearchTool, MoveFileTool, ReadFileTool, WriteFileTool, ListDirectoryTool]
    selected_tools=["read_file", "write_file", "list_directory"],
).get_tools()
print(f"LocalFileSystem目录: {LocalFileSystem}")

def Qwen_ML_node(state: QwenML_translationoptions)->Command[Literal['END','evaluator']]:
    end = state.end
    if end:
        trans_agent_state:State = {"messages": "翻译结束"}
        return
    else:
        text, source_lang, target_lang, domains= state.text, state.source_lang, state.target_lang, state.domains,
    return

if __name__ == '__main__':
    prompt = '请总结今日国际新闻3条'
    Qwen_plus = langchain_qwen_llm(model="qwen-plus-latest", enable_thinking=True, )
    Qwen_turbo_noThink = langchain_qwen_llm(model="qwen-turbo", )
    Qwen_VL = langchain_qwen_llm(model="qwen-vl-ocr-latest", )
    Qwen_MT = langchain_qwen_llm(model='qwen-mt-plus')
    QwenML_transOption_agent = langgraph_agent(model=Qwen_turbo_noThink.model,
                                               structure_output=QwenML_translationoptions,
                                               system_instruction="""
                                               你对接收的Input进行分析，并做如下分析和输出：
                                               1) 如果Input不是对一段文本进行语言翻译的请求，请结构化输出: end=True;否则，请结构化输出: end=False ;
                                               2) 从Input中分析待翻译的文本，并使用一段自然语言(必须为英文)总结下待翻译文本的领域，语气，从而使得翻译的风格更符合某个领域的特性；并将该总结输出至结构化输出domains;
                                               3) 从Input中分析出翻译需求的源语言，目标语言，整理出待翻译的文本，并分别结构化输出到 source_lang, target_lang, text;
                                               """)
    QwenML_trans_agent = langgraph_agent(model=Qwen_MT.model,
                                         system_instruction="""
                                         你是个专业的翻译，善于在多语言之间翻译各个领域的文档
                                         """)
    Qwen_VL_agent = langgraph_agent(model=Qwen_VL.model,
                                    structure_output=QwenML_translationoptions,
                                    system_instruction="""
                                    你善于识图理解，请识别输入的图片，获取其全部内容，然后做如下分析和输出：
                                    1) 如果Input不是对一段文本进行语言翻译的请求，请结构化输出: end=True;否则，请结构化输出: end=False ;
                                    2) 从Input中分析待翻译的文本，并使用一段自然语言(必须为英文)总结下待翻译文本的领域，语气，从而使得翻译的风格更符合某个领域的特性；并将该总结输出至结构化输出domains;
                                    3) 从Input中分析出翻译需求的源语言，目标语言，整理出待翻译的文本，并分别结构化输出到 source_lang, target_lang, text;
                                    """)
    # graph_draw_path = r"E:/Python_WorkSpace/modelscope/LangGraph/graph_draw.png"
    # # langgraph_agent.agent.get_graph().draw_mermaid_png(output_file_path=graph_draw_path)

    # asyncio.run(Qwen_VL_agent.astreamPrint(prompt))
    asyncio.run(QwenML_transOption_agent.multi_turn_conversation())
    ## 测试 webBaseLoader:
    # url= r"https://www.eastcom.com"
    # docs = asyncio.run(web_txtLoader(url))
    # print(docs[0])
    ## 测试 docx2txtLoader:
    # file_path = r"E:/Working Documents/Eastcom/产品/无集/2019年中国专网通信产业全景图谱.docx"
    # docs = asyncio.run(docx_txtLoader(file_path))
    # print(docs[0])
