import os
import uuid
import operator
import json
from typing import Literal, Union

import typing_extensions
from typing_extensions import TypedDict
from typing import Annotated
from pydantic import BaseModel

from langchain_community import chat_models
from langchain_community.document_loaders import WebBaseLoader, Docx2txtLoader

from langchain_core.runnables import RunnableConfig
from langchain_qwq import ChatQwen, ChatQwQ
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, AnyMessage


from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command

from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.errors import GraphRecursionError

from langchain_mcp_adapters.client import MultiServerMCPClient
from dataclasses import dataclass

import asyncio








class langchain_qwen_llm:
    def __init__(self,
                 model: str = 'qwen-turbo',
                 base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",  # 阿里云国内站点(默认为国际站点),
                 api_key: str = None,
                 streaming: bool = False,
                 enable_thinking: bool = False,
                 thinking_budget: int = 100,
                 extra_body: dict = None,
                 tools: list = None,
                 structure_output: dict[str, typing_extensions.Any] | BaseModel | type | None = None,
                 ):
        """
        langchain-qwq库中的ChatQwQ与ChatQwen针对Qwen3进行了优化；然而，其缺省的base_url却是阿里云的国际站点；国内使用需要更改base_url为国内站点
        :param model: str
        :param base_url: str
        :param api_key: str ;当None时，则从环境变量中读取：DASH_SCOPE_API_KEY
        :param streaming: bool
        :param enable_thinking: bool; Qwen3 model only
        :param thinking_budget: int
        :param extra_body: dict; 缺省{"enable_search": True}
        :param tools: list
        :param structure_output: TypedDict
        """
        if extra_body is None:
            extra_body = {
                "enable_search": True
            }
        ChatQwen_param = {
            'model': model,
            'base_url': base_url,
            'streaming': streaming,
            'enable_thinking': enable_thinking,
            'thinking_budget': thinking_budget,
            'extra_body': extra_body,
        }
        if api_key:
            ChatQwen_param['api_key'] = api_key
        self.model = ChatQwen(**ChatQwen_param)
        if tools is not None:
            self.model = self.model.bind_tools(tools)

        if structure_output is not None:
            self.model = self.model.with_structured_output(structure_output)

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


class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


class langgraph_agent:
    def __init__(self,
                 model: Union[str, chat_models] = 'qwen-turbo',
                 tools: list = None,
                 checkpointer: InMemorySaver | None = None,
                 structure_output: dict[str, typing_extensions.Any] | BaseModel | type | None = None,
                 system_instruction: str | list[AnyMessage] = "You are a helpful assistant."
                 ):
        """
        :param model: str
        :param tools: list
        :param structure_output: TypedDict; requires the model to support `.with_structured_output`
        :param system_instruction: str
        """
        if checkpointer is None:
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

    async def astreamOutput(self, input: str | State,
                            stream_modes: Literal['values', 'updates', 'custom', 'messages', 'debug'] | list[
                                Literal['values', 'updates', 'custom', 'messages', 'debug']] = 'updates',
                            thread_id: str = None, config: RunnableConfig = None,
                            print_mode: list[Literal['token', 'think', 'model_output', 'tools', 'None']] | Literal[
                                'token', 'think', 'model_output', 'tools', 'None'] = 'None'):
        """
        异步流式打印Agent response,可以同时接受'updates','messages'两种stream_mode,以便同时stream token,和输出structured_response;
        messages特点: 1) stream输出llm token,包括reason_content,以及模型content;
                     2) 不能输出generate_structured_response;
        updates特点: 不是token级别stream输出,是每个步骤输出.例如,只输出两条update messages:
                        a)模型输出(带思考在一条message内) b)structured_response
        千文模型 response json: {'agent':{'messages':[AIMessage(content='',
                                             additional_kwargs={'reasoning_content': '正在思考中...'},
                                             response_metadata={'finish_reason','model_name'},
                                             id='',
                                             usage_metadata={},
                                             output_token_details={})]}}
        :param input:
        :param stream_modes: str,或者包含多个stream_mode的列表
        :param thread_id: Short-term memory (thread-level persistence) enables agents to track multi-turn conversations
        :param config: config: RunnableConfig=None; 当thread_id=None时，可以直接赋值config = {"configurable": {"thread_id": thread_id}}
        :param print_mode: list，包含一个多多个选项；
            'token': to print streaming token_level for "messages"
            'think': to print streaming think
            'tools': to print tools
            'model_output': to print model_output
            'None': no printing
        :return: 根据stream_mode返回不同元组：
            1) if stream_mode = 'updates', return: updates_think_content, updates_modelOutput, updates_finish_reason, structrued_response
            2) if stream_mode = 'messages', return: msg_think_content, msg_modelOutput, msg_finish_reason, structed_response(为与updates对应，无意义)
            3) if stream_mode = ['updates', 'messages'], return ((updates returns), (messages returns))
        """
        if isinstance(stream_modes, str):
            stream_modes = [stream_modes]
        if isinstance(print_mode, str):
            print_mode = [print_mode]

        if isinstance(input, str):
            message = {"messages": input}
        else:
            message = input
        if config is None and thread_id is not None:
            config = {"configurable": {"thread_id": thread_id}}
        response = self.agent.astream(input=message, config=config, stream_mode=stream_modes)
        # response json: {'agent':{'messages':[AIMessage(content='',
        #                                      additional_kwargs={'reasoning_content': '正在思考中...'},
        #                                      response_metadata={'finish_reason','model_name'},
        #                                      id='',
        #                                      usage_metadata={},
        #                                      output_token_details={})

        isFirst_updates_think = True
        isFirst_updates_modelOutput = True
        isFirst_updates_toolCalls = True
        isFirst_msg_think = True
        isFirst_msg_toolCalls = True
        isFirst_msg_modelOutput = True
        updates_think_content = ""
        updates_modelOutput = ""
        updates_finish_reason = ""
        structured_response = {}

        msg_think_content = ""
        msg_modelOutput = ""
        msg_finish_reason = ""

        async for stream_mode, msg in response:
            # print(f'response: {msg}')
            if stream_mode == 'updates':
                agent_msgs = []
                tool_msgs = []
                # 处理节点(agent)消息
                if 'agent' in msg:
                    if 'messages' in msg['agent']:
                        agent_msgs = msg['agent']["messages"]  # 消息列表
                    for agent_msg in agent_msgs:
                        #  输出reasoning:
                        if hasattr(agent_msg,
                                   'additional_kwargs') and "reasoning_content" in agent_msg.additional_kwargs:
                            if isFirst_updates_think:
                                print("\n*Starting to think...*  \n")
                                isFirst_updates_think = False
                            updates_think_content = agent_msg.additional_kwargs["reasoning_content"]
                            if 'think' in print_mode and 'token' not in print_mode:
                                print(updates_think_content, end="", flush=True)
                        #  输出content:
                        if hasattr(agent_msg, 'content') and agent_msg.content:
                            if isFirst_updates_modelOutput:
                                print("\n*Starting model output...*  \n")
                                isFirst_updates_modelOutput = False
                            updates_modelOutput = agent_msg.content
                            if 'model_output' in print_mode and 'token' not in print_mode:
                                print(updates_modelOutput, end="", flush=True)
                        if hasattr(agent_msg, 'response_metadata'):
                            updates_finish_reason = agent_msg.response_metadata.get('finish_reason', None)
                            if updates_finish_reason == "stop":
                                print("\n*Ending model output...*  \n")
                                isFirst_updates_modelOutput = True
                            if updates_finish_reason == "tool_calls":
                                print("\n*Ending tool calls...*  \n")
                                isFirst_updates_toolCalls = True
                # 处理tools消息
                if 'tools' in msg:
                    if 'messages' in msg['tools']:
                        tool_msgs = msg['tools']["messages"]  # 消息列表
                    for tool_msg in tool_msgs:
                        if hasattr(tool_msg, 'name'):
                            if isFirst_updates_toolCalls:
                                print(f"tool: {tool_msg.name}, invoked ")
                                isFirst_updates_toolCalls = False
                        if hasattr(tool_msg, 'content') and tool_msg.content:
                            if 'tools' in print_mode:
                                print(tool_msg.content, end="", flush=True)

                # 处理{generate_structured_response:{'structured_response': None}}消息:
                if 'generate_structured_response' in msg:
                    if 'structured_response' in msg['generate_structured_response']:
                        structured_response = msg['generate_structured_response']['structured_response']
                        # print(f"\nstructured_response: {structured_response}\n")
                    else:
                        print(f"\nagent没有生成structured_response\n")

            if stream_mode == 'messages':
                llm_token, metadata = msg
                # print(f"llm_token:{llm_token}")
                # print(f"metadata: {metadata}")
                # 输出reasoning:
                if hasattr(llm_token, "additional_kwargs"):
                    add_kwargs = llm_token.additional_kwargs
                    if 'reasoning_content' in add_kwargs:
                        msg_think_content = add_kwargs['reasoning_content']
                        if isFirst_msg_think:
                            print("\n*Start Thinking...*  \n")
                            isFirst_msg_think = False
                        if 'think' in print_mode:
                            print(msg_think_content, end="", flush=True)
                    if 'tool_calls' in add_kwargs:
                        if isFirst_msg_toolCalls:
                            print("\n*Start tools_call...*  \n")
                            isFirst_msg_toolCalls = False
                        for dict in add_kwargs['tool_calls']:
                            if 'function' in dict:
                                if 'arguments' in dict['function']:
                                    if 'tools' in print_mode:
                                        print(dict['function']['arguments'], end="", flush=True)

                    if hasattr(llm_token, "response_metadata"):
                        resp_metadata = llm_token.response_metadata
                        msg_finish_reason = resp_metadata.get('finish_reason', None)
                        if msg_finish_reason == "stop":
                            print("\n*Ending model output...*  \n")
                            isFirst_msg_modelOutput = True
                        if msg_finish_reason == "tool_calls":
                            print("\n*Ending tool_calls...*  \n")

                # 输出content:
                if hasattr(llm_token, 'content'):
                    if llm_token.content:
                        msg_modelOutput = llm_token.content
                        if isFirst_msg_modelOutput:
                            print("\n*Starting model output...*  \n")
                            isFirst_msg_modelOutput = False
                        if 'model_output' in print_mode:
                            print(msg_modelOutput, end="", flush=True)
                        print(msg_modelOutput, end="", flush=True)

            if 'updates' in stream_modes and 'messages ' not in stream_modes:
                yield updates_think_content, updates_modelOutput, updates_finish_reason, structured_response
            if 'updates' not in stream_modes and 'messages ' in stream_modes:
                yield msg_think_content, msg_modelOutput, msg_finish_reason, structured_response
            if 'updates' in stream_modes and 'messages' in stream_modes:
                yield (updates_think_content, updates_modelOutput, updates_finish_reason, structured_response), (
                    msg_think_content, msg_modelOutput, msg_finish_reason)

    async def multi_turn_conversation(self, stream_modes: Literal[
                                                              'values', 'updates', 'custom', 'messages', 'debug'] |
                                                          list[
                                                              Literal[
                                                                  'values', 'updates', 'custom', 'messages', 'debug']] = 'updates',
                                      print_mode: Literal['token', 'think', 'model_output', 'tools', 'None'] = 'None',
                                      thread_id: str | None = None):
        """
        多轮对话(似乎不设置thread_id时，也是具备会话的记忆)
        :param stream_modes: Literal['values', 'updates', 'custom', 'messages', 'debug']
        :param thread_id: Short-term memory (thread-level persistence) enables agents to track multi-turn conversations
        :return:
        """
        if isinstance(stream_modes, str):
            stream_modes = [stream_modes]
        while True:
            try:
                user_input = input("\nUser: ")
                if user_input.lower() in ["quit", "exit", "q"]:
                    print("Goodbye!")
                    break
                # 以下待完成：根据print_mode参数，print不同内容
                response = self.astreamOutput(user_input, stream_modes=stream_modes,
                                              print_mode=print_mode, thread_id=thread_id, config=config)

                if 'updates' in stream_modes and 'messages ' not in stream_modes:
                    async for updates_think_content, updates_modelOutput, updates_finish_reason, structured_response in response:
                        print(f"")
                if 'updates' not in stream_modes and 'messages ' in stream_modes:
                    msg_think_content, msg_modelOutput, msg_finish_reason, structured_response = response
                    async for msg_think_content, msg_modelOutput, msg_finish_reason, structured_response in response:
                        print(f"")
                if 'updates' in stream_modes and 'messages' in stream_modes:
                    async for (
                            updates_think_content, updates_modelOutput, updates_finish_reason, structured_response), (
                            msg_think_content, msg_modelOutput, msg_finish_reason) in response:
                        print(f"")


            except Exception as e:
                print(f"发生错误: {str(e)}")
                break
            except KeyboardInterrupt:
                print("\n用户中断，退出程序")
                break


class nodeloopState(State):
    loop_count: Annotated[int, operator.add]



class QwenML_trasOptions(BaseModel):
    response: list[AnyMessage] | str = ""  # Qwen_ML模型就非翻译请求的响应 = “”
    text: str = ""  # 待翻译的文本
    source_lang: str = "auto"  # "Chinese"
    target_lang: str = ""  # "English"
    domains: str = ""  # 翻译的风格具备某领域的特性，自然语言(英文)描述
    translate_request: bool = False  # 表明输入是否是关于语言翻译的请求，如True，则结束对话
    img: bool = False  # 是否prompt中带有图片，handoff至Qwen_VL


@dataclass
class EvaluationFeedback:
    feedback: str
    score: Literal["pass", "needs_improvement", "end"]




def QwenML_transOption_node(state: State, config: RunnableConfig) -> Command[Literal['Qwen_ML_node', 'Qwen_VL_agent']]:
    print(f"+ **prompt预分析:**")
    response = QwenML_transOption_agent.agent.invoke(input=state, config=config)
    if response['structured_response'] is None:  # 有时候，由于prompt对于结构化输出的类的各个item生成有遗漏，会导致structure_response=None
        structured_response = json.loads(response['messages'][-1].content)
    else:
        structured_response = response['structured_response']
        if isinstance(structured_response, BaseModel):
            structured_response = structured_response.model_dump()  # 将BaseModel对象转换为字典,为使得Pydantic类型接受get方法
    update_state = {'messages': [AIMessage(structured_response.get('response', 'no response is available'))]}
    if isinstance(structured_response, dict):
        if structured_response.get('translate_request', False):
            goto = "Qwen_ML_node"
            # 构造一个包含所有必要字段的新状态
            update_state = QwenML_trasOptions(
                response=structured_response.get('response', 'no response is available'),
                text=structured_response.get('text', ''),
                source_lang=structured_response.get('source_lang', 'auto'),
                target_lang=structured_response.get('target_lang', ''),
                domains=structured_response.get('domains', ''),
                translate_request=structured_response.get('translate_request', False),
                img=structured_response.get('img', False),
            )
            # 本节点即goto， 也update
            command_params = {'update': update_state, 'goto': goto}
        else:  # 本节点，不带有goto参数，仅update; command后，不再handoff,自行结束;
            command_params = {'update': update_state}
        if structured_response.get('img', False):  # 本节点，不带有update参数，仅goto; command后，仅handoff, 不更新state;
            goto = "Qwen_VL_agent"
            command_params = {'goto': goto}
            # command_params['update'] = state  # 如果需要识图，无需update
    else:  # 本节点，不带有goto参数，仅update; command后，不再handoff,自行结束;
        command_params = {'update': update_state}
    return Command(**command_params)


def Qwen_ML_node(state: QwenML_trasOptions, config: RunnableConfig) -> Command[Literal['evaluator']]:
    print(f"+ **进入Qwen_ML_node，启动首次翻译:**")
    if not state.translate_request:  # 本节点，不带有goto参数，仅update; command后，不再handoff,自行结束;
        command_params = {'update': {"messages": [AIMessage(content=state.response)]}}
    else:
        text, source_lang, target_lang, domains = state.text, state.source_lang, state.target_lang, state.domains
        extra_body = {
            "translation_options": {
                "source_lang": source_lang,
                "target_lang": target_lang,
                "domains": domains,
            }
        }
        Qwen_MT = langchain_qwen_llm(model='qwen-mt-plus',
                                     streaming=False,
                                     extra_body=extra_body, )
        message = [{"role": "user", "content": text}]
        try:
            response = Qwen_MT.model.invoke(input=message, config=config)
            # print(f"**首次翻译:**\n  {response.content}")
            update = {"messages": [AIMessage(content=response.content)],
                      "loop_count": 1,  # operator.add会自动增加1,分别统计循环次数
                      }
            command_params = {'update': update, 'goto': 'evaluator'}
        except Exception as e:
            response = str(e)

            update = {"messages": [AIMessage(content=response)],
                      "loop_count": 0,  # operator.add会自动增加0,分别统计循环次数
                      }
            command_params = {'update': update, }
    return Command(**command_params)


async def evaluator_node(state: nodeloopState, config: RunnableConfig) -> Command[Literal['translator']]:
    loop_count = state.get("loop_count", 0)
    print(f"+ **进入翻译评估阶段, 当前第{loop_count}次翻译评估**")
    response = evaluator.astreamOutput(input=state, stream_modes='updates', print_mode='None', config=config)
    command_params = {'update': [AIMessage("agent doesn't generate structured_response, exit!")]}
    async for think, modelOutput, finish_reason, structured_response in response:
        if structured_response is not None and structured_response:  # 非空{},非None
            score = structured_response['score']
            feedback = structured_response['feedback']
            if score == 'end' or feedback == 'pass':
                update = {"messages": [AIMessage(feedback)],
                          'loop_count': -loop_count,
                          }  # 节点结束时，state输出将loop_count清零; ;
                command_params = {'update': update}  # 节点无goto, command后，不再handoff,自行结束;
            elif score == "needs_improvement":
                goto = "translator"
                update = {"messages": [HumanMessage(feedback)],
                          'loop_count': 0}  # operator.add会自动增加0,统计循环次数;(评估次数为翻译次数相等,增加0)
                command_params = {'goto': goto, 'update': update}

            print(f"**评估score: {score}**")
            # print(f"**评估feedback:\n  {response['structured_response']['feedback']}")
        yield Command(**command_params)


async def translator_node(state: nodeloopState, config: RunnableConfig) -> Command[Literal['evaluator']]:
    loop_account = state.get("loop_count", 0)
    print(f"+ **进入翻译改进阶段, 当前第{loop_account}次改进**")
    response = translator.astreamOutput(input=state, stream_modes='updates', print_mode='None', config=config)
    async for think, modelOutput, finish_reason, structured_response in response:
        if modelOutput:
            update = {"messages": [AIMessage(modelOutput)],
                      "loop_count": 1,  # operator.add会自动增加1,统计循环次数
                      }
            # print(f"**翻译改进:**\n  {modelOutput}")
            yield Command(
                goto='evaluator',
                update=update, )



async def langgraph_astream(graph: StateGraph | CompiledStateGraph, state: State,
                            stream_mode: Literal['messages', 'updates'] = "updates",
                            print_mode: list[Literal['token', 'think', 'model_output', 'tools', 'None']] | Literal[
                                'token', 'think', 'model_output', 'tools', 'None'] = 'None',
                            config: dict = None):
    """
    包含多个节点的langgraph异步stream输出;
    可以同时接受'updates','messages'两种stream_mode,以便同时stream token,和输出structured_response;
        messages特点: 1) stream输出llm token,包括reason_content,以及模型content;
                     2) 不能输出generate_structured_response;
        updates特点: 不是token级别stream输出,是每次update的步骤输出.例如,只输出两条update messages:
                        a)模型输出(带思考在一条message内) b)structured_response
    :param graph: 多节点的compiled的langgraph对象；
    :param state: 输入state
    :param stream_mode: str,或者包含多个stream_mode的列表
    :param print_mode: list,包含一个或者多个选项
        'token': to print streaming token_level for "messages"
        'think': to print streaming think
        'tools': to print tools
        'model_output': to print model_output
        'None': no printing
    :param config: 典型如: config = {"configurable": {"thread_id": thread_id},
                                    "recursion_limit": 20}
    :return:
        node_name: graph每次update时的node name: 缺省值为: graph.name
        updates_think_content: graph每次更新状态中如有思考时的输出；缺省值：None；
        updates_modelOutput: graph每次更新状态中如有模型输出(AIMessage/HumanMessage)时的输出；缺省值：None；
        updates_finish_reason: graph每次更新状态中如有finish_reason时的输出；例如'stop', 'tool_calls', 缺省值：None；
    """
    if isinstance(stream_mode, str):
        stream_mode = [stream_mode]
    if isinstance(print_mode, str):
        print_mode = [print_mode]
    node_name = graph.name
    updates_think_content = None
    updates_modelOutput = None
    updates_finish_reason = None

    try:
        async for stream_mode, chunk in graph.astream(state,
                                                      stream_mode=stream_mode,
                                                      config=config):
            if stream_mode == "messages":
                token, metadata = chunk
                node_name = metadata['langgraph_node']
                # print(f"graph运行节点: {metadata['langgraph_node']}")
                if token.content:
                    print(token.content, end="", flush=True)

            if stream_mode == "updates":
                # print(f"graph当前update了node: {[*chunk.keys()]}")
                for node in chunk.keys():
                    print(f"graph当前update的node: {node}")
                    node_name = node
                    # print(f"其State为:{chunk[node]}")

                    if isinstance(chunk[node], dict):
                        if 'messages' in chunk[node]:
                            modeloutput = chunk[node]['messages']  # list
                            for msg in modeloutput:
                                #  输出reasoning:
                                if hasattr(msg,
                                           'additional_kwargs') and "reasoning_content" in msg.additional_kwargs:
                                    updates_think_content = msg.additional_kwargs["reasoning_content"]
                                    if 'think' in print_mode and 'token' not in print_mode:
                                        print(updates_think_content, end="", flush=True)
                                #  输出content:
                                if hasattr(msg, 'content') and msg.content:
                                    updates_modelOutput = msg.content
                                    if 'model_output' in print_mode and 'token' not in print_mode:
                                        print(updates_modelOutput, end="", flush=True)
                                if hasattr(msg, 'response_metadata'):
                                    updates_finish_reason = msg.response_metadata.get('finish_reason', None)
                                    if updates_finish_reason == "stop":
                                        if not "None" in print_mode:
                                            print("\n*Ending model output...*  \n")
                                    if updates_finish_reason == "tool_calls":
                                        if not "None" in print_mode:
                                            print("\n*Ending tool calls...*  \n")

                                yield node_name, updates_think_content, updates_modelOutput, updates_finish_reason

        print(f"graph: {graph.name} 正常完成 !")

    except GraphRecursionError:
        response = "Recursion Error"
        print(f"graph: {graph.name} 响应错误:{response} !")
        yield node_name, updates_think_content, updates_modelOutput, updates_finish_reason


def translation_graph(State: State, name="translation_graph", checkpointer: None | bool | InMemorySaver = None):
    builder = StateGraph(State, )
    builder.add_node("QwenML_transOption_node", QwenML_transOption_node)
    builder.add_node("Qwen_VL_agent", Qwen_VL_agent.agent)
    builder.add_node("Qwen_ML_node", Qwen_ML_node)
    builder.add_node("evaluator", evaluator_node)
    builder.add_node("translator", translator_node)

    builder.add_edge(START, 'QwenML_transOption_node')
    builder.add_edge("Qwen_VL_agent", "Qwen_ML_node")

    translation_agent = builder.compile(name=name, checkpointer=checkpointer)


    return translation_agent


Qwen_plus = langchain_qwen_llm(model="qwen-plus-latest", enable_thinking=True, streaming=True)

Qwen_turbo_noThink = langchain_qwen_llm(model="qwen-turbo", )

Qwen_VL = langchain_qwen_llm(model="qwen-vl-ocr-latest", )
checkpointer = InMemorySaver()
QwenML_transOption_agent = langgraph_agent(model=Qwen_turbo_noThink.model,
                                           checkpointer=checkpointer,
                                           structure_output=QwenML_trasOptions,
                                           system_instruction="""
                                           你是一个优秀的助手。你对接收的prompt进行分析，并按照如下指示,将结果按照structured_response事先定义的Pydantic类，进行结构化输出每个字段：
                                           1) 首先对prompt中是否包含图片，做出判断，如果prompt包含有图片，则结构化输出字段: img=True,否则，img=False;
                                           2) 再对prompt是否包含有语言翻译的请求，做出判断，如果包含有语言翻译请求，结构化输出字段：translate_request=True；否则,translate_request=False;
                                           3) 如果prompt包含有语言翻译的请求，请从中整理出待翻译的文本，并使用一段自然英文(必须为英文)总结下待翻译文本的领域，语气，从而使得翻译的风格更符合某个领域的特性；并将该总结输出至结构化输出字段domains;否则,domains="";
                                           4) 如果prompt包含有语言翻译的请求，请从中分析出翻译请求的源语言，目标语言，以及整理出的待翻译的文本，并分别结构化输出到字段 source_lang, target_lang, text; 否则, source_lang='auto', target_lang='', text='';
                                           5) 如果prompt包含有语言翻译的请求，请结构化输出字段：response=''; 否则，尽你所能，进行回答问题或者提供帮助，并将响应内容结构化输出到response;
                                           6) 事先定义的用于结构化输出Pydantic的各个字段(field)中，如果以上指示中有遗漏，请使用默认值，最后完整结构化输出该事先定义的Pydantic类。
                                           """)

Qwen_VL_agent = langgraph_agent(model=Qwen_VL.model,
                                checkpointer=checkpointer,
                                structure_output=QwenML_trasOptions,
                                system_instruction="""
                                你接收到的prompt将同时包括text文本,以及图片img或者文件file.你善于识图理解，请识图并理解输入的图片或文件，获取其全部内容，并且结合接收的prompt,按照structured_response事先定义的Pydantic类，进行结构化输出每个字段：
                                           1) 显然，你已经收到了图片img或者文件file; 请结构化输出字段: img=True;
                                           2) 请对prompt是否包含有语言翻译的请求，做出判断，如果包含有语言翻译请求，结构化输出字段：translate_request=True；否则,translate_request=False;
                                           3) 如果prompt包含有语言翻译的请求，结合识图理解的内容，请从中整理出待翻译的文本，并使用一段自然英文(必须为英文)总结下待翻译文本的领域，语气，从而使得翻译的风格更符合某个领域的特性；并将该总结输出至结构化输出字段domains;否则,domains="";
                                           4) 如果prompt包含有语言翻译的请求，结合识图理解的内容，请从中分析出翻译请求的源语言，目标语言，以及整理出的待翻译的文本，并分别结构化输出到字段 source_lang, target_lang, text; 否则, source_lang='auto', target_lang='', text='';
                                           5) 如果prompt包含有语言翻译的请求，结合识图理解的内容，请结构化输出字段：response=''; 否则，尽你所能，就提问的text文本，结合识图理解的内容，进行回答问题或者提供帮助，并将响应内容结构化输出到response;
                                           6) 事先定义的用于结构化输出Pydantic的各个字段(field)中，如果以上指示中有遗漏，请使用默认值，最后完整结构化输出该事先定义的Pydantic类。                                    
                                """)

evaluator = langgraph_agent(model=Qwen_plus.model,
                            checkpointer=checkpointer,
                            structure_output=EvaluationFeedback,
                            system_instruction="""
                            你是一个翻译评价家，根据你收到的包含原文以及翻译的内容，评估翻译质量是否合格，并给出评价意见, 你将结构化输出: score: 评估结论,包含pass,needs_improvement,end; feedback: 反馈意见;
                                a.如果你对翻译内容评估不太满意，认为需改进(needs_improvement)的话，你需要结构化输出: score="needs_improvement", feedback为反馈意见，指明翻译内容需要改进的地方;
                                b.如果你对翻译内容比较满意，则结构化输出: score="pass", feedback=你满意的翻译内容;
                                c.如果你认为，不需要给出评估意见，请结构化输出: score="end", feedback=原因或理由; 然后,你可以结束进一步推理,停止任何响应,停止任何输出,结束你的工作,退出.
                                d.评价的要求需要严格，尽量不要在首次评价中就给与翻译质量合格(pass)的决定。
                            """)
async def get_mcp_tool():
    # get_tools是一个协程对象，需要被await,需要异步函数
    # 并且StructuredTool does not support sync invocation
    client = MultiServerMCPClient(
        {
            "tempSave": {
                "command": "python",
                # Replace with absolute path to your math_server.py file
                "args": ["E:/Python_WorkSpace/modelscope/mcp/mcpServer_localSave.py"],
                "transport": 'stdio',
            },
        }
    )
    tool = await client.get_tools()
    return tool


translator = langgraph_agent(model=Qwen_plus.model,
                             checkpointer=checkpointer,
                             system_instruction="""
                              1)你是一名优异的文档翻译官，具备各类语言的文字，文档的翻译能力；并且具备根据原文的文体，原文内容的领域，阅读对象，语气，使用恰当的目标语言和文字，术语，语气来翻译原文的能力，翻译结果专业，贴切。
                              2)你也会根据输入的评估意见，改进建议，针对性的对翻译结果进行改善;
                              """)

if __name__ == '__main__':
    thread_id = uuid.uuid4().hex  # 128 位的随机数，通常用 32 个十六进制数字表示
    config = {"configurable": {"thread_id": thread_id},
              "recursion_limit": 20}

