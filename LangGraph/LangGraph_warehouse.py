import os
from typing import Literal, Union

from langchain_community import chat_models
from langchain_qwq import ChatQwen, ChatQwQ
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.messages import AnyMessage

from langgraph.prebuilt import create_react_agent
from langgraph.prebuilt.chat_agent_executor import AgentState
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver

from pydantic import BaseModel
import pydantic
from typing import TypedDict
import asyncio

class langchain_qwen_llm:
    def __init__(self,
                 model: str= 'qwen-turbo-latest',
                 base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",  # 阿里云国内站点(默认为国际站点),
                 streaming: bool = True,
                 enable_thinking: bool = False,
                 thinking_budget: int = 100,
                 extra_body: dict = None,
                 tools: list = None,
                 structure_output: TypedDict = None,
                 system_instruction: str | list[AnyMessage] = "You are a helpful assistant."
                 ):
        """
        langchain-qwq库中的ChatQwQ与ChatQwen针对Qwen3进行了优化；然而，其缺省的base_url却是阿里云的国际站点；国内使用需要更改base_url为国内站点
        :param model: str
        :param base_url: str
        :param streaming: bool
        :param enable_thinking: bool
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
            self.model.bind_structure_output(structure_output)

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
        :param stream_mode: str
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



class graph_agent:
    def __init__(self,
                 model: Union[str,chat_models] = 'qwen-turbo-latest',
                 tools: list = None,
                 structure_output: TypedDict = None,
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
            # print(f'response,msg: {msg}')
            agent_msg = {}
            if 'agent' in msg:
                if 'messages' in msg['agent']:
                    agent_msg = msg['agent']["messages"][-1]  # 最后一条消息
            else:
                print(f'msg: {msg}')
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


if __name__ == '__main__':
    prompt = '请总结今日国际新闻3条'
    Qwen_llm = langchain_qwen_llm(enable_thinking=True)
    Qwen_agent = graph_agent(model=Qwen_llm.model)
    # graph_draw_path = r"E:/Python_WorkSpace/modelscope/LangGraph/graph_draw.png"
    # # graph_agent.agent.get_graph().draw_mermaid_png(output_file_path=graph_draw_path)

    asyncio.run(Qwen_agent.astreamPrint(prompt))
    # asyncio.run(Qwen_llm.astreamPrint(prompt))
