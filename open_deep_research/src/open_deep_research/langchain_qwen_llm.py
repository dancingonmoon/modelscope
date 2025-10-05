from langchain.chat_models import init_chat_model
from langchain_qwq import ChatQwen, ChatQwQ
import typing_extensions
from typing_extensions import TypedDict, Optional
from pydantic import BaseModel
from typing import Literal


def langchain_qwen_llm(
        model: str = 'qwen-turbo',
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",  # 阿里云国内站点(默认为国际站点),
        api_key: str = None,
        streaming: bool = True,
        enable_thinking: bool = False,
        thinking_budget: Optional[int] = 100,
        extra_body: dict = None,
        tools: list = None,
        structure_output: dict[str, typing_extensions.Any] | BaseModel | type | None = None,
        structure_output_method: Literal["function_calling", "json_mode", "json_schema"] = "function_calling",
        max_retries: int = None,
        max_tokens: Optional[int] = None,
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
            :param structure_output_method: The method for steering model generation, one of:
                    - "function_calling": Uses DashScope Qwen's `tool-calling features <https:// help. aliyun. com/ zh/ model-studio/ qwen-function-calling>`_.
                    - "json_mode": Uses DashScope Qwen's `JSON mode feature <https:// help. aliyun. com/ zh/ model-studio/ json-mode>`_.
            :param max_tokens:  Max number of tokens to generate.
            :param max_retries: Max number of retries
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
    if max_tokens is not None:
        ChatQwen_param['max_tokens'] = max_tokens
    if max_retries is not None:
        ChatQwen_param['max_retries'] = max_retries

    model = ChatQwen(**ChatQwen_param)
    if tools is not None:
        model = model.bind_tools(tools)

    if structure_output is not None:
        model = model.with_structured_output(structure_output, method=structure_output_method)

    return model
