import os
import sys
from pathlib import Path
import uuid

sys.path.append(str(Path(__file__).parent.parent))  # 添加项目根目录

import gradio as gr  # gradio 5.5.0 需要python 3.10以上

from zhipuai import ZhipuAI
from google import genai  # 新版
from google.genai.types import Tool, GenerateContentConfig, GoogleSearch
from agents import Agent, Runner
import base64
from typing import Literal
import json
import logging

from LangGraph_warehouse import translation_graph, State, checkpointer, langgraph_astream, EvaluationFeedback
from langgraph.graph.state import CompiledStateGraph, StateGraph
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


def gradio_msg2LLM_msg(gradio_msg: dict = None,
                       msg_format: Literal["openai_agents", "gemini", "glm", "langchain"] = "openai_agents",
                       genai_client: genai.Client = None, zhipuai_client: ZhipuAI = None):
    """
    一次gradio的多媒体message(包含text,file)，转换成各类LLM要求的message格式
    :param gradio_msg: gradio.MultiModalText.value,例如: {"text": "sample text", "files": [{path: "files/file.jpg", orig_name: "file.jpg", url: "http://image_url.jpg", size: 100}]}
    :param msg_format: "openai_agents", "gemini", "glm"
    :param genai_client: genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))
    :param zhipuai_client: ZhipuAI(api_key=os.environ.get("ZHUIPU_API_KEY"))
    :return:  与msg_format兼容的message格式，以及History
    """
    supported_img = [".bmp", ".png", ".jpe", ".jpeg", ".jpg", ".tif", ".tiff", ".webp", ".heic"]
    jpg_variant = ['.jpe', '.jpeg', '.jpg']
    tif_variant = ['.tif', '.tiff']
    contents = []
    input_item = []

    text = gradio_msg.get("text", '')
    files = gradio_msg.get("files", [])
    # openAI-Agents gradio_message 格式处理:
    if msg_format == "openai_agents":
        if files:
            for file in files:
                file_path = Path(file)
                if file_path.exists() and file_path.is_file():
                    file_suffix = file_path.suffix.lower()
                    if file_suffix in supported_img:  # 处理Image:
                        with open(file_path, "rb") as image_file:
                            base64_img = base64.b64encode(image_file.read()).decode("utf-8")
                        if file_suffix in jpg_variant:
                            file_suffix = "jpeg"
                        elif file_suffix in tif_variant:
                            file_suffix = "tiff"
                        content = {
                            # "type": "image_url", # qwen的OpenAI格式,与openai-agent不同
                            # "image_url": {"url": f"data:image/{img_format};base64,{base64_img}"} # qwen的OpenAI格式,与openai-agent不同
                            "type": "input_image",
                            "detail": "auto",
                            "image_url": f"data:image/{file_suffix};base64,{base64_img}"}  # openAI-Aents格式
                        contents.append(content)
                    else:
                        # 可以处理其它格式文件，例如:使用file.upload
                        print("✅ 暂时只处理指定格式的IMG格式")
                        # break
                else:
                    print("✅ 文档路径不存在")
                    # break

            input_item.append({"role": "user", "content": contents})
        input_item.append({"role": "user", "content": text})

    # gemini gradio_message 格式处理:
    elif msg_format == "gemini":
        # genai_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        if files:
            for file in files:
                file_path = Path(file)
                if file_path.exists() and file_path.is_file():
                    # Gemini 1.5 Pro 和 1.5 Flash 最多支持 3,600 个文档页面。文档页面必须采用以下文本数据 MIME 类型之一：
                    # PDF - application/pdf,JavaScript - application/x-javascript、text/javascript,Python - application/x-python、text/x-python,
                    # TXT - text/plain,HTML - text/html, CSS - text/css,Markdown - text/md,CSV - text/csv,XML - text/xml,RTF - text/rtf
                    content = genai_client.files.upload(file=file_path)  # 环境变量缺省设置GEMINI_API_KEY
                    contents.append(content)
                else:
                    print("✅ 文档路径不存在")
                    # break
        contents.append(text)
        input_item.append({"role": "user", "content": contents})
        # print(f"genai input_item: {input_item}")

    # glm gradio_message 格式处理:
    elif msg_format == "glm":
        # zhipuai_client = ZhipuAI(api_key=os.getenv("ZHIPU_API_KEY"))
        if files:
            contents = "请结合以下文件或图片内容回答：\n"  # for glm
            for file_No, file in enumerate(files):
                file_path = Path(file)
                if file_path.exists() and file_path.is_file():
                    # 格式限制：.PDF .DOCX .DOC .XLS .XLSX .PPT .PPTX .PNG .JPG .JPEG .CSV .PY .TXT .MD .BMP .GIF
                    # 大小：单个文件50M、总数限制为100个文件
                    file_object = zhipuai_client.files.create(
                        file=Path(file), purpose="file-extract"
                    )
                    # 获取文本内容
                    content = json.loads(
                        zhipuai_client.files.content(file_id=file_object.id).content
                    )["content"]

                    if content is None or content == "":
                        contents += f"第{file_No + 1}个文件或图片内容无可提取之内容\n\n"
                    else:
                        contents += f"第{file_No + 1}个文件或图片内容如下：\n" f"{content}\n\n"
                else:
                    print("✅ 文档路径不存在")
        if contents:
            contents = f'{text}. {contents}'
        else:
            contents = f'{text}'
        input_item.append({"role": "user", "content": contents})
        # print(f"gradio_msg2 input_item:{input_item}")

    # LangGraph-QWQ/Qwen gradio_message 格式处理:
    elif msg_format == "langchain":
        if files:
            for file in files:
                file_path = Path(file)
                if file_path.exists() and file_path.is_file():
                    file_suffix = file_path.suffix.lower()
                    if file_suffix in supported_img:  # 处理Image:
                        with open(file_path, "rb") as image_file:
                            base64_img = base64.b64encode(image_file.read()).decode("utf-8")
                        if file_suffix in jpg_variant:
                            file_suffix = "jpeg"
                        elif file_suffix in tif_variant:
                            file_suffix = "tiff"
                        content = {
                            "type": "image_url",  # qwen的OpenAI格式,与openai-agent不同
                            "image_url": {
                                "url": f"data:image/{file_suffix};base64,{base64_img}"}}  # qwen的OpenAI格式,与openai-agent不同
                        # "type": "input_image",
                        # "detail": "auto",
                        # "image_url": f"data:image/{file_suffix};base64,{base64_img}"}  # openAI-Aents格式
                        contents.append(content)
                    else:
                        # 可以处理其它格式文件，例如:使用file.upload
                        print("✅ 暂时只处理指定格式的IMG格式")
                        # break
                else:
                    print("✅ 文档路径不存在")
                    # break
            # contents.append({"type": "text", "text": text})
        # else:  # 对于Qwen模型，当promt为列表时，例如VL模型，必须{"type": "text", "text": text};否则必须为字符串非列表
        # contents.append(text)

        contents.append({"type": "text", "text": text})
        state = HumanMessage(content=contents)
        input_item.append(state)

    return input_item


def add_message(history_gradio: list[gr.ChatMessage] = None, history_llm: list[dict] = None,
                gradio_message: str | dict[str, str | list] = None, model: str = None):
    # gradio gr.MultiModalTextbox() 输出:
    # value= {"text": "sample text", "files": [{'path': "files/ file. jpg", 'orig_name': "file. jpg", 'url': "http:// image_url. jpg ", 'size': 100}]},
    # chatbot gr.Chatbot() 输入与输出：
    # ChatMessage = {"role": "user", "content":str|FileData"}
    #                                  {"Path": str, "url": str,
    #                                  "size": int, mime_type": str, "is-stream": bool}}

    if history_llm is None:
        history_llm = []
    if history_gradio is None:
        history_gradio = []
    if isinstance(gradio_message, dict):
        text = gradio_message.get("text", None)
        files = gradio_message.get("files", [])
    elif isinstance(gradio_message, str):
        text = gradio_message
        files = []
    else:
        text = None
        files = []

    if files:
        for file in files:
            if isinstance(file, str):
                history_gradio.append({"role": "user", "content": {"path": file}})
            elif isinstance(file, dict):
                history_gradio.append({"role": "user", "content": {"path": file.get("path"), "url": file.get("url"),
                                                                   "mime_type": file.get("mime_type"), }})
    if text is not None:
        history_gradio.append({"role": "user", "content": text})

    if 'agent' in model.lower():
        llm_message = gradio_msg2LLM_msg(gradio_message, msg_format="openai_agents")
    elif 'glm' in model.lower():
        llm_message = gradio_msg2LLM_msg(gradio_message, msg_format="glm", zhipuai_client=zhipuai_client)
    elif 'gemini' in model.lower():
        llm_message = gradio_msg2LLM_msg(gradio_message, msg_format="gemini", genai_client=genai_client)
    elif 'translator' in model.lower():  # translator_agent由openai_agents SDK生成;
        llm_message = gradio_msg2LLM_msg(gradio_message, msg_format="openai_agents")
    elif "langchain" in model.lower() or "langgraph" in model.lower():
        llm_message = gradio_msg2LLM_msg(gradio_message, msg_format="langchain")
    else:
        llm_message = [{"role": "user", "content": text}]

    for item in llm_message:
        history_llm.append(item)
    # print(f"gradio_message:{gradio_message}")
    # print(f"add message_v2 history_llm:{history_llm}")
    # print(f"add message_v2 history_gradio:{history_gradio}")
    return (
        history_gradio,
        history_llm,
        gr.MultimodalTextbox(value=None, interactive=False),
        gr.Button(interactive=True, visible=True),

    )


def get_last_user_messages(history):
    """
    在刚刚完成role:user输入后的history的列表中，寻找最后一个assistant消息之后的全部user消息。(history列表中，最后的消息总是user消息)
    :param history:
    :return:
    """
    user_msg = []
    if history:  # 非空列表
        for msg in reversed(history):
            if msg.get("role", None) == "user":
                user_msg.append(msg)
            elif msg["role"] == "assistant":
                break
    return user_msg[::-1]


def undo_history(history: list[dict], ):
    """
    移除history中最后一个role不是user的那组消息。最后一组消息role，如为user， 不变化；否则，/assistant/system/model,则移除该组消息
    :param history:
    :return:
    """
    index = -1
    if history:  # 非空列表
        for index, msg in enumerate(reversed(history)):
            # print(f"undo_history type: {[type(msg) for msg in history]}:{history}")
            # 此处问题： 当model为gemini时，history：[UserConent(parts=[...] role='user'),Content(parts=[...] role='model')]
            if isinstance(msg, dict):
                if msg.get("role", None) != "user":
                    continue
                else:
                    break
            elif isinstance(msg, genai.types.Content):
                continue
            elif isinstance(msg, genai.types.UserContent):
                break
    return history[:index + 1 + 1]


def inference(history_gradio: list[dict], history_llm: list[dict], new_topic: bool, model: str = None,
              stop_inference: bool = False):
    """
    仅针对非异步函数的gemini以及glm模型，以实现yield生成器实现stream输出；当那种async异步函数定义的inference需要在async异步函数中执行，否则报错；
    :param history_gradio:
    :param history_llm:
    :param new_topic:
    :param model:
    :param stop_inference:
    :return:
    """
    if 'gemini' in model.lower():
        for history_gradio, history_llm in gemini_inference(history_gradio, history_llm, new_topic, model=model,
                                                            genai_client=genai_client,
                                                            stop_inference_flag=stop_inference):
            yield history_gradio, history_llm
    elif 'glm' in model.lower():
        for history_gradio, history_llm in glm_inference(history_gradio, history_llm, new_topic, model,
                                                         zhipuai_client=zhipuai_client,
                                                         stop_inference_flag=stop_inference):
            yield history_gradio, history_llm


async def async_inference(history_gradio: list[dict], history_llm: list[dict], new_topic: bool, model: str = None,
                          stop_inference: bool = False):
    """
    异步函数中，对于glm及gemini这种非异步的函数inference会导致非流式输出,yield生成器，经过异步函数后，可以推理，但不再是stream输出，而是一次性输出；对于openaiAgents这种异步函数定义的推理，也需要在异步函数中执行infrence，实现stream输出；
    :param history_gradio:
    :param history_llm:
    :param new_topic:
    :param model:
    :param stop_inference:
    :return:
    """
    if 'gemini' in model.lower() and 'translator' not in model.lower():
        for history_gradio, history_llm in gemini_inference(history_gradio, history_llm, new_topic, model=model,
                                                            genai_client=genai_client,
                                                            stop_inference_flag=stop_inference):
            yield history_gradio, history_llm
    elif 'glm' in model.lower():
        for history_gradio, history_llm in glm_inference(history_gradio, history_llm, new_topic, model,
                                                         zhipuai_client=zhipuai_client,
                                                         stop_inference_flag=stop_inference):
            yield history_gradio, history_llm


    elif "langgraph" in model.lower() and "translation" in model.lower():
        async for history_gradio, history_llm in translation_langgraph_inference(history_gradio, history_llm,
                                                                                 graph=translation_graph,
                                                                                 new_topic=new_topic,
                                                                                 stop_inference_flag=stop_inference,
                                                                                 stream_mode="updates",
                                                                                 config=config):
            yield history_gradio, history_llm


async def openai_agents_inference(
        history_gradio: list[dict], history_llm: list[dict], new_topic: bool, agent: Agent = None,
        stop_inference_flag: bool = False):
    try:
        present_message = get_last_user_messages(history_llm)
        if new_topic:
            input_message = present_message
        else:
            input_message = history_llm
        response = Runner.run_streamed(agent, input_message)

        present_response = ""
        history_gradio.append({"role": "assistant", "content": present_response})
        history_llm.append({"role": "assistant", "content": present_response})
        async for event in response.stream_events():
            if stop_inference_flag:
                yield history_gradio, history_llm  # 先yield 再return ; 直接return history会导致history不输出
                return
            if event.type == "raw_response_event" :
                print(event.data.delta, end="", flush=True)
                present_response += event.data.delta
            elif event.type == "agent_updated_stream_event":
                print(f"Agent updated: {event.new_agent.name}")
                present_response += f"\n**Agent updated: {event.new_agent.name}**\n"
                continue
            elif event.type == "run_item_stream_event":
                if event.item.type == "tool_call_item":
                    print("-- Tool was called")
                    present_response += "\n**-- Tool was called**"
                elif event.item.type == "tool_call_output_item":
                    print(f"-- Tool output: {event.item.output}")
                    present_response += f"\n**-- Tool output: {event.item.output}**\n"
                # elif event.item.type == "message_output_item": # 如果完成后一次性输出
                #     print(f"\n-- Message output:\n {ItemHelpers.text_message_output(event.item)}")
                #     present_response += f"\n-- Message output:\n {ItemHelpers.text_message_output(event.item)}"
                else:
                    pass  # Ignore other event types
            history_gradio[-1] = {"role": "assistant", "content": present_response}
            history_llm[-1] = {"role": "assistant", "content": present_response}
            yield history_gradio, history_llm

    except Exception as e:
        logging.error("Exception encountered:", str(e))
        history_gradio.append({"role": "assistant", "content": f"出现错误,错误内容为: {str(e)}"})
        # print(history_llm)
        yield history_gradio, history_llm


async def translation_langgraph_inference(history_gradio: list[dict | gr.ChatMessage], history_llm: list[dict | State],
                                          new_topic: bool, stop_inference_flag: bool = False,
                                          graph: StateGraph | CompiledStateGraph = None,
                                          stream_mode: Literal['messages', 'updates'] = "updates",
                                          config: dict = None):
    try:
        present_message = history_llm[-1]

        if new_topic:
            input_message = {'messages': present_message}
        else:
            input_message = {"messages": history_llm}

        async for node_name, updates_think_content, updates_modelOutput, updates_finish_reason in langgraph_astream(
                graph=translation_graph,
                state=input_message,
                stream_mode=stream_mode,
                print_mode=["think", 'model_output'],
                config=config):
            gradio_message = gr.ChatMessage(
                role="assistant",
                content=f"## * graph_node: {node_name}\n",
            )
            history_gradio.append(gradio_message)
            # print(f'append.gradio_message:{history_gradio}')
            yield history_gradio, history_llm

            if stop_inference_flag:
                yield history_gradio, history_llm  # 先yield 再return ; 直接return history会导致history不输出
                return
            if updates_think_content:
                gradio_message = gr.ChatMessage(
                    role="assistant",
                    content=updates_think_content,
                    metadata={"title": "🧠 Thinking",
                              "log": f"@ graph node: {node_name}",
                              "status": "pending"}
                )
                history_gradio.append(gradio_message)
                # thinking 无需append history_llm
                yield history_gradio, history_llm

            if updates_modelOutput:
                loop_count = graph.get_state(config=config).values.get('loop_count', None)
                if loop_count:
                    content = ""
                    if "evaluator" in node_name:
                        content = f"### + {node_name}, 第{loop_count}次评估:\n"
                    if "translator" in node_name:
                        content = f"### + {node_name}, 第{loop_count}次翻译:\n"
                    gradio_message = gr.ChatMessage(
                        role="assistant",
                        content=content,
                    )
                    history_gradio.append(gradio_message)
                    yield history_gradio, history_llm
                gradio_message = gr.ChatMessage(
                    role="assistant",
                    content=updates_modelOutput,
                )
                history_gradio.append(gradio_message)
                #  graph的node之间，按照自有的workflow传递state;
                # history_llm.append(graph.get_state(config=config))
                yield history_gradio, history_llm

            if updates_finish_reason:
                if updates_finish_reason == "stop":
                    gradio_message = gr.ChatMessage(
                        role="assistant",
                        content="",
                        metadata={"title": "🧠 End Module Output",
                                  "status": "done"}
                    )

                if updates_finish_reason == "tool_calls":
                    gradio_message = gr.ChatMessage(
                        role="assistant",
                        content="",
                        metadata={"title": "🧠 End Tool Calls",
                                  "status": "done"}
                    )

                history_gradio.append(gradio_message)
                # thinking 无需append history_llm
                yield history_gradio, history_llm

        #  graph执行完毕之后, graph的state,装载入history_llm;histoy_gradio标记
        gradio_message = gr.ChatMessage(
            role="assistant",
            content=f"{graph.name},执行完成!",
        )
        history_gradio.append(gradio_message)
        history_llm = graph.get_state(config=config).values['messages']  # get_state输出list，包含thread_id下的全部state
        # history_llm.append(message)
        yield history_gradio, history_llm


    except Exception as e:
        logging.error("Exception encountered:", str(e))
        history_gradio.append({"role": "assistant", "content": f"出现错误,错误内容为: {str(e)}"})
        # print(history_llm)
        yield history_gradio, history_llm


async def translator_agents_inference(
        history_gradio: list[dict], history_llm: list[dict], new_topic: bool,
        translator_agent: Agent = None, evaluator_agent: Agent = None,
        stop_inference_flag: bool = False):
    """
    translator_agent为翻译模型，Stream输出翻译结果；再送入evaluator_agent评估模型。evaluator_agent的output_type的key: score及feedback.
    根据score的值[pass,needs_improvement,end]来判断是否需要进一步改进翻译，或者通过（合格），或者结束（end)
    :param history_gradio:
    :param history_llm:
    :param new_topic:
    :param translator_agent: 翻译模型Agent
    :param evaluator_agent: 评估模型Agent
    :param stop_inference_flag:
    :return:
    """
    try:
        while True:
            # stream 输出 translator_agent
            async for history_gradio, history_llm in openai_agents_inference(history_gradio, history_llm, new_topic,
                                                                             agent=translator_agent,
                                                                             stop_inference_flag=stop_inference_flag):
                yield history_gradio, history_llm

            # 同步输出 evaluator_agent
            evaluator_result = await Runner.run(evaluator_agent, history_llm)
            result: EvaluationFeedback = evaluator_result.final_output
            print(f"**Evaluator score:** {result.score}")
            history_gradio.append({"role": "assistant", "content": f"**Evaluator score: {result.score}**"})

            if result.score == "pass":
                print("**translation is good enough, exiting.**")
                history_gradio.append({"role": "assistant", "content": "**translation is good enough, exiting.**"})
                yield history_gradio, history_llm
                break
            elif result.score == "end":
                print("**evaluation progress comes to an end, exiting.**")
                history_gradio.append(
                    {"role": "assistant", "content": "**evaluation progress comes to an end, exiting.**"})
                yield history_gradio, history_llm
                break
            elif result.score == "needs_improvement":
                print(f"**Evaluator feedback:** {result.feedback}")
                history_gradio.append({"role": "assistant", "content": f"**Evaluator feedback**: {result.feedback}"})

                print("**Re-running with feedback**")
                history_gradio.append({"role": "assistant", "content": "**Re-running with feedback**"})

                # 以user身份，向translator_agents输入feedback:
                history_llm.append({"role": "user", "content": f"Feedback: {result.feedback}"})
                yield history_gradio, history_llm

        yield history_gradio, history_llm



    except Exception as e:
        logging.error("Exception encountered:", str(e))
        history_gradio.append({"role": "assistant", "content": f"出现错误,错误内容为: {str(e)}"})
        # print(history_llm)
        yield history_gradio, history_llm


def gemini_inference(
        history_gradio: list[dict], history_llm: list[dict], new_topic: bool, genai_client: genai.Client = None,
        model: str = None, stop_inference_flag: bool = False, ):
    try:
        google_search_tool = Tool(
            google_search=GoogleSearch()
        )
        #  当retry后，最后一条消息变成了Conent(parts=[...], role='model')
        present_message = history_llm[-1].get("content")
        if new_topic:
            # gemeni_model = genai.GenerativeModel(model)
            # streaming_chat = gemeni_model.start_chat(history_llm=None, )
            streaming_chat = genai_client.chats.create(model=model, history=None,
                                                       config=GenerateContentConfig(
                                                           tools=[google_search_tool],
                                                       )
                                                       )
        else:
            # history_llm包含present_message,取出present_message之后，剩余的为Chat的history
            streaming_chat = genai_client.chats.create(model=model, history=history_llm[:-1],
                                                       config=GenerateContentConfig(
                                                           tools=[google_search_tool],
                                                       )
                                                       )
            # print(f"gemini_inference present_message: {present_message}")
            # print(f"gemini_inference history_llm: {history_llm}")

        # present_message = get_last_user_messages(history_llm)
        response = streaming_chat.send_message_stream(present_message)
        history_llm = streaming_chat.get_history()

        present_response = ""
        history_gradio.append({"role": "assistant", "content": present_response})
        # history_llm.append({"role": "assistant", "content": present_response})
        for chunk in response:
            if stop_inference_flag:
                # print(f"return之前history:{history_llm}")
                yield history_gradio, history_llm  # 先yield 再return ; 直接return history会导致history不输出
                return
            out = chunk.text
            if out:
                present_response += out  # extract text from streamed litellm chunks
                history_gradio[-1] = {"role": "assistant", "content": present_response}
                # history_llm[-1] = {"role": "assistant", "content": present_response}
                yield history_gradio, history_llm
    except Exception as e:
        logging.error("Exception encountered:", str(e))
        history_gradio.append({"role": "assistant", "content": f"出现错误,错误内容为: {str(e)}"})
        yield history_gradio, history_llm


def glm_inference(history_gradio: list[dict], history_llm: list[dict],
                  new_topic: bool, model: str = None, zhipuai_client: ZhipuAI = None,
                  stop_inference_flag: bool = False, ):
    # global present_message
    try:
        if new_topic:
            glm_prompt = get_last_user_messages(history_llm)
        else:
            glm_prompt = history_llm

        present_response = ""
        history_gradio.append({"role": "assistant", "content": present_response})
        history_llm.append({"role": "assistant", "content": present_response})
        # print(f"present_message:{present_message};glm_prompt之后:{glm_prompt}")
        for chunk in zhipuai_messages_api(glm_prompt, model=model, zhipuai_client=zhipuai_client):
            if stop_inference_flag:
                # print(f"return之前history:{history_llm}")
                yield history_gradio, history_llm  # 先yield 再return ; 直接return history会导致history不输出
                return
            out = chunk.choices[0].delta.content
            if out:
                present_response += out  # extract text from streamed litellm chunks
                history_gradio[-1] = {"role": "assistant", "content": present_response}
                history_llm[-1] = {"role": "assistant", "content": present_response}
                yield history_gradio, history_llm
    except Exception as e:
        logging.error("Exception encountered:", str(e))
        history_gradio.append({"role": "assistant", "content": f"出现错误,错误内容为: {str(e)}"})
        history_llm.append({"role": "assistant", "content": f"出现错误,错误内容为: {str(e)}"})
        yield history_gradio, history_llm


def zhipuai_messages_api(messages: str | list[dict], model: str, zhipuai_client: ZhipuAI = None):
    prompt = []
    if "alltools" in model:
        if isinstance(messages, str):
            prompt.append(
                [{"role": "user", "content": [{"type": "text", "text": messages}]}]
            )
        elif isinstance(messages, list) and all(
                isinstance(message, dict) for message in messages
        ):
            for message in messages:
                prompt.append(
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": message["content"]}],
                    }
                )

        tools = [{"type": "web_browser"}]
    else:
        if isinstance(messages, str):
            prompt.append([{"role": "user", "content": messages}])
        elif isinstance(messages, list) and all(
                isinstance(message, dict) for message in messages
        ):
            prompt = messages

        tools = [
            {
                "type": "web_search",
                "web_search": {
                    "enable": True
                    # "search_result": True,
                },
            }
        ]

    # 同步调用 （stream模式）
    response = zhipuai_client.chat.completions.create(
        model=model,  # 填写需要调用的模型名称
        tools=tools,
        messages=prompt,
        stream=True,
    )
    return response


def vote(data: gr.LikeData):
    if data.liked:
        logging.info(f"You upvoted this response:  {data.index}, {data.value} ")
    else:
        logging.info(f"You downvoted this response: {data.index}, {data.value}")


def handle_undo(history, undo_data: gr.UndoData):
    return history[: undo_data.index], history[undo_data.index]["content"]


async def handle_retry(
        history_gradio: list[dict],
        history_llm: list[dict],
        new_topic: bool,
        model: str,
        stop_inference_flag: bool,
        retry_data: gr.RetryData,
):
    last_history_gradio = history_gradio[: retry_data.index + 1]
    last_history_llm = undo_history(history_llm)
    # print(f"last_history_llm:{last_history_llm}")
    # print(f"last_history_gradio:{last_history_gradio}")
    if 'agent' in model or 'translator' in model:  # 异步inference
        async for _gradio, _llm in async_inference(last_history_gradio, last_history_llm, new_topic, model,
                                                   stop_inference_flag):
            yield _gradio, _llm
    else:  # 同步inference
        for _gradio, _llm in inference(last_history_gradio, last_history_llm, new_topic, model,
                                       stop_inference_flag):
            yield _gradio, _llm


def stop_inference_flag_True():
    stop_inference_flag = True
    return stop_inference_flag


def stop_inference_flag_False():
    stop_inference_flag = False
    return stop_inference_flag


def on_selectDropdown(history_gradio: list[dict], history_llm: list[dict], evt: gr.SelectData):
    # global streaming_chat
    # global model
    model = evt.value
    logging.info(f"下拉菜单选择了{evt.value},当前状态是evt.selected:{evt.selected}")
    history_gradio: list[dict] = []
    history_llm: list[dict] = []
    return history_gradio, history_llm, model


def on_topicRadio(value, evt: gr.EventData):
    logging.error(f"The {evt.target} component was selected, and its value was {value}.")


def gradio_UI():
    with gr.Blocks() as demo:
        gr.Markdown("# Translator Robot 🤗")
        chatbot = gr.Chatbot(
            elem_id="Evaluator&Translator Chatbot",
            label="Hi,look at here!",
            type="messages",
            placeholder="# **想翻译点什么?**",
            show_copy_button=True,
            show_copy_all_button=True,
            show_share_button=True,
            autoscroll=True,
            height=400,
            render_markdown=True,
            avatar_images=(
                None,
                "https://em-content.zobj.net/source/twitter/376/hugging-face_1f917.png",
            ),
        )

        history_llm = gr.State([])
        model = gr.State("langgraph_translation_graph")  # 初始值
        stop_inference_flag = gr.State(False)

        stop_inference_button = gr.Button(
            value="停止推理",
            variant="secondary",
            size="sm",
            visible=False,
            interactive=True,
            min_width=100,
        )

        with gr.Row():
            topicCheckbox = gr.Checkbox(
                label="新话题", show_label=True, scale=1, min_width=90
            )
            chat_input = gr.MultimodalTextbox(
                # value= {"text": "sample text", "files": [{'path': "files/ file. jpg", 'orig_name': "file. jpg", 'url': "http:// image_url. jpg ", 'size': 100}]},
                file_types=["file"],
                interactive=True,
                file_count="multiple",
                lines=1,
                placeholder="Enter gradio_message or upload file...",
                show_label=False,
                scale=20,
            )
            models_dropdown = gr.Dropdown(
                choices=[
                    "langgraph_translation_graph",
                    "openAI-Agents",
                    "glm-4-flash",
                    "glm-4-air",
                    "glm-4-plus",
                    "glm-4-alltools",

                ],
                value="langgraph_translation_graph",
                multiselect=False,
                scale=1,
                show_label=False,
                label="models",
                interactive=True,
            )

        models_dropdown.select(on_selectDropdown, [chatbot, history_llm], [chatbot, history_llm, model])
        stop_inference_button.click(stop_inference_flag_True, None, stop_inference_flag)
        chatbot.undo(handle_undo, chatbot, [chatbot, chat_input])
        chatbot.like(vote, None, None)
        chatbot.retry(
            handle_retry,
            [chatbot, history_llm, topicCheckbox, model, stop_inference_flag],
            [chatbot, history_llm],
        )

        chat_msg = chat_input.submit(
            add_message,
            [chatbot, history_llm, chat_input, model],
            [chatbot, history_llm, chat_input, stop_inference_button],
            queue=False,
        )
        bot_msg = chat_msg.then(
            async_inference,  # 如果是glm即gemini同步函数推理的模型，可以使用inference函数，这样可以实现流式输出；使用async_inference函数，可以正常推理，但不能实现流式输出。
            [chatbot, history_llm, topicCheckbox, model, stop_inference_flag],
            [chatbot, history_llm],
            api_name="bot_response",
        )

        bot_msg.then(lambda: gr.Checkbox(value=False), None, [topicCheckbox])
        bot_msg.then(lambda: gr.MultimodalTextbox(interactive=True), None, [chat_input])
        bot_msg.then(
            lambda: gr.Button(visible=False, interactive=True),
            None,
            [stop_inference_button],
        )
        bot_msg.then(stop_inference_flag_False, None, stop_inference_flag)

    return demo


if __name__ == "__main__":
    # zhipuAI client:
    zhipuai_client = ZhipuAI(api_key=os.getenv("ZHIPUAI_API_KEY"))
    # # 测试zhipuai
    # model = "glm-4-flash"
    # response = zhipuai_api("请联网搜索，回答：美国大选最新情况", model=model)
    # for chunk_size in response:
    #     out = chunk_size.choices[0].delta.content

    # gemini client:
    genai_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    # openai_agents:
    # 1
    # agent_client = qwen_VL()
    # 2
    # translator_agent, evaluator_agent = gemini_translate_agent()
    # translator = gemini_translator(translate_agent=translate_agent.agent,
    #                                  evaluate_agent=evaluate_agent.agent,
    #                                  )
    # langgraph_graph:
    thread_id = uuid.uuid4()  # 128 位的随机数，通常用 32 个十六进制数字表示
    config = {"configurable": {"thread_id": thread_id},
              "recursion_limit": 20}
    translation_graph = translation_graph(State, name="translation_graph", checkpointer=checkpointer)

    demo = gradio_UI()
    demo.queue().launch(server_name='0.0.0.0')
