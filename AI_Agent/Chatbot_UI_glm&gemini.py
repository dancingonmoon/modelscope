import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))  # 添加项目根目录

import asyncio

import gradio as gr  # gradio 5.5.0 需要python 3.10以上
from gradio.data_classes import GradioModel,GradioRootModel, FileData, FileDataDict


from zhipuai import ZhipuAI
# import google.generativeai as genai # 旧版
from google import genai  # 新版
from agents import OpenAIChatCompletionsModel, Agent, Runner, set_default_openai_client, set_tracing_disabled, \
    function_tool, TResponseInputItem, ItemHelpers
from openai.types.responses import ResponseTextDeltaEvent
from openAI_Agents.openAI_Agents_practice import openAI_Agents_create, save2file, _Qwen_MT_func

import base64
from typing import Literal
import json
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


def gradio_msg2LLM_msg(gradio_msg: dict = None,
                       msg_format: Literal["openai_agents", "gemini", "glm"] = "openai_agents",
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

    text = gradio_msg.get("text", None)
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

    return input_item


def add_message_v2(history_gradio: list[gr.ChatMessage] = None, history_llm: list[dict] = None,
                   gradio_message: str|dict[str,str|list]=None, model:str= None):

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
                history_gradio.append({"role": "user", "content": {"path": file.get("path"), "url": file.get("url"),"mime_type": file.get("mime_type"), }})
    if text is not None:
        history_gradio.append({"role": "user", "content": text})

    if 'agent' in model.lower():
       llm_message = gradio_msg2LLM_msg(gradio_message, msg_format="openai_agents")
    elif 'glm' in model.lower():
        llm_message = gradio_msg2LLM_msg(gradio_message, msg_format="glm", zhipuai_client=zhipuai_client)
    elif 'gemini' in model.lower():
        llm_message = gradio_msg2LLM_msg(gradio_message, msg_format="gemini", genai_client=genai_client)
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

def undo_history(history:list[dict],):
    """
    移除history中最后一个role不是user的那组消息。最后一组消息role，如为user， 不变化；否则，/assistant/system/model,则移除该组消息
    :param history:
    :return:
    """
    index = -1
    if history:  # 非空列表
        for index, msg in enumerate(reversed(history)):
            if msg.get("role", None) != "user":
                continue
            else:
                break
    return history[:index+1+1]
def inference(history_gradio: list[dict], history_llm: list[dict], new_topic: bool, model:str=None, stop_inference: bool = False ):
    if 'gemini' in model.lower():
        for history_gradio, history_llm in gemini_inference(history_gradio, history_llm, new_topic, model=model,genai_client=genai_client,  stop_inference_flag=stop_inference):
            yield history_gradio, history_llm
    elif 'glm' in model.lower():
        for history_gradio, history_llm in glm_inference(history_gradio, history_llm, new_topic, model,zhipuai_client=zhipuai_client, stop_inference_flag=stop_inference):
            yield history_gradio, history_llm
    elif 'agent' in model.lower():
        return openai_agents_inference(history_gradio, new_topic, agent=agent_client, stop_inference_flag=stop_inference)


async def openai_agents_inference(
        history: list, new_topic: bool, agent: Agent = None):
    try:
        if new_topic:
            #  present_message取自全局变量
            response = Runner.run_streamed(agent, [present_message])
        else:
            # response = asyncio.run(Runner.run(agent, history_llm[:-1].append(present_message)))
            response = Runner.run_streamed(agent, history)  # 检查是否history与chatbot_display一致

        present_response = ""
        history.append({"role": "assistant", "content": present_response})
        async for event in response.stream_events():
            if stop_inference_flag:
                # print(f"return之前history:{history_llm}")
                yield history  # 先yield 再return ; 直接return history会导致history不输出
                return
            if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
                print(event.data.delta, end="", flush=True)
                present_response += event.data.delta
            elif event.type == "agent_updated_stream_event":
                print(f"Agent updated: {event.new_agent.name}")
                present_response += f"\n{event.new_agent.name}"
                continue
            elif event.type == "run_item_stream_event":
                if event.item.type == "tool_call_item":
                    print("-- Tool was called")
                    present_response += "\n-- Tool was called"
                elif event.item.type == "tool_call_output_item":
                    print(f"-- Tool output: {event.item.output}")
                    present_response += f"\n-- Tool output: {event.item.output}"
                elif event.item.type == "message_output_item":
                    print(f"-- Message output:\n {ItemHelpers.text_message_output(event.item)}")
                    present_response += f"\n-- Message output:\n {ItemHelpers.text_message_output(event.item)}"
                else:
                    pass  # Ignore other event types
            history[-1] = {"role": "assistant", "content": present_response}
            yield history

    except Exception as e:
        logging.error("Exception encountered:", str(e))
        history.append({"role": "assistant", "content": f"出现错误,错误内容为: {str(e)}"})
        # print(history_llm)
        yield history


def gemini_inference(
        history_gradio: list[dict], history_llm:list[dict], new_topic: bool, genai_client: genai.Client = None, model: str = None, stop_inference_flag: bool = False,):
    # global streaming_chat


    try:
        present_message = history_llm[-1].get('content')
        if new_topic:
            # gemeni_model = genai.GenerativeModel(model)
            # streaming_chat = gemeni_model.start_chat(history_llm=None, )
            streaming_chat = genai_client.chats.create(model=model, history=None)
        else:
            # history_llm包含present_message,取出present_message之后，剩余的为Chat的history
            streaming_chat = genai_client.chats.create(model=model, history=history_llm[:-1])
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
                yield history_gradio, history_llm # 先yield 再return ; 直接return history会导致history不输出
                return
            out = chunk.text
            if out:
                present_response += out  # extract text from streamed litellm chunks
                history_gradio[-1] = {"role": "assistant", "content": present_response}
                # history_llm[-1] = {"role": "assistant", "content": present_response}
                yield history_gradio,history_llm
    except Exception as e:
        logging.error("Exception encountered:", str(e))
        history_gradio.append({"role": "assistant", "content": f"出现错误,错误内容为: {str(e)}"})
        yield history_gradio, history_llm


def glm_inference(history_gradio:list[dict], history_llm: list[dict],
                  new_topic: bool,  model:str=None, zhipuai_client: ZhipuAI = None, stop_inference_flag: bool = False,):
    global present_message
    try:
        if new_topic:
            glm_prompt = get_last_user_messages(history_llm)
            # glm_prompt = present_message  # 取自全局变量
        else:
            # glm模型文件作为prompt，非通过type方式，而是通过件文件内容放在到prompt内
            # history中连续的{"role": "user", "content"：""},是文件链接或内容的删除
            # glm_prompt = [
            #     gradio_message
            #     for gradio_message in history_llm[:-1]  # 最后一条直接取自全局变量present_message
            #     if not (
            #             gradio_message["role"] == "user" and isinstance(gradio_message["content"], tuple)
            #     )
            # ]
            # glm_prompt.extend(present_message)
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

    response = zhipuai_client.chat.completions.create(
        model=model,  # 填写需要调用的模型名称
        tools=tools,
        messages=prompt,
        stream=True,
    )
    return response


def openai_agents():
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
    return Qwen3_agent


def vote(data: gr.LikeData):
    if data.liked:
        logging.info(f"You upvoted this response:  {data.index}, {data.value} ")
    else:
        logging.info(f"You downvoted this response: {data.index}, {data.value}")


def handle_undo(history, undo_data: gr.UndoData):
    return history[: undo_data.index], history[undo_data.index]["content"]


def handle_retry(
        history_gradio: list[dict],
        history_llm: list[dict],
        new_topic: bool,
        model: str,
        stop_inference_flag: bool,
        retry_data: gr.RetryData,
):
    last_history_gradio = history_gradio[: retry_data.index+1]
    last_history_llm = undo_history(history_llm)
    # print(f"last_history_llm:{last_history_llm}")
    # print(f"last_history_gradio:{last_history_gradio}")
    for _gradio, _llm in inference(last_history_gradio, last_history_llm, new_topic, model,stop_inference_flag):
        yield _gradio, _llm
def stop_inference_flag_True():
    stop_inference_flag = True
    return stop_inference_flag

def stop_inference_flag_False():
    stop_inference_flag = False
    return stop_inference_flag


def on_selectDropdown(evt: gr.SelectData) -> None:
    # global streaming_chat
    # global model
    model = evt.value
    logging.info(f"下拉菜单选择了{evt.value},当前状态是evt.selected:{evt.selected}")
    # if 'gemini' in model:
    #     try:
    #         # gemeni_model = genai.GenerativeModel(model)
    #         streaming_chat = genai_client.chats.create(model=model,history_llm=None, )
    #     except Exception as e:
    #         logging.error(e.args)
    # print(f"model:{model}")
    return model


def on_topicRadio(value, evt: gr.EventData):
    logging.error(f"The {evt.target} component was selected, and its value was {value}.")


def gradio_UI():
    with gr.Blocks() as demo:
        gr.Markdown("# 多模态Robot 🤗")
        chatbot = gr.Chatbot(
            elem_id="Multimodal Chatbot",
            label="Hi,look at here!",
            type="messages",
            placeholder="# **想问点什么?**",
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
        model = gr.State("glm-4-flash") # 初始值
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
                    "openAI-Agents",
                    "glm-4-flash",
                    "glm-4-air",
                    "glm-4-plus",
                    "glm-4-alltools",
                    "gemini-1.5-pro",
                    "gemini-2.0-flash",
                    "gemini-2.0-flash",
                    "gemini-2.5-flash-preview-05-20",
                    "gemini-2.5-pro:非免费",

                ],
                value="glm-4-flash",
                multiselect=False,
                scale=1,
                show_label=False,
                label="models",
                interactive=True,
            )

        models_dropdown.select(on_selectDropdown, None, model)
        stop_inference_button.click(stop_inference_flag_True, None, stop_inference_flag)
        chatbot.undo(handle_undo, chatbot, [chatbot, chat_input])
        chatbot.like(vote, None, None)
        chatbot.retry(
            handle_retry,
            [chatbot, history_llm, topicCheckbox, model, stop_inference_flag],
            [chatbot,history_llm],
        )

        chat_msg = chat_input.submit(
            add_message_v2,
            [chatbot, history_llm,chat_input,model],
            [chatbot, history_llm, chat_input, stop_inference_button],
            queue=False,
        )
        bot_msg = chat_msg.then(
            inference,
            [chatbot, history_llm,topicCheckbox,model, stop_inference_flag],
            [chatbot,history_llm],
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
    # for chunk in response:
    #     out = chunk.choices[0].delta.content

    # gemini client:
    genai_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    # openai_agents:
    agent_client = openai_agents()
    # 全局变量

    stop_inference_flag = False  # 停止推理初始值，全局变量
    # model = 'glm-4-flash'  # 初始假定值，作为全局变量
    # model = 'openai_agents'  # 初始假定值，作为全局变量
    # streaming_chat = None  # gemini直播聊天对象；全局变量
    # present_message = {}  # 当前消息，全局变量;因为chatbot显示的message与送入模型的message会有所不同;

    demo = gradio_UI()

    demo.queue().launch(server_name='0.0.0.0')
