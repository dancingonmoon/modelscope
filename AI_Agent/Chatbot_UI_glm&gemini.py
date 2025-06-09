import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))  # 添加项目根目录

from agents import OpenAIChatCompletionsModel, Agent, Runner, set_default_openai_client, set_tracing_disabled, \
    function_tool, TResponseInputItem, ItemHelpers
from openai.types.responses import ResponseTextDeltaEvent
import gradio as gr  # gradio 5.5.0 需要python 3.10以上
from zhipuai import ZhipuAI
import base64
from typing import Literal
import json
# import google.generativeai as genai # 旧版
from google import genai  # 新版
# from openAI_Agents.openAI_Agents_practice import openAI_Agents_create
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


def gradio_msg2LLM_msg(gradio_msg: dict = None,
                       msg_format: Literal["openai_agents", "gemini", "glm"] = "openai_agents",
                       geniai_client: genai.Client = None, zhipuai_client: ZhipuAI = None):
    """
    一次gradio的多媒体message(包含text,file)，转换成各类LLM要求的message格式
    :param gradio_msg: gradio.MultiModalText.value,例如: {"text": "sample text", "files": [{path: "files/file.jpg", orig_name: "file.jpg", url: "http://image_url.jpg", size: 100}]}
    :param msg_format: "openai_agents", "gemini", "glm"
    :param geniai_client: genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))
    :param zhipuai_client: ZhipuAI(api_key=os.environ.get("ZHUIPU_API_KEY"))
    :return:  与msg_format兼容的message格式，以及History
    """
    supported_img = [".bmp", ".png", ".jpe", ".jpeg", ".jpg", ".tif", ".tiff", ".webp", ".heic"]
    jpg_variant = ['.jpe', '.jpeg', '.jpg']
    tif_variant = ['.tif', '.tiff']
    contents = []
    input_item = []

    text = gradio_msg.get("text", None)
    files = gradio_msg.get("files", None)
    # openAI-Agents message 格式处理:
    if msg_format == "openai_agents":
        if files is not None:
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
                        print("✅ 不合适的IMG格式")
                        # break
                else:
                    print("✅ 文档路径不存在")
                    # break

            input_item.append({"role": "user", "content": contents})
        input_item.append({"role": "user", "content": text})

    # gemini message 格式处理:
    elif msg_format == "gemini":
        # geniai_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        contents = []
        for file in files:
            file_path = Path(file)
            if file_path.exists() and file_path.is_file():
                # Gemini 1.5 Pro 和 1.5 Flash 最多支持 3,600 个文档页面。文档页面必须采用以下文本数据 MIME 类型之一：
                # PDF - application/pdf,JavaScript - application/x-javascript、text/javascript,Python - application/x-python、text/x-python,
                # TXT - text/plain,HTML - text/html, CSS - text/css,Markdown - text/md,CSV - text/csv,XML - text/xml,RTF - text/rtf
                content = geniai_client.files.upload(path=file)  # 环境变量缺省设置GEMINI_API_KEY
                contents.append(content)
            else:
                print("✅ 文档路径不存在")
                # break
        contents.append(text)
        input_item.append({"role": "user", "content": contents})

    # glm message 格式处理:
    elif msg_format == "glm":
        # zhipuai_client = ZhipuAI(api_key=os.getenv("ZHIPU_API_KEY"))
        contents = "请结合以下文件或图片内容回答：\n\n"  # for glm
        for file, file_No in enumerate(files):
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
        contents = f'{text}.'.join(contents)
        input_item.append({"role": "user", "content": contents})

    return input_item


def add_message_v2(history: list[dict] = None, message: dict = None):
    global present_message
    global model

    if history is None:
        history = [{
            "role": "user",
            "content": "",
        }]
    text = message.get("text")
    files = message.get("files")
    if 'agent' in model:
        present_message = gradio_msg2LLM_msg(message, msg_format="openai_agents")
    elif 'glm' in model:
        present_message = gradio_msg2LLM_msg(message, msg_format="glm", zhipuai_client=zhipuai_client)
    elif 'gemini' in model:
        present_message = gradio_msg2LLM_msg(message, msg_format="gemini", geniai_client=genai_client)

    # history.append(present_message)
    chatbot_display_prompt = text
    if files and 'glm' in model:  # 假设gemini 以及 agent 模型的message格式，在gradio的chatbot_display上，不因files而显示大量内容（可以显示图片）
        chatbot_display_prompt = f"{text}\n\n({files['path']})"
    history.append({
        "role": "user",
        "content": chatbot_display_prompt,
    })  # chatbot上只显示text ,不显示files_prompt,以及files_boject, 避免显示内容过长
    return (
        history,
        gr.MultimodalTextbox(value=None, interactive=False),
        gr.Button(interactive=True, visible=True),

    )


def add_message(history, message):
    global present_message
    global model
    present_message = {
        "role": "user",
        "content": "",
    }
    if history is None:
        history = [present_message]
    text = message.get("text")
    files = message.get("files")
    if files:
        files_prompt = "请结合以下文件或图片内容回答：\n\n"  # for glm
        files_object = []  # for gemini
        for file_No, file in enumerate(files):
            history.append(
                {"role": "user", "content": {"path": file, "alt_text": file}}
            )  # chatbot上先显示该图片
            # 文件处理
            try:
                if 'gemini' in model:
                    # Gemini 1.5 Pro 和 1.5 Flash 最多支持 3,600 个文档页面。文档页面必须采用以下文本数据 MIME 类型之一：
                    # PDF - application/pdf,JavaScript - application/x-javascript、text/javascript,Python - application/x-python、text/x-python,
                    # TXT - text/plain,HTML - text/html, CSS - text/css,Markdown - text/md,CSV - text/csv,XML - text/xml,RTF - text/rtf
                    file_object = genai.upload_file(path=file)
                    files_object.append(file_object)

                elif 'glm' in model:
                    # 格式限制：.PDF .DOCX .DOC .XLS .XLSX .PPT .PPTX .PNG .JPG .JPEG .CSV .PY .TXT .MD .BMP .GIF
                    # 大小：单个文件50M、总数限制为100个文件
                    file_object = zhipuai_client.files.create(
                        file=Path(file), purpose="file-extract"
                    )
                    # 获取文本内容
                    file_content = json.loads(
                        zhipuai_client.files.content(file_id=file_object.id).content
                    )["content"]

                    if file_content is None or file_content == "":
                        files_prompt += f"第{file_No + 1}个文件或图片内容无可提取之内容\n\n"
                    else:
                        files_prompt += f"第{file_No + 1}个文件或图片内容如下：\n" f"{file_content}\n\n"

            except Exception as e:
                logging.error(e.args)
                present_message = {
                    "role": "assistant",
                    "content": e.args[0],
                }
                history.append(present_message)
                return (history, gr.MultimodalTextbox(
                    value=None, interactive=False
                ),
                        None)  # 因此此处输出的仅仅是错误，但不影响后续程序执行，导致模型输入部分是空值，出错

        if text is None or text == "":
            if 'gemini' in model:
                present_message = {
                    "role": "user",
                    "content": files_object}
            elif 'glm' in model:
                present_message = {
                    "role": "user",
                    "content": files_prompt,  # GLM模型不支持content里面file 或者Path
                }

        else:
            if 'gemini' in model:
                present_message = {
                    "role": "user",
                    "content": [text] + files_object,  # 列表合并
                }
            elif 'glm' in model:
                present_message = {
                    "role": "user",
                    "content": f"{text},{files_prompt}",  # GLM模型不支持content里面file 或者Path
                }
    else:
        if text is not None:
            present_message = {
                "role": "user",
                "content": f"{text}",
            }
    # history.append(present_message)
    history.append({
        "role": "user",
        "content": f"{text}",
    })  # chatbot上只显示text ,不显示files_prompt,以及files_boject
    return (
        history,
        gr.MultimodalTextbox(value=None, interactive=False),
        gr.Button(interactive=True, visible=True),
    )


def inference(history: list, new_topic: bool):
    if 'gemini' in model:
        yield from gemini_inference(history, new_topic)
    elif 'glm' in model:
        yield from glm_inference(history, new_topic)
    elif 'agent' in model:
        yield from openai_agents_inference(history, new_topic)


def openai_agents_inference(
        history: list, new_topic: bool, agent: Agent = None):
    try:
        if new_topic:
            #  present_message取自全局变量
            response = Runner.run_streamed(agent, [present_message])
        else:
            # response = asyncio.run(Runner.run(agent, history[:-1].append(present_message)))
            response = Runner.run_streamed(agent, history)  # 检查是否history与chatbot_display一致

        present_response = ""
        history.append({"role": "assistant", "content": present_response})
        for event in response.stream_events():
            if stop_inference_flag:
                # print(f"return之前history:{history}")
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
        # print(history)
        yield history


def gemini_inference(
        history: list, new_topic: bool, genai_client: genai.Client = None):
    # global streaming_chat

    try:
        if new_topic:
            # gemeni_model = genai.GenerativeModel(model)
            # streaming_chat = gemeni_model.start_chat(history=None, )
            streaming_chat = genai_client.chats.create(model=model, history=None)
        else:
            streaming_chat = genai_client.chats.create(model=model, history=history[:-1].append(present_message))

        # present_message = history[-1]['content'] # present_message取自全局变量
        response = streaming_chat.send_message_stream(present_message['content'])

        present_response = ""
        history.append({"role": "assistant", "content": present_response})
        for chunk in response:
            if stop_inference_flag:
                # print(f"return之前history:{history}")
                yield history  # 先yield 再return ; 直接return history会导致history不输出
                return
            out = chunk.text
            if out:
                present_response += out  # extract text from streamed litellm chunks
                history[-1] = {"role": "assistant", "content": present_response}
                yield history
    except Exception as e:
        logging.error("Exception encountered:", str(e))
        history.append({"role": "assistant", "content": f"出现错误,错误内容为: {str(e)}"})
        # print(history)
        yield history


# def openai_agents_inference(history: list, new_topic: bool):

def glm_inference(
        history: list, new_topic: bool, zhipu_client: ZhipuAI = None):
    global present_message
    try:
        if new_topic:
            # present_message = [history[-1]]
            glm_prompt = [present_message]  # 取自全局变量
        else:
            # glm模型文件作为prompt，非通过type方式，而是通过件文件内容放在到prompt内
            # history中连续的{"role": "user", "content"：""},是文件链接或内容的删除
            glm_prompt = [
                message
                for message in history[:-1]  # 最后一条直接取自全局变量present_message
                if not (
                        message["role"] == "user" and isinstance(message["content"], tuple)
                )
            ]
            glm_prompt.append(present_message)

        present_response = ""
        history.append({"role": "assistant", "content": present_response})
        for chunk in zhipuai_messages_api(glm_prompt, model=model,zhipuai_client=zhipuai_client):
            if stop_inference_flag:
                # print(f"return之前history:{history}")
                yield history  # 先yield 再return ; 直接return history会导致history不输出
                return
            out = chunk.choices[0].delta.content
            if out:
                present_response += out  # extract text from streamed litellm chunks
                history[-1] = {"role": "assistant", "content": present_response}
                yield history
    except Exception as e:
        logging.error("Exception encountered:", str(e))
        history.append({"role": "assistant", "content": f"出现错误,错误内容为: {str(e)}"})
        # print(history)
        yield history


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


def vote(data: gr.LikeData):
    if data.liked:
        logging.error(f"You upvoted this response:  {data.index}, {data.value} ")
    else:
        logging.error("You downvoted this response: " + data.value)
        logging.error(f"You downvoted this response: {data.index}, {data.value}")


def handle_undo(history, undo_data: gr.UndoData):
    return history[: undo_data.index], history[undo_data.index]["content"]


def handle_retry(
        history: str | list[dict],
        new_topic: bool,
        retry_data: gr.RetryData,
):
    new_history = history[: retry_data.index]
    previous_prompt = history[retry_data.index]
    new_history.append(previous_prompt)
    if 'glm' in model:
        yield from glm_inference(new_history, new_topic)
    elif 'gemini' in model:
        yield from gemini_inference(new_history, new_topic)


def stop_inference_flag_True():
    global stop_inference_flag
    stop_inference_flag = True


def stop_inference_flag_False():
    global stop_inference_flag
    stop_inference_flag = False


def on_selectDropdown(evt: gr.SelectData) -> None:
    # global streaming_chat
    global model
    model = evt.value
    logging.info(f"下拉菜单选择了{evt.value},当前状态是evt.selected:{evt.selected}")
    # if 'gemini' in model:
    #     try:
    #         # gemeni_model = genai.GenerativeModel(model)
    #         streaming_chat = genai_client.chats.create(model=model,history=None, )
    #     except Exception as e:
    #         logging.error(e.args)


def gradio_UI():
    with gr.Blocks() as demo:
        gr.Markdown("# 多模态Robot 🤗")
        chatbot = gr.Chatbot(
            elem_id="Multimodal Chatbot",
            label="Hi,look at here!",
            bubble_full_width=False,
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
                placeholder="Enter message or upload file...",
                show_label=False,
                scale=20,
            )
            models_dropdown = gr.Dropdown(
                choices=[
                    "glm-4-flash",
                    "glm-4-air",
                    "glm-4-plus",
                    "glm-4-alltools",
                    "gemini-1.5-flash",
                    "gemini-1.5-pro",
                    "gemini-1.5-flash-latest",
                    "gemini-1.5-pro-latest",
                ],
                value="glm-4-flash",
                multiselect=False,
                scale=1,
                show_label=False,
                label="models",
                interactive=True,
            )

        models_dropdown.select(on_selectDropdown, None, None)
        stop_inference_button.click(stop_inference_flag_True, None, None)
        chatbot.undo(handle_undo, chatbot, [chatbot, chat_input])
        chatbot.like(vote, None, None)
        chatbot.retry(
            handle_retry,
            [chatbot, topicCheckbox],
            [chatbot],
        )

        chat_msg = chat_input.submit(
            add_message_v2,
            [chatbot, chat_input],
            [chatbot, chat_input, stop_inference_button],
            queue=False,
        )
        bot_msg = chat_msg.then(
            inference,
            [chatbot, topicCheckbox],
            [chatbot],
            api_name="bot_response",
        )
        bot_msg.then(lambda: gr.Checkbox(value=False), None, [topicCheckbox])
        bot_msg.then(lambda: gr.MultimodalTextbox(interactive=True), None, [chat_input])
        bot_msg.then(
            lambda: gr.Button(visible=False, interactive=True),
            None,
            [stop_inference_button],
        )
        bot_msg.then(stop_inference_flag_False, None, None)

        return demo


def on_topicRadio(value, evt: gr.EventData):
    logging.error(f"The {evt.target} component was selected, and its value was {value}.")


if __name__ == "__main__":
    # zhipuAI client:
    zhipuai_client = ZhipuAI(api_key=os.getenv("ZHIPUAI_API_KEY"))
    # # 测试zhipuai
    # model = "glm-4-flash"
    # response = zhipuai_api("请联网搜索，回答：美国大选最新情况", model=model)
    # for chunk in response:
    #     out = chunk.choices[0].delta.content

    # gemini client:
    genai_client = genai.Client(api_key="GEMINI_API_KEY")

    # 全局变量
    stop_inference_flag = False  # 停止推理初始值，全局变量
    model = 'glm-4-flash'  # 初始假定值，作为全局变量
    # streaming_chat = None  # gemini直播聊天对象；全局变量
    present_message = {}  # 当前消息，全局变量;因为chatbot显示的message与送入模型的message会有所不同;

    demo = gradio_UI()

    demo.queue().launch(server_name='0.0.0.0')
