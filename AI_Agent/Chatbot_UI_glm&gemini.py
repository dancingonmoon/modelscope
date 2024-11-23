import gradio as gr  # gradio 5.5.0 需要python 3.10以上
from zhipuai import ZhipuAI
from GLM.GLM_callFunc import config_read
from pathlib import Path
import json
import google.generativeai as genai
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


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
                None ) # 因此此处输出的仅仅是错误，但不影响后续程序执行，导致模型输入部分是空值，出错


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
                    "content": [text] + files_object, # 列表合并
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
            }) # chatbot上只显示text ,不显示files_prompt,以及files_boject
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


def gemini_inference(
        history: list, new_topic: bool, ):
    global streaming_chat
    try:
        if new_topic:
            gemeni_model = genai.GenerativeModel(model)
            streaming_chat = gemeni_model.start_chat(history=None, )

        # present_message = history[-1]['content'] # present_message取自全局变量
        response = streaming_chat.send_message(present_message['content'], stream=True)

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


def glm_inference(
        history: list, new_topic: bool):
    global present_message
    try:
        if new_topic:
            # present_message = [history[-1]]
            glm_prompt = [present_message] # 取自全局变量
        else:
            # glm模型文件作为prompt，非通过type方式，而是通过件文件内容放在到prompt内
            # history中连续的{"role": "user", "content"：""},是文件链接或内容的删除
            glm_prompt = [
                message
                for message in history[:-1] # 最后一条直接取自全局变量present_message
                if not (
                        message["role"] == "user" and isinstance(message["content"], tuple)
                )
            ]
            glm_prompt.append(present_message)

        present_response = ""
        history.append({"role": "assistant", "content": present_response})
        for chunk in zhipuai_messages_api(glm_prompt, model=model):
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


def zhipuai_messages_api(messages: str | list[dict], model: str):
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



def on_selectDropdown(evt: gr.SelectData)-> None:
    global streaming_chat
    global model
    model = evt.value
    logging.info(f"下拉菜单选择了{evt.value},当前状态是evt.selected:{evt.selected}")
    if 'gemini' in model:
        try:
            gemeni_model = genai.GenerativeModel(model)
            streaming_chat = gemeni_model.start_chat(history=None, )
        except Exception as e:
            logging.error(e.args)




def on_topicRadio(value, evt: gr.EventData):
    logging.error(f"The {evt.target} component was selected, and its value was {value}.")


if __name__ == "__main__":
    config_path_gemini = r"l:/Python_WorkSpace/config/geminiAPI.ini"
    config_path_zhipuai = r"l:/Python_WorkSpace/config/zhipuai_SDK.ini"
    zhipu_apikey = config_read(
        config_path_zhipuai, section="zhipuai_SDK_API", option1="api_key"
    )
    zhipuai_client = ZhipuAI(api_key=zhipu_apikey)
    # # 测试zhipuai
    # model = "glm-4-flash"
    # response = zhipuai_api("请联网搜索，回答：美国大选最新情况", model=model)
    # for chunk in response:
    #     out = chunk.choices[0].delta.content

    # gemini API
    geminiAPI = config_read(config_path_gemini, section="gemini_API", option1="api_key")
    genai.configure(api_key=geminiAPI)
    # genai.types.GenerationConfig()

    # 全局变量
    stop_inference_flag = False  #停止推理初始值，全局变量
    model = 'glm-4-flash'  # 初始假定值，作为全局变量
    streaming_chat = None  # gemini直播聊天对象；全局变量
    present_message = None  # 当前消息，全局变量;因为chatbot显示的message与送入模型的message会有所不同;

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
            height=600,
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
            add_message,
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

    demo.queue().launch()
