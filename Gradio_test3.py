import gradio as gr  # gradio 5.5.0 需要python 3.10以上
from zhipuai import ZhipuAI
from GLM.GLM_callFunc import config_read
from pathlib import Path
import json


def add_message(history, message):
    present_message = {
        "role": "user",
        "content": "",
    }
    if history is None:
        history = [present_message]
    text = message.get("text")
    files = message.get("files")
    if files:
        files_prompt = "请结合以下文件或图片内容回答：\n\n"
        for file_No, file in enumerate(files):
            history.append(
                {"role": "user", "content": {"path": file}}
            )  # chatbot上先显示该图片
            # 文件处理
            # 格式限制：.PDF .DOCX .DOC .XLS .XLSX .PPT .PPTX .PNG .JPG .JPEG .CSV .PY .TXT .MD .BMP .GIF
            # 大小：单个文件50M、总数限制为100个文件
            try:
                file_object = zhipuai_client.files.create(
                    file=Path(file), purpose="file-extract"
                )
                # 获取文本内容
                file_content = json.loads(
                    zhipuai_client.files.content(file_id=file_object.id).content
                )["content"]
            except Exception as e:
                print(e.args)
                present_message = {
                    "role": "assistant",
                    "content": e.args[0],
                }
                history.append(present_message)
                return history, gr.MultimodalTextbox(
                    value=None, interactive=False
                )  # 因此此处输出的仅仅是错误，但不影响后续程序执行，导致模型输入部分是空值，出错

            if file_content is None or file_content == "":
                files_prompt += f"第{file_No+1}个文件或图片内容无可提取之内容\n\n"
            else:
                files_prompt += f"第{file_No+1}个文件或图片内容如下：\n" f"{file_content}\n\n"
        if text is None or text == "":
            present_message = {
                "role": "user",
                "content": files_prompt,  # GLM模型不支持content里面file 或者Path
            }
        else:
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
    history.append(present_message)
    return (
        history,
        gr.MultimodalTextbox(value=None, interactive=False),
        gr.Button(interactive=True, visible=True),
    )


def glm_inference(
    history: list, new_topic: bool, model: str, stop_inference_flag: bool
):
    try:
        if new_topic:
            present_message = [history[-1]]
        else:
            # glm模型文件作为prompt，非通过type方式，而是通过件文件内容放在到prompt内
            # history中连续的{"role": "user", "content"：""},是文件链接或内容的删除
            present_message = [
                message
                for message in history
                if not (
                    message["role"] == "user" and isinstance(message["content"], tuple)
                )
            ]

        present_response = ""
        history.append({"role": "assistant", "content": present_response})
        for chunk in zhipuai_messages_api(present_message, model=model):
            if stop_inference_flag == True : # 没有发挥作用,当stop_inference_button.click()时，代码没有执行或者没有响应.
                # break
                return history
            out = chunk.choices[0].delta.content
            if out:
                present_response += out  # extract text from streamed litellm chunks
                history[-1] = {"role": "assistant", "content": present_response}
                yield history
    except Exception as e:
        print("Exception encountered:", str(e))
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
        print(f"You upvoted this response:  {data.index}, {data.value} ")
    else:
        print("You downvoted this response: " + data.value)
        print(f"You downvoted this response: {data.index}, {data.value}")


def handle_undo(history, undo_data: gr.UndoData):
    return history[: undo_data.index], history[undo_data.index]["content"]


def handle_retry(
    history: str | list[dict],
    new_topic: bool,
    model: str,
    retry_data: gr.RetryData,
    stop_inference_bool: bool,
):
    # yield history, gr.MultimodalTextbox(value=None, interactive=False), gr.Button(interactive=True, visible=True)
    new_history = history[: retry_data.index]
    previous_prompt = history[retry_data.index]
    new_history.append(previous_prompt)
    stop_inference_flag = stop_inference_bool
    if isinstance(stop_inference_bool,gr.components.state.State):
        stop_inference_flag = stop_inference_bool.value
    elif isinstance(stop_inference_bool,bool):
        stop_inference_flag = stop_inference_bool

    # yield gr.Button(visible=True)
    yield from glm_inference(new_history, new_topic, model, stop_inference_flag)


def on_stop_inference_button():
    stop_inference_bool.value = True
    print("Stop inference button clicked. stop_inference_bool set to True.")
    return gr.State(value=True)


def on_topicRadio(value, evt: gr.EventData):
    print(f"The {evt.target} component was selected, and its value was {value}.")


if __name__ == "__main__":
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
            render_markdown=True,
            avatar_images=(
                None,
                "https://em-content.zobj.net/source/twitter/376/hugging-face_1f917.png",
            ),
        )
        # 用于中止推理,仅仅在推理过程中显现作用;gr.Button仅仅用于UI显示,bool变量的传送以gr.State传递参数;
        # 没有使用checkbox一次搞定是因为,需要借用button按钮
        stop_inference_button = gr.Button(
            value="停止推理",
            variant="secondary",
            size="sm",
            visible=False,
            interactive=True,
            min_width=100,
        )
        stop_inference_bool = gr.State(value=False)

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
                    "gemini-1.5-pro",
                ],
                value="glm-4-flash",
                multiselect=False,
                scale=1,
                show_label=False,
                label="models",
            )

        stop_inference_button.click(
            on_stop_inference_button, None, [stop_inference_bool]
        )
        chatbot.undo(handle_undo, chatbot, [chatbot, chat_input])
        chatbot.like(vote, None, None)
        chatbot.retry(
            handle_retry,
            [chatbot, topicCheckbox, models_dropdown, stop_inference_bool],
            [chatbot],
        )

        chat_msg = chat_input.submit(
            add_message,
            [chatbot, chat_input],
            [chatbot, chat_input, stop_inference_button],
            queue=False,
        )
        bot_msg = chat_msg.then(
            glm_inference,
            [chatbot, topicCheckbox, models_dropdown, stop_inference_bool],
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
        bot_msg.then(lambda: gr.State(value=False), None, [stop_inference_bool])

    demo.queue().launch()
