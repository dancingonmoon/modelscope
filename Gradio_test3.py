import gradio as gr # gradio 5.5.0 需要python 3.10以上
from zhipuai import ZhipuAI
from GLM.GLM_callFunc import config_read
from pathlib import Path
import json


def add_message(history, message):
    text = message.get("text")
    files = message.get("files")
    present_message = {
        "role": "user",
        "content": "",
    }
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
                return history, gr.MultimodalTextbox(value=None, interactive=False) # 因此此处输出的仅仅是错误，但不影响后续程序执行，导致模型输入部分是空值，出错

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
    return history, gr.MultimodalTextbox(value=None, interactive=False)


def inference(history: list, new_topic: bool):
    try:
        if new_topic:
            present_message = [history[-1]]
        else:
            # glm模型文件作为prompt，非通过type方式，而是通过件文件内容放在到prompt内
            # history中连续的{"role": "user", "content"：""},是文件链接或内容的删除
            present_message = [message for message in history if
                               not (message["role"] == "user" and isinstance(message["content"], tuple))]

            # present_message = history # 变量赋值，只会对同一个对象指定两个变量名
            # for message in present_message:
            #     if message["role"] == "user" and isinstance(message["content"],tuple):
            #         present_message.remove(message)


        present_response = ""
        history.append({"role": "assistant", "content": present_response})
        for chunk in zhipuai_messages_api(present_message, model=model):
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


def zhipuai_api(question: str, model: str):
    if "alltools" in model:
        messages = [{"role": "user", "content": [{"type": "text", "text": question}]}]
        tools = [{"type": "web_browser"}]
    else:
        messages = [{"role": "user", "content": question}]
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
        messages=messages,
        stream=True,
    )
    return response

def zhipuai_messages_api(messages: str|list[dict], model: str):
    prompt = []
    if "alltools" in model:
        if isinstance(messages, str):
            prompt.append([{"role": "user", "content": [{"type": "text", "text": messages}]}])
        elif isinstance(messages, list) and all(isinstance(message, dict) for message in messages):
            for message in messages:
                prompt.append([{"role": "user", "content": [{"type": "text", "text": message["content"]}]}])

        tools = [{"type": "web_browser"}]
    else:
        if isinstance(messages, str):
            prompt.append([{"role": "user", "content": messages}])
        elif isinstance(messages, list) and all(isinstance(message, dict) for message in messages):
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
    return history[:undo_data.index], history[undo_data.index]['content']

def handle_retry(history: str|list[dict], new_topic:bool, retry_data: gr.RetryData):
    new_history = history[:retry_data.index]
    previous_prompt = history[retry_data.index]
    new_history.append(previous_prompt)

    yield from inference(new_history, new_topic)

def on_topicRadio(value, evt:gr.EventData):
    print( f"The {evt.target} component was selected, and its value was {value}.")

if __name__ == "__main__":
    config_path_zhipuai = r"l:/Python_WorkSpace/config/zhipuai_SDK.ini"
    zhipu_apikey = config_read(
        config_path_zhipuai, section="zhipuai_SDK_API", option1="api_key"
    )
    zhipuai_client = ZhipuAI(api_key=zhipu_apikey)
    model = "glm-4-flash"
    # # 测试zhipuai
    # response = zhipuai_api("请联网搜索，回答：美国大选最新情况", model=model)
    # for chunk in response:
    #     out = chunk.choices[0].delta.content

    with gr.Blocks() as demo:
        gr.Markdown("# 多模态Robot 🤗")
        chatbot = gr.Chatbot(
            elem_id="Multimodal Chatbot",
            label="聊天框",
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

        with gr.Row():
            topicRadio = gr.Checkbox(label="新话题",show_label=True,)


        chat_input = gr.MultimodalTextbox(
            # value= {"text": "sample text", "files": [{'path': "files/ file. jpg", 'orig_name': "file. jpg", 'url': "http:// image_url. jpg ", 'size': 100}]},
            file_types=["file"],
            interactive=True,
            file_count="multiple",
            lines=1,
            placeholder="Enter message or upload file...",
            show_label=False,
        )
        chatbot.undo(handle_undo, chatbot,[chatbot,chat_input])
        chatbot.retry(handle_retry, [chatbot,topicRadio],[chatbot])

        chat_msg = chat_input.submit(
            add_message,
            [chatbot, chat_input],
            [chatbot, chat_input],
            queue=False,
        )
        bot_msg = chat_msg.then(
            inference,
            [chatbot,topicRadio],
            [chatbot],
            api_name="bot_response",
        )
        bot_msg.then(lambda: gr.Checkbox(value=False), None, [topicRadio])
        bot_msg.then(lambda: gr.MultimodalTextbox(interactive=True), None, [chat_input])

        chatbot.like(vote, None, None)

    demo.queue().launch()



