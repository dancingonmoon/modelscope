import gradio as gr
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
            file_object = zhipuai_client.files.create(
                file=Path(file), purpose="file-extract"
            )
            # 获取文本内容
            file_content = json.loads(
                zhipuai_client.files.content(file_id=file_object.id).content
            )["content"]
            # files_content.append(file_content)

            files_prompt += f"第{file_No+1}个文件或图片内容如下：\n" f"{file_content}\n\n"
        if text is None:
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


def inference(
    history: list,
):
    try:
        present_message = history[-1]["content"]
        present_response = ""
        history.append({"role": "assistant", "content": present_response})
        for chunk in zhipuai_api(present_message, model=model):
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


def vote(data: gr.LikeData):
    if data.liked:
        print(f"You upvoted this response:  {data.index}, {data.value} ")
    else:
        print("You downvoted this response: " + data.value)
        print(f"You downvoted this response: {data.index}, {data.value}")
def handle_undo(history, undo_data: gr):
    return history[:undo_data.index], history[undo_data.index]['content']

def on_topicRadio(click):
    print(click)

if __name__ == "__main__":
    config_path_zhipuai = r"e:/Python_WorkSpace/config/zhipuai_SDK.ini"
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
        gr.Markdown("# 试试这个多模态输入 🤗")
        chatbot = gr.Chatbot(
            elem_id="Multimodal Chatbot",
            label="**理想王国**",
            bubble_full_width=False,
            type="messages",
            placeholder="**想问点什么?**",
            show_copy_button=True,
            show_copy_all_button=True,
            show_share_button=True,
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

        chat_msg = chat_input.submit(
            add_message,
            [chatbot, chat_input],
            [chatbot, chat_input],
            queue=False,
        )
        bot_msg = chat_msg.then(
            inference,
            [chatbot],
            [chatbot],
            api_name="bot_response",
        )
        bot_msg.then(lambda: gr.MultimodalTextbox(interactive=True), None, [chat_input])

        chatbot.like(vote, None, None)

    demo.queue().launch()
