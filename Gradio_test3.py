import gradio as gr
from zhipuai import ZhipuAI
from GLM.GLM_callFunc import config_read


def add_message(history, message):
    text = message.get("text")
    file = message.get("file")
    present_message = {
        "role": "user",
        "content": f"{text},请结合文件：{file}回答",  # GLM模型不支持content里面file 或者Path
    }

    # flattened_history = [item for sublist in history for item in sublist]
    # flattened_history = [item for item in history]  # 这不是与history相等吗？
    if text is None and file is not None:
        present_message = {
            "role": "user",
            "content": f"{file}",  # GLM模型不支持content里面file 或者Path
        }
    if text is not None and file is None:
        present_message = {
            "role": "user",
            "content": f"{text}",  # GLM模型不支持content里面file 或者Path
        }
    history.append(present_message)
    # return history, gr.MultimodalTextbox(value=None, interactive=False)
    return history, present_message


def inference(history, present_message):
    try:
        present_response = ""
        for chunk in zhipuai_api(present_message):
            out = chunk["choices"][0]["delta"]["content"]
            if out:
                present_response += out  # extract text from streamed litellm chunks

                yield history.append({"role": "assistant", "content": present_response})
    except Exception as e:
        print("Exception encountered:", str(e))
        return history.append({"role": "assistant", "content": f"出现错误,错误内容为: {str(e)}"})


def zhipuai_api(question):
    response = zhipuai_client.chat.completions.create(
        model="glm-4-air",  # 填写需要调用的模型名称
        messages=[
            {"role": "user", "content": question},
        ],
        tools=[
            {
                "type": "web_search",
                "web_search": {
                    "enable": True,
                    "search_result": True,
                },
            }
        ],
        stream=True,
    )
    return response


def vote(data: gr.LikeData):
    if data.liked:
        print(f"You upvoted this response:  {data.index}, {data.value} ")
    else:
        print("You downvoted this response: " + data.value)
        print(f"You downvoted this response: {data.index}, {data.value}")


def bot(history: list):
    response = "**That's cool!**"
    history.append({"role": "assistant", "content": ""})
    for character in response:
        history[-1]["content"] += character
        time.sleep(0.05)
        yield history


if __name__ == "__main__":
    config_path_zhipuai = r"e:/Python_WorkSpace/config/zhipuai_SDK.ini"

    zhipu_apikey = config_read(
        config_path_zhipuai, section="zhipuai_SDK_API", option1="api_key"
    )
    zhipuai_client = ZhipuAI(api_key=zhipu_apikey)

    with gr.Blocks() as demo:
        chatbot = gr.Chatbot(
            elem_id="chatbot",
            bubble_full_width=False,
            type="messages",
            placeholder="chatbot 占位符：",
        )

        chat_input = gr.MultimodalTextbox(
            # value= {"text": "sample text", "files": [{'path': "files/ file. jpg", 'orig_name': "file. jpg", 'url': "http:// image_url. jpg ", 'size': 100}]},
            file_types=["file"],
            interactive=True,
            file_count="multiple",
            lines=2,
            placeholder="Enter message or upload file...",
            show_label=False,
        )

        chat_msg = chat_input.submit(
            add_message, [chatbot, chat_input], [chatbot, chat_input]
        )
        bot_msg = chat_msg.then(
            inference,
            [chatbot, chat_input],
            chatbot,
            api_name="bot_response",
        )
        bot_msg.then(lambda: gr.MultimodalTextbox(interactive=True), None, [chat_input])

        chatbot.like(vote, None, None)

    demo.queue().launch()
