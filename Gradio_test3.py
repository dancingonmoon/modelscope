import gradio as gr
from zhipuai import ZhipuAI
from GLM.GLM_callFunc import config_read

def test(message,file):
    return message

def inference(history,message):
    try:
        flattened_history = [item for sublist in history for item in sublist]
        text = message.get("text")
        file = message.get("file")
        if text:
            full_message = " ".join(flattened_history + [text])
        else:
            full_message = flattened_history
        messages_prompt = [
            {"role": "user", "content": full_message}
        ]  # litellm message format

        partial_message = ""
        for chunk in zhipuai_api(messages_prompt):
            out = chunk["choices"][0]["delta"]["content"]
            if out:
                partial_message += out  # extract text from streamed litellm chunks

                yield history.append({"role": "assistant", "content": partial_message})
    except Exception as e:
        print("Exception encountered:", str(e))
        yield history.append({"role": "assistant", "content": f"出现错误,错误内容为: {str(e)}"})


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

import gradio as gr
import time

# Chatbot demo with multimodal input (text, markdown, LaTeX, code blocks, image, audio, & video). Plus shows support for streaming text.


def print_like_dislike(x: gr.LikeData):
    print(x.index, x.value, x.liked)


def add_message(history, message):
    for x in message["files"]:
        history.append({"role": "user", "content": {"path": x}})
    if message["text"] is not None:
        history.append({"role": "user", "content": message["text"]})
    return history, gr.MultimodalTextbox(value=None, interactive=False)


def bot(history: list):
    response = "**That's cool!**"
    history.append({"role": "assistant", "content": ""})
    for character in response:
        history[-1]["content"] += character
        time.sleep(0.05)
        yield history






if __name__ == "__main__":
    config_path_zhipuai = r"l:/Python_WorkSpace/config/zhipuai_SDK.ini"

    zhipu_apikey = config_read(
        config_path_zhipuai, section="zhipuai_SDK_API", option1="api_key"
    )
    zhipuai_client = ZhipuAI(api_key=zhipu_apikey)

    # gradio web:
    # with (gr.Blocks() as demo):
    #     chatbot = gr.Chatbot(
    #         height=400, placeholder=f"<strong>我是大模型:XXX </strong><br>问我吧?"
    #     )
    #     chatbot.like(vote, None, None)
    #     gr.ChatInterface(
    #         # inference,
    #         test,
    #         chatbot=chatbot,
    #         # textbox=gr.Textbox(placeholder="Enter text here...", container=False, scale=5),
    #         multimodal=True,
    #         description=f"""
    #     多模态模型输入测试.""",
    #         title="BOT",
    #         examples=["今天的日期与未来5天杭州天气"],
    #         # retry_btn="Retry",
    #         # undo_btn="Undo",
    #         # clear_btn="Clear",
    #         # fill_width=True,
    #         # stop_btn='Stop',
    #         theme="soft",
    #     )
    # demo.queue().launch()

    with gr.Blocks() as demo:
        chatbot = gr.Chatbot(elem_id="chatbot", bubble_full_width=False, type="messages",placeholder="chatbot 占位符：")

        chat_input = gr.MultimodalTextbox(
            # value= {"text": "sample text", "files": [{'path': "files/ file. jpg", 'orig_name': "file. jpg", 'url': "http:// image_url. jpg ", 'size': 100}]},
            file_types= ['file'],
            interactive=True,
            file_count="multiple",
            lines=2,
            placeholder="Enter message or upload file...",
            show_label=False,
        )

        chat_msg = chat_input.submit(
            add_message, [chatbot, chat_input], [chatbot, chat_input]
        )
        bot_msg = chat_msg.then(inference, [chatbot, chat_input], [chatbot, chat_input], api_name="bot_response")
        bot_msg.then(lambda: gr.MultimodalTextbox(interactive=True), None, [chat_input])

        chatbot.like(print_like_dislike, None, None)

    demo.queue().launch()
