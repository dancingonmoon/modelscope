import gradio as gr
from zhipuai import ZhipuAI
from GLM.GLM_callFunc import config_read
import time


def add_message(history, message):
    text = message.get("text")
    file = message.get("file")
    present_message = {
        "role": "user",
        "content": f"{text},请结合文件：{file}回答",  # GLM模型不支持content里面file 或者Path
    }

    if text is None and file is not None:
        present_message = {
            "role": "user",
            "content": f"{file}",  # GLM模型不支持content里面file 或者Path
        }
    if text is not None and file is None:
        present_message = {
            "role": "user",
            "content": f"{text}",
        }
    history.append(present_message)
    return history, gr.MultimodalTextbox(value=None, interactive=False)
    # return history, present_message


def inference(
    history: list,
):
    try:
        present_message = history[-1]
        present_response = ""
        history.append({"role": "assistant", "content": present_response})
        for chunk in zhipuai_api(present_message,model=model):
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


def zhipuai_api(question:str, model:str):
    if "alltools" in model:
        messages = (
            [{"role": "user", "content": [{"type": "text", "text": question}]}],
        )
    else:
        messages = (
            [
                {"role": "user", "content": question},
            ],
        )
    response = zhipuai_client.chat.completions.create(
        # model="glm-4-alltools",  # 填写需要调用的模型名称
        model=model,  # 填写需要调用的模型名称
        messages=messages,
        # tools=[
        #     {
        #         "type": "web_search",
        #         "web_search": {
        #             "enable": True,
        #             # "search_result": True,
        #         },
        #     }
        # ],
        tools=[{"type": "web_browser"}],
        stream=True,
    )
    return response


def vote(data: gr.LikeData):
    if data.liked:
        print(f"You upvoted this response:  {data.index}, {data.value} ")
    else:
        print("You downvoted this response: " + data.value)
        print(f"You downvoted this response: {data.index}, {data.value}")


if __name__ == "__main__":
    config_path_zhipuai = r"e:/Python_WorkSpace/config/zhipuai_SDK.ini"
    zhipu_apikey = config_read(
        config_path_zhipuai, section="zhipuai_SDK_API", option1="api_key"
    )
    zhipuai_client = ZhipuAI(api_key=zhipu_apikey)
    model = 'glm-4-alltools'
    # 测试zhipuai
    response = zhipuai_api("请联网搜索美国大选最新情况", model=model)
    for chunk in response:
        out = chunk.choices[0].delta.content

    with gr.Blocks() as demo:
        chatbot = gr.Chatbot(
            elem_id="Multimodal Chatbot",
            bubble_full_width=False,
            type="messages",
            placeholder="**想问点什么?**",
            show_copy_button=True,
            show_copy_all_button=True,
            show_share_button=True,
        )

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
