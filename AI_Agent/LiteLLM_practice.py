from GLM.GLM_callFunc import config_read
from litellm import completion
import litellm
import os
from pathlib import Path
import base64
import gradio as gr


def inference(message, history):
    try:
        flattened_history = [item for sublist in history for item in sublist]
        full_message = " ".join(flattened_history + [message])
        messages_litellm = [{"role": "user", "content": full_message}]  # litellm message format
        partial_message = ""
        for chunk in litellm.completion(model=model,
                                        # add `openai/` prefix to model so litellm knows to route to OpenAI
                                        api_key=zhipuaiAPI,  # api key to your openai compatible endpoint
                                        api_base="https://open.bigmodel.cn/api/paas/v4/",
                                        # set API Base of your Custom OpenAI Endpoint
                                        messages=messages_litellm,
                                        max_new_tokens=512,
                                        temperature=.7,
                                        top_k=100,
                                        top_p=.9,
                                        repetition_penalty=1.18,
                                        stream=True):
            out = chunk['choices'][0]['delta']['content']
            if out:
                partial_message += out  # extract text from streamed litellm chunks
            yield partial_message
    except Exception as e:
        print("Exception encountered:", str(e))
        yield f"出现错误,错误内容为: {str(e)}"


def vote(data: gr.LikeData):
    if data.liked:
        print("You upvoted this response: " + data.value["value"])
    else:
        print("You downvoted this response: " + data.value["value"])


if __name__ == "__main__":
    config_path_serp = r"l:/Python_WorkSpace/config/SerpAPI.ini"
    config_path_gemini = r"l:/Python_WorkSpace/config/geminiAPI.ini"
    config_path_zhipuai = r"l:/Python_WorkSpace/config/zhipuai_SDK.ini"

    # gemini API
    geminiAPI = config_read(
        config_path_gemini, section="gemini_API", option1="api_key"
    )

    os.environ['GEMINI_API_KEY'] = geminiAPI
    model = "gemini/gemini-1.5-pro-latest"
    # response = completion(
    #     model=model,
    #     messages=[{"role": "user", "content": "请介绍喀山这个城市"}]
    # )
    # print(response.choices[0].message.content)

    # audio_path = "H:/music/让我们荡起双桨 - 黑鸭子.mp3"
    #
    # litellm.set_verbose = True  # 👈 See Raw call
    #
    # audio_bytes = Path(audio_path).read_bytes()
    # encoded_data = base64.b64encode(audio_bytes).decode("utf-8")
    # # print("Audio Bytes = {}".format(audio_bytes))
    #
    # response = litellm.completion(
    #     model=model,
    #     messages=[
    #         {
    #             "role": "user",
    #             "content": [
    #                 {"type": "text", "text": "这个音频是啥内容,详细介绍下."},
    #                 {
    #                     "type": "image_url",
    #                     "image_url": "data:audio/mp3;base64,{}".format(encoded_data),  # 👈 SET MIME_TYPE + DATA
    #                 },
    #             ],
    #         }
    #     ],
    # )
    # print(response.choices[0].message.content)

    # openAI compatible endpoint   : GLM

    model = "openai/glm-4-air"

    # functionTools = litellm.supports_function_calling(model=model) # 👈 Check if model supports function calling
    # print(functionTools)

    zhipuaiAPI = config_read(
        config_path_zhipuai, section="zhipuai_SDK_API", option1="api_key"
    )

    # response = litellm.completion(
    #     model=model,  # add `openai/` prefix to model so litellm knows to route to OpenAI
    #     api_key=zhipuaiAPI,  # api key to your openai compatible endpoint
    #     api_base="https://open.bigmodel.cn/api/paas/v4/", # set API Base of your Custom OpenAI Endpoint
    #     messages=[
    #         {
    #             "role": "user",
    #             "content": "请联网搜索,告诉我今天日期",
    #         }
    #     ],
    #     # 经测试,functions 在openai/compatibale模型上不支持
    #     tools=[{
    #         "type": "web_search",
    #         "web_search": {
    #             "enable": True  # 默认为关闭状态（False） 禁用：False，启用：True。
    #         }
    #     }]
    # )
    # print(response.choices[0].message.content)
    # print(response)

    # gradio web:
    with (gr.Blocks() as demo):
        chatbot = gr.Chatbot(height=400, placeholder=f"<strong>我是大模型:{model}</strong><br>Ask Me Anything")
        chatbot.like(vote, None, None)
        gr.ChatInterface(
            inference,
            chatbot=chatbot,
            textbox=gr.Textbox(placeholder="Enter text here...", container=False, scale=5),
            # multimodal=True,
            description=f"""
        CURRENT PROMPT TEMPLATE: {model}.
        An incorrect prompt template will cause performance to suffer.
        Check the API specifications to ensure this format matches the target LLM.""",
            title="Simple Chatbot Test Application",
            examples=["Define 'deep learning' in once sentence."],
            retry_btn="Retry",
            undo_btn="Undo",
            clear_btn="Clear",
            fill_width=True,
            stop_btn='Stop',
            theme="soft",
            )
    demo.queue().launch()
