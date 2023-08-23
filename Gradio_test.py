import random

def random_response(message, history):
    return random.choice(["Yes", "No"])

import time
import gradio as gr

def slow_echo(message, history):
    for i in range(len(message)):
        time.sleep(0.3)
        yield "You typed: " + message[: i+1]

# import gradio as gr

# demo = gr.load("stabilityai/stable-diffusion-xl-base-1.0", src="models")
# demo = gr.load("openai/whisper-large-v2", src="models")
# demo = gr.load("ysharma/Explore_llamav2_with_TGI", src="spaces")
#
# demo.queue()
# demo.launch()

import requests

API_URL = "https://api-inference.huggingface.co/models/openai/whisper-large-v2"
headers = {"Authorization": "Bearer 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"}

def query(filename):
    with open(filename, "rb") as f:
        data = f.read()
    response = requests.post(API_URL, headers=headers, data=data)
    return response.json()

# with gr.Blocks(theme='soft', title='whisper应用') as demo:
#     gr.Markdown("""
#             # 测试whisper
#             """)
#     with gr.Row():
#         audio_input = gr.Text(placeholder='文件路径')
#         text_ouput = gr.Text(placeholder='录音解析为:')
#     button = gr.Button("解析")
#     button.click(query,audio_input,text_ouput)
# demo.queue().launch()
filename = r"F:\Music\Dangerous.mp3"
output = query(filename)
print(output)




