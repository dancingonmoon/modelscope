import configparser
from zhipuai import ZhipuAI

config_path = "E:/Python_WorkSpace/config/zhipuai_SDK.ini"
config = configparser.ConfigParser()
config.read(config_path)
api_key = config.get('zhipuai_SDK_API', 'api_key')

client = ZhipuAI(api_key=api_key)
# response = client.chat.completions.create(
#     model='glm-4',
#     messages=[
#         {"role": "system", "content": "你的名字叫大侠."},
#         {"role": "user", "content": "你的名字? 请回答明天的天气"},
#     ],
#     stream = False,
# )
# print(response.choices[0].message)
# for chunk in response:
#     print(chunk.choices[0].delta.content)

import base64
import io
from PIL import Image

def image_to_base64(image_path):
    """
    Convert an image to base64 encoding.
    """
    with Image.open(image_path) as image:
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")  # or format="PNG", or "JPEG", depending on your image.
        img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str

# img_path = "C:/Users/danci/Pictures/分配生名额.png"
img_path = "C:/Users/danci/Pictures/Screenshots/屏幕截图_20230112_135119.png"
base64_image = image_to_base64(img_path)
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": "请将图片中的内容以json的格式输出"
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": base64_image
                }

            }
        ]
    }
]
# print(base64_image)

response = client.chat.completions.create(
    model="glm-4v",
    messages=messages,
)

print(response.choices[0].message.content)
# print(client.files.list)