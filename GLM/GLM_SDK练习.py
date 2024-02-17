import configparser
from zhipuai import ZhipuAI

config_path = r"L:/Python_WorkSpace/zhipuai_SDK.ini"
config = configparser.ConfigParser()
config.read(config_path, encoding='utf-8')
api_key = config.get('zhipuai_SDK_API', 'api_key')
client = ZhipuAI(api_key=api_key)
# 1 通用模型 LLM
# response = client.chat.completions.create(
#     model='glm-4',
#     messages=[
#         {"role": "system", "content": "你的名字叫大侠."},
#         {"role": "user", "content": "请你预测GPT-5会比GPT-4改进性能的比例"},
#     ],
#     stream=False,
# )
# print(response.choices[0].message)
# for chunk in response:
#     print(chunk.choices[0].delta.content)

# 2 图像文本模型 Vision-LLM
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
img_path = r"C:/Users/shoub/Pictures/Screenshots/屏幕截图 2023-06-28 202503.png"
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

# response = client.chat.completions.create(
#     model="glm-4v",
#     messages=messages,
# )
#
# print(response.choices[0].message.content)
# print(client.files.list)

# 3 向量模型

response = client.embeddings.create(
    model="embedding-2", #填写需要调用的模型名称
    input="只是在人群中多看了你一眼",
)
print(response.data[0].embedding)