import configparser
from zhipuai import ZhipuAI
from datasets import Dataset
import numpy as np
from semantic_search_by_zhipu import semantic_search

# config_path = r"L:/Python_WorkSpace/zhipuai_SDK.ini"
config_path = r"L:/Python_WorkSpace/config/zhipuai_SDK.ini"
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
# img_path = r"C:/Users/shoub/Pictures/Screenshots/屏幕截图 2023-06-28 202503.png"
# base64_image = image_to_base64(img_path)
# messages = [
#     {
#         "role": "user",
#         "content": [
#             {
#                 "type": "text",
#                 "text": "请将图片中的内容以json的格式输出"
#             },
#             {
#                 "type": "image_url",
#                 "image_url": {
#                     "url": base64_image
#                 }
#
#             }
#         ]
#     }
# ]
# print(base64_image)

# response = client.chat.completions.create(
#     model="glm-4v",
#     messages=messages,
# )
#
# print(response.choices[0].message.content)
# print(client.files.list)

# 3 向量模型
# data = ["只是在人群中多看了你一眼", "已经无法把你忘记", "你是我心头的唯一","爱意横生",'恶从胆边生']
# data_vec = []
#
# for d in data:
#     response = client.embeddings.create(
#         model="embedding-2",  # 填写需要调用的模型名称
#         input=d
#         # input="hello"
#     )
#     data_vec.append(response.data[0].embedding)  # 输出字典,'embedding':每个向量长度为1024的列表
#     # print(response.data[0].embedding)
#
# dataset = Dataset.from_dict({'embedding': data_vec,'txt':data})
# # dataset = Dataset.from_dict({'embedding':data})
# dataset.add_faiss_index(column="embedding")
# question = "无法挽留"
# response = client.embeddings.create(
#     model="embedding-2",  # 填写需要调用的模型名称
#     input=question
# )
# question_embedding = np.array(response.data[0].embedding, dtype=np.float32)  # get_nearest_examples(需要是numpy)
#
# scores, samples = dataset.get_nearest_examples("embedding", question_embedding, k=5)
# for score, sample in zip(scores, samples['txt']):
#     print(score,sample)

# 语义搜索,自定义函数:


sentences= [
    "太阳能电池板是一种可再生能源，对环境有益。",
    "风力涡轮机利用风能发电。",
    "地热供暖利用来自地球的热量为建筑物供暖。",
    "水电是一种可持续能源，依靠水流发电。",
    ]
query = "风能对环境有什么好处？"
# scores, samples = semantic_search(client, query=query, sentences=sentences)
# for score, sample in zip(scores, samples):
#     print(score,sample)
# print(samples)

# retrieval:
knowledge_id = 1759942607489871872 #
question = "Tucker Carson与普京的会面,都谈了些什么?"

prompt_template = """
从文档
{{knowledge}}
中找问题
{{question}}
的答案，
找到答案就仅使用文档语句回答，找不到答案就用自身知识回答并告诉用户该信息不是来自文档。
不要复述问题，直接开始回答。
"""

# response = client.chat.completions.create(
#     model="glm-4",  # 填写需要调用的模型名称
#     messages=[
#         {"role": "user", "content": question},
#     ],
#     tools=[
#             {
#                 "type": "retrieval",
#                 "retrieval": {
#                     "knowledge_id": "your knowledge id",
#                     "prompt_template": prompt_template,
#                 }
#             }
#             ],
#     stream=False,
# )
# # for chunk in response:
# #     print(chunk.choices[0].delta)
# print(response.choices[0].message.content)

# web_search
question = "Tucker Carlson与普京的会面,都谈了些什么?"
response = client.chat.completions.create(
    model="glm-4",  # 填写需要调用的模型名称
    messages=[
        {"role": "user", "content": question},
    ],
    tools=[
            {
                "type": "web_search",
                "web_search": {
                    "enable": True,
                    "search_query": "塔克卡尔森与普京的访谈内容",
                }
            }
            ],
    stream=False,
)
# for chunk in response:
#     print(chunk.choices[0].delta)
print(response.choices[0].message.content)