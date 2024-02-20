import configparser
from zhipuai import ZhipuAI
from datasets import Dataset
import numpy as np

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
def semantic_search(client, query, sentences, k=3,):
    """
    使用zhipuAI向量模型, 在sentences列中搜索与query最相似的k个句子
    :param
        client: object; zhipuAI client (已经送入API_KEY)
        query: str; 欲搜索的关键词或者句子
        sentences: list; 包含所有欲搜索句子列表
        k: int; 返回最相似的句子数量
    :return: scores, nearest_examples中的text; 得分,以及对应的句子 (score越小,越佳)
    """
    sentences_vec = []
    for sentence in sentences:
        response = client.embeddings.create(
            model="embedding-2",
            input=sentence
            )
        sentences_vec.append(response.data[0].embedding)  # 输出字典,'embedding':每个向量长度为1024的列表

    dataset = Dataset.from_dict({'embedding': sentences_vec, 'txt': sentences})
    dataset.add_faiss_index(column="embedding")

    response = client.embeddings.create(
        model="embedding-2",  # 填写需要调用的模型名称
        input=query
    )
    query_embedding = np.array(response.data[0].embedding, dtype=np.float32)  # get_nearest_examples(需要是numpy)

    scores, nearest_examples = dataset.get_nearest_examples("embedding", query_embedding, k=k)
    return scores, nearest_examples['txt']

sentences= [
    "太阳能电池板是一种可再生能源，对环境有益。",
    "风力涡轮机利用风能发电。",
    "地热供暖利用来自地球的热量为建筑物供暖。",
    "水电是一种可持续能源，依靠水流发电。",
    ]
query = "风能对环境有什么好处？"
scores, samples = semantic_search(client, query=query, sentences=sentences)
for score, sample in zip(scores, samples):
    print(score,sample)
print(samples)