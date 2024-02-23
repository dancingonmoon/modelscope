from datasets import Dataset
import numpy as np
from SerpAPI_fn import get_api_key,serpapi_GoogleSearch,serpapi_BaiduSearch


# 语义搜索,自定义函数:
def semantic_search(client, query, sentences, k=3, ):
    """
    使用zhipuAI向量模型Embedding-2, 在sentences列中搜索与query最相似的k个句子.(使用huggingface的Dataset调用的faiss方法)
    1) sentences list 送入embedding-2模型,获得长度1024的向量列表;
    2) Dataset.add_faiss_index,生成faiss索引;
    3) query 送入embedding-2模型,获得长度1024的向量;
    4) Dataset.get_nearest_examples,获得最佳的k个dataset的样本
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


class semanticSearch_via_SerpAPI_by_zhipuai:
    """
    关键字Google或者Baidu搜索引擎之后,再语义搜索:
    1) query于Google或者Baidu获得link,title,以及snippet,将title与snippet合并后,生成字典,包含key:link与title_snippet; (SerpAPI_fn.py)
    2) 字典的content,送入semantic_search 获得最佳的k个样本;
    3) 对k中的n个样本的link,进行request,获取webpage的主要内容,并生成字典,keys: link,title_snippet,link_content
    """
    def __init__(self,query,
                 key_serp_path=None, key_serp_section='Serp_API',key_serp_option='api_key',
                key_zhipu_path=None,key_zhipu_section='zhipuai_SDK_API',key_zhipu_option='api_key',

                 ):
        self.query = query

    def get_api_key(self,
                    key_path=None, key_section='Serp_API',key_option='api_key',):
        api_key = get_api_key(config_path, key_section, key_option)
        return api_key

if __name__ == "__main__":
    config_path = r"l:/Python_WorkSpace/config/SerpAPI.ini"
    query = 'Tucker Carson与普京的会面,都谈了些什么?'