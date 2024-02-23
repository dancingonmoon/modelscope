from datasets import Dataset
import numpy as np
import configparser
from serpapi import GoogleSearch, BaiduSearch


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
    2) 字典的content,送入semantic_search 获得最佳的k个样本 (zhipuai embedding-2向量模型);
    3) 对k中的n个样本的link,进行request,获取webpage的主要内容,并生成字典,keys: link,title_snippet,link_content
    """
    def __init__(self,engine="Baidu",):
        """
        :param engine: "Baidu","Google",or "None",分别表示,以baidu,google为搜索引擎,搜索指定query,或者None表示不从搜索引擎获取数据
        """
        if engine in ["Google","Baidu"]:
            serp_api =


    def get_api_key(self,key_path=None, key_section='Serp_API',key_option='api_key'):
        """
            从配置文件config.ini中,读取api_key;避免程序代码中明文显示key,secret.
            args:
                key_path: config.ini的文件路径(包含文件名,即: directory/config.ini)
                key_section: config.ini中的section名称;
                key_option: config.ini中的option名称;
            out:
                返回option对应的value值;此处为api_key
            """
        config = configparser.ConfigParser()
        config.read(key_path, encoding="utf-8")  # utf-8支持中文
        return config[key_section][key_option]

    def serpapi_GoogleSearch(self, api_key, query,
                             location='Hong Kong', hl='zh-cn', gl='cn', tbs=None, tbm=None, num=30, ):
        """
        使用SerpAPI进行Google搜索
        args:
            config_path: config.ini的文件路径(包含文件名,即: directory/config.ini)
            section: config.ini中section名称;
            option: config.ini中option名称;
            query: 搜索的问题或关键字
            location: Parameter defines from where you want the search to originate.
            hl:Parameter defines the country to use for the Google search. It's a two-letter country code. (e.g., us for the
                United States, uk for United Kingdom, or fr for France)
            gl:Parameter defines the language to use for the Google search. It's a two-letter language code. (e.g., en for
                English, es for Spanish, or fr for French). Head to the Google languages page for a full list of supported
                Google languages.
            num:Parameter defines the maximum number of results to return. (e.g., 10 (default) returns 10 results
            tbs:(to be searched) parameter defines advanced search parameters that aren't possible in the regular query
            field. (e.g., advanced search for patents, dates, news, videos, images, apps, or text contents).
            tbm:(to be matched) parameter defines the type of search you want to do.
                It can be set to:
                (no tbm parameter): regular Google Search,
                isch: Google Images API,
                lcl - Google Local API
                vid: Google Videos API,
                nws: Google News API,
                shop: Google Shopping API,
                pts: Google Patents API,
                or any other Google service.

        out:
            result: a structured JSON of the google search results
        """
        param = {
            "q": query,
            "location": location,
            "api_key": api_key,
            "hl": hl,
            "gl": gl,
            "num": num,
            "tbm": tbm,
            "tbs": tbs
        }
        search = GoogleSearch(param)
        result = search.get_dict()
        return result

if __name__ == "__main__":
    config_path = r"l:/Python_WorkSpace/config/SerpAPI.ini"
    query = 'Tucker Carson与普京的会面,都谈了些什么?'