from datasets import Dataset
import numpy as np
from SerpAPI_fn import serpapi_GoogleSearch, serpapi_BaiduSearch, config_read
from zhipuai import ZhipuAI
from collections import defaultdict
from typing import Union


# 语义搜索,自定义函数:
def semantic_search(client, query, sentences, embedding_key=None, k=3, ):
    """
    使用zhipuAI向量模型Embedding-2, 在sentences列中搜索与query最相似的k个句子.(使用huggingface的Dataset调用的faiss方法)
    1) sentences为一字典,包含embedding的key,将其转换成huggingface的Dataset;
    2) Dataset.add_faiss_index,生成faiss索引;
    3) query 送入embedding-2模型,获得长度1024的向量;
    4) Dataset.get_nearest_examples,获得最佳的k个dataset的样本
    :param
        client: object; zhipuAI client (已经送入API_KEY)
        query: str; 欲搜索的关键词或者句子
        sentences: 字典,送入embedding-2模型,获得长度1024的向量列表;
        embedding_key: str; sentences字典中用于embedding的key的名称,该key下的values将用于embedding并用于语义搜索
        k: int; 返回最相似的句子数量
    :return: scores, nearest_examples中的text; 得分,以及对应的句子 (score越小,越佳)
    """
    sentences_vec = []
    if embedding_key is not None:
        for sentence in sentences[embedding_key]:
            response = client.embeddings.create(
                model="embedding-2",
                input=sentence
            )
            sentences_vec.append(response.data[0].embedding)  # 输出字典,'embedding':每个向量长度为1024的列表

    sentences['embedding'] = sentences_vec
    dataset = Dataset.from_dict(sentences)
    dataset.add_faiss_index(column="embedding")

    response = client.embeddings.create(
        model="embedding-2",  # 填写需要调用的模型名称
        input=query
    )
    query_embedding = np.array(response.data[0].embedding, dtype=np.float32)  # get_nearest_examples(需要是numpy)

    scores, nearest_examples = dataset.get_nearest_examples("embedding", query_embedding, k=k)
    return scores, nearest_examples


class chatGLM_by_semanticSearch_amid_SerpAPI:
    """
    关键字Google或者Baidu搜索引擎之后,再语义搜索:
    1) query于Google或者Baidu获得link,title,以及snippet,将title与snippet合并后,生成字典,包含key:link与title_snippet; (SerpAPI_fn.py)
    2) 字典的content,送入semantic_search 获得最佳的k个样本 (zhipuai embedding-2向量模型);
    3) 对k中的n个样本的link,进行request,获取webpage的主要内容,并生成字典,keys: link,title_snippet,link_content
    4) 将n个样本的title_snippet,link_content,送入GLM-4,或者GLM-3-Turbo,获得模型回答;
    """

    def __init__(self, zhipuai_client, serp_key_path=None, serp_key_section='Serp_API', serp_key_option='api_key',
                 ):
        """
        zhipuai_client: 初始化之后的zhipuai_client = ZhipuAI(api_key=zhipuai_apiKey); (避免多次初始化zhipuai_client)
        engine: "Baidu","Google",or "None",分别表示,以baidu,google为搜索引擎,搜索指定query,或者None表示不从搜索引擎获取数据
        """

        self.zhipuai_client = zhipuai_client
        self.serp_api_key = config_read(config_path=serp_key_path, section=serp_key_section, option1=serp_key_option)

    def web_search(self, query, websearch_engine: str = 'baidu',
                   location='Hong Kong', hl='zh-cn', gl='cn', tbs=None, tbm=None, num=30,
                   ct=2, rn=50, ):
        """
        Google Search,或者Baidu Search,将搜索结果,提取title,snippet,生成字典
        websearch_engine: "baidu" or "google", 二选一;
        :return:
        """
        search_results_dict = defaultdict(list)
        if websearch_engine.lower() == 'Google'.lower():
            search_result = serpapi_GoogleSearch(self.serp_api_key, query,
                                                 location=location, hl=hl, gl=gl,
                                                 tbs=tbs, tbm=tbm, num=num, )
            if 'organic_results' in search_result:
                for result_dict in search_result['organic_results']:
                    search_results_dict['link'].append(result_dict['link'])
                    search_results_dict['title'].append(result_dict['title'])
                    search_results_dict['snippet'].append(result_dict['snippet'])
                    search_results_dict['title_snippet'].append(
                        ';'.join([result_dict['title'], result_dict['snippet']]))
        elif websearch_engine.lower() == 'Baidu'.lower():
            search_results = serpapi_BaiduSearch(self.serp_api_key, query, ct=ct, rn=rn, )
            if 'organic_results' in search_results:
                for result_dict in search_results['organic_results']:
                    search_results_dict['link'].append(result_dict['link'])
                    search_results_dict['title'].append(result_dict['title'])
                    if 'snippet' in result_dict:
                        search_results_dict['snippet'].append(result_dict['snippet'])
                        search_results_dict['title_snippet'].append(
                            ';'.join([result_dict['title'], result_dict['snippet']]))
                    else:
                        search_results_dict['snippet'].append('')
                        search_results_dict['title_snippet'].append(
                            ';'.join([result_dict['title'], '']))

        return search_results_dict  # 字典 keys: link, title, snippet, title_snippet

    def semantic_websearch(self, query, search_results_dict, k=3, ):

        if 'title_snippet' in search_results_dict:
            scores, nearest_examples = semantic_search(self.zhipuai_client, query, sentences=search_results_dict,
                                                       embedding_key='title_snippet', k=k)
        else:
            return '搜索结果中无title_snippet键', query

        return scores, nearest_examples  # list中包含字典

    def chatGLMAPI_completion_create(self, question=None, query=None, glm_model='GLM-4-air', GLM_websearch_enable=False,
                                     stream=False, websearch_result_show=False):
        """
        同步调用:chatGLM API 接口,返回生成接口
        :param question:
        :param query:
        :param glm_model:
        :param GLM_websearch_enable: GLM内置的web_search函数功能是否开启
        """
        web_search_parameters = {"enable": GLM_websearch_enable,
                                 "search_query": query, }
        try:
            response = self.zhipuai_client.chat.completions.create(
                model=glm_model,  # 填写需要调用的模型名称
                messages=[
                    {"role": "user", "content": question},
                ],
                tools=[{
                    'type': 'web_search',
                    'web_search': {
                        'enable': GLM_websearch_enable,
                        'search_query': query,
                        'search_result': websearch_result_show
                    }}],
                stream=stream,
            )
            if 'error' in response:
                result = response['error']
            else:
                result = response.choices[0].message.content
        except Exception as e:
            result = f"An error occurred: {e.args}"
        return result

    def chatGLM_RAG(self, question, query, glm_model='GLM-4-air', stream=False, GLM_websearch_enable=False,
                    nearest_examples=None, reference_key='title_snippet', ):
        """
        将web_search搜索,语义匹配之后的n个句子作为参考信息,送入GLM模型生成回复
        RAG_websearch_enable: 是否启动baidu,或者google搜索,进行RAG;
        GLM_websearch_enable: 是否启动GLM模型内置的web_search函数;
        :query:
        :nearest_examples:
        :reference_key:
        :return:
        """
        question_template = question
        if nearest_examples:  # 当nearest_examples=None时,表示不启动RAG搜索
            reference = ';'.join(nearest_examples[reference_key])
            question_template = f"请基于以下参考信息生成回答: {reference}\n问题: {question}"

        result = self.chatGLMAPI_completion_create(question=question_template, query=query,
                                                   glm_model=glm_model, GLM_websearch_enable=GLM_websearch_enable,
                                                   stream=stream, websearch_result_show=False)

        return result

    def chatGLM_RAG_oneshot(self, question, query, websearch_engine: Union[str, None] = None, glm_model='GLM-4-air',
                            GLM_websearch_enable=False, k=3, rn=10):
        if websearch_engine is not None:
            search_results_dict = self.web_search(query, websearch_engine=websearch_engine, rn=rn)
            scores, nearest_samples = self.semantic_websearch(query, search_results_dict, k=k, )
        else:
            scores, nearest_samples = None, None

        # for score, sample in zip(scores, nearest_samples['title_snippet']):
        #     print(f'----语义搜索最佳的{k}个结果:---------')
        #     print(score, sample)
        result = self.chatGLM_RAG(question, query, glm_model=glm_model, stream=False,
                                  GLM_websearch_enable=GLM_websearch_enable,
                                  nearest_examples=nearest_samples, reference_key='title_snippet')
        # print(f'-------web_search_enable={web_search_enable}后,模型{glm_model}/RAG的回答:--------')
        # print(result)
        if websearch_engine is None:
            engine_id = "未经"
        else:
            engine_id = websearch_engine
        output_text = f"模型{glm_model}+{engine_id}搜索引擎RAG+GLM内置web_search={GLM_websearch_enable}后,回答为:\n{result}"

        return scores, nearest_samples, result, output_text


if __name__ == "__main__":
    config_path_serp = r"l:/Python_WorkSpace/config/SerpAPI.ini"
    config_path_zhipuai = r"l:/Python_WorkSpace/config/zhipuai_SDK.ini"
    zhipu_apiKey = config_read(config_path_zhipuai, section="zhipuai_SDK_API", option1="api_key")
    zhipuai_client = ZhipuAI(api_key=zhipu_apiKey)

    # question = 'Tucker Carlson与普京的会面,都谈了些什么?'
    # query = '塔克卡尔森与普京的会面,都谈了些什么?'  # 用于web_search
    question = '这一届货币政策委员会成员介绍'
    query = '这一届货币政策委员会成员介绍'  # 用于web_search
    semantic_search_engine = chatGLM_by_semanticSearch_amid_SerpAPI(zhipuai_client, serp_key_path=config_path_serp,
                                                                    )
    # search_results_dict = semantic_search_engine.web_search(query, rn=10)
    # scores, nearest_samples = semantic_search_engine.semantic_websearch(query, search_results_dict, k=3, )
    # for score, sample in zip(scores, nearest_samples['title_snippet']):
    #     print('----语义搜索最佳的k个结果:---------')
    #     print(score, sample)
    # result = semantic_search_engine.chatGLM_RAG(question, query, 'GLM-3-Turbo', web_search_enable=True,
    #                                             nearest_examples=nearest_samples, reference_key='title_snippet')
    # print('-------chatGLM-RAG的回答:--------')

    scores, nearest_samples, result, output_text = semantic_search_engine.chatGLM_RAG_oneshot(question, query,
                                                                                              websearch_engine='google',
                                                                                              glm_model='glm-4-air',
                                                                                              GLM_websearch_enable=True,
                                                                                              k=3, rn=10)

    print(output_text)
