from datasets import Dataset
import numpy as np
from SerpAPI_fn import serpapi_GoogleSearch, serpapi_BaiduSearch, get_api_key
from zhipuai import ZhipuAI
from collections import defaultdict


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

    def __init__(self, engine="Baidu",
                 serp_key_path=None, serp_key_section='Serp_API', serp_key_option='api_key',
                 zhipu_key_path=None, zhipu_key_section='zhipuai_SDK_API', zhipu_key_option='api_key',
                 ):
        """
        :param engine: "Baidu","Google",or "None",分别表示,以baidu,google为搜索引擎,搜索指定query,或者None表示不从搜索引擎获取数据
        """
        zhipu_api_key = get_api_key(config_file_path=zhipu_key_path, section=zhipu_key_section,
                                    option=zhipu_key_option)
        self.zhipuai_client = ZhipuAI(api_key=zhipu_api_key)
        if engine in ["Google", "Baidu"]:
            self.serp_api_key = get_api_key(config_file_path=serp_key_path, section=serp_key_section,
                                            option=serp_key_option)
        self.engine = engine
        # self.query = query

    def web_search(self, query,
                   location='Hong Kong', hl='zh-cn', gl='cn', tbs=None, tbm=None, num=30,
                   ct=2, rn=50, ):
        """
        Google Search,或者Baidu Search,将搜索结果,提取title,snippet,生成字典
        :return:
        """
        search_results_dict = defaultdict(list)
        if self.engine.lower() == 'Google'.lower():
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
        elif self.engine.lower() == 'Baidu'.lower():
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

    def chatGLM_RAG(self, question, query, glm_model='GLM-4', web_search_enable=False, nearest_examples=None,
                    reference_key='title_snippet', ):
        """
        将web_search搜索,语义匹配之后的n个句子作为参考信息,送入GLM模型生成回复
        :param query:
        :param nearest_examples:
        :param reference_key:
        :return:
        """
        reference = ';'.join(nearest_examples[reference_key])
        if reference is not None:
            question_template = f"请基于以下参考信息生成回答: {reference}\n问题: {question}"
        else:
            question_template = question
        web_search_parameters = {"enable": web_search_enable,
                                 "search_query": query, }
        response = self.zhipuai_client.chat.completions.create(
            model=glm_model,  # 填写需要调用的模型名称
            messages=[
                {"role": "user", "content": question_template},
            ],
            tools=[
                {
                    "type": "web_search",
                    "web_search": web_search_parameters,
                }
            ],
            stream=False,
        )
        # for chunk in response:
        #     print(chunk.choices[0].delta)
        result = response.choices[0].message.content

        return result

    def chatGLM_RAG_oneshot(self, question, query, glm_model='GLM-4', web_search_enable=False, k=3, rn=10):
        search_results_dict = self.web_search(query, rn=rn)
        scores, nearest_samples = self.semantic_websearch(query, search_results_dict, k=k, )
        # for score, sample in zip(scores, nearest_samples['title_snippet']):
        #     print(f'----语义搜索最佳的{k}个结果:---------')
        #     print(score, sample)
        result = self.chatGLM_RAG(question, query, glm_model=glm_model, web_search_enable=web_search_enable,
                                  nearest_examples=nearest_samples, reference_key='title_snippet')
        print(f'-------web_search_enable={web_search_enable}后,模型{glm_model}/RAG的回答:--------')
        print(result)
        return scores, nearest_samples,result


if __name__ == "__main__":
    config_path_serp = r"e:/Python_WorkSpace/config/SerpAPI.ini"
    config_path_zhipuai = r"e:/Python_WorkSpace/config/zhipuai_SDK.ini"

    question = 'Tucker Carlson与普京的会面,都谈了些什么?'
    query = '塔克卡尔森与普京的会面,都谈了些什么?'  # 用于web_search
    semantic_search_engine = chatGLM_by_semanticSearch_amid_SerpAPI(engine='Baidu', serp_key_path=config_path_serp,
                                                                    zhipu_key_path=config_path_zhipuai, )
    # search_results_dict = semantic_search_engine.web_search(query, rn=10)
    # scores, nearest_samples = semantic_search_engine.semantic_websearch(query, search_results_dict, k=3, )
    # for score, sample in zip(scores, nearest_samples['title_snippet']):
    #     print('----语义搜索最佳的k个结果:---------')
    #     print(score, sample)
    # result = semantic_search_engine.chatGLM_RAG(question, query, 'GLM-3-Turbo', web_search_enable=True,
    #                                             nearest_examples=nearest_samples, reference_key='title_snippet')
    # print('-------chatGLM-RAG的回答:--------')

    semantic_search_engine.chatGLM_RAG_oneshot(question,query,'GLM-3-Turbo', web_search_enable=True,k=3,rn=10)
