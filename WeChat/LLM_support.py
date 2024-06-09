#!/usr/bin/python3
# -*- coding: utf-8 -*-

import re

import sys
sys.path.append("../GLM")  # 将上一级目录的/GLM目录添加到系统路径中
from semantic_search_by_zhipu import chatGLM_by_semanticSearch_amid_SerpAPI




def chatGLM_RAG_generate(
    question,
    query,
    search_engine=None,
    LLM="glm-4-air",
    config_path_serp=r"l:/Python_WorkSpace/config/SerpAPI.ini",
    config_path_zhipuai=r"l:/Python_WorkSpace/config/zhipuai_SDK.ini",
):
    """
    search_engine: [None, "Google", "Baidu"],三者之一; 当search_engine为None时，不进行搜索，直接回答问题
    LLM: zhipuAI的模型选择: 'glm-3-Turbo', 或者 'glm-4'
    question: 问题;
    query: 用于web_search的query
    """

    semantic_search_engine = chatGLM_by_semanticSearch_amid_SerpAPI(
        engine=search_engine,
        serp_key_path=config_path_serp,
        zhipu_key_path=config_path_zhipuai,
    )
    websearch_flag = False
    if search_engine is None:
        websearch_flag = False
    elif search_engine.lower() in ["google", "baidu"]:
        websearch_flag = True
    (
        scores,
        nearest_samples,
        result,
        output_text,
    ) = semantic_search_engine.chatGLM_RAG_oneshot(
        question, query, LLM, web_search_enable=websearch_flag, k=3, rn=10
    )
    return output_text

def split_string(
    text,
    method=0,
):
    """
    method=0: 判断字符串中尾部是否有|<search_engine>|, 将字符串输出成question,以及search_engine两部分
    method=1: 判断字符串中尾部是否有" 1" or " 2", " 1"表示search_engine="baidu"; " 2"表示search_engine="google" 将字符串输出成question,以及search_engine两部分
    :param text:
    :return:
    """
    search_engine = None
    question = text

    if method == 0:  # |<baidu>| or |<google>|
        pattern = r"\|<.+>\|"
        match = re.search(pattern, text)
        if match:
            search_engine = match.group(0)[2:-2]
            question = text[: match.start()]

    elif method == 1:  # "空格1" or "空格2"
        if text[-2:] == " 1":
            search_engine = "baidu"
            question = text[-3]
        elif text[-2:] == " 2":
            search_engine = "google"
            question = text[-3]

    return question, search_engine


if __name__ == "__main__":

    question = "俄乌战争北约会出兵吗?"
    query = "俄乌战争北约立场"
    output_text = chatGLM_RAG_generate(
        question, query, search_engine=None, LLM="glm-3-turbo"
    )
    print(output_text)
