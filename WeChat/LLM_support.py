from datasets import Dataset
import numpy as np
from zhipuai import ZhipuAI
from collections import defaultdict

import sys
sys.path.append("../GLM")  # 将上一级目录的/GLM目录添加到系统路径中
from semantic_search_by_zhipu import chatGLM_by_semanticSearch_amid_SerpAPI


def chatGLM_RAG_generate(
    question,
    query,
    search_engine=None,
    LLM="glm-3-Turbo",
    config_path_serp=r"e:/Python_WorkSpace/config/SerpAPI.ini",
    config_path_zhipuai=r"e:/Python_WorkSpace/config/zhipuai_SDK.ini",
):
    """
    search_engine: [None, "Google", "Baidu"],三者之一; 当search_engine为None时，不进行搜索，直接回答问题
    LLM: zhipuAI的模型选择: 'glm-3-Turbo', 或者 'glm-4'
    question: 问题;
    query: 用于web_search的query
    """
    config_path_serp = config_path_serp
    config_path_zhipuai = config_path_zhipuai

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