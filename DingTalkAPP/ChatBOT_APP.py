# !/usr/bin/env python
import re
import sys

sys.path.append('../GLM')  # 将上一级目录的/GLM目录添加到系统路径中
from semantic_search_by_zhipu import chatGLM_by_semanticSearch_amid_SerpAPI

import argparse
import logging
from dingtalk_stream import AckMessage
import dingtalk_stream
import configparser
import zhipuai


def config_read(config_path, section='DingTalkAPP_chatGLM', option1='Client_ID', option2='client_secret'):
    config = configparser.ConfigParser()
    config.read(config_path, encoding='utf-8')
    client_ID = config.get(section=section, option=option1)
    client_secret = config.get(section=section, option=option2)
    return client_ID, client_secret


def chatGLM_RAG_generate(question, query, search_engine=None, LLM='glm-3-Turbo',
                         config_path_serp=r"e:/Python_WorkSpace/config/SerpAPI.ini",
                         config_path_zhipuai=r"e:/Python_WorkSpace/config/zhipuai_SDK.ini"):
    """
    search_engine: [None, "Google", "Baidu"],三者之一; 当search_engine为None时，不进行搜索，直接回答问题
    LLM: zhipuAI的模型选择: 'glm-3-Turbo', 或者 'glm-4'
    question: 问题;
    query: 用于web_search的query
    """
    config_path_serp = config_path_serp
    config_path_zhipuai = config_path_zhipuai

    semantic_search_engine = chatGLM_by_semanticSearch_amid_SerpAPI(engine=search_engine,
                                                                    serp_key_path=config_path_serp,
                                                                    zhipu_key_path=config_path_zhipuai, )
    websearch_flag = False
    if search_engine is None:
        websearch_flag = False
    elif search_engine.lower() in ["google", "baidu"]:
        websearch_flag = True
    scores, nearest_samples, result, output_text = semantic_search_engine.chatGLM_RAG_oneshot(question, query, LLM,
                                                                                              web_search_enable=websearch_flag,
                                                                                              k=3, rn=10)
    return output_text


def characterGLMAPI_completion_create(api_key, prompt, history_prompt, bot_info, bot_name, user_info="用户", user_name="用户"):
    """
    角色扮演模型characterglm需要zhipuai库版本<=1.07
    实现同步调用下的角色模型输出,包含历史聊天记录并入输入
    api_key: zhipuai_api_key
    prompt: list;调用角色扮演模型时，将当前对话信息列表作为提示输入给模型; 按照 {"role": "user", "content": "你好"}
                    的键值对形式进行传参; 总长度超过模型最长输入限制后会自动截断，需按时间由旧到新排序
    history_prompt: list;
    bot_info:角色信息
    bot_name:角色名称
    user_info:用户信息
    user_name:用户名称，默认值为"用户"

    """
    zhipuai.api_key = api_key
    prompt = history_prompt.extend(prompt)
    try:
        response = zhipuai.model_api.invoke(
            model="characterglm",
            prompt=prompt,
            temperature=0.9,
            top_p=0.7,
            meta={
                "bot_info": bot_info,
                "bot_name": bot_name,
                "user_info": user_info,
                "user_name": user_name,
            },
        )
        result = response.choices[0].content
    except Exception as e:
        result = f"An error occurred: {e.args}"

    return result


def setup_logger():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    # 控制台handler:
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(
        logging.Formatter('%(asctime)s %(name)-8s %(levelname)-8s %(message)s [%(filename)s:%(lineno)d]'))
    logger.addHandler(console_handler)

    # 文件handler:
    file_handler = logging.FileHandler('log.log')
    file_handler.setFormatter(
        logging.Formatter('%(asctime)s %(name)-8s %(levelname)-8s %(message)s [%(filename)s:%(lineno)d]'))
    logger.addHandler(file_handler)
    return logger


def define_options():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--client_id', dest='client_id', required=True,
        help='app_key or suite_key from https://open-dev.digntalk.com'
    )
    parser.add_argument(
        '--client_secret', dest='client_secret', required=True,
        help='app_secret or suite_secret from https://open-dev.digntalk.com'
    )
    options = parser.parse_args()
    return options


class EchoTextHandler(dingtalk_stream.ChatbotHandler):
    def __init__(self, logger: logging.Logger = None):
        super(dingtalk_stream.ChatbotHandler, self).__init__()
        if logger:
            self.logger = logger

    async def process(self, callback: dingtalk_stream.CallbackMessage):
        incoming_message = dingtalk_stream.ChatbotMessage.from_dict(callback.data)
        question = incoming_message.text.content.strip()
        query = question
        logger.info(question)
        text = '收到您问题了,请待我调用GLM-3-Turbo大模型来回答:'
        self.reply_text(text, incoming_message)
        question, search_engine = split_string(question)
        # self.reply_text(search_engine, incoming_message)
        # self.reply_text(question, incoming_message)

        text = chatGLM_RAG_generate(question=question, query=query, search_engine=search_engine, LLM='glm-3-Turbo',
                                    config_path_serp=config_path_serp, config_path_zhipuai=config_path_zhipuai)
        self.reply_text(text, incoming_message)
        logger.info(text)
        return AckMessage.STATUS_OK, 'OK'


def split_string(text):
    pattern = r'\|<.+>\|'
    match = re.search(pattern, text)
    if match:
        search_engine = match.group(0)[2:-2]
        question = text[:match.start()]

    else:
        question = text
        search_engine = None
    return str(question), str(search_engine)


if __name__ == '__main__':
    logger = setup_logger()
    # options = define_options()
    config_path_dtApp = r"e:/Python_WorkSpace/config/DingTalk_APP.ini"
    config_path_serp = r"e:/Python_WorkSpace/config/SerpAPI.ini"
    config_path_zhipuai = r"e:/Python_WorkSpace/config/zhipuai_SDK.ini"
    client_id, client_secret = config_read(config_path_dtApp)

    bot_info = "杨幂,1986年9月12日出生于北京市，中国内地影视女演员、流行乐歌手、影视制片人。2005年，杨幂进入北京电影学院表演系本科班就读。2006年，因出演金庸武侠剧《神雕侠侣》崭露头角。2008年，凭借古装剧《王昭君》获得第24届中国电视金鹰奖观众喜爱的电视剧女演员奖提名 。2009年，在“80后新生代娱乐大明星”评选中被评为“四小花旦”。2011年，凭借穿越剧《宫锁心玉》赢得广泛关注 ，并获得了第17届上海电视节白玉兰奖观众票选最具人气女演员奖。2012年，不仅成立杨幂工作室，还凭借都市剧《北京爱情故事》获得了多项荣誉 。2015年，主演的《小时代》系列电影票房突破18亿人民币 。2016年，其主演的职场剧《亲爱的翻译官》取得全国年度电视剧收视冠军 。2017年，杨幂主演的神话剧《三生三世十里桃花》获得颇高关注；同年，她还凭借科幻片《逆时营救》获得休斯顿国际电影节最佳女主角奖 。2018年，凭借古装片《绣春刀Ⅱ：修罗战场》获得北京大学生电影节最受大学生欢迎女演员奖 [4]；。2014年1月8日，杨幂与刘恺威在巴厘岛举办了结婚典礼。同年6月1日，在香港产下女儿小糯米。"
    bot_name = "大幂"
    user_info = "粉丝"
    user_name = "用户"

    credential = dingtalk_stream.Credential(client_id, client_secret)
    client = dingtalk_stream.DingTalkStreamClient(credential)
    client.register_callback_handler(dingtalk_stream.chatbot.ChatbotMessage.TOPIC, EchoTextHandler(logger))
    client.start_forever()
