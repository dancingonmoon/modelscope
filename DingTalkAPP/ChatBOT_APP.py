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


def config_read(config_path, section='DingTalkAPP_chatGLM', option1='Client_ID', option2='client_secret'):
    config = configparser.ConfigParser()
    config.read(config_path, encoding='utf-8')
    client_ID = config.get(section=section, option=option1)
    client_secret = config.get(section=section, option=option2)
    return client_ID, client_secret


def chatGLM_RAG_generate(question, query, search_engine=None,
                         config_path_serp=r"e:/Python_WorkSpace/config/SerpAPI.ini",
                         config_path_zhipuai=r"e:/Python_WorkSpace/config/zhipuai_SDK.ini"):
    """
    search_engine: [None, "Google", "Baidu"],三者之一; 当search_engine为None时，不进行搜索，直接回答问题
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
    scores, nearest_samples, result, output_text = semantic_search_engine.chatGLM_RAG_oneshot(question, query,
                                                                                              'GLM-3-Turbo',
                                                                                              web_search_enable=websearch_flag,
                                                                                              k=3, rn=10)
    return output_text


def setup_logger():
    logger = logging.getLogger()
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter('%(asctime)s %(name)-8s %(levelname)-8s %(message)s [%(filename)s:%(lineno)d]'))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
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
        text = '收到您问题了,请待我调用GLM-3-Turbo大模型来回答:'
        self.reply_text(text, incoming_message)
        question, search_engine = split_string(question)
        # self.reply_text(search_engine, incoming_message)
        # self.reply_text(question, incoming_message)
        # if search_engine == "None":
        #     search_engine = None
        text = chatGLM_RAG_generate(question=question, query=query, search_engine=search_engine,
                                    config_path_serp=config_path_serp, config_path_zhipuai=config_path_zhipuai)
        self.reply_text(text, incoming_message)
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
    config_path_dtApp = r"l:/Python_WorkSpace/config/DingTalk_APP.ini"
    config_path_serp = r"l:/Python_WorkSpace/config/SerpAPI.ini"
    config_path_zhipuai = r"l:/Python_WorkSpace/config/zhipuai_SDK.ini"
    client_id, client_secret = config_read(config_path_dtApp)

    credential = dingtalk_stream.Credential(client_id, client_secret)
    client = dingtalk_stream.DingTalkStreamClient(credential)
    client.register_callback_handler(dingtalk_stream.chatbot.ChatbotMessage.TOPIC, EchoTextHandler(logger))
    client.start_forever()
    #
    # question = "请问美国|哈哈|如何?  |<Google>|ss"
    #
    # question, search_engine = split_string(question)
    # print(question, search_engine)
