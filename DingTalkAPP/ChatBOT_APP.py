# !/usr/bin/env python
# -*- coding: utf-8 -*-
import re
import sys
import time
import datetime
import requests

sys.path.append('../GLM')  # 将上一级目录的/GLM目录添加到系统路径中
# from semantic_search_by_zhipu import chatGLM_by_semanticSearch_amid_SerpAPI

import argparse
import logging
from dingtalk_stream import AckMessage
import dingtalk_stream
import configparser
import zhipuai
from chatbotClass_utilies import ChatbotMessage_Utilies, ChatbotHandler_utilies, OpenAPI_SendMessage

from TTS_SSML import TTS_threadsRUN, get_audio_duration, emotion_classification


def config_read(config_path, section='DingTalkAPP_chatGLM', option1='Client_ID', option2=None):
    """
    option2 = None 时,仅输出第一个option1的值; 否则输出section下的option1与option2两个值
    """
    config = configparser.ConfigParser()
    config.read(config_path, encoding='utf-8')
    option1_value = config.get(section=section, option=option1)
    if option2 is not None:
        option2_value = config.get(section=section, option=option2)
        return option1_value, option2_value
    else:
        return option1_value


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


def characterGLMAPI_completion_create(zhipuai, prompt: list, history_prompt: list = None,
                                      bot_info: str = None, bot_name: str = None,
                                      user_info: str = "用户", user_name: str = "用户") -> str:
    """
    角色扮演模型characterglm需要zhipuai库版本<=1.07
    实现同步调用下的角色模型输出,包含历史聊天记录并入输入
    zhipuai: zhipuai库(1.0.7版)已经load api_key, 即:zhipuai.api_key = api_key
    prompt: list;调用角色扮演模型时，将当前对话信息列表作为提示输入给模型; 按照 {"role": "user", "content": "你好"}
                    的键值对形式进行传参; 总长度超过模型最长输入限制后会自动截断，需按时间由旧到新排序
    history_prompt: list; 当history_prompt=None时,表明是不附加历史,仅仅是prompt作为characterglm输入;否则,history_prompt为list
    bot_info:角色信息
    bot_name:角色名称
    user_info:用户信息
    user_name:用户名称，默认值为"用户"
    """
    # zhipuai.api_key = api_key
    if history_prompt is None:
        history_prompt = []
    history_prompt.extend(prompt)

    try:
        response = zhipuai.model_api.invoke(
            model="characterglm",
            prompt=history_prompt,
            temperature=0.9,
            top_p=0.7,
            meta={
                "bot_info": bot_info,
                "bot_name": bot_name,
                "user_info": user_info,
                "user_name": user_name,
            },
        )
        # logger.info(response)
        if 'error' in response:
            result = response['error']
        else:
            result = response['data']['choices'][0]['content']

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
    """
    GLM-3-Turbo LLM 聊天
    """

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
        question, search_engine = split_string(question, method=1)
        # self.reply_text(search_engine, incoming_message)
        # self.reply_text(question, incoming_message)

        text = chatGLM_RAG_generate(question=question, query=query, search_engine=search_engine, LLM='glm-3-Turbo',
                                    config_path_serp=config_path_serp, config_path_zhipuai=config_path_zhipuai)
        # self.reply_text(text, incoming_message)
        self.reply_markdown('GLM回答:', text, incoming_message)
        logger.info(text)
        return AckMessage.STATUS_OK, 'OK'


history_prompt = []  # 初始值定义为空列表,以与后续列表进行extend()拼接


class PromptTextHandler(dingtalk_stream.ChatbotHandler):
    """
    characterGLM 聊天
    历史prompt并入prompt,对话循环
    """

    def __init__(self, logger: logging.Logger = None):
        super(dingtalk_stream.ChatbotHandler, self).__init__()
        if logger:
            self.logger = logger

    async def process(self, callback: dingtalk_stream.CallbackMessage):
        global history_prompt
        callback_data = callback.data
        incoming_message = dingtalk_stream.ChatbotMessage.from_dict(callback_data)
        text = incoming_message.text.content.strip()
        if change_topic_str_Detect(text):
            history_prompt = []
        prompt = [{"role": "user", "content": text}]

        logger.info(f"user:{text}")
        text = characterGLMAPI_completion_create(zhipuai=zhipuai, prompt=prompt, history_prompt=history_prompt,
                                                 bot_name=bot_name, bot_info=bot_info, user_name=user_name,
                                                 user_info=user_info)
        # logger.info(f"之前:{text}")
        text = re.sub(r'^["]+|[\\n+]|[\n+"]$', '', text)  # 去除字符串中的换行符或者回车符
        # logger.info(f"去除标点:{text}")
        history_prompt.extend([{"role": "assistant", "content": text}])
        self.reply_text(text, incoming_message)
        logger.info(f"assistant:{text}")
        # logger.info(history_prompt)
        return AckMessage.STATUS_OK, 'OK'


def split_string(text, method=0, ):
    """
    method=0: 判断字符串中尾部是否有|<search_engine>|, 将字符串输出成question,以及search_engine两部分
    method=1: 判断字符串中尾部是否有" 1" or " 2", " 1"表示search_engine="baidu"; " 2"表示search_engine="google" 将字符串输出成question,以及search_engine两部分
    :param text:
    :return:
    """
    search_engine = None
    question = text

    if method == 0:  # |<baidu>| or |<google>|
        pattern = r'\|<.+>\|'
        match = re.search(pattern, text)
        if match:
            search_engine = match.group(0)[2:-2]
            question = text[:match.start()]

    elif method == 1:  # "空格1" or "空格2"
        if text[-2:] == " 1":
            search_engine = "baidu"
            question = text[-3]
        elif text[-2:] == " 2":
            search_engine = "google"
            question = text[-3]

    return question, search_engine


def change_topic_str_Detect(text: str) -> bool:
    """
    判读字符串是否包括|<换个话题>|,或者|<change topic>|
    """
    pattern0 = r'\|<换个话题>\|'
    pattern1 = r'\|<change topic>\|'
    match0 = re.search(pattern0, text)
    match1 = re.search(pattern1, text)
    if match0 or match1:
        return True
    else:
        return False


class VoiceChatHandler(ChatbotHandler_utilies):
    """
    语音 聊天
    """

    def __init__(self, accessKey_id, accessKey_secret, region_id='cn-shanghai', appKey=None, tts_name=None,
                 audio_path=None, aformat='wav', voice='xiaoyun', speech_rate=0, pitch_rate=0, wait_complete=False,
                 enable_subtitle=False, enable_ptts=False, callbacks: list = [], logger: logging.Logger = None
                 ):
        super(VoiceChatHandler, self).__init__()
        self.accessKey_id = accessKey_id
        self.accessKey_secret = accessKey_secret
        self.region_id = region_id
        self.appKey = appKey
        self.tts_name = tts_name
        self.audio_path = audio_path
        self.aformat = aformat
        self.voice = voice
        self.speech_rate = speech_rate
        self.pitch_rate = pitch_rate
        self.wait_complete = wait_complete
        self.enable_subtitle = enable_subtitle
        self.enable_ptts = enable_ptts
        self.callbacks = callbacks

        self.logger = logger

    async def process(self, callback: dingtalk_stream.CallbackMessage):
        global history_prompt
        callback_data = callback.data
        incoming_message = ChatbotMessage_Utilies.from_dict(callback_data)

        text = incoming_message.text.content.strip()
        if change_topic_str_Detect(text):
            history_prompt = []
        prompt = [{"role": "user", "content": text}]

        self.logger.info(f"user:{text}")
        text = characterGLMAPI_completion_create(zhipuai=zhipuai, prompt=prompt, history_prompt=history_prompt,
                                                 bot_name=bot_name, bot_info=bot_info, user_name=user_name,
                                                 user_info=user_info)
        text = re.sub(r'^["]+|[\\n+]|[\n+"]$', '', text)  # 去除字符串中的换行符或者回车符
        # self.logger.info(f"生成的text:{text}")
        history_prompt.extend([{"role": "assistant", "content": text}])

        # Text To Speech:
        tts_instance = TTS_threadsRUN(self.accessKey_id, self.accessKey_secret, appkey=self.appKey,
                                      tts_name=self.tts_name,
                                      audio_path=self.audio_path, aformat=self.aformat, voice=self.voice,
                                      speech_rate=self.speech_rate,
                                      pitch_rate=self.pitch_rate, wait_complete=self.wait_complete,
                                      enable_subtitle=self.enable_subtitle,
                                      enable_ptts=self.enable_ptts, callbacks=self.callbacks, logger=self.logger)
        # 情绪识别:
        ssml_label, _, _ = emotion_classification(accessKey_id, accessKey_secret, text)
        # print(ssml_label, emo_label, s_dict)
        tts_instance.start(text, ssml_label, ssml_intensity=1.5)
        while tts_instance.completion_status is False:
            # self.logger.info(f"tts_instance.completion_status: {tts_instance.completion_status}")
            time.sleep(0.05)
        # self.logger.info(f"tts_instance.completion_status: {tts_instance.completion_status}")
        # 1)获取存盘的音频文件的时长; 2)TTS, 上传获取mediaId,:
        if self.audio_path is None:
            duration = get_audio_duration(tts_instance.BytesIO, sample_rate=16000)
            duration = int(duration)
            print(f'duration:{duration}')
            mediaId = self.upload2media_id(media_content=tts_instance.BytesIO.read(), media_type='voice')
        else:
            duration = get_audio_duration(self.audio_path, sample_rate=16000)
            duration = int(duration)
            mediaId = self.upload2media_id(media_content=self.audio_path, media_type='voice')
        logger.info(f"voice media_id: {mediaId}")
        # 发送voice message:
        # self.reply_voice_http(mediaId, duration, incoming_message)  # http方式发送voice message reqeust格式有误;
        self.reply_voice_SDK(mediaId, duration, incoming_message)
        self.reply_text(text, incoming_message)
        self.logger.info(f"assistant:{text}")
        # logger.info(history_prompt)
        return AckMessage.STATUS_OK, 'OK'


if __name__ == '__main__':

    characterGLM_chat_flag = True  # True时,characterglm,需要zhipuai库版本<=1.07
    voiceMessage_chat_flag = True

    if characterGLM_chat_flag is False:
        from semantic_search_by_zhipu import chatGLM_by_semanticSearch_amid_SerpAPI

    logger = setup_logger()
    # options = define_options()
    config_path_dtApp = r"l:/Python_WorkSpace/config/DingTalk_APP.ini"
    config_path_serp = r"l:/Python_WorkSpace/config/DingTalk_APP.ini"
    config_path_zhipuai = r"l:/Python_WorkSpace/config/DingTalk_APP.ini"
    config_path_aliyunsdk = r"l:/Python_WorkSpace/config/aliyunsdkcore.ini"

    # bot_info = "杨幂,1986年9月12日出生于北京市，中国内地影视女演员、流行乐歌手、影视制片人。2005年，杨幂进入北京电影学院表演系本科班就读。2006年，因出演金庸武侠剧《神雕侠侣》崭露头角。2008年，凭借古装剧《王昭君》获得第24届中国电视金鹰奖观众喜爱的电视剧女演员奖提名 。2009年，在“80后新生代娱乐大明星”评选中被评为“四小花旦”。2011年，凭借穿越剧《宫锁心玉》赢得广泛关注 ，并获得了第17届上海电视节白玉兰奖观众票选最具人气女演员奖。2012年，不仅成立杨幂工作室，还凭借都市剧《北京爱情故事》获得了多项荣誉 。2015年，主演的《小时代》系列电影票房突破18亿人民币 。2016年，其主演的职场剧《亲爱的翻译官》取得全国年度电视剧收视冠军 。2017年，杨幂主演的神话剧《三生三世十里桃花》获得颇高关注；同年，她还凭借科幻片《逆时营救》获得休斯顿国际电影节最佳女主角奖 。2018年，凭借古装片《绣春刀Ⅱ：修罗战场》获得北京大学生电影节最受大学生欢迎女演员奖 [4]；。2014年1月8日，杨幂与刘恺威在巴厘岛举办了结婚典礼。同年6月1日，在香港产下女儿小糯米。"
    bot_info = "刘亦菲（Crystal Liu,1987年8月25日-）,生于湖北省武汉市,毕业于北京电影学院,美籍华裔女演员、歌手.2002年,因出演电视剧《金粉世家》中白秀珠一角踏入演艺圈.2003年,因主演武侠剧《天龙八部》王语嫣崭露头角.2004年,凭借仙侠剧《仙剑奇侠传》赵灵儿一角获得了高人气.2005年,因在《神雕侠侣》中饰演小龙女受到关注.2006年,发行首张国语专辑《刘亦菲》和日语专辑《All My Words》.2008年起,转战大银幕,凭借好莱坞电影《功夫之王》成为首位荣登IMDB电影新人排行榜榜首的亚洲女星.2020年3月,为电影《花木兰》演唱中文主题曲《自己》；同年9月,主演的电影《花木兰》在Disney+上线,在剧中饰演花木兰；11月,凭借《花木兰》获首届评论家选择超级奖动作电影最佳女演员提名；2022年12月21日,凭借《梦华录》获第十三届澳门国际电视节金莲花最佳女主角奖.9月1日,凭借《梦华录》获得首届金熊猫奖电视剧单元最佳女主角提名.2023年12月28日,凭借《去有风的地方》获得第十四届澳门国际电视节'金莲花'最佳女主角奖"
    bot_name = "茜茜"
    user_info = "喜欢刘亦菲的男孩一枚"
    user_name = "用户"
    voice = 'zhiyan_emo'  # zhiyan的声音,略微的更女性化些;
    today = datetime.datetime.today().strftime('%y%m%d')
    # tts_out_path = f'tts_{voice}_{today}.wav'
    tts_out_path = None
    speech_data = 0

    if characterGLM_chat_flag:  # 角色扮演机器人聊天
        zhipuai_key = config_read(config_path_zhipuai, section="zhipuai_SDK_API", option1="api_key", option2=None)
        zhipuai.api_key = zhipuai_key
        client_id, client_secret = config_read(config_path_dtApp, section="DingTalkAPP_charGLM", option1='client_id',
                                               option2='client_secret')
        credential = dingtalk_stream.Credential(client_id, client_secret)
        client = dingtalk_stream.DingTalkStreamClient(credential)
        if voiceMessage_chat_flag is False:  # 角色扮演机器人, 文本聊天
            client.register_callback_handler(dingtalk_stream.chatbot.ChatbotMessage.TOPIC, PromptTextHandler(logger))
        else:  # 角色扮演机器人,  语音聊天
            accessKey_id, accessKey_secret = config_read(config_path_aliyunsdk, section='aliyunsdkcore',
                                                         option1='AccessKey_ID',
                                                         option2='AccessKey_Secret')
            appKey = config_read(config_path_aliyunsdk, section='APP_tts', option1='AppKey')
            client.register_callback_handler(ChatbotMessage_Utilies.TOPIC,
                                             VoiceChatHandler(accessKey_id, accessKey_secret, region_id='cn-shanghai',
                                                              appKey=appKey, tts_name='Example',
                                                              audio_path=tts_out_path, aformat='wav', voice=voice,
                                                              speech_rate=0, pitch_rate=0, wait_complete=False,
                                                              enable_subtitle=False, enable_ptts=False,
                                                              callbacks=[], logger=logger))

    else:  # GLM 办公助手, 文本聊
        client_id, client_secret = config_read(config_path_dtApp, section="DingTalkAPP_chatGLM", option1='client_id',
                                               option2='client_secret')
        credential = dingtalk_stream.Credential(client_id, client_secret)
        client = dingtalk_stream.DingTalkStreamClient(credential)
        client.register_callback_handler(dingtalk_stream.chatbot.ChatbotMessage.TOPIC, EchoTextHandler(logger))

    client.start_forever()
