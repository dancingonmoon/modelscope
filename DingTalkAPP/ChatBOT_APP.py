# !/usr/bin/env python
# -*- coding: utf-8 -*-
import re
import sys
import time
import datetime
from zhipuai import ZhipuAI

# import requests

sys.path.append("../GLM")  # 将上一级目录的/GLM目录添加到系统路径中
from semantic_search_by_zhipu import chatGLM_by_semanticSearch_amid_SerpAPI

import argparse
import logging
from dingtalk_stream import AckMessage
import dingtalk_stream
import configparser
import zhipuai
from chatbotClass_utilies import (
    ChatbotMessage_Utilies,
    ChatbotHandler_utilies,
    OpenAPI_SendMessage,
)

from TTS_SSML import (
    aliyun_TTS_threadsRUN,
    get_audio_duration,
    emotion_classification,
    wav2ogg,
    azure_TTS,
)


def config_read(
        config_path, section="DingTalkAPP_chatGLM", option1="Client_ID", option2=None
):
    """
    option2 = None 时,仅输出第一个option1的值; 否则输出section下的option1与option2两个值
    """
    config = configparser.ConfigParser()
    config.read(config_path, encoding="utf-8")
    option1_value = config.get(section=section, option=option1)
    if option2 is not None:
        option2_value = config.get(section=section, option=option2)
        return option1_value, option2_value
    else:
        return option1_value


def characterGLMAPI_completion_create(
        zhipuai_client,
        messages: list,
        history_messages: list = None,
        bot_info: str = None,
        bot_name: str = None,
        user_info: str = "用户",
        user_name: str = "用户",
) -> str:
    """
    角色扮演模型characterglm
    实现同步调用下的角色模型输出,包含历史聊天记录并入输入
    zhipuai_client: 已经load api_key 即: zhipuai_client = ZhipuAI(api_key=zhi);
    messages: list;调用角色扮演模型时，将当前对话信息列表作为提示输入给模型; 按照 {"role": "user", "content": "你好"}
                    的键值对形式进行传参; 总长度超过模型最长输入限制后会自动截断，需按时间由旧到新排序
    history_messages: list; 当history_messages=None时,表明是不附加历史,仅仅是messages作为characterglm输入;否则,history_messages为list
    bot_info:角色信息
    bot_name:角色名称
    user_info:用户信息
    user_name:用户名称，默认值为"用户"
    """
    # zhipuai.api_key = api_key
    if history_messages is None:
        history_messages = []
    history_messages.extend(messages)

    try:
        response = zhipuai_client.chat.completions.create(
            model="characterglm",
            messages=history_messages,
            temperature=0.9,
            top_p=0.7,
            meta={
                "bot_info": bot_info,
                "bot_name": bot_name,
                "user_info": user_info,
                "user_name": user_name,
            },
            stream=False
        )
        # logger.info(response)
        if "error" in response:
            result = response["error"]
        else:
            result = response.choices[0].message.content

    except Exception as e:
        result = f"An error occurred: {e.args}"

    return result


def setup_logger():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    # 控制台handler:
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s %(name)-8s %(levelname)-8s %(gradio_message)s [%(filename)s:%(lineno)d]"
        )
    )
    logger.addHandler(console_handler)

    # 文件handler:
    file_handler = logging.FileHandler("log.log", encoding="utf-8")
    file_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s %(name)-8s %(levelname)-8s %(gradio_message)s [%(filename)s:%(lineno)d]"
        )
    )
    logger.addHandler(file_handler)
    return logger


def define_options():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--client_id",
        dest="client_id",
        required=True,
        help="app_key or suite_key from https://open-dev.digntalk.com",
    )
    parser.add_argument(
        "--client_secret",
        dest="client_secret",
        required=True,
        help="app_secret or suite_secret from https://open-dev.digntalk.com",
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
        self.logger.info(question)
        text = "收到您问题了,请待我调用GLM-4-air大模型来回答:"
        self.reply_text(text, incoming_message)
        question, search_engine, glm_web_search = split_string(question, method=1)
        # self.reply_text(search_engine, incoming_message)
        # self.reply_text(question, incoming_message)

        scores, nearest_samples, result, text = semantic_search_engine.chatGLM_RAG_oneshot(
            question=question,
            query=query,
            websearch_engine=search_engine,
            glm_model="glm-4-air",
            GLM_websearch_enable=glm_web_search
        )
        # self.reply_text(text, incoming_message)
        self.reply_markdown("GLM回答:", text, incoming_message)
        self.logger.info(text)
        return AckMessage.STATUS_OK, "OK"


history_messages = []  # 初始值定义为空列表,以与后续列表进行extend()拼接


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
        global history_messages
        callback_data = callback.data
        incoming_message = dingtalk_stream.ChatbotMessage.from_dict(callback_data)
        # 文本,语音消息分别判断后处理:
        if incoming_message.message_type == "text":
            text = incoming_message.text.content.strip()
        elif incoming_message.message_type == "audio":
            text = incoming_message.audio_recognition
            self.reply_text(f"你:{text}", incoming_message)

        if change_topic_str_Detect(text):
            history_messages = []
        prompt = [{"role": "user", "content": text}]

        logger.info(f"user:{text}")
        text = characterGLMAPI_completion_create(
            zhipuai_client=zhipuai_client,
            messages=prompt,
            history_messages=history_messages,
            bot_name=bot_name,
            bot_info=bot_info,
            user_name=user_name,
            user_info=user_info,
        )
        # logger.info(f"之前:{text}")
        text = re.sub(r'^["]+|[\\n+]|[\n+"]$', "", text)  # 去除字符串中的换行符或者回车符
        # logger.info(f"去除标点:{text}")
        history_messages.extend([{"role": "assistant", "content": text}])
        self.reply_markdown("response", text, incoming_message)
        logger.info(f"assistant:{text}")
        # logger.info(history_prompt)
        return AckMessage.STATUS_OK, "OK"


def split_string(
        text,
        method=0,
):
    """
    method=0: 判断字符串中尾部是否有|<search_engine><web_search>|, 将字符串输出成question,以及search_engine, web_search三部分
    method=1: 判断字符串中尾部是否有" 1" or " 2", "3","4", " 1"表示search_engine="baidu", web_search=False; " 2"表示search_engine="google" , web_search=False;
                                                      " 3"表示search_engine="baidu", web_search=False; " 4"表示search_engine="google" , web_search=False;
    :param text:
    :return:
    """
    search_engine = None
    web_search = False
    question = text

    if method == 0:  # |<baidu>| or |<google>| or |<baidu><web_search>| or |<baidu><web_search>|
        pattern = r"\|<.+>\|"
        match = re.search(pattern, text)
        if match:
            tail_text = match.group(0)[2:-2]
            question = text[: match.start()]
            if 'web_search' in tail_text:
                search_engine = tail_text[:-12]
                web_search = True
            else:
                search_engine = tail_text
                web_search = False



    elif method == 1:  # "空格1" or "空格2" or "空格3" or "空格4"
        if text[-2:] == " 1":
            search_engine = "baidu"
            web_search = False
            question = text[-3]
        elif text[-2:] == " 2":
            search_engine = "google"
            web_search = False
            question = text[-3]
        elif text[-2:] == " 3":
            search_engine = "google"
            web_search = True
            question = text[-3]
        elif text[-2:] == " 4":
            search_engine = "google"
            web_search = True
            question = text[-3]

    return question, search_engine, web_search


def change_topic_str_Detect(text: str) -> bool:
    """
    判读字符串是否包括|<换个话题>|,或者|<change topic>|
    """
    pattern0 = r"\|<换个话题>\|"
    pattern1 = r"\|<change topic>\|"
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

    def __init__(
            self,
            zhipuai_client,
            history_messages=[],
            bot_name=None,
            bot_info=None,
            user_name=None,
            user_info=None,
            ssml_enabled=False,
            TTS_process_instance=None,
            logger: logging.Logger = None,
    ):
        """

        :param logger:
        :param zhipuai_client:
        :param history_messages:
        :param bot_name:
        :param bot_info:
        :param user_name:
        :param user_info:
        :param ssml_enabled: 是否使用ssml标记以获得语音style
        :param TTS_process_instance: 自定义TTS_process类的初始化后的实例,用于实现语音合成,其中aliyun TTS与azure TTS可选
        """
        super(VoiceChatHandler, self).__init__()

        self.logger = logger

        self.zhipuai = zhipuai_client
        self.history_messages = history_messages
        self.bot_name = bot_name
        self.bot_info = bot_info
        self.user_name = user_name
        self.user_info = user_info
        self.ssml_enabled = ssml_enabled

        self.TTS_process_instance = TTS_process_instance

    async def process(self, callback: dingtalk_stream.CallbackMessage):
        global history_messages
        callback_data = callback.data
        incoming_message = ChatbotMessage_Utilies.from_dict(callback_data)
        text = ""
        # 文本,语音消息分别判断后处理:
        if incoming_message.message_type == "text":
            text = incoming_message.text.content.strip()
        elif incoming_message.message_type == "audio":
            text = incoming_message.audio_recognition
            self.reply_text(f"你:{text}", incoming_message)
        # 判断是否重启对话,并构造prompt
        if change_topic_str_Detect(text):
            history_messages = []
        messages = [{"role": "user", "content": text}]

        self.logger.info(f"user:{text}")
        text = characterGLMAPI_completion_create(
            zhipuai_client=self.zhipuai,
            messages=messages,
            history_messages=self.history_messages,
            bot_name=self.bot_name,
            bot_info=self.bot_info,
            user_name=self.user_name,
            user_info=self.user_info,
        )
        text = re.sub(r'^["]+|[\\n+]|[\n+"]$', "", text)  # 去除字符串中的换行符或者回车符
        # self.logger.info(f"生成的text:{text}")
        history_messages.extend([{"role": "assistant", "content": text}])

        audio_content, duration = self.TTS_process_instance.start(
            text, ssml_enabled=self.ssml_enabled, ssml_intensity=1, styledegree=1
        )
        mediaId = self.upload2media_id(media_content=audio_content, media_type="voice")
        self.logger.info(f"uploaded aliyun voice media_id: {mediaId}")
        # 发送voice gradio_message:
        # self.reply_voice_http(mediaId, duration, incoming_message)  # http方式发送voice gradio_message reqeust格式有误;
        self.reply_voice_SDK(mediaId, duration, incoming_message)
        self.reply_text(text, incoming_message)
        self.logger.info(f"assistant:{text}")
        # logger.info(history_prompt)
        return AckMessage.STATUS_OK, "OK"


class TTS_process:
    def __init__(
            self,
            aliyun_accessKey_id,
            aliyun_accessKey_secret,
            aliyun_region_id="cn-shanghai",
            aliyun_appKey=None,
            tts_name=None,
            aformat="wav",
            aliyun_voice="xiaoyun",
            speech_rate=0,
            pitch_rate=0,
            wait_complete=False,
            enable_subtitle=False,
            enable_ptts=False,
            callbacks: list = [],
            aliyun_azure: bool = True,
            azure_key=None,
            azure_region=None,
            azure_voice: str = "zh-CN-XiaoxiaoNeural",
            logger: logging.Logger = None,
    ):
        self.aliyun_accessKey_id = aliyun_accessKey_id
        self.aliyun_accessKey_secret = aliyun_accessKey_secret
        self.aliyun_region_id = aliyun_region_id
        self.aliyun_appKey = aliyun_appKey
        self.tts_name = tts_name
        self.aformat = aformat
        self.aliyun_voice = aliyun_voice
        self.speech_rate = speech_rate
        self.pitch_rate = pitch_rate
        self.wait_complete = wait_complete
        self.enable_subtitle = enable_subtitle
        self.enable_ptts = enable_ptts
        self.callbacks = callbacks

        self.aliyun_azure = aliyun_azure

        self.logger = logger

        if aliyun_azure:  # 初始化aliyun　TTS
            pass  # 由于aliyun_TTS_threadsRUN为多线程,每次初始化一个线程即运行一次调用,所以放在后面start里面初始化

        else:  # 初始化 azure TTS:
            self.azure_TTS_instance = azure_TTS(
                azure_key, azure_region, voice_name=azure_voice, logger=logger
            )

    def start(self, text, ssml_enabled=True, ssml_intensity=1, styledegree=1):

        # 情绪识别:
        if ssml_enabled:
            ssml_label, _, _ = emotion_classification(
                self.aliyun_accessKey_id,
                self.aliyun_accessKey_secret,
                text,
                aliyun_azure=self.aliyun_azure,
            )
        else:
            ssml_label = None
        if self.aliyun_azure:  # 使用aliyun　TTS
            # Text To Speech:
            aliyun_tts_instance = aliyun_TTS_threadsRUN(
                self.aliyun_accessKey_id,
                self.aliyun_accessKey_secret,
                self.aliyun_region_id,
                appkey=self.aliyun_appKey,
                tts_name=self.tts_name,
                aformat=self.aformat,
                voice=self.aliyun_voice,
                speech_rate=self.speech_rate,
                pitch_rate=self.pitch_rate,
                wait_complete=self.wait_complete,
                enable_subtitle=self.enable_subtitle,
                enable_ptts=self.enable_ptts,
                callbacks=self.callbacks,
                logger=self.logger,
            )
            # 将情绪加入到多情感语音合成中去,强度值根据效果调整
            aliyun_tts_instance.start(
                text, ssml_label, ssml_intensity=ssml_intensity
            )
            while aliyun_tts_instance.completion_status is False:
                # self.logger.info(f"tts_instance.completion_status: {tts_instance.completion_status}")
                time.sleep(0.0005)  # 如果是是本地硬盘I/O,建议值设为0.05
            # 将wav格式的音频转化成ogg格式,便于手机端传送(手机端钉钉在wav格式时,当只有几个字的时间很短时,"语音播放异常,请重试"
            audio_content, duration = wav2ogg(aliyun_tts_instance.BytesIO)

        else:  # 使用azure TTS:
            audio_content, duration = self.azure_TTS_instance.start(
                text=text, style=ssml_label, styledegree=1
            )
        return audio_content, duration


if __name__ == "__main__":

    characterGLM_chat_flag = True
    voiceMessage_chat_flag = True
    aliyun_azure = True  # True:使用aliyun TTS;False:使用azure TTS
    ssml_enabled = False  # True

    logger = setup_logger()
    # options = define_options()
    config_path_dtApp = r"l:/Python_WorkSpace/config/DingTalk_APP.ini"
    config_path_serp = r"l:/Python_WorkSpace/config/SerpAPI.ini"
    config_path_zhipuai = r"l:/Python_WorkSpace/config/zhipuai_SDK.ini"
    config_path_aliyunsdk = r"l:/Python_WorkSpace/config/aliyunsdkcore.ini"
    config_path_azure = r"l:/Python_WorkSpace/config/Azure_Resources.ini"

    zhipu_apiKey = config_read(config_path_zhipuai, section="zhipuai_SDK_API", option1="api_key", option2=None)
    zhipuai_client = ZhipuAI(api_key=zhipu_apiKey)

    # bot_info = "杨幂,1986年9月12日出生于北京市，中国内地影视女演员、流行乐歌手、影视制片人。2005年，杨幂进入北京电影学院表演系本科班就读。2006年，因出演金庸武侠剧《神雕侠侣》崭露头角。2008年，凭借古装剧《王昭君》获得第24届中国电视金鹰奖观众喜爱的电视剧女演员奖提名 。2009年，在“80后新生代娱乐大明星”评选中被评为“四小花旦”。2011年，凭借穿越剧《宫锁心玉》赢得广泛关注 ，并获得了第17届上海电视节白玉兰奖观众票选最具人气女演员奖。2012年，不仅成立杨幂工作室，还凭借都市剧《北京爱情故事》获得了多项荣誉 。2015年，主演的《小时代》系列电影票房突破18亿人民币 。2016年，其主演的职场剧《亲爱的翻译官》取得全国年度电视剧收视冠军 。2017年，杨幂主演的神话剧《三生三世十里桃花》获得颇高关注；同年，她还凭借科幻片《逆时营救》获得休斯顿国际电影节最佳女主角奖 。2018年，凭借古装片《绣春刀Ⅱ：修罗战场》获得北京大学生电影节最受大学生欢迎女演员奖 [4]；。2014年1月8日，杨幂与刘恺威在巴厘岛举办了结婚典礼。同年6月1日，在香港产下女儿小糯米。"
    bot_info = "刘亦菲（Crystal Liu,1987年8月25日-）,生于湖北省武汉市,毕业于北京电影学院,美籍华裔女演员、歌手.2002年,因出演电视剧《金粉世家》中白秀珠一角踏入演艺圈.2003年,因主演武侠剧《天龙八部》王语嫣崭露头角.2004年,凭借仙侠剧《仙剑奇侠传》赵灵儿一角获得了高人气.2005年,因在《神雕侠侣》中饰演小龙女受到关注.2006年,发行首张国语专辑《刘亦菲》和日语专辑《All My Words》.2008年起,转战大银幕,凭借好莱坞电影《功夫之王》成为首位荣登IMDB电影新人排行榜榜首的亚洲女星.2020年3月,为电影《花木兰》演唱中文主题曲《自己》；同年9月,主演的电影《花木兰》在Disney+上线,在剧中饰演花木兰；11月,凭借《花木兰》获首届评论家选择超级奖动作电影最佳女演员提名；2022年12月21日,凭借《梦华录》获第十三届澳门国际电视节金莲花最佳女主角奖.9月1日,凭借《梦华录》获得首届金熊猫奖电视剧单元最佳女主角提名.2023年12月28日,凭借《去有风的地方》获得第十四届澳门国际电视节'金莲花'最佳女主角奖"
    bot_name = "茜茜"
    user_info = "喜欢刘亦菲的男孩一枚"
    user_name = "用户"
    # voice = "zhiyan_emo"  # zhiyan的声音,略微的更女性化些;
    voice = "zhixiaoxia"  # 对话数字人, 不支持多情感; 显然这个voice更适合日常对话, 语气生活化些.
    # voice = "voice-f90ed52" # 个性化声音


    if characterGLM_chat_flag:  # 角色扮演机器人聊天
        client_id, client_secret = config_read(
            config_path_dtApp,
            section="DingTalkAPP_charGLM",
            option1="client_id",
            option2="client_secret",
        )
        credential = dingtalk_stream.Credential(client_id, client_secret)
        client = dingtalk_stream.DingTalkStreamClient(credential)
        if voiceMessage_chat_flag is False:  # 角色扮演机器人, 文本聊天
            client.register_callback_handler(
                dingtalk_stream.chatbot.ChatbotMessage.TOPIC, PromptTextHandler(logger)
            )
        else:  # 角色扮演机器人,  语音聊天
            aliyun_accessKey_id, aliyun_accessKey_secret = config_read(
                config_path_aliyunsdk,
                section="aliyunsdkcore",
                option1="AccessKey_ID",
                option2="AccessKey_Secret",
            )
            aliyun_appKey = config_read(
                config_path_aliyunsdk, section="APP_tts", option1="AppKey"
            )
            azure_key, azure_region = config_read(
                config_path=config_path_azure,
                section="Azure_TTS",
                option1="key",
                option2="region",
            )

            TTS_process_instance = TTS_process(
                aliyun_accessKey_id,
                aliyun_accessKey_secret,
                aliyun_region_id="cn-shanghai",
                aliyun_appKey=aliyun_appKey,
                tts_name="Example",
                aformat="wav",
                aliyun_voice=voice,
                speech_rate=0,
                pitch_rate=0,
                wait_complete=False,
                enable_subtitle=False,
                enable_ptts=False,
                callbacks=[],
                aliyun_azure=aliyun_azure,
                azure_key=azure_key,
                azure_region=azure_region,
                logger=logger,
            )

            client.register_callback_handler(
                ChatbotMessage_Utilies.TOPIC,
                VoiceChatHandler(
                    zhipuai_client=zhipuai_client,
                    logger=logger,
                    history_messages=history_messages,
                    bot_info=bot_info,
                    bot_name=bot_name,
                    user_name=user_name,
                    user_info=user_info,
                    ssml_enabled=ssml_enabled,
                    TTS_process_instance=TTS_process_instance,
                ),
            )

    else:  # GLM 办公助手, 文本聊

        semantic_search_engine = chatGLM_by_semanticSearch_amid_SerpAPI(zhipuai_client, serp_key_path=config_path_serp,)
        client_id, client_secret = config_read(
            config_path_dtApp,
            section="DingTalkAPP_chatGLM",
            option1="client_id",
            option2="client_secret",
        )
        credential = dingtalk_stream.Credential(client_id, client_secret)
        client = dingtalk_stream.DingTalkStreamClient(credential)
        client.register_callback_handler(
            dingtalk_stream.chatbot.ChatbotMessage.TOPIC, EchoTextHandler(logger)
        )

    client.start_forever()
