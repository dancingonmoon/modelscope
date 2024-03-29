# ! /usr/bin/env python
# coding=utf-8
import os
import time
import json
from aliyunsdkcore.client import AcsClient
from aliyunsdkcore.request import CommonRequest
import nls
import threading

from ChatBOT_APP import config_read, setup_logger


def emo_label_out(data):
    scores = data['scores']
    max_value = max(scores)
    max_index = scores.index(max_value)
    label = data['labels'][max_index]
    return label


# semantic_cls = pipeline(Tasks.text_classification, 'damo/nlp_structbert_emotion-classification_chinese-base', model_revision='v1.0.0')

# def emo_classification(text):
#     """
#     中文情绪分类: labels': ['恐惧', '高兴', '悲伤', '喜好', '厌恶', '愤怒', '惊讶']
#     """
#     emo_dict = semantic_cls(input=text)  # 输出dict
#     label = emo_label_out(emo_dict)
#     # print(label)
#     return label


# SSML（Speech Synthesis Markup Language）:pip install aliyun-python-sdk-core
# <emotion>用于多情感声音合成，该标签是可选标签，不支持多情感声音合成的发音人使用情感标签会导致合成请求报错。
# ```<emotion category="happy" intensity="1.0">今天天气真不错！</emotion>```
# intensity: [0.01,2.0] 指定情绪强度。默认值为1.0，表示预定义的情绪强度。最小值为0.01，导致目标情绪略有倾向。最大值为2.0，导致目标情绪强度加倍。
# zhimiao_emo: serious，sad，disgust，jealousy，embarrassed，happy，fear，surprise，neutral，frustrated，
#               affectionate，gentle，angry，newscast，customer-service，story，living
# zhimi_emo: angry，fear，happy，hate，neutral，sad，surprise
# zhiyan_emo: neutral，happy，angry，sad，fear，hate，surprise，arousal
# zhibei_emo: neutral，happy，angry，sad，fear，hate，surprise
# zhitian_emo: neutral，happy，angry，sad，fear，hate，surprise

# 使用阿里云公共SDK获取Token，采用RPC风格的API调用:


# 获取 accessKey_id, accessKey_secret:
config_path_aliyunsdk = r"e:/Python_WorkSpace/config/aliyunsdkcore.ini"
accessKey_id, accessKey_secret = config_read(config_path_aliyunsdk, section='aliyunsdkcore', option1='AccessKey_ID',
                                             option2='AccessKey_Secret')


def get_token_viaSDK(accessKey_id, accessKey_secret, region_id='cn-shanghai'):
    """
    使用阿里云公共SDK获取Token,采用PRC分隔的API调用;获取的token有效期24小时
    """
    # 创建AcsClient实例
    client = AcsClient(ak=accessKey_id, secret=accessKey_secret, region_id=region_id)
    # 创建request，并设置参数。
    request = CommonRequest()
    request.set_method('POST')
    request.set_domain('nls-meta.cn-shanghai.aliyuncs.com')
    request.set_version('2019-02-28')
    request.set_action_name('CreateToken')

    try:
        response = client.do_action_with_exception(request)
        # print(response)

        jss = json.loads(response)
        if 'Token' in jss and 'Id' in jss['Token']:
            token = jss['Token']['Id']
            expireTime = jss['Token']['ExpireTime']
            # print("token = " + token)
            # #将时间戳转换为struct_time
            # struct_time = time.localtime(expireTime)
            # #将struct_time转换为自定义格式的字符串
            # str_time = time.strftime("%Y-%m-%d %H:%M:%S", struct_time)
            # print(f"expireTime:{str_time}")
            return token, expireTime
    except Exception as e:
        # print(e)
        return e


token, expirTime = get_token_viaSDK(accessKey_id, accessKey_secret)
appKey = config_read(config_path_aliyunsdk, section='APP_tts', option1='AppKey')
# URL = "wss://nls-gateway-cn-shanghai.aliyuncs.com/ws/v1"
URL = "wss://nls-gateway.aliyuncs.com/ws/v1"  # 就近地域智能接入,自动根据地域选择shanghai/beijing/shenzhen
TEXT = '大壮正想去摘取花瓣，谁知阿丽和阿强突然内讧，阿丽拿去手枪向树干边的阿强射击，两声枪响，阿强直接倒入水中'
TEXT_ssml = """<speak>
  相传北宋年间，
  <say-as interpret-as="date">1121-10-10</say-as>，
  <say-as interpret-as="address">开封城</say-as>
  郊外的早晨笼罩在一片
  <sub alias="双十一">11.11</sub>
  前买买买的欢乐海洋中。一支运货的骡队刚进入城门
  <soundEvent src="http://nls.alicdn.com/sound-event/bell.wav"/>
  一个肤白貌美
  <phoneme alphabet="py" ph="de5">地</phoneme>
  姑娘便拦下第一排的小哥<say-as interpret-as="name">阿发。</say-as>
</speak>"""

tts_out_path = './tts_out.wav'
femail_speakers = ["zhixiaobai", "zhixiaoxia", "zhixiaomei", "zhigui", "aixia", "zhimiao_emo", "zhiyan_emo",
                   "zhibei_emo", "zhitian_emo", "xiaoyun", "ruoxi", "sijia", "aiqi", "aijia", "ninger", "ruilin",
                   "siyue", "aiya", "aimei", "aiyu", "aiyue", "aijing", "xiaomei", "xiaobei", "aiwei", "guijie",
                   "stella", "xiaoxian", "maoxiaomei", ]


class TTS_threadsRUN():
    """
    TTS,将音频存入指定目录的文件.
    每次初始化,启动单一thread,可以在多次初始化后,多线程处理密集I/O;
    """

    def __init__(self, tts_name, audio_path, aformat='wav', voice='xiaoyun',
                 speech_rate=0, pitch_rate=0, wait_complete=True,
                 enable_subtitle=False, enable_ptts=False, callbacks=None):
        """
        tts_name: 一个名字而已,仅用于标识
        callbacks: list; nls.NlsSpeechSynthesizer()中的callback_args,用于自定义回调时,传入自定义参数
        """
        self.__thread = threading.Thread(target=self.__tts_run)
        self.__tts_name = tts_name
        self.__audio_path = audio_path
        self.__aformat = aformat
        self.__voice = voice
        self.__speech_rate = speech_rate
        self.__pitch_rate = pitch_rate
        self.__wait_complete = wait_complete,
        self.__enable_subtitle = enable_subtitle
        self.__enable_ptts = enable_ptts
        self.__callbacks = callbacks

    def __tts_run(self):
        """
        调用语音合成API,1)初始化语音合成,2)启动语音合成
        :return: Boolean, 成功或者失败
        """
        print("thread:{} start..".format(self.__tts_name))
        tts = nls.NlsSpeechSynthesizer(url=URL,
                                       token=token,
                                       appkey=appKey,
                                       # on_metainfo=self.fun_on_metainfo,
                                       on_data=self.fun_on_data,
                                       # on_completed=self.fun_on_completed,
                                       on_error=self.fun_on_error,
                                       on_close=self.fun_on_close,
                                       callback_args=self.__callbacks)
        print("{}: session start".format(self.__tts_name))
        r = tts.start(self.__text, aformat=self.__aformat, voice=self.__voice,
                      speech_rate=self.__speech_rate, pitch_rate=self.__pitch_rate, wait_complete=self.__wait_complete,
                      ex={"enable_subtitle": self.__enable_subtitle, "enable_ptts": self.__enable_ptts})
        print("{}: tts done with result:{}".format(self.__tts_name, r))

    def start(self, text):
        self.__text = text
        self.__f = open(self.__audio_path, 'wb')
        self.__thread.start()

    def fun_on_metainfo(self, message, *args):
        """
        如果start方法中通过ex参数传递enable_subtitle，则会返回对应字幕信息。回调参数包含以下两种：
        """
        print(f"callback_args:{args}; subtitle:{message}")
        # return message

    def fun_on_data(self, data, *args):
        """
        当存在合成数据后的回调参数。回调参数包含以下两种：
        1)对应start方法中aformat的二进制音频数据 2)用户自定义参数
        """
        try:
            self.__f.write(data)
            print(f"write data finish")
        except Exception as e:
            print("write data failed:", e)

    def fun_on_error(self, message, *args):
        """
        当SDK或云端出现错误时的回调参数。回调参数包含以下两种：
        """
        print(f"error:{message}")

    def fun_on_completed(self, message, *args):
        """
        语音合成完成时,执行函数
        """
        print("on_completed:args=>{} message=>{}".format(args, message))

    def fun_on_close(self, *args):
        # global con_fig
        print("on_close: args=>{}".format(args))
        try:
            self.__f.close()
        except Exception as e:
            print("close file failed since:", e)

        # con_fig = False
        # return con_fig


# nls.enableTrace(True)
# tts = nls.NlsSpeechSynthesizer(url=URL,
#                                token=token,
#                                appkey=appKey,
#                                on_metainfo=fun_on_metainfo,
#                                on_data=fun_on_data,
#                                # on_completed=fun_on_completed,
#                                on_error=fun_on_error,
#                                on_close=fun_on_close,
#                                callback_args=[appKey, tts_out_path])
# result = tts.start(text=TEXT, aformat='wav', voice='zhitian_emo',
#                    speech_rate=0, pitch_rate=0, wait_complete=True,
#                    ex={"enable_subtitle": True, "enable_ptts": False})


# nls.enableTrace(True)
# voice = 'zhiyan_em' # zhiyan的声音,略微的更女性化些;
for voice in femail_speakers:
    con_fig = True
    tts = TTS_threadsRUN(tts_name='测试例子', audio_path=f'./tts_{voice}_out.wav', voice=voice, wait_complete=False,
                         enable_subtitle=False, callbacks=[con_fig])
    tts.start(TEXT_ssml)
    while con_fig:
        time.sleep(2)
        if not con_fig:
            break
