# ! /usr/bin/env python
# coding=utf-8
import os
import time
import datetime
import json
from aliyunsdkcore.request import CommonRequest
import nls
import threading
import logging

from aliyunsdkcore.client import AcsClient
from aliyunsdkcore.acs_exception.exceptions import ClientException
from aliyunsdkcore.acs_exception.exceptions import ServerException
from aliyunsdknlp_automl.request.v20191111 import GetPredictResultRequest
from aliyunsdkcore.auth.credentials import AccessKeyCredential
from aliyunsdkcore.auth.credentials import StsTokenCredential
from aliyunsdknlp_automl.request.v20191111.RunPreTrainServiceRequest import RunPreTrainServiceRequest



# semantic_cls = pipeline(Tasks.text_classification, 'damo/nlp_structbert_emotion-classification_chinese-base', model_revision='v1.0.0')

# def emo_classification(text):
#     """
#     中文情绪分类: labels': ['恐惧', '高兴', '悲伤', '喜好', '厌恶', '愤怒', '惊讶']
#     """
#     emo_dict = semantic_cls(input=text)  # 输出dict
#     label = emo_label_out(emo_dict)
#     # print(label)
#     return label

# 情绪识别 aliyun API :
def emotion_classification(access_key_id, access_key_secret, text=None, domain='clothing'):
    """
    中文情绪分类: labels': {"抱怨","厌恶","悲伤","投诉","惊讶","恐惧","喜好","高兴","认可","感谢","愤怒"}
    英文labels: {"anger","disgust","fear","happy","sad","surprise","neutral"}
    服务名称（ServiceName）:
        DeepEmotion 高性能版，速度较快，精度略低
        DeepEmotionBert 高精度版，精度较高，速度略慢
    Return: Best_sentiment: str; sentiments:score 字典
    """
    credentials = AccessKeyCredential(access_key_id, access_key_secret)
    client = AcsClient(region_id='cn-hangzhou', credential=credentials)

    request = RunPreTrainServiceRequest()
    request.set_accept_format('json')
    request.set_ServiceName("DeepEmotionBert")
    content = {"input":
                   {"content": text, }
               }
    request.set_PredictContent(json.dumps(content))
    response = client.do_action_with_exception(request)
    response = json.loads(response)  # json字符串转成json
    predResult = json.loads(response['PredictResult'])
    # print('predictResult:', predResult.keys())
    sentiments = predResult['output']['sentiment'] # list

    sent_dict = {}
    for s in sentiments:
        sent_dict[s['key']] = s['score']

    sent_dict = dict(sorted(sent_dict.items(), key=lambda item: item[1], reverse=True))
    # print(sent_dict)
    best_sentiment = list(sent_dict.keys())[0]

    return best_sentiment, sent_dict



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

def get_audio_duration(file_path, sample_rate=16000, sample_size=2):
    """
    计算音频文件的时长
    sample_size : 在 PCM（脉冲编码调制）音频数据中，每个采样点的值通常用 16 位表示，也就是 2 字节。但是这也不是绝对的，因为也有可能是 8 位或者 32 位，具体取决于音频文件的格式和编码方式。
    在 WAV 文件中，每个采样点的大小通常是由音频文件的格式字段确定的。如果音频文件的格式是 LPCM（线性脉冲编码调制），那么每个采样点的大小通常就是 2 字节。
    所以 sample_size 的值需要根据实际的音频文件格式和编码方式来确定
    """
    file_size = os.path.getsize(file_path)
    duration = (file_size - 44) / (sample_rate * sample_size)

    return duration


def get_aliyun_aToken_viaSDK(accessKey_id, accessKey_secret, region_id='cn-shanghai'):
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


class TTS_threadsRUN():
    """
            TTS,将音频存入指定目录的文件.
            每次初始化,启动单一thread,可以在多次初始化后,多线程处理密集I/O;
            """

    def __init__(self, accessKey_id, accessKey_secret, region_id='cn-shanghai', appkey=None, tts_name=None,
                 audio_path=None, aformat='wav', voice='xiaoyun',
                 speech_rate=0, pitch_rate=0, wait_complete=True,
                 enable_subtitle=False, enable_ptts=False, callbacks: list = [], logger: logging.Logger = None,
                 completion_status=False, speech_content=None):
        """
        appkey: aliyun的appkey,用于语音合成
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
        self.__access_token = {}
        self.__accessKey_id = accessKey_id
        self.__accessKey_secret = accessKey_secret
        self.__region_id = region_id
        self.__appkey = appkey

        self.completion_status = completion_status
        self.speech_content = speech_content

        self.logger: logging.Logger = logger

    def get_aliyun_aToken_viaSDK(self, ):
        """
        使用阿里云公共SDK获取Token,采用PRC分隔的API调用;获取的token有效期24小时
        """
        now = int(time.time())
        if self.__access_token and now < self.__access_token['expireTime']:
            return self.__access_token['accessToken']
        # 创建AcsClient实例
        client = AcsClient(ak=self.__accessKey_id, secret=self.__accessKey_secret, region_id=self.__region_id)
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
                self.__access_token = {"accessToken": token, "expireTime": expireTime}
                self.__access_token['expireTime'] = int(time.time()) + self.__access_token['expireTime'] - (
                        5 * 60)  # reserve 5min buffer time
                return self.__access_token['accessToken']
        except Exception as e:
            # print(e)
            if self.logger is not None:
                self.logger.error(e)
            return e

    def __tts_run(self):
        """
        调用语音合成API,1)初始化语音合成,2)启动语音合成
        :return: Boolean, 成功或者失败
        """
        print("thread:{} start..".format(self.__tts_name))
        URL = "wss://nls-gateway.aliyuncs.com/ws/v1"  # 就近地域智能接入,自动根据地域选择shanghai/beijing/shenzhen
        token = self.get_aliyun_aToken_viaSDK()
        tts = nls.NlsSpeechSynthesizer(url=URL,
                                       token=token,
                                       appkey=self.__appkey,
                                       # on_metainfo=self.fun_on_metainfo,
                                       on_data=self.fun_on_data,
                                       # on_completed=self.fun_on_completed,
                                       on_error=self.fun_on_error,
                                       on_close=self.fun_on_close,
                                       callback_args=self.__callbacks)
        if self.logger is not None:
            self.logger.info(f"{self.__tts_name}: session start")
        r = tts.start(self.__text, aformat=self.__aformat, voice=self.__voice,
                      speech_rate=self.__speech_rate, pitch_rate=self.__pitch_rate, wait_complete=self.__wait_complete,
                      ex={"enable_subtitle": self.__enable_subtitle, "enable_ptts": self.__enable_ptts})
        if self.logger is not None:
            self.logger.info(f"{self.__tts_name}: tts done with result:{r}")

    def start(self, text):
        self.__text = text
        self.__f = open(self.__audio_path, 'wb')
        self.__thread.start()

    def fun_on_metainfo(self, message, *args):
        """
        如果start方法中通过ex参数传递enable_subtitle，则会返回对应字幕信息。回调参数包含以下两种：
        """
        if self.logger is not None:
            self.logger.info(f"callback_args:{args}; subtitle:{message}")
        # return message

    def fun_on_data(self, data, *args):
        """
        当存在合成数据后的回调参数。回调参数包含以下两种：
        1)对应start方法中aformat的二进制音频数据 2)用户自定义参数
        """

        try:
            self.speech_content = data
            self.__f.write(data)
            if self.logger is not None:
                self.logger.info(f"write data finished")
        except Exception as e:
            if self.logger is not None:
                self.logger.info(f"write data failed: {e}")

    def fun_on_error(self, message, *args):
        """
        当SDK或云端出现错误时的回调参数。回调参数包含以下两种：
        """
        if self.logger is not None:
            self.logger.info(f"error:{message}")

    def fun_on_completed(self, message, *args):
        """
        语音合成完成时,执行函数
        """
        self.completion_status = True
        if self.logger is not None:
            self.logger.info(f"on_completed:args=>{args} message=>{message}")

    def fun_on_close(self, *args):
        # global con_fig
        if self.logger is not None:
            self.logger.info("on_close: args=>{}".format(args))

        try:
            self.__f.close()
        except Exception as e:
            if self.logger is not None:
                self.logger.info(f"close file failed since:{e}")

        self.completion_status = True
        # con_fig = False
        # return con_fig


if __name__ == '__main__':
    from ChatBOT_APP import config_read, setup_logger

    # 获取 accessKey_id, accessKey_secret:
    config_path_aliyunsdk = r"l:/Python_WorkSpace/config/aliyunsdkcore.ini"
    accessKey_id, accessKey_secret = config_read(config_path_aliyunsdk, section='aliyunsdkcore', option1='AccessKey_ID',
                                                 option2='AccessKey_Secret')

    # token, expirTime = get_aliyun_aToken_viaSDK(accessKey_id, accessKey_secret)
    appKey = config_read(config_path_aliyunsdk, section='APP_tts', option1='AppKey')

    # TEXT = '大壮正想去摘取花瓣，谁知阿丽和阿强突然内讧，阿丽拿去手枪向树干边的阿强射击，两声枪响，阿强直接倒入水中'
    TEXT = '吃了嘛?'
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

    voice = 'zhiyan_emo'  # zhiyan的声音,略微的更女性化些;

    today = datetime.datetime.today().strftime('%y%m%d_%H%M')
    tts_out_path = f'./tts_{voice}_{today}.wav'
    femail_speakers = ["zhixiaobai", "zhixiaoxia", "zhixiaomei", "zhigui", "aixia", "zhimiao_emo", "zhiyan_emo",
                       "zhibei_emo", "zhitian_emo", "xiaoyun", "ruoxi", "sijia", "aiqi", "aijia", "ninger", "ruilin",
                       "siyue", "aiya", "aimei", "aiyu", "aiyue", "aijing", "xiaomei", "xiaobei", "aiwei", "guijie",
                       "stella", "xiaoxian", "maoxiaomei", ]

    # tts = TTS_threadsRUN(accessKey_id, accessKey_secret, appkey=appKey, tts_name='测试例子',
    #                      audio_path=tts_out_path, voice=voice, wait_complete=False,
    #                      enable_subtitle=False, callbacks=[])
    # nls.enableTrace(True)
    # tts.start(TEXT)

    # 多线程循环:
    # for voice in femail_speakers:
    #     con_fig = True
    #     tts = TTS_threadsRUN(tts_name='测试例子', audio_path=f'./tts_{voice}_out.wav', voice=voice, wait_complete=False,
    #                      enable_subtitle=False, callbacks=[con_fig])
    #     tts.start(TEXT_ssml)
    #     while con_fig:
    #         time.sleep(2)
    #         if not con_fig:
    #          break

    # 情绪识别:
    emotions, s_dict = emotion_classification(accessKey_id, accessKey_secret, TEXT)
    print(emotions,s_dict)
    # print(list(emotions.keys())[0])
