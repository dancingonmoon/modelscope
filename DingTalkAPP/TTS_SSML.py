# ! /usr/bin/env python
# coding=utf-8
import os
import time
import json
from aliyunsdkcore.client import AcsClient
from aliyunsdkcore.request import CommonRequest
import nls

from ChatBOT_APP import config_read, setup_logger


# def emo_label_out(data):
#     scores = data['scores']
#     max_value = max(scores)
#     max_index = scores.index(max_value)
#     label = data['labels'][max_index]
#     return label


# semantic_cls = pipeline(Tasks.text_classification, 'damo/nlp_structbert_emotion-classification_chinese-base', model_revision='v1.0.0')

def emo_classification(text):
    """
    中文情绪分类: labels': ['恐惧', '高兴', '悲伤', '喜好', '厌恶', '愤怒', '惊讶']
    """
    emo_dict = semantic_cls(input=text)  # 输出dict
    label = emo_label_out(emo_dict)
    # print(label)
    return label


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


token = get_token_viaSDK(accessKey_id, accessKey_secret)
appKey = config_read(config_path_aliyunsdk, section='APP_tts', option1='AppKey')
# URL = "wss://nls-gateway-cn-shanghai.aliyuncs.com/ws/v1"
URL = "wss://nls-gateway.aliyuncs.com/ws/v1"  # 就近地域智能接入,自动根据地域选择shanghai/beijing/shenzhen
TEXT = '大壮正想去摘取花瓣，谁知阿丽和阿强突然内讧，阿丽拿去手枪向树干边的阿强射击，两声枪响，阿强直接倒入水中'
tts_out_path = './tts_out.wav'


def fun_on_metainfo(message, *args):
    """
    如果start方法中通过ex参数传递enable_subtitle，则会返回对应字幕信息。回调参数包含以下两种：
    1)JSON形式的字符串 2)用户自定义参数
    其中，用户自定义参数为callback_args字段中返回的参数内容。
    """
    return f"appKey:{args}; subtitle:{message}"


def fun_on_data(data, *args):
    """
    当存在合成数据后的回调参数。回调参数包含以下两种：
    1)对应start方法中aformat的二进制音频数据 2)用户自定义参数
    """
    tts_out_path = args[1]
    try:
        with open(tts_out_path, 'wb') as f:
            f.write(data)
    except Exception as e:
        print("write data failed:", e)


tts = nls.NlsSpeechSynthesizer(url=URL,
                               token=token,
                               appkey=appKey,
                               on_metainfo=fun_on_metainfo,
                               on_data=fun_on_data,
                               # on_completed=fun_on_completed,
                               # on_error=fun_on_error,
                               # on_close=fun_on_close,
                               callback_args=[appKey, tts_out_path])
tts.start(text=TEXT, aformat='wav', voice='zhitian_emo',
          speech_rate=0, pitch_rate=0, wait_complete=True,
          ex={"enable_subtitle": True, "enable_ptts": False})
