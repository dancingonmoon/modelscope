# from modelscope.pipelines import pipeline
# from modelscope.utils.constant import Tasks
from ChatBOT_APP import config_read, setup_logger


def emo_label_out(data):
    scores = data['scores']
    max_value = max(scores)
    max_index = scores.index(max_value)
    label = data['labels'][max_index]
    return label


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
# ! /usr/bin/env python
# coding=utf-8
import os
import time
import json
from aliyunsdkcore.client import AcsClient
from aliyunsdkcore.request import CommonRequest

# 获取 accessKey_id, accessKey_secret:
config_path_aliyunsdk = r"e:/Python_WorkSpace/config/aliyunsdkcore.ini"
accessKey_id, accessKey_secret = config_read(config_path_aliyunsdk, section='aliyunsdkcore', option1='AccessKey_ID',
                                             option2='AccessKey_Secret')

# 创建AcsClient实例
client = AcsClient(ak=accessKey_id, secret=accessKey_secret, region_id="cn-shanghai")
# 创建request，并设置参数。
request = CommonRequest()
request.set_method('POST')
request.set_domain('nls-meta.cn-shanghai.aliyuncs.com')
request.set_version('2019-02-28')
request.set_action_name('CreateToken')

try:
    response = client.do_action_with_exception(request)
    print(response)

    jss = json.loads(response)
    if 'Token' in jss and 'Id' in jss['Token']:
        token = jss['Token']['Id']
        expireTime = jss['Token']['ExpireTime']
        print("token = " + token)
        print("expireTime = " + str(expireTime))
        print(time.strftime())
except Exception as e:
    print(e)
