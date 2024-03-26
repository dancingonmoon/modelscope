from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks


# 中文情绪分类:
# labels': ['恐惧', '高兴', '悲伤', '喜好', '厌恶', '愤怒', '惊讶']

def emo_label_out(data):
    scores = data['scores']
    max_value = max(scores)
    max_index = scores.index(max_value)
    label = data['labels'][max_index]
    return label


semantic_cls = pipeline(Tasks.text_classification, 'damo/nlp_structbert_emotion-classification_chinese-base', model_revision='v1.0.0')
text_emo = semantic_cls(input='新年快乐！')
print(text_emo)

# SSML（Speech Synthesis Markup Language）:
# <emotion>用于多情感声音合成，该标签是可选标签，不支持多情感声音合成的发音人使用情感标签会导致合成请求报错。
# ```<emotion category="happy" intensity="1.0">今天天气真不错！</emotion>```
# intensity: [0.01,2.0] 指定情绪强度。默认值为1.0，表示预定义的情绪强度。最小值为0.01，导致目标情绪略有倾向。最大值为2.0，导致目标情绪强度加倍。
# zhimiao_emo: serious，sad，disgust，jealousy，embarrassed，happy，fear，surprise，neutral，frustrated，
#               affectionate，gentle，angry，newscast，customer-service，story，living
# zhimi_emo: angry，fear，happy，hate，neutral，sad，surprise
# zhiyan_emo: neutral，happy，angry，sad，fear，hate，surprise，arousal
# zhibei_emo: neutral，happy，angry，sad，fear，hate，surprise
# zhitian_emo: neutral，happy，angry，sad，fear，hate，surprise