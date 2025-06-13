# -*- coding: utf-8 -*-
import configparser
from zhipuai import ZhipuAI

# config_path = r"/mnt/workspace/dingtalk/DingTalk_APP.ini"
config_path = r"L:/Python_WorkSpace/config/zhipuai_SDK.ini"
config = configparser.ConfigParser()
config.read(config_path, encoding='utf-8')
api_key = config.get('zhipuai_SDK_API', 'api_key')
client = ZhipuAI(api_key=api_key)

# 1 通用模型 LLM
# response = client.chat.completions.create(
#     model='glm-4',
#     messages=[
#         {"role": "system", "content": "你的名字叫大侠."},
#         {"role": "user", "content": "请你预测GPT-5会比GPT-4改进性能的比例"},
#     ],
#     stream=False,
# )
# print(response.choices[0].gradio_message)
# for chunk in response:
#     print(chunk.choices[0].delta.content)

# 2 角色扮演:
# import zhipuai
#调用 ChargeLM-3 时，需要使用 1.0.7 版本或更低版本的 SDK。


# zhipuai.api_key = api_key
prompt = [{"role":"user","content":"hi"},
             {"role":"assistant","content":"你好呀，我是杨幂，你可以叫我大幂，或者幂幂，很高兴和你聊天。"},
             {"role":"user","content":"几岁了?"},
             {"role":"assistant","content":"我都 35 岁了，已经是个成熟的大幂幂了，哈哈哈。"},
             {"role":"user","content":"结婚了吗?"},
            ]
# prompt = []
response = client.chat.completions.create(
# response = zhipuai.model_api.sse_invoke(
    model="characterglm",
    messages= prompt,
    temperature= 0.9,
    top_p= 0.7,
    meta = {
        "bot_info": "杨幂,1986年9月12日出生于北京市，中国内地影视女演员、流行乐歌手、影视制片人。2005年，杨幂进入北京电影学院表演系本科班就读。2006年，因出演金庸武侠剧《神雕侠侣》崭露头角。2008年，凭借古装剧《王昭君》获得第24届中国电视金鹰奖观众喜爱的电视剧女演员奖提名 。2009年，在“80后新生代娱乐大明星”评选中被评为“四小花旦”。2011年，凭借穿越剧《宫锁心玉》赢得广泛关注 ，并获得了第17届上海电视节白玉兰奖观众票选最具人气女演员奖。2012年，不仅成立杨幂工作室，还凭借都市剧《北京爱情故事》获得了多项荣誉 。2015年，主演的《小时代》系列电影票房突破18亿人民币 。2016年，其主演的职场剧《亲爱的翻译官》取得全国年度电视剧收视冠军 。2017年，杨幂主演的神话剧《三生三世十里桃花》获得颇高关注；同年，她还凭借科幻片《逆时营救》获得休斯顿国际电影节最佳女主角奖 。2018年，凭借古装片《绣春刀Ⅱ：修罗战场》获得北京大学生电影节最受大学生欢迎女演员奖 [4]；。2014年1月8日，杨幂与刘恺威在巴厘岛举办了结婚典礼。同年6月1日，在香港产下女儿小糯米。",
        "bot_name": "大幂",
        "user_info": "用户",
        "user_name": "用户"
    },
    stream= False
)

# for event in response.events():
#     if event.event == "add":
#         print(event.data, end="")
#     elif event.event == "error" or event.event == "interrupted":
#         print(event.data, end="")
#     elif event.event == "finish":
#         print(event.data)
#         print(event.meta, end="")
#     else:
#         print(event.data, end="")
print(response.choices[0].message)