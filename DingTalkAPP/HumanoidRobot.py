# !/usr/bin/env python
# -*- coding: utf-8 -*-
from ChatBOT_APP import (
    config_read,
    setup_logger,
    VoiceChatHandler,
    PromptTextHandler,
    EchoTextHandler,
)
import sys
import zhipuai
import dingtalk_stream
from chatbotClass_utilies import ChatbotMessage_Utilies

sys.path.append("../GLM")  # 将上一级目录的/GLM目录添加到系统路径中
# from semantic_search_by_zhipu import chatGLM_by_semanticSearch_amid_SerpAPI
history_prompt = []  # 初始值定义为空列表,以与后续列表进行extend()拼接

if __name__ == "__main__":
    characterGLM_chat_flag = True  # True时,characterglm,需要zhipuai库版本<=1.07
    voiceMessage_chat_flag = True
    aliyun_azure = True # 个性化ptts,目前只在aliyun TTS

    if characterGLM_chat_flag is False:
        from semantic_search_by_zhipu import chatGLM_by_semanticSearch_amid_SerpAPI

    logger = setup_logger()
    # options = define_options()
    config_path_dtApp = r"l:/Python_WorkSpace/config/DingTalk_APP.ini"
    config_path_serp = r"l:/Python_WorkSpace/config/SerpAPI.ini"
    config_path_zhipuai = r"l:/Python_WorkSpace/config/zhipuai_SDK.ini"
    config_path_aliyunsdk = r"l:/Python_WorkSpace/config/aliyunsdkcore.ini"
    config_path_azure = r"e:/Python_WorkSpace/config/Azure_Resources.ini"
    # bot_info = """
    # 杨幂,1986年9月12日出生于北京市，中国内地影视女演员、流行乐歌手、影视制片人。2005年，杨幂进入北京电影学院表演系本科班就读。2006年，因出演金庸武侠剧《神雕侠侣》崭露头角。
    # 2008年，凭借古装剧《王昭君》获得第24届中国电视金鹰奖观众喜爱的电视剧女演员奖提名 。2009年，在“80后新生代娱乐大明星”评选中被评为“四小花旦”。
    # 2011年，凭借穿越剧《宫锁心玉》赢得广泛关注 ，并获得了第17届上海电视节白玉兰奖观众票选最具人气女演员奖。2012年，不仅成立杨幂工作室，还凭借都市剧《北京爱情故事》获得了多项荣誉 。
    # 2015年，主演的《小时代》系列电影票房突破18亿人民币 。2016年，其主演的职场剧《亲爱的翻译官》取得全国年度电视剧收视冠军 。2017年，杨幂主演的神话剧《三生三世十里桃花》获得颇高关注；
    # 同年，她还凭借科幻片《逆时营救》获得休斯顿国际电影节最佳女主角奖 。2018年，凭借古装片《绣春刀Ⅱ：修罗战场》获得北京大学生电影节最受大学生欢迎女演员奖 [4]；
    # 2014年1月8日，杨幂与刘恺威在巴厘岛举办了结婚典礼。同年6月1日，在香港产下女儿小糯米。
    # """"
    # bot_info = "丁涛, 个人简历如下"
    bot_info = "丁涛, 个人简历如下:职业背景：23年+（14年+上市公司，9年自主创业）通信行业海外业务拓展及管理工作经验。具备从0到1创建及全面负责海外子公司日常经营、市场开拓、项目管理、客户关系管理、核心团队建设的实战经历和能力。受过完整通信行业技术训练，通过通信领域高级工程师认证，英语流利，无障碍和外国人交流沟通。市场开拓：独立开拓诺基亚、爱立信、中兴、华为及伊朗、非洲落地国运营商（含：MTN、MCI、ET）等通信领域知名客户。拥有开拓、建设、运营海外加密货币矿场经验。熟悉非洲与中东国际环境，在伊朗政府、金融机构、通信运营商具有较为丰富的人脉资源。项目管理：主导合同额百万级通信项目，参与合同额千万级通信项目。能从需求、质量、时间、进度、成本、风险、干系人、资源8个维度把控项目，确保项目顺利实施。团队建设：从0到1搭建及管理100人海外本地化团队（含：销售团队、项目执行团队）。根据团队成员特点，针对性培训，提升团队成员专业水平、销售及项目交付能力。知识管理：对标未来工作需要，学习人工智能领域知识，熟悉机器学习理论和机器学习算法，理解深度学习典型算法，掌握Tensorflow、keras深度学习框架，具备使用典型算法对数据集进行模型训练和优化的能力。综合素养：有激情、有拼劲，以结果为导向，系统思考，善于发现问题和解决问题。具有优秀的领导、组织和协调沟通能力，自驱力强，抗压性佳。工作经历2021.05-至今	东方通信股份有限公司 杭州 公司简介：国有控股上市公司。公司业务主要包括：专网通信及信息安全产品和解决方案；公网通信相关产品及ICT服务；金融电子设备及软件产品；智能制造业务。业务经理        所属部门：国际业务部	汇报上级：分管副总裁	下属人数：4人	业务规划：瞄准中东、东南亚2个海外区域，从市场、客户、项目3个维度分级规划业务，据此制定年度业务目标、策略、预算和市场开拓计划。核心业务包括：面向银行的金融自助解决方案和终端销售、TETRA专网通信系统销售。市场开拓：面向银行的金融自助解决方案和终端：通过寻找当地具有售后服务能力的代理商、直接投标2种方式开拓3家海外银行，客户名称：中东某国家某些银行。疫情期间, 销售金融自助柜员机产品1500万（合同额）。TETRA专网通信系统：通过直接投标方式，开拓政府、军队、铁路、地铁领域海外客户。	项目管理：共完成3个项目，项目名称：中东某国家某些银行金融自助解决方案和终端项目。"
    bot_name = "丁涛"
    user_info = "聊天对象"
    user_name = "用户"
    # voice = 'zhiyan_emo'  # zhiyan的声音,略微的更女性化些;
    voice = "voice-f90ed52"  # 个性化声音
    # today = datetime.datetime.today().strftime('%y%m%d')
    # tts_out_path = f'tts_{voice}_{today}.wav'
    tts_out_path = None

    if characterGLM_chat_flag:  # 角色扮演机器人聊天
        zhipuai_key = config_read(
            config_path_zhipuai,
            section="zhipuai_SDK_API",
            option1="api_key",
            option2=None,
        )
        zhipuai.api_key = zhipuai_key
        client_id, client_secret = config_read(
            config_path_dtApp,
            section="DingTalkAPP_HumanoidRobot",
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
            accessKey_id, accessKey_secret = config_read(
                config_path_aliyunsdk,
                section="aliyunsdkcore",
                option1="AccessKey_ID",
                option2="AccessKey_Secret",
            )
            appKey = config_read(
                config_path_aliyunsdk, section="APP_tts", option1="AppKey"
            )
            azure_key, azure_region = config_read(
                config_path=config_path_azure,
                section="Azure_TTS",
                option1="key",
                option2="region",
            )
            client.register_callback_handler(
                ChatbotMessage_Utilies.TOPIC,
                VoiceChatHandler(
                    accessKey_id,
                    accessKey_secret,
                    aliyun_region_id="cn-shanghai",
                    aliyun_appKey=appKey,
                    tts_name="Example",
                    audio_path=tts_out_path,
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
                    zhipuai=zhipuai,
                    history_prompt=history_prompt,
                    bot_info=bot_info,
                    bot_name=bot_name,
                    user_name=user_name,
                    user_info=user_info,
                ),
            )

    else:  # GLM 办公助手, 文本聊
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
