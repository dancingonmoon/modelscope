# -*- coding: utf-8 -*-
# This file is auto-generated, don't edit it. Thanks.
import sys

from typing import List
import asyncio
import datetime
import requests
import json

from alibabacloud_dingtalk.robot_1_0.client import Client as dingtalkrobot_1_0Client
from alibabacloud_dingtalk.robot_1_0 import models as dingtalkrobot__1__0_models
from alibabacloud_tea_util import models as util_models

from alibabacloud_dingtalk.oauth2_1_0.client import Client as dingtalkoauth2_1_0Client
from alibabacloud_tea_openapi import models as open_api_models
from alibabacloud_dingtalk.oauth2_1_0 import models as dingtalkoauth_2__1__0_models
from alibabacloud_tea_util.client import Client as UtilClient
import sys
# sys.path.append("L:/Python_WorkSpace/dingtalk-sdk-python3")
# import dingtalk.api

from ChatBOT_APP import config_read, setup_logger


# 获取accessToken:
class get_accessToken:
    def __init__(self):
        pass

    @staticmethod
    def create_client() -> dingtalkoauth2_1_0Client:
        """
        使用 Token 初始化账号Client
        @return: Client
        @throws Exception
        """
        config = open_api_models.Config()
        config.protocol = 'https'
        config.region_id = 'central'
        return dingtalkoauth2_1_0Client(config)

    @staticmethod
    def main(Client_ID, Client_Secret):
        client = get_accessToken.create_client()
        get_access_token_request = dingtalkoauth_2__1__0_models.GetAccessTokenRequest(
            app_key=Client_ID,
            app_secret=Client_Secret,
        )
        try:
            response = client.get_access_token(get_access_token_request)
            # aToken = response.body.access_token
            # expireIn = response.body.expire_in # 有效时长 7200秒.即2小时;
            return response.body
        except Exception as err:
            # if not UtilClient.empty(err.code) and not UtilClient.empty(err.message):
            # err 中含有 code 和 message 属性，可帮助开发定位问题
            return err


class SendMessage_groupChat:
    def __init__(self):
        pass

    @staticmethod
    def create_client() -> dingtalkrobot_1_0Client:
        """
        使用 Token 初始化账号Client
        @return: Client
        @throws Exception
        """
        config = open_api_models.Config()
        config.protocol = 'https'
        config.region_id = 'central'
        return dingtalkrobot_1_0Client(config)

    @staticmethod
    def main(aToken, openConversationID,robotCode,coolAppCode=None,msg_key='sampleText',msg_param=None) -> None:
        """
        msg_key: 消息模板key: sampleText, sampleMarkdown, sampleImageMsg, sampleLink, sampleActionCard, sampleAudio, sampleFile, sampleVideo
        msg_param: dict; 对应于每一种消息模板中定义的参数: 例如: sampleMarkdown模板定义参数:
                        {"text": "hello text", "title": "hello title"};代码会自动将其打上引号作为字符串输入
        """
        client = SendMessage_groupChat.create_client()
        private_chat_send_headers = dingtalkrobot__1__0_models.PrivateChatSendHeaders()
        private_chat_send_headers.x_acs_dingtalk_access_token = aToken
        private_chat_send_request = dingtalkrobot__1__0_models.PrivateChatSendRequest(
            msg_param=f"{msg_param}",
            msg_key=msg_key,
            open_conversation_id=openConversationID,
            robot_code=robotCode,
            cool_app_code=coolAppCode
        )
        try:
            client.private_chat_send_with_options(private_chat_send_request, private_chat_send_headers,
                                                  util_models.RuntimeOptions())
        except Exception as err:
            # if not UtilClient.empty(err.code) and not UtilClient.empty(err.message):
                # err 中含有 code 和 message 属性，可帮助开发定位问题
            print(err)

    @staticmethod
    async def main_async(aToken, openConversationID,robotCode,coolAppCode=None,msg_key='sampleText',msg_param=None) -> None:
        client = SendMessage_userChat.create_client()
        private_chat_send_headers = dingtalkrobot__1__0_models.PrivateChatSendHeaders()
        private_chat_send_headers.x_acs_dingtalk_access_token = aToken
        private_chat_send_request = dingtalkrobot__1__0_models.PrivateChatSendRequest(
            msg_param=f"{msg_param}",
            msg_key=msg_key,
            open_conversation_id=openConversationID,
            robot_code=robotCode,
            cool_app_code=coolAppCode
        )
        try:
            await client.private_chat_send_with_options_async(private_chat_send_request, private_chat_send_headers,
                                                              util_models.RuntimeOptions())
        except Exception as err:
            if not UtilClient.empty(err.code) and not UtilClient.empty(err.message):
                # err 中含有 code 和 message 属性，可帮助开发定位问题
                pass


class SendMessage_userChat:
    def __init__(self):
        pass

    @staticmethod
    def create_client() -> dingtalkrobot_1_0Client:
        """
        使用 Token 初始化账号Client
        @return: Client
        @throws Exception
        """
        config = open_api_models.Config()
        config.protocol = 'https'
        config.region_id = 'central'
        return dingtalkrobot_1_0Client(config)

    @staticmethod
    def main(aToken, robot_code, user_ids, msg_key="sampleText", msg_param=None) -> None:
        """
        :param aToken:
        :param robot_code:
        :param user_ids: list; 包含发送的user_id的列表
        :param msg_key: 消息模板key: sampleText,sampleMarkdown,sampleImageMsg,sampleLink,sampleActionCard,sampleAudio,sampleFile,sampleVideo
        :param msg_param: dict; 对应于每一种消息模板中定义的参数: 例如: sampleMarkdown模板定义参数:
                            {"text": "hello text","title": "hello title"}; 代码会自动将其打上引号作为字符串输入
        :return:
        """
        client = SendMessage_userChat.create_client()
        batch_send_otoheaders = dingtalkrobot__1__0_models.BatchSendOTOHeaders()
        batch_send_otoheaders.x_acs_dingtalk_access_token = aToken
        batch_send_otorequest = dingtalkrobot__1__0_models.BatchSendOTORequest(
            robot_code=robot_code,
            user_ids=user_ids,
            msg_key=msg_key,
            msg_param=f"{msg_param}"
        )
        try:
            client.batch_send_otowith_options(batch_send_otorequest, batch_send_otoheaders,
                                              util_models.RuntimeOptions())
        except Exception as err:
            # if not UtilClient.empty(err.code) and not UtilClient.empty(err.message):
            # err 中含有 code 和 message 属性，可帮助开发定位问题
            print(err)

    @staticmethod
    async def main_async(aToken, robot_code, user_ids, msg_key="sampleText", msg_param=None) -> None:
        """

        :param aToken:
        :param robot_code:
        :param user_ids: list; 包含发送的user_id的列表
        :param msg_key: 消息模板key: sampleText,sampleMarkdown,sampleImageMsg,sampleLink,sampleActionCard,sampleAudio,sampleFile,sampleVideo
        :param msg_param: dict; 对应于每一种消息模板中定义的参数: 例如: sampleMarkdown模板定义参数:
                            {"text": "hello text","title": "hello title"}; 代码会自动将其打上引号作为字符串输入
        :return:
        """
        client = SendMessage_userChat.create_client()
        batch_send_otoheaders = dingtalkrobot__1__0_models.BatchSendOTOHeaders()
        batch_send_otoheaders.x_acs_dingtalk_access_token = aToken
        batch_send_otorequest = dingtalkrobot__1__0_models.BatchSendOTORequest(
            robot_code=robot_code,
            user_ids=user_ids,
            msg_key=msg_key,
            msg_param=f"{msg_param}",
        )
        try:
            await client.batch_send_otowith_options_async(batch_send_otorequest, batch_send_otoheaders,
                                                          util_models.RuntimeOptions())
        except Exception as err:
            if not UtilClient.empty(err.code) and not UtilClient.empty(err.message):
                # err 中含有 code 和 message 属性，可帮助开发定位问题
                pass


def upload2media_id(access_Token, media_content, media_type, ):
    """
    media upload, 直接RestAPI接口; Request Body 参数: media (FileItem类型),构建media包括:
        media_type: str; union['voice','image','file','video'];
        media_content: str或者object; 媒体文件的path,或者二进制media文件.
    """
    if isinstance(media_content, str):  # 若是文件路径;
        media_content = {'media': open(media_content, 'rb'), }
    elif isinstance(media_content, bytes):  # 若是bytes
        media_content = {'media': media_content}
    else:
        print("Invalid media file format")

    api = f"https://oapi.dingtalk.com/media/upload?access_token={access_Token}&type={media_type}"
    response = requests.post(api, files=media_content)
    # print(response.text)
    text_dict = json.loads(response.text)  # 将str转成dict
    if 'media_id' in text_dict:
        media_id = text_dict['media_id']
    else:
        media_id = text_dict
        print(text_dict)

    return media_id


if __name__ == '__main__':
    logger = setup_logger()

    config_path_dtApp = r"e:/Python_WorkSpace/config/DingTalk_APP.ini"
    # config_path_serp = r"e:/Python_WorkSpace/config/SerpAPI.ini"
    # config_path_zhipuai = r"e:/Python_WorkSpace/config/zhipuai_SDK.ini"

    client_id, client_secret = config_read(config_path_dtApp, section="DingTalkAPP_charGLM", option1='client_id',
                                           option2='client_secret')

    response = get_accessToken.main(client_id, client_secret)
    print(response)
    aToken = response.access_token
    robot_code = "dingqskohvdceaa7kahx"
    user_ids = ["030751651766858"]
    openConversationID = 'cidi6Du7cw2weMSqKitjMsnf9H/VYmT/URCV/E8CSGPjKA='
    msg_key = "sampleMarkdown"
    markdown_text = [
        '## 人类计划生育的失败',
        '### 人口的缩减试验的失败',
        '* 70年代计划生育政策',
        '* [*2023年, 人口大逆转*](https://www.zhihu.com/tardis/bd/art/445391746)',
        '* ![alt 人口断崖式下跌](https://img2.baidu.com/it/u=341909850,623946234&fm=253&fmt=auto&app=120&f=JPEG?w=1280&h=800)',
        f'> *消息发送时间:{datetime.datetime.now()}* '
    ]
    markdown_text = '  \n  '.join(markdown_text)
    msg_param = {
        "title": "信息摘要(测试):",
        "text": markdown_text
    }

    # media upload, 直接RestAPI接口; body参数: media (FileItem类型), type (str: ['voice','image','file','video']);
    # audio_path = r"E:/Python_WorkSpace/modelscope/DingTalkAPP/tts_aijing_out.wav"
    # media_id = upload2media_id(aToken, audio_path, 'voice')
    # print(media_id)
    #
    # # 构建语音消息模板:
    # msg_key = "sampleAudio"
    # msg_param = {"mediaId": media_id,
    #              "duration": "23"
    #              }

    # SendMessage_userChat.main(aToken, robot_code, user_ids, msg_key, msg_param)
    SendMessage_groupChat.main(aToken,openConversationID,robot_code, msg_key=msg_key, msg_param=msg_param)
    # asyncio.run(SendMessage_userChat.main_async(aToken,robot_code,user_ids,msg_key,msg_param,))

