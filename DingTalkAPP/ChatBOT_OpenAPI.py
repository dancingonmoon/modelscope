# -*- coding: utf-8 -*-
# This file is auto-generated, don't edit it. Thanks.
import sys

from typing import List
import asyncio

from alibabacloud_dingtalk.robot_1_0.client import Client as dingtalkrobot_1_0Client
from alibabacloud_dingtalk.robot_1_0 import models as dingtalkrobot__1__0_models
from alibabacloud_tea_util import models as util_models

from alibabacloud_dingtalk.oauth2_1_0.client import Client as dingtalkoauth2_1_0Client
from alibabacloud_tea_openapi import models as open_api_models
from alibabacloud_dingtalk.oauth2_1_0 import models as dingtalkoauth_2__1__0_models
from alibabacloud_tea_util.client import Client as UtilClient
import sys
sys.path.append("L:/Python_WorkSpace/dingtalk-sdk-python3")
import dingtalk.api


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
    def main(aToken, client_ID, client_secret) -> None:
        client = SendMessage_groupChat.create_client()
        private_chat_send_headers = dingtalkrobot__1__0_models.PrivateChatSendHeaders()
        private_chat_send_headers.x_acs_dingtalk_access_token = aToken
        private_chat_send_request = dingtalkrobot__1__0_models.PrivateChatSendRequest(
            msg_param='{"content":"钉钉，让进步发生"}',
            msg_key='sampleText',
            open_conversation_id='cid6******==',
            robot_code="dingqskohvdceaa7kahx",
            cool_app_code='COOLAPP-1-******9000J'
        )
        try:
            client.private_chat_send_with_options(private_chat_send_request, private_chat_send_headers,
                                                  util_models.RuntimeOptions())
        except Exception as err:
            if not UtilClient.empty(err.code) and not UtilClient.empty(err.message):
                # err 中含有 code 和 message 属性，可帮助开发定位问题
                pass

    @staticmethod
    async def main_async(
            args: List[str],
    ) -> None:
        client = SendMessage_userChat.create_client()
        private_chat_send_headers = dingtalkrobot__1__0_models.PrivateChatSendHeaders()
        private_chat_send_headers.x_acs_dingtalk_access_token = '<your access token>'
        private_chat_send_request = dingtalkrobot__1__0_models.PrivateChatSendRequest(
            msg_param='{"content":"钉钉，让进步发生"}',
            msg_key='sampleText',
            open_conversation_id='cid6******==',
            robot_code='dingue4kfzdxbyn0pjqd',
            cool_app_code='COOLAPP-1-******9000J'
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
        :return:
        """
        client = SendMessage_userChat.create_client()
        batch_send_otoheaders = dingtalkrobot__1__0_models.BatchSendOTOHeaders()
        batch_send_otoheaders.x_acs_dingtalk_access_token = aToken
        batch_send_otorequest = dingtalkrobot__1__0_models.BatchSendOTORequest(
            robot_code=robot_code,
            user_ids=user_ids,
            msg_key=msg_key,
            msg_param=msg_param
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
        :param msg_param: string; 对应于每一种消息模板中定义的参数: 例如: sampleMarkdown模板定义参数:
                            '{"text": "hello text","title": "hello title"}'
        :return:
        """
        client = SendMessage_userChat.create_client()
        batch_send_otoheaders = dingtalkrobot__1__0_models.BatchSendOTOHeaders()
        batch_send_otoheaders.x_acs_dingtalk_access_token = aToken
        batch_send_otorequest = dingtalkrobot__1__0_models.BatchSendOTORequest(
            robot_code=robot_code,
            user_ids=user_ids,
            msg_key=msg_key,
            msg_param=msg_param,
        )
        try:
            await client.batch_send_otowith_options_async(batch_send_otorequest, batch_send_otoheaders,
                                                          util_models.RuntimeOptions())
        except Exception as err:
            if not UtilClient.empty(err.code) and not UtilClient.empty(err.message):
                # err 中含有 code 和 message 属性，可帮助开发定位问题
                pass



if __name__ == '__main__':
    logger = setup_logger()

    config_path_dtApp = r"l:/Python_WorkSpace/config/DingTalk_APP.ini"
    # config_path_serp = r"e:/Python_WorkSpace/config/SerpAPI.ini"
    # config_path_zhipuai = r"e:/Python_WorkSpace/config/zhipuai_SDK.ini"

    client_id, client_secret = config_read(config_path_dtApp, section="DingTalkAPP_charGLM", option1='client_id',
                                           option2='client_secret')

    response = get_accessToken.main(client_id, client_secret)
    print(response)
    aToken = response.access_token
    robot_code = "dingqskohvdceaa7kahx"
    user_ids = ["030751651766858"]
    msg_key = "sampleText"
    # msg_key = "sampleAudio"
    msg_param = '{"content": "试试上传语音发声",}'
    #    msg_param = {
    #    "mediaId": "@IR_P********nFkfhsisbf4A",
    #    "duration":"xxxxx"
    # }
    SendMessage_userChat.main(aToken,robot_code,user_ids,msg_key,msg_param)
    # asyncio.run(SendMessage_userChat.main_async(aToken,robot_code,user_ids,msg_key,msg_param,))


    # media upload:

    req = dingtalk.api.OapiMediaUploadRequest("https://oapi.dingtalk.com/media/upload")

    req.type = "audio"
    req.media = dingtalk.api.FileItem('abc.jpg', open('abc.jpg', 'rb'))
    try:
        resp = req.getResponse(access_token)
        print(resp)
    except Exception, e:
        print(e)

