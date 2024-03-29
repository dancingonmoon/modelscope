# -*- coding: utf-8 -*-
# This file is auto-generated, don't edit it. Thanks.
import sys

from typing import List

from alibabacloud_dingtalk.robot_1_0.client import Client as dingtalkrobot_1_0Client
from alibabacloud_tea_openapi import models as open_api_models
from alibabacloud_dingtalk.robot_1_0 import models as dingtalkrobot__1__0_models
from alibabacloud_tea_util import models as util_models
from alibabacloud_tea_util.client import Client as UtilClient

from .ChatBOT_APP import config_read, setup_logger


class Sample:
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
    def main(
        args: List[str],
    ) -> None:
        client = Sample.create_client()
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
            client.private_chat_send_with_options(private_chat_send_request, private_chat_send_headers, util_models.RuntimeOptions())
        except Exception as err:
            if not UtilClient.empty(err.code) and not UtilClient.empty(err.message):
                # err 中含有 code 和 message 属性，可帮助开发定位问题
                pass

    @staticmethod
    async def main_async(
        args: List[str],
    ) -> None:
        client = Sample.create_client()
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
            await client.private_chat_send_with_options_async(private_chat_send_request, private_chat_send_headers, util_models.RuntimeOptions())
        except Exception as err:
            if not UtilClient.empty(err.code) and not UtilClient.empty(err.message):
                # err 中含有 code 和 message 属性，可帮助开发定位问题
                pass

if __name__ == '__main__':

    logger = setup_logger()

    config_path_dtApp = r"e:/Python_WorkSpace/config/DingTalk_APP.ini"
    config_path_serp = r"e:/Python_WorkSpace/config/SerpAPI.ini"
    config_path_zhipuai = r"e:/Python_WorkSpace/config/zhipuai_SDK.ini"

    client_id, client_secret = config_read(config_path_dtApp, section="DingTalkAPP_charGLM", option1='client_id',
                                           option2='client_secret')

    Sample.main(sys.argv[1:])