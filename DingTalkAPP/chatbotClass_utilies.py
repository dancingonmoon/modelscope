import io

import dingtalk_stream
from dingtalk_stream.chatbot import TextContent, ImageContent, RichTextContent, AtUser, HostingContext, ChatbotMessage, \
    ConversationMessage
from dingtalk_stream import ChatbotHandler
import requests
import json

from alibabacloud_dingtalk.robot_1_0.client import Client as dingtalkrobot_1_0Client
from alibabacloud_dingtalk.robot_1_0 import models as dingtalkrobot__1__0_models
from alibabacloud_tea_util import models as util_models

from alibabacloud_dingtalk.oauth2_1_0.client import Client as dingtalkoauth2_1_0Client
from alibabacloud_tea_openapi import models as open_api_models
from alibabacloud_dingtalk.oauth2_1_0 import models as dingtalkoauth_2__1__0_models
from alibabacloud_tea_util.client import Client as UtilClient


class ChatbotMessage_Utilies(ChatbotMessage):
    """
    在类ChatbotMessage里,增加除image以外的voice,video等media消息的支持
    """

    def __init__(self, ):
        super(ChatbotMessage_Utilies, self).__init__()

        # 增加部分:
        self.audio_duration = None
        self.audio_downloadCode = None
        self.audio_recognition = None

        self.video_duration = None
        self.video_downloadCode = None
        self.video_videoType = None

        self.file_downloadCode = None
        self.file_fileName = None

    @classmethod
    def from_dict(cls, d):
        msg = ChatbotMessage_Utilies()
        data = ''
        for name, value in d.items():
            if name == 'isInAtList':
                msg.is_in_at_list = value
            elif name == 'sessionWebhook':
                msg.session_webhook = value
            elif name == 'senderNick':
                msg.sender_nick = value
            elif name == 'robotCode':
                msg.robot_code = value
            elif name == 'sessionWebhookExpiredTime':
                msg.session_webhook_expired_time = int(value)
            elif name == 'msgId':
                msg.message_id = value
            elif name == 'senderId':
                msg.sender_id = value
            elif name == 'chatbotUserId':
                msg.chatbot_user_id = value
            elif name == 'conversationId':
                msg.conversation_id = value
            elif name == 'isAdmin':
                msg.is_admin = value
            elif name == 'createAt':
                msg.create_at = value
            elif name == 'conversationType':
                msg.conversation_type = value
            elif name == 'atUsers':
                msg.at_users = [AtUser.from_dict(i) for i in value]
            elif name == 'chatbotCorpId':
                msg.chatbot_corp_id = value
            elif name == 'senderCorpId':
                msg.sender_corp_id = value
            elif name == 'conversationTitle':
                msg.conversation_title = value
            elif name == 'msgtype':
                msg.message_type = value
                if value == 'text':
                    msg.text = TextContent.from_dict(d['text'])
                elif value == 'picture':
                    msg.image_content = ImageContent.from_dict(d['content'])
                elif value == 'richText':
                    msg.rich_text_content = RichTextContent.from_dict(d['content'])
                # 新增部分:
                elif value == 'audio':
                    msg.audio_duration = d['content']['duration']
                    msg.audio_downloadCode = d['content']['downloadCode']
                    msg.audio_recognition = d['content']['recognition']
                elif value == 'video':
                    msg.video_duration = d['content']['duration']
                    msg.video_downloadCode = d['content']['downloadCode']
                    msg.video_videoType = d['content']['videoType']
                elif value == 'file':
                    msg.file_fileName = d['content']['fileName']
                    msg.file_downloadCode = d['content']['downloadCode']
            elif name == 'senderStaffId':
                msg.sender_staff_id = value
            elif name == 'hostingContext':
                msg.hosting_context = HostingContext()
                msg.hosting_context.user_id = value["userId"]
                msg.hosting_context.nick = value["nick"]
            elif name == 'conversationMsgContext':
                msg.conversation_msg_context = []
                for v in value:
                    conversation_msg = ConversationMessage()
                    conversation_msg.read_status = v["readStatus"]
                    conversation_msg.send_time = v["sendTime"]
                    conversation_msg.sender_user_id = v["senderUserId"]

                    msg.conversation_msg_context.append(conversation_msg)
            else:
                msg.extensions[name] = value
        return msg

    def get_media_list(self, ):
        """
        借鉴get_image_list方法,增加支持media,获取接收voice/video消息的downloadCode
        (注:接收"msgtype":[text,richText,picture,audio,video,file]
        :return:
        """
        if self.message_type == 'audio':
            return {'msg_type': 'audio', 'downloadCode': self.audio_downloadCode}
        elif self.message_type == 'video':
            return {'msg_type': 'video', 'downloadCode': self.video_downloadCode}
        elif self.message_type == 'file':
            return {'msg_type': 'file', 'downloadCode': self.file_downloadCode}
        else:
            return {'downloadCode': None}


class ChatbotHandler_utilies(ChatbotHandler):
    """
    补充增加方法,实现对audio,video,file接收的媒体信息:
    1) 根据downloadCode获取下载链接;
    2) 下载audio/video/file,再上传,获取media_id;
    3) 对存为media_id的audio/video/file,发送消息
    """

    def __init__(self, ):
        super(ChatbotHandler_utilies, self).__init__()

    def extract_media_from_incoming_message(self, incoming_message: ChatbotMessage_Utilies,
                                            media_save_folder=None) -> list:
        """
        获取用户发送的媒体文件，重新上传，获取新的media_ids列表;或者不上传,直接下载到本地存储
        :param incoming_message:
        media_save_folder: 不为None,是Path的话,则media存盘,并不上传空间获得media_ids列表;media_save_path需为文件目录,文件名自动生成
        :return: media_id list
        """
        media_dic = incoming_message.get_media_list()
        msg_type = media_dic['msg_type']
        download_code = media_dic['downloadCode']
        # self.logger.info(f"downloadCode: {download_code}")
        if download_code is None:
            return None

        filetype = None
        if msg_type == 'audio':
            filetype = 'voice'
            # filename = 'voice.wav'
        elif msg_type == 'video':
            filetype = 'video'
            # filename = 'video.mp4'
        elif msg_type == 'file':
            filetype = 'file'
            # filename = None

        download_url = self.get_image_download_url(download_code)  # get_image_download_url应该可以获取任何media的下载链接
        # self.logger.info(f"download_url: {download_url}")
        media_content = requests.get(download_url)
        # self.logger.info(f"media_content.content: {media_content.content}")
        # 上传媒体的类型filetype: str; union['voice','image','file','video']
        # FileItem类型: 'media': (filename, image_content, mimetype)
        media_id = None
        if media_save_folder is None:
            media_id = self.upload2media_id(media_content.content, filetype)
            return media_id

        elif isinstance(media_save_folder, str):
            with open(f"{media_save_folder}/{filetype}.{filetype}", 'wb') as f:
                f.write(media_content.content)
                self.logger.info(f"{filetype}.{filetype} 下载完成")

    def upload2media_id(self, media_content, media_type, ):
        """
        media upload, 返回media_id.
        使用RestAPI接口; Request Body 参数: media (FileItem类型),构建media包括:
                media_type: str; union['voice','image','file','video'];
                media_content: str或者object; 媒体文件的path,或者二进制media文件content.
        """
        if isinstance(media_content, str):  # 若是文件路径;
            media_content = {'media': open(media_content, 'rb'), }
        elif isinstance(media_content, io.BytesIO):  # 若是bytes
            # form-data中媒体文件标识，有filename、filelength、content-type等信息。
            # 这里的文件名'audio.wav'是随便定义, 无特别意义
            media = ('audio.wav', media_content, 'audio/wav')
            media_content = {'media': media}
        else:
            self.logger.error("Invalid media file format")

        access_token = self.dingtalk_client.get_access_token()
        api = f"https://oapi.dingtalk.com/media/upload?access_token={access_token}&type={media_type}"
        response = requests.post(api, files=media_content)
        # print(response.text)
        text_dict = json.loads(response.text)  # 将str转成dict
        media_id = None
        if 'media_id' in text_dict:
            media_id = text_dict['media_id']
        else:
            self.logger.info(f"response upon uploading: {text_dict}")

        return media_id

    def reply_voice_http(self,
                         mediaId: str, duration: int,
                         incoming_message: ChatbotMessage_Utilies):
        """
        通过OpenAPI RestAPI http构造requests,post语音消息 #未成功,需技术支持
        """
        access_token = self.dingtalk_client.get_access_token()
        if not access_token:
            self.logger.error(
                'reply_voice failed, cannot get dingtalk access token')
            return None

        msgKey = 'sampleText'
        msgParm = "{\"content\":\"@lAfPDf0i-NZzbiPOKV79d85HUSK4\"}"

        request_headers = {
            'Host': 'api.dingtalk.com',
            'x-acs-dingtalk-access-token': access_token,
            'Content-Type': 'application/json',
        }
        values = {
            'robotCode': incoming_message.robot_code,
            'msgKey': msgKey,
            'msgParm': msgParm,
        }
        url = "https://api.dingtalk.com"
        if incoming_message.conversation_type == "1":  # 1 单聊; 2 群聊
            values['userIds'] = [incoming_message.sender_staff_id]
            url += '/v1.0/robot/oToMessages/batchSend'
        elif incoming_message.conversation_type == "2":  # 1 单聊; 2 群聊
            values["openConversationId"] = incoming_message.conversation_id
            url += '/v1.0/robot/groupMessages/send'

        self.logger.info(f"url:{url}; values:{values}; request_headers:{request_headers}")
        try:
            response = requests.post(url, headers=request_headers, json=values)
            response.raise_for_status()
        except Exception as e:
            self.logger.error('reply sampleAudio failed, error=%s', e)
            return None
        return response.json()

    def reply_voice_SDK(self,
                        mediaId: str, duration: int,
                        incoming_message: ChatbotMessage_Utilies):
        """
        通过OpenAPI python SDK,来发送语音消息
        """
        access_token = self.dingtalk_client.get_access_token()
        if not access_token:
            self.logger.error(
                'reply_voice failed, cannot get dingtalk access token')
            return None

        msgKey = 'sampleAudio'
        msgParm = {
            'mediaId': mediaId,
            'duration': f"{duration}",
        }
        robotCode = incoming_message.robot_code
        userId = [incoming_message.sender_staff_id]
        openConversationId = incoming_message.conversation_id
        if incoming_message.conversation_type == "1":  # 1 单聊; 2 群聊
            response = OpenAPI_SendMessage.sendMessage_userChat(access_token, robot_code=robotCode, user_ids=userId,
                                                                msg_key=msgKey, msg_param=msgParm)
        elif incoming_message.conversation_type == "2":  # 1 单聊; 2 群聊
            response = OpenAPI_SendMessage.sendMessage_groupChat(access_token, open_conversation_id=openConversationId,
                                                                 robot_code=robotCode,
                                                                 msg_key=msgKey, msg_param=msgParm)

        self.logger.info(f"OpenAPI SDK response: {response}")

        return response


class OpenAPI_SendMessage:
    """
    使用OpenAPI, python SDK (非RestAPI https),来实现批量发送人与机器人会话中机器人消息(包括文本,图片,语音,视频,文件)
    """

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
    def sendMessage_userChat(access_token, robot_code, user_ids, msg_key="sampleText", msg_param=None):
        """
        :access_token:
        :robot_code:
        :user_ids: list; 包含发送的user_id的列表
        :msg_key: 消息模板key: sampleText,sampleMarkdown,sampleImageMsg,sampleLink,sampleActionCard,sampleAudio,sampleFile,sampleVideo
        :msg_param: dict; 对应于每一种消息模板中定义的参数: 例如: sampleMarkdown模板定义参数:
                            {"text": "hello text","title": "hello title"}; 代码会自动将其打上引号作为字符串输入
        :return:
        """
        client = OpenAPI_SendMessage.create_client()
        batch_send_otoheaders = dingtalkrobot__1__0_models.BatchSendOTOHeaders()
        batch_send_otoheaders.x_acs_dingtalk_access_token = access_token
        batch_send_otorequest = dingtalkrobot__1__0_models.BatchSendOTORequest(
            robot_code=robot_code,
            user_ids=user_ids,
            msg_key=msg_key,
            msg_param=f"{msg_param}"
        )
        try:
            response = client.batch_send_otowith_options(batch_send_otorequest, batch_send_otoheaders,
                                                         util_models.RuntimeOptions())
            response = response.body

        except Exception as err:
            # if not UtilClient.empty(err.code) and not UtilClient.empty(err.message):
            # err 中含有 code 和 message 属性，可帮助开发定位问题
            response = err

        return response

    @staticmethod
    def sendMessage_groupChat(access_token, open_conversation_id, robot_code, coolAppCode=None, msg_key='sampleText',
                              msg_param=None):
        """
        msg_key: 消息模板key: sampleText, sampleMarkdown, sampleImageMsg, sampleLink, sampleActionCard, sampleAudio, sampleFile, sampleVideo
        msg_param: dict; 对应于每一种消息模板中定义的参数: 例如: sampleMarkdown模板定义参数:
                        {"text": "hello text", "title": "hello title"};代码会自动将其打上引号作为字符串输入
        """
        client = OpenAPI_SendMessage.create_client()
        private_chat_send_headers = dingtalkrobot__1__0_models.PrivateChatSendHeaders()
        private_chat_send_headers.x_acs_dingtalk_access_token = access_token
        private_chat_send_request = dingtalkrobot__1__0_models.PrivateChatSendRequest(
            msg_param=f"{msg_param}",
            msg_key=msg_key,
            open_conversation_id=open_conversation_id,
            robot_code=robot_code,
            cool_app_code=coolAppCode
        )
        try:
            response = client.private_chat_send_with_options(private_chat_send_request, private_chat_send_headers,
                                                             util_models.RuntimeOptions())
            response = response.body
        except Exception as err:
            # if not UtilClient.empty(err.code) and not UtilClient.empty(err.message):
            # err 中含有 code 和 message 属性，可帮助开发定位问题
            response = err
        return response
