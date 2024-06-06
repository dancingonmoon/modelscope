import requests
import logging
import configparser
import time
import hashlib
import xmltodict


def config_read(config_path, section="weChatOA", option1="AppID", option2="AppSecret"):
    """
    option2 = None 时,仅输出第一个option1的值; 否则输出section下的option1与option2两个值
    """
    config = configparser.ConfigParser()
    config.read(config_path, encoding="utf-8")
    option1_value = config.get(section=section, option=option1)
    if option2 is not None:
        option2_value = config.get(section=section, option=option2)
        return option1_value, option2_value
    else:
        return option1_value


def setup_logger():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    # 控制台handler:
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s %(name)-8s %(levelname)-8s %(message)s [%(filename)s:%(lineno)d]"
        )
    )
    logger.addHandler(console_handler)

    # 文件handler:
    file_handler = logging.FileHandler("log.log")
    file_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s %(name)-8s %(levelname)-8s %(message)s [%(filename)s:%(lineno)d]"
        )
    )
    logger.addHandler(file_handler)
    return logger


def get_WeChat_accessToken(AppID, AppSecret, existing_aToken_dict=None):
    """

    :param AppID:
    :param AppSecret:
    :param existing_aToken_dict: 已经存在的,历史access_token字典,用于判断是否重复获取;
                          字典结构:{"accessToken": token, "expireTime": expireTime, "expires_in": expireIn}
    :return:
    """
    now = int(time.time())
    if (
        existing_aToken_dict
        and "expireTime" in existing_aToken_dict
        and now < existing_aToken_dict["expireTime"]
    ):
        print(f"已存access_token:\n{existing_aToken_dict}")
        return existing_aToken_dict

    url = "https://api.weixin.qq.com/cgi-bin/token"
    url = f"{url}?grant_type=client_credential&appid={AppID}&secret={AppSecret}"

    try:
        now = int(time.time())
        response = requests.get(url)
        result = response.json()
        if "access_token" in result:
            aToken = result["access_token"]
            expireIn = result["expires_in"]  # 有效时长 7200秒.即2小时;
            result["expireTime"] = now + expireIn - (5 * 60)  # reserve 5min buffer time

        print(f"新生access_token:\n{result}")
        return result
    except Exception as err:
        # err 中含有 code 和 message 属性，可帮助开发定位问题
        return err


def get_signature(token, timestamp, nonce):
    """
    模拟实现微信URL参数中的signature签名过程：
    1. 将token、timestamp（URL参数中的）、nonce（URL参数中的）三个参数进行字典序排序，排序后结果为:["1717401318","369471988","lockup"]
    2. 将三个参数字符串拼接成一个字符串："1717401318369471988lockup"
    3. 进行sha1签名计算：4e2106caf85c97a38d03e375cc7234663ac31fef
    4. 开发者需按照此流程计算签名并与URL参数中的signature进行对比验证，相等则验证通过
    :return:
    """
    sort_list = [str(token), str(timestamp), str(nonce)]
    sort_list.sort()
    sha1 = hashlib.sha1()  # 进行sha1计算签名
    # 将字符串转换为字节
    byte_list = map(lambda x: x.encode("utf-8"), sort_list)
    # 将字节连接起来
    byte_str = b"".join(byte_list)
    # 进行sha1计算签名
    sha1.update(byte_str)
    hashcode = sha1.hexdigest()
    return hashcode


def weChatOA_text_reply(message_dict, text_content):
    """
    微信公众号向开发者服务器发送消息时，开发者服务器回复指定XML格式消息给微信服务器,微信服务器可以实现消息回复.
    本函数实现文本消息的回复
    :message_dict: 微信服务器向开发者服务器发送的POST请求体中,包含了XML格式的消息体;该消息体转换成字典;此处为文本消息的字典
    :text: 开发者服务器回复的文本消息
    :return: 微信服务器要求的XML格式,实现文本消息回复
    """
    if message_dict["MsgType"] == "text":
        ToUserName = message_dict["ToUserName"]
        FromUserName = message_dict["FromUserName"]
        # Xml文本方式:
        # XmlForm = """
        #             <xml>
        #                 <ToUserName><![CDATA[{ToUserName}]]></ToUserName>
        #                 <FromUserName><![CDATA[{FromUserName}]]></FromUserName>
        #                 <CreateTime>{CreateTime}</CreateTime>
        #                 <MsgType><![CDATA[text]]></MsgType>
        #                 <Content><![CDATA[{Content}]]></Content>
        #             </xml>
        #             """
        # reply_xml = XmlForm.format(ToUserName=FromUserName, FromUserName=ToUserName, CreateTime=int(time.time()), Content=text_content)
        # xmltodict库方式:
        reply_dict = {
            "ToUserName": FromUserName,
            "FromUserName": ToUserName,
            "CreateTime": int(time.time()),
            "MsgType": "text",
            "Content": text_content,
        }
        reply_xml_dict = {"xml": reply_dict}
        reply_xml = xmltodict.unparse(reply_xml_dict, )
        return reply_xml
    # elif message_dict["MsgType"] == "image":
    #     print({"MsgType": message_dict["MsgType"], "MediaId": message_dict["MediaId"]})
    #     return {"MsgType": message_dict["MsgType"], "MediaId": message_dict["MediaId"]}
    # elif message_dict["MsgType"] == "voice":
    #     print({"MsgType": message_dict["MsgType"], "MediaId": message_dict["MediaId"]})
    #     return {"MsgType": message_dict["MsgType"], "MediaId": message_dict["MediaId"]}
    # elif message_dict["MsgType"] == "video":
    #     print({"MsgType": message_dict["MsgType"], "MediaId": message_dict["MediaId"]})
    #     return {"MsgType": message_dict["MsgType"], "MediaId": message_dict["MediaId"]}
    # else:
    #     return "Invalid MsgType"


if __name__ == "__main__":
    config_path = "l:/Python_WorkSpace/config/WeChat_OpenAPI.ini"
    AppID, AppSecret = config_read(
        config_path, section="weChatOA", option1="AppID", option2="AppSecret"
    )
    existing_aToken_dict = {
        "access_token": "81_z5s4xNwPmXeR1HS6Y-abCMgGQATvHYx9nAi1bt0o9qiQ-xRwuqTRmISmdu7kk35Vk786UtREW4r6ZzMKHTN39H-l9XnHHSzEWP-_Q_lrCKdMP7elwF_5l7kcR5AHTOgACASEZ",
        "expires_in": 7200,
        "expireTime": 1717592756,
    }
    access_token = get_WeChat_accessToken(AppID, AppSecret, existing_aToken_dict)
    # token = "lockup"
    # timestamp = 1717401318
    # nonce = 369471988
    # signature = get_signature(str(token), str(timestamp), str(nonce))
    # print(signature)
