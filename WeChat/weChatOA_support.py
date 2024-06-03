import requests
import json
import logging
import configparser
import time
import hashlib


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
    # sort_list = ''.join(sort_list).encode('utf-8') #将字符串转换成字节串;因为map(sha1,sort_list)需要的是字节串
    sha1 = hashlib.sha1()  # 进行sha1计算签名
    # 将字符串转换为字节
    byte_list = map(lambda x: x.encode("utf-8"), sort_list)
    # 将字节连接起来
    byte_str = b"".join(byte_list)
    # 进行sha1计算签名
    sha1.update(byte_str)
    hashcode = sha1.hexdigest()
    return hashcode


if __name__ == "__main__":
    config_path = "e:/Python_WorkSpace/config/WeChat_OpenAPI.ini"
    AppID, AppSecret = config_read(
        config_path, section="weChatOA", option1="AppID", option2="AppSecret"
    )
    existing_aToken_dict = {'access_token': '81_g_pW4WO_TXhOqcmTGs4XJZBkmuKwL2Wbkr5FLvCWwPeoGFbc_GdsGzgyWV6BeaCHId-oRFceoE7JeLol0LNHloOxvXqtv8j5lsVWI1E8Sul4F-Mjkqgcp5fvTL8NRCbAHATXI', 'expires_in': 7200, 'expireTime': 1717410212}
    access_token = get_WeChat_accessToken(AppID, AppSecret, existing_aToken_dict)
    # token = "lockup"
    # timestamp = 1717401318
    # nonce = 369471988
    # signature = get_signature(str(token), str(timestamp), str(nonce))
    # print(signature)
