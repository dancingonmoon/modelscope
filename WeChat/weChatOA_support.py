import requests
import json
import logging
import configparser
import time

def config_read(
    config_path, section="weChatOA", option1="AppID", option2="AppSecret"
):
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
def get_WeChatOAaccessToken(AppID, AppSecret, existing_aToken_dict=None):
    """

    :param AppID:
    :param AppSecret:
    :param existing_aToken_dict: 已经存在的,历史access_token字典,用于判断是否重复获取;
                          字典结构:{"accessToken": token, "expireTime": expireTime, "expires_in": expireIn}
    :return:
    """
    now = int(time.time())
    if existing_aToken_dict and 'expireTime' in existing_aToken_dict and now < existing_aToken_dict["expireTime"]:
        return existing_aToken_dict

    url = 'https://api.weixin.qq.com/cgi-bin/token'
    url = f'{url}?grant_type=client_credential&appid={AppID}&secret={AppSecret}'

    try:
        now = int(time.time())
        response = requests.get(url)
        result = response.json()
        if "access_token" in result:
            aToken = result["access_token"]
            expireIn = result["expires_in"] # 有效时长 7200秒.即2小时;
            result["expireTime"] = now + expireIn - (5 * 60)  # reserve 5min buffer time

        print(result)
        return result
    except Exception as err:
        # err 中含有 code 和 message 属性，可帮助开发定位问题
        return err

if __name__ == "__main__":
    config_path = "L:/Python_WorkSpace/config/WeChat_OpenAPI.ini"
    AppID, AppSecret = config_read(config_path, section="weChatOA", option1="AppID", option2="AppSecret")
    existing_aToken_dict ={'access_token': '81_fgEGgQVTCMQgJmJPGdEXv8TIP3862awOLtogE5TnwMIBhh7lixGckYYa5653eKvgcvHd72hdWy7s9SRAk9UL_aTlS78JJQ6kI_RH874uMoJ7Hh8nqoeW4RUplwsOMPbAHAJHK',
                           'expires_in': 7200,
                           'expireTime': 1717346580}
    access_token = get_WeChatOAaccessToken(AppID, AppSecret, existing_aToken_dict)
