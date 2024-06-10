from typing import Union, Annotated
from fastapi import FastAPI, Request, Response
from pydantic import BaseModel
from weChatOA_support import get_signature, weChatOA_text_reply, setup_logger, config_read
import xmltodict
from zhipuai import ZhipuAI
import time

import sys

sys.path.append('../GLM')
from GLM.GLM_callFunc import GLM_callFunc_SSE_SYN

description = """
## 微信公众号开发者服务器.🦬
 + **get 开发者服务器token验证**
 + **post 微信公众号被动回复**
"""
app = FastAPI(
    title="微信公众号开发者服务器",
    description=description,
)




@app.get("/wx")
async def token_validation(signature: str, timestamp: int, nonce: int, echostr: int):
    """
    每次微信服务器向开发者服务器消息推动,微信服务器会对开发者服务器发起验证，请在提交前按以下方式开发： 微信服务器将发送GET请求到填写的服务器地址URL上， GET请求携带参数如下
    signature: 签名
    timestamp: 时间戳
    nonce: 随机数
    echostr: 必须设置成int, (微信开发文档中定义为随机字符串,但其会导致返回时在字符串外添加引号,导致2层引号,与echostr不一致,验证失败)
    :return: echostr
    """
    try:
        token = "lockup"  # 请按照公众平台官网\基本配置中信息填写

        hashcode = get_signature(token, timestamp, nonce)
        if hashcode == signature:  # 签名比较合法:构造回包返回微信服务器，回包消息体内容为URL链接中的echostr参数
            return echostr  # 将字符串转化为字节串输出
        else:
            return "hashcode!=signature"
    except Exception as Argument:
        return Argument


@app.post("/wx")
async def post_message(
        request: Request,
):
    xml_message = await request.body()
    message_dict = xmltodict.parse(xml_message)["xml"]
    # logger.info(f"开发者服务器post收到:\n{message_dict}")

    if message_dict["MsgType"] == "text":
        question = message_dict["Content"]
        logger.info(f"收到问题: {question}")
        query = question
        out = GLM_callFunc_SSE_SYN(zhipuai_client, question, query, LLM_model=LLM_model,
                                   web_search_enable=web_search_enable, web_search_result_show=web_search_result_show,
                                   time_threshold=time_threshold)
        answer = out[0]
        logger.info(f"模型回答: {answer}")

        reply_xml = weChatOA_text_reply(message_dict, answer)
        # headers = {"Content-Type": "text/xml; charset=utf-8"} # text/xml 其实就是html格式
        return Response(reply_xml, media_type="application/xml", )
    else:
        text_content = "受微信被动回复5秒限制,开发非文本消息被动回复,超时概率大,暂停..."
        ToUserName = message_dict["ToUserName"]
        FromUserName = message_dict["FromUserName"]
        reply_dict = {
            "ToUserName": FromUserName,
            "FromUserName": ToUserName,
            "CreateTime": int(time.time()),
            "MsgType": "text",
            "Content": text_content,
        }
        reply_xml_dict = {"xml": reply_dict}
        reply_xml = xmltodict.unparse(reply_xml_dict, )
        # headers = {"Content-Type": "text/xml; charset=utf-8"} # text/xml 其实就是html格式
        return Response(reply_xml, media_type="application/xml", )


if __name__ == "__main__":
    import uvicorn

    logger = setup_logger()
    config_path_zhipuai = r"l:/Python_WorkSpace/config/zhipuai_SDK.ini"
    zhipu_apiKey = config_read(config_path_zhipuai, section="zhipuai_SDK_API", option1="api_key")
    zhipuai_client = ZhipuAI(api_key=zhipu_apiKey)
    web_search_enable = True
    web_search_result_show = False
    time_threshold = 5
    LLM_model = "glm-4-air"

    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8080,
    )
