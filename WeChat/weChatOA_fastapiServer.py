from typing import Union, Annotated
from fastapi import FastAPI, Path
from pydantic import BaseModel
from weChatOA_support import get_signature

description = """
## 微信公众号开发者服务器.🦬
 + **get 开发者服务器token验证**
 + **get **
"""
app = FastAPI(
    title="微信公众号开发者服务器",
    description=description,
)


class Item(BaseModel):
    ToUserName: str  # 开发者微信号
    FromUserName: str  # 发送方账号（一个OpenID）
    CreateTime: int  # 消息创建时间 （整型）
    MsgType: str  # 消息类型，文本为text
    Content: str  # 文本消息内容
    MsgId: int  # 消息id，64位整型
    MsgDataId: int  # 消息的数据ID（消息如果来自文章时才有）
    Idx: int  # 多图文时第几篇文章，从1开始（消息如果来自文章时才有）


@app.get("/wx")
async def token_validation(signature: str, timestamp: int, nonce: int, echostr: str):
    """
    每次微信服务器向开发者服务器消息推动,微信服务器会对开发者服务器发起验证，请在提交前按以下方式开发： 微信服务器将发送GET请求到填写的服务器地址URL上， GET请求携带参数如下
    :param signature: 签名
    :param timestamp: 时间戳
    :param nonce: 随机数
    :param echostr: 随机字符串
    :return:
    """
    try:
        token = "lockup"  # 请按照公众平台官网\基本配置中信息填写

        hashcode = get_signature(token, timestamp, nonce)
        print(f"handle/GET func: hashcode:{hashcode}, signature:{signature}")
        if hashcode == signature:  # 签名比较合法:构造回包返回微信服务器，回包消息体内容为URL链接中的echostr参数
            return echostr
        else:
            return "hashcode!=signature"
    except Exception as Argument:
        return Argument


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8080,
    )
