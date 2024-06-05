from typing import Union, Annotated
from fastapi import FastAPI, Request
from pydantic import BaseModel, Json
from weChatOA_support import get_signature
import json

description = """
## 微信公众号开发者服务器.🦬
 + **get 开发者服务器token验证**
 + **get **
"""
app = FastAPI(
    title="微信公众号开发者服务器",
    description=description,
)


class MessageBody(BaseModel):
    ToUserName: str  # 开发者微信号
    FromUserName: str  # 发送方账号（一个OpenID）
    CreateTime: int  # 消息创建时间 （整型）
    MsgType: str  # 消息类型，文本为text

    MsgId: int  # 消息id，64位整型
    MsgDataId: Union[int, None] = None  # 消息的数据ID（消息如果来自文章时才有）
    Idx: Union[int, None] = None  # 多图文时第几篇文章，从1开始（消息如果来自文章时才有）


class TextMessage(MessageBody):
    Content: str  # 文本消息内容


class ImageMessage(MessageBody):
    PicUrl: str  # 图片链接（由系统生成）
    MediaId: int  # 	图片消息媒体id，可以调用获取临时素材接口拉取数据。


class VoiceMessage(MessageBody):
    MediaId: int  # 	语音消息媒体id，可以调用获取临时素材接口拉取数据，Format为amr时返回8K采样率amr语音。
    Format: str  # 	语音格式，如amr，speex等
    MediaId16K: int  # 	16K采样率语音消息媒体id，可以调用获取临时素材接口拉取数据，返回16K采样率amr/speex语音。


class VideoMessage(MessageBody):
    MediaId: int  # 	视频消息媒体id，可以调用获取临时素材接口拉取数据。
    ThumbMediaId: int  # 	视频消息缩略图的媒体id，可以调用多媒体文件下载接口拉取数据。


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

class testvar(BaseModel):
    message: Json
@app.post("/wx")
async def post_message(
    message: testvar
):
    print(message)
    return message

    # if message.MsgType == "text":
    #     print({"MsgType": message.MsgType, "Content": message.Content})
    #     return {"MsgType": message.MsgType, "Content": message.Content}
    # elif message.MsgType == "image":
    #     print({"MsgType": message.MsgType, "MediaId": message.MediaId})
    #     return {"MsgType": message.MsgType, "MediaId": message.MediaId}
    # elif message.MsgType == "voice":
    #     print({"MsgType": message.MsgType, "MediaId": message.MediaId})
    #     return {"MsgType": message.MsgType, "MediaId": message.MediaId}
    # elif message.MsgType == "video":
    #     print({"MsgType": message.MsgType, "MediaId": message.MediaId})
    #     return {"MsgType": message.MsgType, "MediaId": message.MediaId}
    # else:
    #     return "Invalid MsgType"

class Item(BaseModel):
    name: str
    description: Union[str, None] = None
    price: float
    tax: Union[float, None] = None


@app.post("/items/")
async def create_item(request:Request,item: Item ):
    print(item, request.url)
    return  {"item": item, "body": request.url}

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8080,
    )
