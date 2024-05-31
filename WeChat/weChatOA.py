from typing import Union, Annotated
from fastapi import FastAPI, Path
from pydantic import BaseModel
import datetime
import hashlib

description = """
## 微信公众号开发者服务器.🦬
 + **get 开发者服务器token验证**
 + **get **
"""
app = FastAPI( title="微信公众号开发者服务器", description=description,)

class Item(BaseModel):
    ToUserName: str 	# 开发者微信号
    FromUserName: str	# 发送方账号（一个OpenID）
    CreateTime: int	# 消息创建时间 （整型）
    MsgType: str	# 消息类型，文本为text
    Content: str	# 文本消息内容
    MsgId: int	# 消息id，64位整型
    MsgDataId: int	# 消息的数据ID（消息如果来自文章时才有）
    Idx: int	# 多图文时第几篇文章，从1开始（消息如果来自文章时才有）
@app.get("/wx")
async def token_validation(
        signature:str,
        timestamp:int,
        nonce:str,
        echostr:str
):
    try:
        # if len(data) == 0:
        #     return "hello, this is handle view"
        token = "lockup" #请按照公众平台官网\基本配置中信息填写

        list = [token, timestamp, nonce]
        list.sort()
        sha1 = hashlib.sha1()
        map(sha1.update, list)
        hashcode = sha1.hexdigest()
        print(f"handle/GET func: hashcode:{hashcode}, signature:{signature}")
        if hashcode == signature:
            return echostr
        else:
            return ""
    except Exception, Argument:
        return Argument



@app.put("/items/{item_id}")
async def update_item(
    item_id: Annotated[int, Path(title="The ID of the item to get", ge=0, le=1000)],
    q: Union[str, None] = None,
    item: Union[Item, None] = None,
):
    results = {"item_id": item_id}
    if q:
        results.update({"q": q})
    if item:
        results.update({"item": item})
    return results

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8080, )