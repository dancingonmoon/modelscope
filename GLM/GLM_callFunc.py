import time
from zhipuai import ZhipuAI
import configparser


# import sys
# sys.path.append("../WeChat/")  # 将上一级目录的/GLM目录添加到系统路径中


def config_read(config_path, section="weChatOA", option1="AppID", option2=None):
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


def GLM_callFunc_Async(zhipuai_client, question, query, LLM_model="glm-4-air", web_search_enable=True, web_search_result_show=False,
                       time_threshold=5):
    """
    实现GLM异步调用,开启web_search工具;异步检索模型输出,在time_threshold时间(微信服务器限定响应时间为5秒), 或输出模型结果,或给微信服务器输出success响应;
    zhipuai_client: 初始化之后的ZhipuAI客户端
    LLM_model: glm-4-0520, glm-4 , glm-4-air, glm-4-airx,  glm-4-flash, 或glm-3-turbo
    web_search_enable: 缺省打开
    web_search_result_show: 是否输出web_search结果,缺省关闭; web_search结果为列表,包含每个网页链接字典,其keys: {content, icon,link,media,refer};
    time_threshold: time_threshold内,持续检索模型异步输出,直到获得模型输出;达到time_threshold,仍没有获得模型输出,则给微信服务器输出success响应;
    return:
        out_text: 模型输出
        out_search: web_search结果,web_search_result_show=False时,为空列表;
    """
    start_time = time.time()
    client = zhipuai_client
    tools = [{
        'type': 'web_search',
        'web_search': {
            'enable': web_search_enable,
            'search_query': query,
            'search_result': web_search_result_show
        }}]
    response = client.chat.asyncCompletions.create(
        model=LLM_model,  # 填写需要调用的模型名称
        messages=[
            {
                "role": "user",
                "content": question,
            }
        ],
        tools=tools,
    )
    task_id = response.id

    while True:
        if time.time() - start_time < time_threshold:
            result_response = client.chat.asyncCompletions.retrieve_completion_result(id=task_id)
            if 'choices' in result_response.dict():
                # print(f"finish_reason: {result_response.choices[0].finish_reason}")
                # print(f"content: {result_response.choices[0].message.content}")
                # print(f"response: {result_response.web_search}")
                out_text = result_response.choices[0].message.content
                if web_search_result_show:
                    out_search = result_response.web_search
                else:
                    out_search = []

                return out_text, out_search
        else:  # 超时未获得模型暑促,返回
            # print(f"服务器未能五秒内返回，为避免微信服务器发起重试, 直接回复success")
            out_text = "success"
            return out_text

def GLM_callFunc_SSE(zhipuai_client, question, query, LLM_model="glm-4-air", web_search_enable=True, web_search_result_show=False,
                       time_threshold=5):
    """
    实现GLM流式调用,开启web_search工具;在time_threshold时间(微信服务器限定响应时间为5秒), 输出模型SSE所能获得的全部输出;
    zhipuai_client: 初始化之后的ZhipuAI客户端
    LLM_model: glm-4-0520, glm-4 , glm-4-air, glm-4-airx,  glm-4-flash, 或glm-3-turbo
    web_search_enable: 缺省打开
    web_search_result_show: 是否输出web_search结果,缺省关闭; web_search结果为列表,包含每个网页链接字典,其keys: {content, icon,link,media,refer};
    time_threshold: time_threshold内,获得模型SSE输出;达到time_threshold后面的SSE输出放弃;
    return:
        out_text: 模型输出
        out_search: web_search结果,web_search_result_show=False时, 为空列表;
    """
    start_time = time.time()
    client = zhipuai_client
    tools = [{
        'type': 'web_search',
        'web_search': {
            'enable': web_search_enable,
            'search_query': query,
            'search_result': web_search_result_show
        }}]
    messages = [
        {
            "role": "user",
            "content": question,
        }
    ]
    response = client.chat.completions.create(
        model= LLM_model,  # 填写需要调用的模型名称
        messages=messages,
        tools=tools,
        tool_choice="auto",
        stream=True,
    )
    out_text = ""
    out_search = []
    for chunk in response:
        chunk_out = chunk.choices[0].delta.content
        out_text = "".join([out_text, chunk_out])
        if chunk.choices[0].finish_reason == 'stop' and web_search_result_show:
            if chunk.web_search is not None:
                out_search = chunk.web_search
        if time.time() - start_time >= time_threshold * 0.89: # 经测试,超过0.90会导致微信, 重发两次响应成功; 超过0.93 对导致微信响应时间5S超时,而不予响应.
            out_text = "".join([out_text, '....\n抱歉! 受微信接口最多5秒响应限制,此处截断....'])
            return out_text, out_search

    return out_text, out_search

if __name__ == "__main__":
    config_path_serp = r"l:/Python_WorkSpace/config/SerpAPI.ini"
    config_path_zhipuai = r"l:/Python_WorkSpace/config/zhipuai_SDK.ini"

    zhipu_apiKey = config_read(config_path_zhipuai, section="zhipuai_SDK_API", option1="api_key")
    zhipuai_client = ZhipuAI(api_key=zhipu_apiKey)

    question = "美国大选最新状况,拜登,川普谁是赢家?"
    query = "美国大选最新选情"
    web_search_enable = False
    web_search_result_show = False
    time_threshold=50


    start_time = time.time()

    out = GLM_callFunc_Async(zhipuai_client, question, query, web_search_enable=web_search_enable,
                web_search_result_show=web_search_result_show, time_threshold=time_threshold)
    if isinstance(out, str):
        print(out)
    else:
        print(out[0])
        print(out[1])
    print(f"模型异步输出耗时:{time.time()-start_time}")

    start_time = time.time()
    out = GLM_callFunc_SSE(zhipuai_client, question, query, web_search_enable=web_search_enable,
                             web_search_result_show=web_search_result_show, time_threshold=time_threshold)

    print(out[0])
    print(out[1])
    print(f"模型SSE输出耗时:{time.time() - start_time}")






