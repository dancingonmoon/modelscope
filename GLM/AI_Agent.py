from GLM_callFunc import config_read
from zhipuai import ZhipuAI
# import zhipuai


def zhipu_agent(
    assistant_id, conversation_id=None, prompt=None, attachment=None, metadata=None
):
    """
    https://open.bigmodel.cn/dev/api/qingyanassistant/assistantapi
    conversation_id: 会话 ID，不传默认创建新会话。需要继续之前的会话时，传入流式输出中的 conversation_id 值。
    attachment: List<Object>
    """
    generate = zhipuai_client.assistant.conversation(
        assistant_id=assistant_id,
        conversation_id=conversation_id,
        model="glm-4-assistant",
        messages=[{"role": "user", "content": [{"type": "text", "text": prompt}]}],
        stream=True,
        attachments=attachment,
        metadata=metadata,
    )
    return generate


if __name__ == "__main__":
    config_path_serp = r"l:/Python_WorkSpace/config/SerpAPI.ini"
    config_path_zhipuai = r"l:/Python_WorkSpace/config/zhipuai_SDK.ini"

    zhipu_apiKey = config_read(
        config_path_zhipuai, section="zhipuai_SDK_API", option1="api_key"
    )
    zhipuai_client = ZhipuAI(api_key=zhipu_apiKey)
    # 查看全部的Agent List
    # response = zhipuai_client.assistant.query_support(assistant_id_list=[])
    # print(response)
    # for assistant in response.data:
    #     print(assistant.assistant_id, assistant.name, assistant.description)

    # AI 搜索：
    assistant_id = "659e54b1b8006379b4b2abd6"
    prompt = "请提供杭州未来5天的天气, 并绘制柱状图"
    generate = zhipu_agent(assistant_id, conversation_id=None, prompt=prompt)

    # output = ""
    for resp in generate:
        # print(resp)
        delta = resp.choices[0].delta
        # print(delta)
        # print(type(delta))
        if resp.status != 'completed':
            if delta.role == 'assistant':
                print(delta.content)
                # output += delta.content
                # print(output)
            if delta.role == 'tool':
                # print(resp)
                print(delta.tool_calls[0])
                if hasattr(delta.tool_calls[0], 'web_browser'):
                    if delta.tool_calls[0].web_browser.outputs:
                        print(delta.tool_calls[0].web_browser.outputs)
                    else:
                        print('in process of searching.......')
                if hasattr(delta.tool_calls[0],'code_interpreter'):
                    if delta.tool_calls[0].code_interpreter.outputs:
                        print(delta.tool_calls[0].code_interpreter.outputs)

