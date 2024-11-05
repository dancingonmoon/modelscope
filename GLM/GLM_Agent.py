from GLM_callFunc import config_read
from zhipuai import ZhipuAI
from pathlib import Path
# from IPython.display import display, Markdown

def glm_agentAPI(
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
    config_path_zhipuai = r"l:/Python_WorkSpace/config/zhipuai_SDK.ini"

    zhipu_apiKey = config_read(
        config_path_zhipuai, section="zhipuai_SDK_API", option1="api_key"
    )
    zhipuai_client = ZhipuAI(api_key=zhipu_apiKey)
    # # 查看全部的Agent List
    # response = zhipuai_client.assistant.query_support(assistant_id_list=[])
    # # print(response)
    # for assistant in response.data:
    #     print(assistant.assistant_id, assistant.name, assistant.description)

    ## 65940acff94777010aa6b796 ChatGLM 嗨~ 我是清言，超开心遇见你！😺 你最近有什么好玩的事情想和我分享吗？ 🌟💬
    ## 659bbf72e76e36c506c1fc5a 诗画通 专业赏析、生动配图，理解古诗词如此简单
    ## 659ce0a2c7ce7c8e8e0fc9db 绘本精灵 只需提供一个主题，为你生成独家故事绘本。
    ## 659d051a5f14eb8ce1235b96 自媒体多面手 你有创意金矿，我是精准挖掘机，为你生成一键通发多平台内容。
    ## 659e54b1b8006379b4b2abd6 AI搜索 连接全网内容，精准搜索，快速分析并总结的智能助手。
    ## 659e74b758eb26bcf4b2ab18 无限流续写 跌宕起伏无限反转，和 AI 共创一部互动小说吧。
    ## 65a265419d72d299a9230616 数据分析 通过分析用户上传文件或数据说明，帮助用户分析数据并提供图表化的能力。也可通过简单的编码完成文件处理的工作。
    ## 65a393b3619c6f13586246cd 程序员助手Sam 我有一个外号叫“编程开发知识搜索引擎”，我很开心能帮助程序员解决日常问题❤️
    ## 65b356af6924a59d52832e54 网文写手 大神写作秘诀：一套模板不断重复。
    ## 65b8a4e975c8530c0656fe60 角色生成器 创造独特角色，激发无限故事可能！
    ## 65bf5a99396389a73ace9352 AiResearch 基于AMiner论文、学者、科研项目等学术资源，提供包括学者信息查询、论文检索总结、研究现状调研、学术网络发现等科研问答。
    ## 65d2f07bb2c10188f885bd89 PPT助手 超实用的PPT生成器，支持手动编辑大纲、自动填充章节内容，更有多个模板一键替换
    ## 663058948bb259b7e8a22730 arXiv论文速读/精析（计算机） 深度解析arXiv论文，让你快速掌握研究动态，节省宝贵时间。
    ## 66437ef3d920bdc5c60f338e AI画手 原AI画图【pro】，AI画图新功能已上线，欢迎搜索AI画图使用。
    ## 664dd7bd5bb3a13ba0f81668 流程图小助手 人人都能掌握的流程图工具，分分钟做出一张清晰的流程图。
    ## 664e0cade018d633146de0d2 思维导图 MindMap 告别整理烦恼，任何复杂概念秒变脑图。
    ## 665473b0a786a901387ca295 小红书文案写手 小红书这些文案都是谁写的啊啊啊！！哦，是我～
    ## 6654898292788e88ce9e7f4c 提示词工程师 人人都是提示词工程师，超强清言结构化提示词专家，一键改写提示词。
    ## 668fdd45405f2e3c9f71f832 英文单词语法助手 输入单词，进行单词查询；输入句子，进行语法检查；输入讲解，进行语法解释。

    # # Agent : AI 搜索
    # assistant_id = "659e54b1b8006379b4b2abd6"
    # prompt = "请提供杭州未来5天的天气, 并绘制柱状图"
    # generate = glm_agentAPI(assistant_id, conversation_id=None, prompt=prompt)
    #
    # # output = ""
    # for resp in generate:
    #     # print(resp)
    #     delta = resp.choices[0].delta
    #     # print(delta)
    #     # print(type(delta))
    #     if resp.status != 'completed':
    #         if delta.role == 'assistant':
    #             print(delta.content)
    #             # output += delta.content
    #             # print(output)
    #         if delta.role == 'tool':
    #             # print(resp)
    #             print(delta.tool_calls[0])
    #             if hasattr(delta.tool_calls[0], 'web_browser'):
    #                 if delta.tool_calls[0].web_browser.outputs:
    #                     print(delta.tool_calls[0].web_browser.outputs)
    #                 else:
    #                     print('in process of searching.......')
    #             if hasattr(delta.tool_calls[0],'code_interpreter'):
    #                 if delta.tool_calls[0].code_interpreter.outputs:
    #                     print(delta.tool_calls[0].code_interpreter.outputs)

    # Agent：文件（pdf,jpg)上传，结合prompt文本：
    pdf_path = r"E:/Working Documents/Eastcom/新业务/刘禹/BPV/test/Испытания пластины редакция сж.pdf"
    jpg_path = r"C:/Users/shoub/Pictures/融合通信架构.png"

    # 上传用于模型微调、知识库、Batch、文件抽取等功能所使用的文件。
    # 格式限制：.PDF .DOCX .DOC .XLS .XLSX .PPT .PPTX .PNG .JPG .JPEG .CSV .PY .TXT .MD .BMP .GIF
    # 文件大小不超过50M，图片大小不超过5M、总数限制为100个文件
    # file_object = zhipuai_client.files.create(
    #     file=open(pdf_path, "rb"),
    #     purpose="file-extract",  # 支持retrieval、batch、fine-tune、file-extract、code-interpreter
    # )
    # 或者直接文件路径
    file_object = zhipuai_client.files.create(file=Path(jpg_path), purpose="file-extract")

    # 文件内容抽取
    file_content = zhipuai_client.files.content(file_id=file_object.id).content.decode()
    print(file_content)

    # 请求文件列表
    result = zhipuai_client.files.list(
        purpose="file-extract",  # 支持batch、file-extract、fine-tune
    )
    print(result)

    # 生成请求消息
    message_content = f"请对\n{file_content}\n的内容进行分析，并按照原格式转换成Markdown格式，图片可以用Markdown语法中的链接表示。"

    response = zhipuai_client.chat.completions.create(
        model="glm-4-long",
        messages=[{"role": "user", "content": message_content}],
    )

    print(response.choices[0].message)

