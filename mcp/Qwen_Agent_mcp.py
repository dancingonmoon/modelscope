import os
from qwen_agent.agents import Assistant
from qwen_agent.gui import WebUI
from qwen_agent.utils.output_beautify import typewriter_print
# `typewriter_print` prints streaming messages in a non-overlapping manner.

llm_cfg = {
    'model': 'qwen-turbo-latest',   # 输入0.0003元;思考模式0.006元;非思考模式0.0006元
    # 'model': 'qwq-plus-latest',   # 输入0.0016元;   输出0.004元
    # 'model': 'qwen-max-latest',   # 输入0.0024元;   输出0.0096元
    # 'model': 'qwen-plus-latest',  # 输入0.0008元;思考模式0.016元;非思考模式0.002元
    'model_server': 'dashscope',
    # 'api_key': ''  # **fill your api key here**
    'generate_cfg': {
            # When using the Dash Scope API, pass the parameter of whether to enable thinking mode in this way
            'enable_thinking': False,
            'enable_search': True, # 开启联网搜索的参数
            'search_options': {
                "forced_search": True, # 强制开启联网搜索
                "enable_source": True, # 使返回结果包含搜索来源的信息，OpenAI 兼容方式暂不支持返回
                "enable_citation": True, # 开启角标标注功能
                "citation_format": "[ref_<number>]", # 角标形式为[ref_i]
                "search_strategy": "pro" # 模型将搜索10条互联网信息
            },

        },

    # Use a model service compatible with the OpenAI API, such as vLLM or Ollama:
    # 'model': 'Qwen3-8B',
    # 'model_server': 'http://localhost:8000/v1',  # base_url, also known as api_base
    # 'api_key': 'EMPTY'
}

tools = [{
    "mcpServers": {
        "filesystem": {
            "command": "npx",
            "args": [
                "-y",
                "@modelcontextprotocol/server-filesystem",
                '.',
            ]
        },
        "mindmap": {
            "command": "uvx",
            "args": ["mindmap-mcp-server", "--return-type", "filePath"]
        },
        "time": {
            "command": "uvx",
            "args": ["mcp-server-time","--local-timezone=Asia/Shanghai"]
                }
                }
        },
    # 'code_interpreter',  # Built-in tools
]

if __name__ == '__main__':
    agent = Assistant(
        llm=llm_cfg,
        function_list=tools,
        name='my Assistant',
        # system_message="按照用户需求，你先画图，再运行代码...."
        # description='使用RAG检索并回答，支持文件类型：PDF/Word/PPT/TXT/HTML。'
        # files = [os.path.join('.', 'doc.pdf')]
    )
    WebUI(agent).run()

    # messages = []  # 这里储存聊天历史。
    # while True:
    #     # 例如，输入请求 "绘制一只狗并将其旋转 90 度"。
    #     query = input('\n用户请求: ')
    #     # 将用户请求添加到聊天历史。
    #     messages.append({'role': 'user', 'content': query})
    #     response = []
    #     response_plain_text = ''
    #     print('机器人回应:')
    #     for response in agent.run(messages=messages):
    #         # 流式输出。
    #         response_plain_text = typewriter_print(response, response_plain_text)
    #     # 将机器人的回应添加到聊天历史。
    #     messages.extend(response)