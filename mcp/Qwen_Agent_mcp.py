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
    "enable_thinking": False,

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
}]

if __name__ == '__main__':
    agent = Assistant(
        llm=llm_cfg,
        function_list=tools
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