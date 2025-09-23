from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langchain_qwq import ChatQwQ, ChatQwen
import asyncio

client = MultiServerMCPClient(
    {
        "Math": {
            "command": "python",
            # Replace with absolute path to your math_server.py file
            "args": ["E:/Python_WorkSpace/modelscope/mcp/mcpServer_math.py"],
            "transport": 'stdio',
        },
        "docx2txt": {
                    "command": "python",
                    # Replace with absolute path to your math_server.py file
                    "args": ["E:/Python_WorkSpace/modelscope/mcp/mcpServer_docx2txt.py"],
                    "transport": 'stdio',
                },
        "tempSave": {
            "command": "python",
            # Replace with absolute path to your math_server.py file
            "args": ["E:/Python_WorkSpace/modelscope/mcp/mcpServer_localSave.py"],
            "transport": 'stdio',
        },
    }
)

async def get_mcp_tool(client):
    # get_tools是一个协程对象，需要被await,需要异步函数
    # 并且StructuredTool does not support sync invocation
    tool = await client.get_tools()
    return tool

# get_tools是一个协程对象，需要被await,需要异步函数
# 并且StructuredTool does not support sync invocation
async def mcp_agent(prompt: str | dict):
    tools = await client.get_tools()
    llm = ChatQwQ(model='qwq-plus',
                  base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",  # 阿里云国内站点(默认为国际站点),
                  )
    agent = create_react_agent(
        model=llm,
        tools=tools,
    )
    # StructuredTool does not support sync invocation,所以需要为异步调用
    return await agent.ainvoke(prompt)


if __name__ == "__main__":
    # prompt = "27的3次方根是多少?"
    prompt = "请总结这篇word文档的内容：E:\Working Documents\Eastcom\新业务\刘禹\俄罗斯公司.docx;然后将总结结果存入本地文件，并给出本地存储文件的路径"
    state = {"messages": [{"role": "user", "content":  prompt}]}
    # 两种方法皆可，推荐方法一
    # # 方法一： 调用mcp_agent：
    response = asyncio.run(mcp_agent(state))
    # # 方法二： 调用get_mcp_tool:
    # llm = ChatQwQ(model='qwq-plus',
    #               base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",  # 阿里云国内站点(默认为国际站点),
    #               )
    # agent = create_react_agent(
    #     model=llm,
    #     tools=asyncio.run(get_mcp_tool(client)),
    # )
    # response = asyncio.run(agent.ainvoke(state))

    # print(response)
    for msg in response["messages"]:
        print(f'Human/AI Message.content:{msg.content}')
        print("msg.additional_kwargs.items:")
        for key, item in msg.additional_kwargs.items():
            print(f'{key}:{item}')

        if hasattr(msg, 'tool_calls'):
            print("Tool Calls:")
            for call in msg.tool_calls:
                for key,item in call.items():
                    print(f"{key}:{item}")
        if hasattr(msg, 'ToolMessage'):
            for key,item in msg.ToolMessage.items():
                print(f"{key}:{item}")
