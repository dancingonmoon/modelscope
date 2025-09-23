from mcp.server import FastMCP
from langchain_community.document_loaders import Docx2txtLoader

# 初始化MCP服务器
mcp = FastMCP('docx2txt')


# 定义函数：
@mcp.tool()
async def docx2txtLoader(file_path: str | list[str] = None, ):
    """
    输入本地文件路径，或者url的word文件，读取内容并转换成txt格式
    https://python.langchain.com/docs/integrations/document_loaders/microsoft_word/
    :param file_path: str;url;Path
    :return: txt文本
    """
    loader = Docx2txtLoader(file_path=file_path, )
    docs = []
    async for doc in loader.alazy_load():
        docs.append(doc)
    return docs

if __name__ == '__main__':
    mcp.run(transport='stdio')