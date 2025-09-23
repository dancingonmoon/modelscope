from mcp.server import FastMCP
from typing import Any
import tempfile
import pathlib

# 初始化MCP服务器
mcp = FastMCP('tempSave')


# 定义函数：
@mcp.tool()
def local_tempSave(txt_content: Any, suffix: str = '.txt', temp_dir: str | pathlib.Path = None):
    """
    将一个对象，譬如模型的文本输出，存入一个临时文件，缺省为txt格式，也可以是Markdown格式，输出临时文件的路径
    :param txt_content
    :param suffix: 临时存储文件的后缀，缺省为.txt
    :param temp_dir: 临时存储文件的临时目录;当为None时,函数会自动分配一个临时目录
    :return: 临时存储temp_saved_file对象的路径，(通过temp_saved_file.name获得)
    """
    if temp_dir is None:
        temp_dir = tempfile.TemporaryDirectory(dir='.', prefix='temp_', delete=False)
        temp_dir_path = temp_dir.name
    else:
        temp_dir_path = temp_dir
    temp_saved_file = tempfile.NamedTemporaryFile(mode='w', delete=False, prefix="mcp_output", suffix=suffix,
                                                  dir=temp_dir_path)
    temp_saved_file.write(txt_content)
    temp_saved_file_path = temp_saved_file.name
    return temp_saved_file.name


if __name__ == '__main__':
    mcp.run(transport='stdio')
