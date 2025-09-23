from mcp.server import FastMCP
import math

# 初始化MCP服务器
mcp = FastMCP('Math')


# 定义函数：
@mcp.tool()
def pow_root(a: float, b: float):
    """
    求a的b次方根
    :param a:
    :param b:
    :return:
    """
    return math.pow(a, 1 / b)


@mcp.tool()
def ln(a: float):
    """
    求ln(a)
    :param a:
    :return:
    """
    return math.log(a)


if __name__ == '__main__':
    mcp.run(transport='stdio')
