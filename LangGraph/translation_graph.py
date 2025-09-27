from LangGraph_warehouse import translation_graph, State

"""
## 创建完整LangGraph智能体项目流程
- Step 1. 创建一个LangGraph项目文件夹
- Step 2. 创建requirements.txt文件
- Step 3. 注册LangSmith（可选）
- Step 4. 创建.env配置文件
- Step 5. 创建graph.py核心文件
- Step 6. 创建langgraph.json文件
- Step 7. 安装langgraph-cli以及其他依赖
- Step 8. 进入项目目录，执行: `LangGraph dev` 即可启动项目
"""

#  在使用LangGraph CLI创建智能体项目时，会自动设置记忆相关内容，并进行持久化记忆存储，无需手动设置，不需要手动设置checkpointer
translation_agent = translation_graph(State=State, name="translation_graph")

