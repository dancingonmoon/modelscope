from AgenticGraphRAG_functions import iterate_json_loader, load_KnowledgeExtraction_KnowledgeGraph_from_json
from AgenticGraphRAG_functions import vector_knowledge_search, graph_knowledge_search
import chromadb
from langchain_chroma import Chroma
from langchain.tools import tool
from langchain.agents import create_agent
from langchain.messages import HumanMessage
from langchain_deepseek import ChatDeepSeek
from pathlib import Path

from langchain_community.embeddings import ZhipuAIEmbeddings

Zhipu_embedding = ZhipuAIEmbeddings(model="embedding-3", )

# 以下在langgraph studio中使用时，才需要内存中预先启动：
# 向量数据库搜索(vector_knowledge_search)，需要显性的在LLM tool中，赋值向量数据库vectorstore
json_path = Path(
        r"E:/Python_WorkSpace/modelscope/Agentic GraphRAG/data/lx_output/阿联酋投资问与答_LangExtract.json")
chromadb_path = r"E:/Python_WorkSpace/modelscope/Agentic GraphRAG/data/UAE_investment_QA"
collection_name = "UAE_investment_QA"
doc_title = '阿联酋投资问与答'
json_data = iterate_json_loader(json_path)
_, KnowledgeGraph_ = load_KnowledgeExtraction_KnowledgeGraph_from_json(json_data,
                                                                       doc_title=doc_title)
# 从chromadb向量数据库中加载
chroma_client = chromadb.PersistentClient(path=chromadb_path)
# 从chromadb向量数据库中加载
vector_store = Chroma(
    collection_name=collection_name,
    client=chroma_client,
    create_collection_if_not_exists=False,
    embedding_function=Zhipu_embedding, )
# Graph搜索(graph_knowledge_search),需要显性的在LLM tool中，赋值KnowledgeGraph类型的数据库

# =============================================================================
# LangChain 1.1 GraphRAG Tools
# =============================================================================

# Tool 1: 向量检索
@tool
def vector_search_tool(query: str) -> str:
    """
    向量语义检索：根据问题搜索相关知识片段。
    返回语义相似的文档内容和溯源信息。
    适合：查找与问题语义相关的内容。
    """
    results = vector_knowledge_search(query, top_k=5, vectorstore=vector_store)  # vectorstore 显性赋值向量数据库

    if not results:
        return "未找到相关信息"

    output_parts = []
    for i, r in enumerate(results, 1):
        part = f"[V{i}] {r['extraction_class']}: {r['extraction_text']}"
        if r.get('char_interval'):
            interval = r['char_interval']
            part += f"\n     位置: 字符 {interval['start_pos']}-{interval['end_pos']}"
        part += f"\n     来源: {r['doc_title']}"
        if r.get('attributes'):
            part += f"\n     属性: {r['attributes']}"
        output_parts.append(part)

    return "\n\n".join(output_parts)


# Tool 2: 图谱检索
@tool
def graph_search_tool(entity: str) -> str:
    """
    知识图谱检索：根据实体名称查找相关实体和关系。
    用于发现实体之间的结构化关联。
    适合：查找某个实体相关的关系、关联实体。
    """
    results = graph_knowledge_search(entity, hop=1, knowledge_graph=KnowledgeGraph_)  # knowledge_graph 显性赋值知识图谱数据库

    # 格式化输出
    output_parts = []

    if results["matched_entities"]:
        output_parts.append("【匹配的实体】")
        for e in results["matched_entities"][:5]:
            mentions = e.get("mentions", [])
            output_parts.append(f"  - {e['name']} (类型: {e.get('type', '未知')}, 提及次数: {len(mentions)})")

    if results["related_relations"]:
        output_parts.append("\n【相关关系】")
        for i, rel in enumerate(results["related_relations"][:5], 1):
            subject = rel.get("subject", "?")
            relation = rel.get("relation", rel.get("type", "相关"))
            obj = rel.get("object", "?")
            output_parts.append(f"  [G{i}] {subject} --[{relation}]--> {obj}")
            if rel.get("text"):
                output_parts.append(f"       原文: {rel['text'][:50]}...")

    if results["connected_entities"]:
        output_parts.append(f"\n【关联实体】: {', '.join(results['connected_entities'][:10])}")

    if not output_parts:
        return f"未找到与 '{entity}' 相关的图谱信息"

    return "\n".join(output_parts)


# Tool 3: 混合检索
@tool
def hybrid_search_tool(query: str) -> str:
    """
    混合检索（GraphRAG）：同时进行向量语义检索和知识图谱检索。
    适合：复杂问题，需要结合语义相似和结构化关系。
    """
    # 向量检索
    vector_result = vector_search_tool.invoke(query)

    # 从问题中提取可能的实体进行图检索
    words = [w for w in query.replace("？", "").replace("?", "").replace("，", " ").replace("。", " ").split() if
             len(w) >= 2]

    graph_results = []
    for word in words[:3]:
        gr = graph_search_tool.invoke(word)
        if "未找到" not in gr:
            graph_results.append(gr)

    # 组合结果
    output = "=== 向量检索结果 ===\n" + vector_result

    if graph_results:
        output += "\n\n=== 图谱检索结果 ===\n" + "\n".join(graph_results)

    return output


# =============================================================================
# 创建 LangChain 1.1 GraphRAG Agent
# =============================================================================


# 创建 DeepSeek 模型实例
llm = ChatDeepSeek(
    model='deepseek-chat',
    temperature=0.3
)

# 创建 Agent
graphrag_agent = create_agent(
    model=llm,
    tools=[vector_search_tool, graph_search_tool, hybrid_search_tool],
    system_prompt="""你是一个 GraphRAG 知识图谱问答助手。

你有以下工具可用：
1. vector_search_tool - 向量语义检索，找语义相似的内容
2. graph_search_tool - 图谱检索，根据实体名找关系
3. hybrid_search_tool - 混合检索，同时使用向量和图谱

回答策略：
- 简单的内容查询：用 vector_search_tool
- 查找实体关系：用 graph_search_tool  
- 复杂问题：用 hybrid_search_tool

回答要求：
1. 综合检索到的信息回答问题
2. 标注信息来源（如 [V1] 表示向量结果，[G1] 表示图谱关系）
3. 如果有溯源位置信息，也一并说明
"""
)


# =============================================================================
# 封装 agent_query 函数
# =============================================================================

def agent_query(question: str, top_k: int = 5):
    """
    GraphRAG Agent 问答

    Args:
        question: 用户问题
        top_k: 检索数量

    Returns:
        包含 question, answer, evidence 的字典
    """
    # 调用 Agent
    result = graphrag_agent.invoke({
        "messages": [HumanMessage(content=question)]
    })

    # 提取最终回答
    answer = result["messages"][-1].content

    # 提取工具调用记录作为 evidence
    tool_calls = []
    for msg in result["messages"]:
        # 工具调用请求
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                tool_calls.append({
                    "type": "call",
                    "tool": tc.get("name", "unknown"),
                    "args": tc.get("args", {})
                })
        # 工具返回结果
        if hasattr(msg, "name") and msg.name:
            tool_calls.append({
                "type": "result",
                "tool": msg.name,
                "content": msg.content[:500] + "..." if len(msg.content) > 500 else msg.content
            })

    return {
        "question": question,
        "answer": answer,
        "evidence": tool_calls
    }


if __name__ == "__main__":
    json_path = Path(
        r"E:/Python_WorkSpace/modelscope/Agentic GraphRAG/data/lx_output/阿联酋投资问与答_LangExtract.json")
    chromadb_path = r"E:/Python_WorkSpace/modelscope/Agentic GraphRAG/data/UAE_investment_QA"
    collection_name = "UAE_investment_QA"
    doc_title = '阿联酋投资问与答'
    json_data = iterate_json_loader(json_path)
    _, KnowledgeGraph_ = load_KnowledgeExtraction_KnowledgeGraph_from_json(json_data,
                                                                           doc_title=doc_title)
    # 从chromadb向量数据库中加载
    chroma_client = chromadb.PersistentClient(path=chromadb_path)
    # 从chromadb向量数据库中加载
    vector_store = Chroma(
        collection_name=collection_name,
        client=chroma_client,
        create_collection_if_not_exists=False,
        embedding_function=Zhipu_embedding, )

    # 测试Agent
    question = "投资阿联酋都有哪些优惠政策？"

    result = agent_query(question, top_k=3)
    print(f"\n回答:\n{result['answer']}\n")

