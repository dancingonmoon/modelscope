from dataclasses import dataclass, field, asdict
from pydantic import BaseModel, Field
from typing import Optional, Iterator
import json
import uuid
from datetime import datetime
from pathlib import Path
import chromadb
from langchain_chroma import Chroma
import langchain_community
from langchain_community.embeddings import ZhipuAIEmbeddings


Zhipu_embedding = ZhipuAIEmbeddings(model="embedding-3", )


# 定义知识提取结果的数据结构
@dataclass
class KnowledgeExtraction:
    """知识提取结果（带溯源）"""
    doc_id: str  # 文档 ID
    doc_title: str  # 文档标题
    extraction_class: str  # 提取类型（实体、关系、事件等）
    extraction_text: str  # 提取的原文文本
    char_interval: Optional[dict] = None  # 原文字符区间（溯源的核心）
    attributes: dict = field(default_factory=dict)  # 属性信息

    def to_dict(self) -> dict:
        """转换为字典格式"""
        return asdict(self)

    def to_searchable_text(self) -> str:
        """生成用于向量化的可搜索文本"""
        parts = [
            f"类型: {self.extraction_class}",
            f"内容: {self.extraction_text}",
            f"来源: {self.doc_title}"
        ]
        if self.attributes:
            for k, v in self.attributes.items():
                parts.append(f"{k}: {v}")
        return " | ".join(parts)


# 构建知识图谱
class relation(BaseModel):
    text: str = Field(description='extraction_text')
    type: str = Field(default="未知", title='关系类型',
                      description="‘关系描述’下‘属性’下‘类型’,attrs.get('类型', '未知')")
    subject1: str = Field(title='关系主体1', description="attrs.get('主体1’)")
    subject2: str = Field(title='关系主体2', description="attrs.get('主体2’)")
    subject3: str = Field(title='关系主体3', description="attrs.get('主体3’)")
    relation: str = Field(title='关系描述', description="’关系描述‘下’属性‘下’关系‘，,attrs.get('关系’)")
    source: str = Field(title='doc_title')


class mention(BaseModel):
    source: str = Field(title='doc_title')
    position: dict = Field(title='char_interval')  # 溯源位置 {"start_pos": 442, "end_pos": 493}


class entity(BaseModel):
    type: str = Field(title='实体类别', description="实体类别,为'实体','数据指标':extraction.class")
    attributes: dict = Field(default_factory=dict, title='实体属性',
                             description="可能包括不同的属性，例如类型，类别，用途，等等:attributes")
    mentions: list[mention]


class KnowledgeGraph(BaseModel):
    entities: dict[str, entity]
    relations: list[relation]


def iterate_json_loader(json_path: Path | str | dict):
    """
    从本地json文件中，依据每行生成json迭代器;
    :param json_path:
    :return:
    """
    if isinstance(json_path, str):
        json_path = Path(json_path)
    if json_path.exists() and json_path.is_file():
        with open(json_path, 'r', encoding='utf-8') as file:
            for line in file:
                yield json.loads(line)
    else:
        raise ValueError("json path is invalid or not existed")


# 从 JSON 文件加载数据并构建 KnowledgeGraph
def load_KnowledgeExtraction_KnowledgeGraph_from_json(json_data: dict | Iterator, doc_title: str = "",
                                                      save_KnowledgeGraph_path: Path | str | None = None) -> (
        list[KnowledgeExtraction], KnowledgeGraph):
    """
    将LangExtract提取的json数据，以KnowledgeExtraction数据类型读出，同时转换陈knowledge_graph的类型, 并同时输出元组，包含KnowlegeExtration列表，以及KnowledgeGraph类型数据
    :param json_data: json数据，可能包括原文档的多页
    :param doc_title: 原文档名
    :return: knowledge_graph数据类型
    """
    # # 初始化KnowledgeExtraction列表：
    KnowledgeExtraction_json = []
    # 初始化knowledge_graph实例：
    KnowledgeGraph_json = KnowledgeGraph(entities={}, relations=[])
    # 遍历 extractions 并分类处理
    if isinstance(json_data, Iterator):
        for data in json_data:
            doc_id = data.get("document_id", "")
            recorded_doc_title = f"{doc_title}_page_{doc_id}"
            extractions = data.get("extractions", [])
            for item in extractions:
                # 转化KnowledgeExtraction数据类型：
                extraction = KnowledgeExtraction(
                    doc_id=doc_id,
                    doc_title=recorded_doc_title,
                    extraction_class=item["extraction_class"],
                    extraction_text=item["extraction_text"],
                    char_interval=item.get("char_interval"),
                    attributes=item.get("attributes", {})
                )
                KnowledgeExtraction_json.append(extraction)
                # 转换knowledge_graph数据类型：
                if extraction.extraction_class == "实体":
                    # 构建 entity 对象
                    entity_name = extraction.extraction_text
                    char_interval = extraction.char_interval
                    if entity_name not in KnowledgeGraph_json.entities:
                        KnowledgeGraph_json.entities[entity_name] = entity(
                            type=extraction.extraction_text,
                            attributes=extraction.attributes,
                            mentions=[]
                        )
                    KnowledgeGraph_json.entities[entity_name].mentions.append(
                        mention(
                            source=recorded_doc_title,
                            position=extraction.char_interval if char_interval else {}
                        )
                    )
                elif extraction.extraction_class == "关系描述":
                    # 构建 relations 对象
                    KnowledgeGraph_json.relations.append(
                        relation(
                            text=extraction.extraction_text,
                            type=extraction.attributes.get("类型", "未知"),
                            subject1=extraction.attributes.get("主体1", ""),
                            subject2=extraction.attributes.get("主体2", ""),
                            subject3=extraction.attributes.get("主体3", ""),
                            relation=extraction.attributes.get("关系", ""),
                            source=extraction.doc_id)
                    )
    # KnowledgeGraph Json本地文件存储：
    if save_KnowledgeGraph_path:
        if isinstance(save_KnowledgeGraph_path, str):
            save_KnowledgeGraph_path = Path(save_KnowledgeGraph_path)
        if Path(save_KnowledgeGraph_path).parent.exists():
            with open(save_KnowledgeGraph_path, 'w', encoding='utf-8') as file:
                json.dump(KnowledgeGraph_json.dict(), file, ensure_ascii=False, indent=4)
            print(f"KnowledgeGraph saved at {save_KnowledgeGraph_path} !")
    # 返回 KnowledgeExtraction_json,以及KnowledgeGraph_json
    return KnowledgeExtraction_json, KnowledgeGraph_json


class chromDB():
    def __init__(self, chromadb_path: str = None, ):
        """
        初始定义chroma client，继承chroma client的全部方法;
        :param chromadb_path: 缺省None，chromaDB存在与内存；否则永久存在于本地路径
        """
        # # Create a ChromaDB client
        if chromadb_path is None:
            self.chroma_client = chromadb.Client()
        else:
            self.chroma_client = chromadb.PersistentClient(path=chromadb_path)
        # # delete collection:
        # self.chroma_client.delete_collection(name="test1")

    def langchain_chroma_vectorstore(self, text: list[str],
                                     collection_name: str = None,
                                     collection_metadata: dict = None,
                                     vectorstore_ids: list = None,
                                     vectorstore_metadatas: list = None,
                                     embedding_function: langchain_community.embeddings = None,
                                     chunk_size: int = 64,
                                     ):
        """
        langchain_community.chroma方法，将text列表写入向量数据库.
        :param collection_name: 写入数据库的collection_name; 如果不存在，则新建该collection;
        :param collection_metadata: collection的metadata,类型为字典,以标记metadata.缺省值为：
                                    {"description": collection_name,
                                   "create": datetime.now().strftime("%Y-%m-%d %H:%M"),
                                   "embedding_function": {"model": embedding_function.model,
                                                          "dimensions": embedding_function.dimensions}}
        :param vectorstore_ids: langchain_chroma.Chroma方法生成向量数据库时，可定义的ids参数，与text的数组一一对应;缺省值为uuid.uuid4()随机数
        :param vectorstore_metadatas: langchain_chroma.Chrom方法生成向量数据库时，可定义的metadatas参数，类型为list[dict]，与text数字一一对应，缺省值为：
                                       {"title": i}
        :param embedding_function: langchain_community.embedings,用于指定向量数据库的embed模型
        :param chunk_size: default 64;不能的embedding_function,受其API约束，可以一次批处理的text列表长度有限制，这里将knowledge_extraction的长度约束到chunk_size大小；
        :return:
        """
        if collection_metadata is None:
            collection_metadata = {"description": collection_name,
                                   "create": datetime.now().strftime("%Y-%m-%d %H:%M"),
                                   "embedding_model": embedding_function.model,
                                   "embedding_dimensions": str(embedding_function.dimensions)}
        if vectorstore_ids is None:
            vectorstore_ids = []
            for i in range(len(text)):
                vectorstore_ids.append(str(uuid.uuid4()))  # 随机数的缺点是每次ids都不同，导致每次重复添加同一组数据；
                # vectorstore_ids.append(f"test{i}")
        if vectorstore_metadatas is None:
            vectorstore_metadatas = []
            for i in range(len(text)):
                vectorstore_metadatas.append({"title": str(i)})
        # # # langchain_chroma vector store 用法：
        vector_store = Chroma(
            collection_name=collection_name,
            embedding_function=embedding_function,
            collection_metadata=collection_metadata,
            client=self.chroma_client,
            create_collection_if_not_exists=True,
        )
        for i in range(0, len(text), chunk_size):
            txt = text[i:i + chunk_size]
            vectorstore_metadata = vectorstore_metadatas[i:i + chunk_size]
            vectorstore_id = vectorstore_ids[i:i + chunk_size]
            vector_store.add_texts(texts=txt,
                                   metadatas=vectorstore_metadata,
                                   ids=vectorstore_id)
        return vector_store

    def KnowledgeExtraction2ChromaDB(self, knowledge_extraction: list[KnowledgeExtraction] = None,
                                     collection_name: str = None,
                                     collection_metadata: dict = None,
                                     embedding_function: langchain_community.embeddings = None,
                                     chunk_size: int = 64,
                                     ):
        """
        从KnowledgeExtraction数据中，提取extraction_text,转换成searchable_text,生成对应的id,metadata,以写入向量数据库;id,metadata依照缺省值
        将KnowledgeExtraction数据类型，写入chromadb向量数据库
        :param knowledge_extraction: 包含KnowledgeExtraction类型的列表，待转换生成向量数据库
        :param collection_name: 写入数据库的collection_name; 如果不存在，则新建该collection;
        :param collection_metadata: collection的metadata,类型为字典,以标记metadata.缺省值为：
                                    {"description": collection_name,
                                    "create": datetime.now().strftime("%Y-%m-%d %H:%M")}
        :param embedding_function: langchain_community.embedings,用于指定向量数据库的embed模型
        :param chunk_size: default 64;不能的embedding_function,受其API约束，可以一次批处理的text列表长度有限制，这里将knowledge_extraction的长度约束到chunk_size大小；
        :return
        """
        ids = []
        searchable_txts = []
        metadatas = []
        for id, extract in enumerate(knowledge_extraction):
            searchable_txt = extract.to_searchable_text()
            metadata = {"doc_title": extract.doc_title,
                        "doc_id": extract.doc_id,
                        "extraction_class": extract.extraction_class,
                        "extraction_text": extract.extraction_text,
                        "char_interval": json.dumps(extract.char_interval) if extract.char_interval else "",  # 溯源关键
                        "attributes": json.dumps(extract.attributes, ensure_ascii=False), }
            searchable_txts.append(searchable_txt)
            ids.append(str(id))  # chromadb要求ids为str类型
            metadatas.append(metadata)

        vector_store = self.langchain_chroma_vectorstore(text=searchable_txts,
                                                         collection_name=collection_name,
                                                         collection_metadata=collection_metadata,
                                                         vectorstore_ids=ids,
                                                         vectorstore_metadatas=metadatas,
                                                         embedding_function=embedding_function,
                                                         chunk_size=chunk_size
                                                         )
        return vector_store

    def load_vectorstore(self, collection_name: str = None,
                         embedding_function: langchain_community.embeddings = None,

                         ):
        """
        使用langchain_chroma.Chroma方法，从chroma已存向量数据库中，加载collection_name的collection
        :param collection_name:
        :return:
        """
        vector_store = Chroma(
            collection_name=collection_name,
            client=self.chroma_client,
            create_collection_if_not_exists=False,
            embedding_function=embedding_function,
        )
        return vector_store


# 定义向量检索函数（含溯源信息）
def vector_knowledge_search(query: str, top_k: int = 5,
                            vectorstore: Chroma = None):
    """
    从向量库检索相关知识，返回含溯源信息的结果;
        依据KnowledgeExtraction数据格式，经过chroma定义的向量模型计算，生成了全部text数据的向量数据库(vectorstore),从向量数据库中寻找top_k个similarity,写入一个json格式的result,json定义的格式如下的return返回所示：
    :param vectorstore: langchain_chroma.Chroma装载的向量数据库，已经指定了embedding_function.如果时本地存储读出的collection,避免未指定与原collection一致的embedding_function,而采用了缺省embedding
    :return 返回json数据,定义如下:
        {   "score": similarity_score,
            "doc_title": doc.metadata.get("doc_title"),
            "extraction_class": doc.metadata.get("extraction_class"),
            "extraction_text": doc.metadata.get("extraction_text"),
            "char_interval": char_interval,  # 溯源关键
            "attributes": json.loads(doc.metadata.get("attributes", "{}")),
        }
    """

    # 执行相似度搜索
    results = vectorstore.similarity_search_with_score(query, k=top_k)

    # 格式化结果（包含溯源信息）
    formatted_results = []
    for doc, score in results:
        similarity_score = 1 / (1 + score)  # 转换为相似度

        # 解析溯源信息
        char_interval_str = doc.metadata.get("char_interval", "")
        char_interval = json.loads(char_interval_str) if char_interval_str else None

        formatted_results.append({
            "score": similarity_score,
            "doc_title": doc.metadata.get("doc_title"),
            "extraction_class": doc.metadata.get("extraction_class"),
            "extraction_text": doc.metadata.get("extraction_text"),
            "char_interval": char_interval,  # 溯源关键
            "attributes": json.loads(doc.metadata.get("attributes", "{}")),
        })

    return formatted_results


# =============================================================================
# 图检索函数（GraphRAG 核心）
# =============================================================================

def graph_knowledge_search(query_entity: str, hop: int = 1,
                           knowledge_graph: KnowledgeGraph | dict = None):
    """
    图检索函数（GraphRAG 核心）
    从知识图谱中检索实体及其关联关系
    :param query_entity: 查询的实体名称（支持模糊匹配）
    :param hop: 跳数，1表示直接关联，2表示二度关联
    :param knowledge_graph: KnowledgeGraph类型，文档经过LangExtraction,再通过实体，关系，转换成KnowledgeGraph数据集合(包含文档全部的实体与关系)
    :return
        相关实体和关系
    """
    results = {
        "matched_entities": [],
        "related_relations": [],
        "connected_entities": set()
    }
    # KnowledgeGraph数据类型，从json文件读出后，已经转换成了字典，不再具有KnowledgeGraph类型属性
    # 将KnowledgeGraph类型统一转换成字典，后面按照字典来处理：
    if isinstance(knowledge_graph, KnowledgeGraph):
        knowledge_graph = knowledge_graph.model_dump()
    # 1. 模糊匹配实体
    for entity_name, entity_data in knowledge_graph["entities"].items():
        if query_entity.lower() in entity_name.lower():
            results["matched_entities"].append({
                "name": entity_name,
                **entity_data
            })

    # 2. 查找相关关系
    matched_names = [e["name"] for e in results["matched_entities"]]

    for relation in knowledge_graph["relations"]:
        subject1 = relation.get("subject1", "")
        subject2 = relation.get("subject2", "")
        subject3 = relation.get("subject3", "")

        for name in matched_names:
            if name.lower() in str(subject1).lower() or name.lower() in str(subject2).lower() or name.lower() in str(
                    subject3).lower():
                results["related_relations"].append(relation)
                if subject1:
                    results["connected_entities"].add(subject1)
                if subject2:
                    results["connected_entities"].add(subject2)
                if subject3:
                    results["connected_entities"].add(subject3)
                break

    # 3. 二度关联
    if hop > 1 and results["connected_entities"]:
        for connected in list(results["connected_entities"]):
            for relation in knowledge_graph["relations"]:
                subject1 = relation.get("subject1", "")
                subject2 = relation.get("subject2", "")
                subject3 = relation.get("subject3", "")
                if connected.lower() in str(subject1).lower() or connected.lower() in str(
                        subject2).lower() or connected.lower() in str(subject3).lower():
                    if relation not in results["related_relations"]:
                        results["related_relations"].append(relation)

    results["connected_entities"] = list(results["connected_entities"])
    return results





if __name__ == "__main__":
    json_path = Path(
        r"E:/Python_WorkSpace/modelscope/Agentic GraphRAG/data/lx_output/阿联酋投资问与答_LangExtract.json")
    chromadb_path = r"E:/Python_WorkSpace/modelscope/Agentic GraphRAG/data/UAE_investment_QA"
    collection_name = "UAE_investment_QA"
    doc_title = '阿联酋投资问与答'
    json_data = iterate_json_loader(json_path)
    KnowledgeExtraction_, KnowledgeGraph_ = load_KnowledgeExtraction_KnowledgeGraph_from_json(json_data,
                                                                                            doc_title=doc_title)
    print(f"KnowledgeExtraction长度:{len(KnowledgeExtraction_)}")  # 3029条提取记录
    print(f"entities数量:{len(KnowledgeGraph_.entities)}")  # 829 实体
    print(f"relations数量:{len(KnowledgeGraph_.relations)}")  # 597 关系
    # print(f"KnowledgeExtraction_[0]:\n{KnowledgeExtraction_[0]}")
    # print(f"KnowledgeExtraction_[0].search_text:\n{KnowledgeExtraction_[0].to_searchable_text()}")

    # 将KnowledgeExtraction数据类型，写入chromadb向量数据库;或者从chromadb向量数据库中加载
    chromadb = chromDB(chromadb_path=chromadb_path)
    # vectorstore = chromadb.KnowledgeExtraction2ChromaDB(knowledge_extraction=KnowledgeExtraction_,
    #                                                     collection_name=collection_name,
    #                                                     collection_metadata=None,  # 采用缺省值，包含embedding_function
    #                                                     embedding_function=Zhipu_embedding,
    #                                                     chunk_size=64,
    #                                                     )
    # 从chromadb向量数据库中加载
    vectorstore = chromadb.load_vectorstore(collection_name=collection_name,
                                            embedding_function=Zhipu_embedding)
    # print(f"vectorstore.collection.metadata:{vectorstore._client.get_collection(collection_name).metadata}")
    query = "阿联酋公司注册费用"
    # vector_knowledge_search_result = vector_knowledge_search(query=query, top_k=3,
    #                                                          vectorstore=vectorstore)
    # print(f"vectore_search:\n{vector_knowledge_search_result}")

    graph_knowledge_search_result = graph_knowledge_search(query_entity=query, hop=1,
                                                           knowledge_graph=KnowledgeGraph_)
    print(f"graph_search:\n{graph_knowledge_search_result}")
