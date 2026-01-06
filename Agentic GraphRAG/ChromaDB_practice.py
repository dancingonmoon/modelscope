import os
from datetime import datetime
from typing import Literal
import uuid
import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

# embedding function，
# Qwen3 默认dimensions=1024,不能更改:
Qwen_embedding_fun = OpenAIEmbeddingFunction(api_key=os.environ["DASHSCOPE_API_KEY"],
                                             model_name="text-embedding-v4",
                                             api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
                                             # dimensions=256, 该参数对于非OpenAI模型无效;
                                             )
# ZhipuAI 默认dimensions=2048,不能更改:
Zhipu_embedding_fun = OpenAIEmbeddingFunction(api_key=os.environ["ZHIPUAI_API_KEY"],
                                              model_name="embedding-3",
                                              api_base="https://open.bigmodel.cn/api/paas/v4/",
                                              # dimensions=1024, # 该参数对于非OpenAI模型无效;
                                              )
# Custom Embedding Functions
from chromadb import Documents, EmbeddingFunction, Embeddings
import dashscope


class QwenEmbeddingFun_custom(EmbeddingFunction):
    """
    自定义的Qwen Embedding Function; 使用dashscope方法，接受Embedding模型参数，自定义输出精度等
    """

    def __init__(self, model: str = "text-embedding-v4",
                 dimension: Literal[2048, 1536, 1024, 768, 512, 256, 128, 64] | None = None,
                 text_type: Literal["query", "document"] | None = None,
                 instruct: str | None = None):
        """
        :param model: 调用的模型
        :param dimension: 指定的向量维度，必须为以下值之一：2048（仅适用于text-embedding-v4）、1536（仅适用于text-embedding-v4）1024、768、512、256、128 或 64，默认值为1024。
        :param text_type: 文本转换为向量后可以应用于检索、聚类、分类等下游任务，对检索这类非对称任务为了达到更好的检索效果建议区分查询文本（query）和底库文本（document）类型，入库、聚类、分类等对称任务可以不用特殊指定，采用系统默认值document即可。
        :param instruct: 添加自定义任务说明，仅在使用 text-embedding-v4 模型且 text_type 为 query 时生效。建议使用英文撰写，通常可带来约 1%–5% 的效果提升。
        """
        self.model = model
        self.dimension = dimension
        self.text_type = text_type
        self.instruct = instruct

    def __call__(self, input: Documents, ) -> Embeddings:
        """
        :param input: 输入待处理的文本。可以是字符串（string）、字符串列表（array）或文件（file）。不同模型版本支持的文本长度和批量大小不同，具体如下：
                        text-embedding-v3 / v4 模型：
                            输入为字符串：最长支持 8,192 Token。
                            输入为字符串列表或文件：最多支持 10 条（行），每条（行）最长支持 8,192 Token。
                        text-embedding-v1 / v2 模型：
                            输入为字符串：最长支持 2,048 Token。
                            输入为字符串列表或文件：最多支持 25 条（行），每条（行）最长支持 2,048 Token。
        :return: 输出的向量列表
        """
        params = {
            "model": self.model,
            "input": input,
        }
        if self.dimension:
            params["dimension"] = self.dimension
        if self.text_type:
            params["text_type"] = self.text_type
        if self.instruct:
            params["instruct"] = self.instruct
        resp = dashscope.TextEmbedding.call(**params)
        return [result['embedding'] for result in resp.output['embeddings']]


QwenEmbeddingFun_2048 = QwenEmbeddingFun_custom(dimension=2048)
# # embedding function debugging:
# test_embedding = QwenEmbeddingFun_2048_instance(["hello world"])
# print(test_embedding)
# print(test_embedding[0].shape)

# text :
text = ["人工智能", "计算机科学与技术", "计算机科学与技术(至诚班)", "工科试验班类(未来空天领军计划)",
        "计算机科学与技术(计算机金融实验班)",
        "工科试验班(AI院士特色班)", "微电子科学与工程(卓越人才培养试验班)", "工科试验班(自主智能系统院士特色班)",
        "口腔医学", "人工智能(卓越人才培养试验班)", "工科试验班(院士特色班)", "汉语言文学", "软件工程",
        "数学与应用数学(拔尖学生培养计划)",
        "工科试验班类(未来工程师项目制育人试验班)", "临床医学", "网络空间安全(卓越人才培养试验班)", "法学",
        "社会科学试验班",
        "工科试验班(AI加先进技术领军班深圳拔尖班)", "技术科学试验班", "工科试验班(未来技术拔尖班)",
        "口腔医学(5+3一体化)",
        "电气工程及其自动化(电气AI启明实验班)", "自动化(钱学森班)", "临床医学（8年）", "理科试验班类(化学与生命科学类)",
        "软件工程(特软班)", "工科试验班(国豪精英班)", "电子信息类(集成电路启明实验班)", "汉语言文学",
        "电子信息类(未来技术实验班)"]

# # Create a ChromaDB client
chromadb_path = r"./data/test.db"
test_client = chromadb.PersistentClient(path=chromadb_path)
# delete collection:
test_client.delete_collection(name="test1")
collection = test_client.get_or_create_collection(name="test1",
                                                  # embedding_function=Qwen_embedding_fun,
                                                  # embedding_function=QwenEmbeddingFun_2048,
                                                  embedding_function=Zhipu_embedding_fun,
                                                  metadata={"description": "test1",

                                                            "create": datetime.now().strftime("%Y-%m-%d %H:%M")})

ids = []
metadatas = []
for i, txt in enumerate(text):
    # ids.append(str(uuid.uuid4())) # 随机数的缺点是每次ids都不同，导致每次重复添加同一组数据；
    ids.append(f"major{i}")
    metadatas.append({"major": i})

# # # delete collection data:
# collection.delete(where={"major": {"$gte": 0}}) # 删除大于等于0的major,即，全部删除
# # # upsert data :　(add or update)
if len(text) > 10:
    for i in range(0, len(text), 10):
        collection.upsert(documents=text[i:i + 10],
                          metadatas=metadatas[i:i + 10],
                          ids=ids[i:i + 10])
print(collection.count())
#
# # query:
result = collection.query(query_texts=["人工智能"],
                          n_results=10)
print(result)
