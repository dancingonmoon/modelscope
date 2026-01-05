import os
import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

# embeddingfuction:
Qwen_embedding = OpenAIEmbeddingFunction(api_key=os.environ["DASHSCOPE_API_KEY"],
                                         model_name="text-embedding-v4",
                                         api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
                                         )
test_embedding = Qwen_embedding(["hello world"])

# Create a ChromaDB client
# chromadb_path = r"./test.db"
# test_client = chromadb.PersistentClient(path=chromadb_path)
# collection = test_client.get_or_create_collection(name="test1")