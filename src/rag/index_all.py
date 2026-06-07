"""一站式索引全部知识到 ChromaDB"""
import os, sys
os.chdir("D:/Ebola")
sys.path.insert(0, "src/rag")

from sentence_transformers import SentenceTransformer
from knowledge_loader import build_all_chunks
from cmrivers_loader import build_cmrivers_chunks
from vector_store import VectorStore

print("=== 构建知识分块 ===")
print("[1/2] 现有知识库...")
chunks = build_all_chunks()

print("\n[2/2] cmrivers 数据...")
cmrivers_chunks = build_cmrivers_chunks()
# 合并到 knowledge 集合中
chunks.extend(cmrivers_chunks)

print(f"\n总计: {len(chunks)} 块")

# 分类
knowledge = [c for c in chunks if c["metadata"]["type"] != "country_profile"]
countries = [c for c in chunks if c["metadata"]["type"] == "country_profile"]
print(f"  knowledge: {len(knowledge)} 块, country: {len(countries)} 块")

print("\n=== 加载嵌入模型 ===")
model = SentenceTransformer("all-MiniLM-L6-v2")
print(f"模型维度: {model.get_sentence_embedding_dimension()}")

def embed_fn(texts):
    return model.encode(texts).tolist()

print("\n=== 写入向量库 ===")
vs = VectorStore()

if knowledge:
    vs.index_chunks(knowledge, embed_fn, collection="knowledge")
if countries:
    vs.index_chunks(countries, embed_fn, collection="country")

print("\n=== 验证 ===")
kc = vs.get_knowledge_collection()
cc = vs.get_country_collection()
print(f"ebola_knowledge: {kc.count()} 条")
print(f"country_profiles: {cc.count()} 条")
print("\n索引完成！")
