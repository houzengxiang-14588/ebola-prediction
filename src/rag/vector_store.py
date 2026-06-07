"""ChromaDB 向量存储 — 持久化向量知识库"""

import os, json
import chromadb
from chromadb.config import Settings


class VectorStore:
    """ChromaDB 向量数据库封装。

    两个 collection:
    - ebola_knowledge: PDF报告 + 专家规则（长文本语义检索）
    - country_profiles: 国家概况（结构化属性检索）
    """

    def __init__(self, persist_dir="chroma_db"):
        self.persist_dir = persist_dir
        os.makedirs(persist_dir, exist_ok=True)
        self.client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(anonymized_telemetry=False),
        )
        self._knowledge_col = None
        self._country_col = None

    def get_knowledge_collection(self):
        """获取或创建知识检索 collection。"""
        if self._knowledge_col is None:
            try:
                self._knowledge_col = self.client.get_collection(
                    "ebola_knowledge"
                )
            except Exception:
                self._knowledge_col = self.client.create_collection(
                    name="ebola_knowledge",
                    metadata={"description": "WHO埃博拉疫情报告+专家规则"},
                )
        return self._knowledge_col

    def get_country_collection(self):
        """获取或创建国家概况 collection。"""
        if self._country_col is None:
            try:
                self._country_col = self.client.get_collection(
                    "country_profiles"
                )
            except Exception:
                self._country_col = self.client.create_collection(
                    name="country_profiles",
                    metadata={"description": "非洲20国埃博拉概况"},
                )
        return self._country_col

    def index_chunks(self, chunks, embed_fn, collection="knowledge", batch_size=32):
        """将知识分块批量写入向量库。

        Args:
            chunks: [{"id": str, "text": str, "metadata": dict}]
            embed_fn: 嵌入函数 text -> embedding vector
            collection: "knowledge" 或 "country"
        """
        col = (
            self.get_knowledge_collection()
            if collection == "knowledge"
            else self.get_country_collection()
        )

        # 清空已有数据后重建
        try:
            existing = col.get()
            if existing.get("ids"):
                col.delete(ids=existing["ids"])
        except Exception:
            pass

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            ids = [c["id"] for c in batch]
            texts = [c["text"] for c in batch]
            metadatas = [c["metadata"] for c in batch]

            # 使用外部嵌入函数生成向量
            embeddings = embed_fn(texts)

            col.add(
                ids=ids,
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas,
            )

        print(f"索引完成: {len(chunks)} 块写入 {collection} 集合")


def load_chunks(filepath="data/knowledge/all_chunks.json"):
    """加载已分块的知识数据。"""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    # 测试
    vs = VectorStore()
    col = vs.get_knowledge_collection()
    print(f"Collection: {col.name}, count: {col.count()}")
