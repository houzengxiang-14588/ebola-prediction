"""知识检索器 — 基于用户查询从向量库检索相关知识"""

import numpy as np
from sentence_transformers import SentenceTransformer

from .vector_store import VectorStore


class KnowledgeRetriever:
    """RAG 检索器：将用户查询转为嵌入向量，从 ChromaDB 召回相关知识。"""

    def __init__(
        self,
        vector_store: VectorStore,
        embed_model: str = "all-MiniLM-L6-v2",
        top_k: int = 5,
    ):
        self.store = vector_store
        self.top_k = top_k

        print(f"加载嵌入模型: {embed_model}")
        self.embedder = SentenceTransformer(embed_model)
        self.embed_dim = self.embedder.get_sentence_embedding_dimension()

    def _embed(self, texts):
        """文本→向量。"""
        if isinstance(texts, str):
            texts = [texts]
        embeddings = self.embedder.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embeddings.tolist()

    def build_query(self, user_cases, metadata=None):
        """从用户数据特征构造检索查询语句。

        Args:
            user_cases: list of daily case counts
            metadata: {"total_cases": int, "max_case": int, "month": int, ...}
        """
        if metadata is None:
            metadata = {}

        total = sum(user_cases) if user_cases else 0
        max_case = max(user_cases) if user_cases else 0
        recent = user_cases[-5:] if len(user_cases) >= 5 else user_cases
        recent_avg = np.mean(recent) if recent else 0

        # 趋势方向
        half = len(user_cases) // 2
        first_half = np.mean(user_cases[:half]) if half > 0 else 0
        second_half = np.mean(user_cases[half:]) if half > 0 else 0
        if second_half > first_half * 1.3:
            direction = "上升"
        elif second_half < first_half * 0.7:
            direction = "下降"
        else:
            direction = "平稳波动"

        query = (
            f"埃博拉病毒 当前{direction}趋势 "
            f"累计{total}例 最高单日{max_case}例 "
            f"近期日均{recent_avg:.1f}例 "
            f"暴发等级评估 传播风险评估 人传人传播链"
        )

        if total > 0:
            query += " 相似暴发规模 历史疫情对比"

        return query

    def retrieve(self, query, collection="knowledge"):
        """执行知识检索。

        Returns:
            [{"content": str, "metadata": dict, "score": float}]
        """
        col = (
            self.store.get_knowledge_collection()
            if collection == "knowledge"
            else self.store.get_country_collection()
        )

        # 检查是否有数据
        if col.count() == 0:
            return []

        query_embedding = self._embed(query)

        results = col.query(
            query_embeddings=query_embedding,
            n_results=self.top_k,
            include=["documents", "metadatas", "distances"],
        )

        retrieved = []
        if results and results.get("ids") and results["ids"][0]:
            for i, doc_id in enumerate(results["ids"][0]):
                retrieved.append({
                    "id": doc_id,
                    "content": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "score": 1 - results["distances"][0][i],  # 距离→相似度
                })

        return retrieved

    def retrieve_for_prediction(self, user_cases, metadata=None):
        """一次调用完成双重检索：知识+国家概况。

        Returns:
            {"knowledge": [...], "countries": [...]}
        """
        query = self.build_query(user_cases, metadata)

        knowledge_results = self.retrieve(query, collection="knowledge")

        # 同时检索相似国家
        country_results = []
        if metadata and metadata.get("max_case", 0) > 0:
            country_query = f"发病率相近 单日最高{metadata['max_case']}例"
            country_results = self.retrieve(country_query, collection="country")

        return {
            "knowledge": knowledge_results,
            "countries": country_results,
        }
