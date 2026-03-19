import chromadb
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


class TableRAGTool:
    """CSV → table_db RAG"""

    def __init__(
        self,
        vs_path: str = "vectorstore/table_db",
        collection_name: str = "table_docs"
    ):
        client = chromadb.PersistentClient(path=vs_path)
        collection = client.get_or_create_collection(collection_name)
        vector_store = ChromaVectorStore(chroma_collection=collection)

        embed_model = HuggingFaceEmbedding(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        self.index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store,
            storage_context=storage_context,
            embed_model=embed_model,
        )
        self.query_engine = self.index.as_query_engine(similarity_top_k=4)

    def query(self, query_text: str) -> str:
        try:
            resp = self.query_engine.query(query_text)
            return str(resp)
        except Exception as e:
            return f"[TableRAGError] 查詢失敗：{e}"
