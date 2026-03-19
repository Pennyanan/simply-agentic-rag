import os
import chromadb

from dotenv import load_dotenv
load_dotenv()

from llama_index.core import (
    SimpleDirectoryReader,
    Document,
    VectorStoreIndex,
    StorageContext,
)
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


def ingest_unstructured(
    data_dir: str = "data/unstructured",
    vs_path: str = "vectorstore/text_db",
    collection_name: str = "text_docs"
):
    """讀取 PDF/TXT/DOCX → 使用 HuggingFace embedding → 寫入 Chroma。"""

    if not os.path.exists(data_dir):
        os.makedirs(data_dir, exist_ok=True)
        print(f"資料夾 {data_dir} 不存在，已建立。請放入文件後再執行。")
        return

    docs = SimpleDirectoryReader(data_dir).load_data()
    print(f"共讀取 {len(docs)} 份文件")

    documents = [Document(text=d.text) for d in docs]

    client = chromadb.PersistentClient(path=vs_path)
    collection = client.get_or_create_collection(collection_name)
    vector_store = ChromaVectorStore(chroma_collection=collection)

    embed_model = HuggingFaceEmbedding(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    print("開始建立/更新 非結構化 向量庫...")
    _ = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        embed_model=embed_model,
        show_progress=True
    )

    print("完成：text_db 已成功建立！")


if __name__ == "__main__":
    ingest_unstructured()
