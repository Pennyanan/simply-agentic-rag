import os
import glob
import pandas as pd
import chromadb

from dotenv import load_dotenv
load_dotenv()

from llama_index.core import (
    Document,
    VectorStoreIndex,
    StorageContext,
)
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


def ingest_structured(
    data_dir: str = "data/structured",
    vs_path: str = "vectorstore/table_db",
    collection_name: str = "table_docs"
):
    """讀取 CSV → 轉 Document → HuggingFace embed → Chroma"""

    if not os.path.exists(data_dir):
        os.makedirs(data_dir, exist_ok=True)
        print(f"資料夾 {data_dir} 不存在，已建立。")
        return

    csv_paths = glob.glob(os.path.join(data_dir, "*.csv"))

    if not csv_paths:
        print("沒有找到任何 CSV 檔")
        return

    print("準備 ingest CSV：")
    for p in csv_paths:
        print(" -", p)

    documents = []

    for p in csv_paths:
        df = pd.read_csv(p)

        for _, row in df.iterrows():
            row_text = ", ".join([f"{col}: {row[col]}" for col in df.columns])
            documents.append(Document(text=row_text))

    print(f"共 {len(documents)} 筆資料列轉換成 Document")

    client = chromadb.PersistentClient(path=vs_path)
    collection = client.get_or_create_collection(collection_name)
    vector_store = ChromaVectorStore(chroma_collection=collection)

    embed_model = HuggingFaceEmbedding(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    print("開始建立/更新 結構化 向量庫...")
    _ = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        embed_model=embed_model,
        show_progress=True
    )

    print("完成：table_db 已成功建立！")


if __name__ == "__main__":
    ingest_structured()
