from ingest.unstructured_ingest import ingest_unstructured
from ingest.structured_ingest import ingest_structured


if __name__ == "__main__":
    print("開始更新所有向量資料庫...")
    ingest_unstructured()
    ingest_structured()
    print("所有向量庫已經完成更新！")
