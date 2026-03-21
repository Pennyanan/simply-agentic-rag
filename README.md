## RAG 模組說明（核心功能）

本專案的核心為 RAG（Retrieval-Augmented Generation）架構，使模型能夠根據本地資料進行查詢與回答，而非僅依賴預訓練知識。

系統目前包含兩種類型的 RAG：

---

### 1. Text RAG（非結構化資料）

- 支援 PDF、TXT 等文件查詢  
- 透過 embedding 建立向量索引（vectorstore）  
- 使用語意相似度進行檢索  

**適用場景：**

- 書籍內容理解  
- 說明文件查詢  
- 長文本知識檢索  

**對應工具：**
`text_rag_query`

---

### 2. Table RAG（結構化資料）

- 支援 CSV 格式資料查詢  
- 將自然語言轉換為資料篩選與聚合操作  
- 可進行排序、篩選與欄位分析  

**適用場景：**

- 書籍 metadata 查詢  
- 評分、價格、分類比較  
- 結構化資料分析  

**對應工具：**
`table_rag_query`

---

## Agent 與 RAG 的整合

本系統透過 LLM Agent 動態決定工具使用方式：

- 判斷是否需要查資料（RAG）  
- 選擇 Text RAG 或 Table RAG  
- 判斷是否需要進一步處理：  

  - 銷售量分析（AutoSales）  
  - 圖表生成（ChartTool）  
  - 外部搜尋（Web Search）  

---
<img width="3819" height="1994" alt="Untitled diagram-2025-12-05-030912 (1)" src="https://github.com/user-attachments/assets/06d4de2d-b4c5-447f-8de5-b710cbf6b189" />
實作步驟如下:
##  下載專案

```bash
git clone https://github.com/你的帳號/simply-agentic-rag.git
cd simply-agentic-rag
```

---

## 建立虛擬環境（建議）

```bash
python -m venv ragenv
```

啟動：

**Windows**

```bash
ragenv\Scripts\activate
```

**Mac / Linux**

```bash
source ragenv/bin/activate
```

---

## 安裝套件

```bash
pip install -r requirement.txt
```

---

## 設定 `.env`

在專案根目錄建立 `.env`

```env
OPENAI_API_KEY=你的API金鑰
```

---

### 說明

```md
本專案使用 OpenAI API，請自行申請 API Key：
https://platform.openai.com/api-keys
```

[OpenAI API Keys](https://platform.openai.com/api-keys?utm_source=chatgpt.com)

---

## 準備資料

建立資料夾：

```bash
data/structured/
```

放入：

* `amazon_books_metadata.csv`
* `amazon_books_reviews.csv`

---

##  啟動系統

```bash
python app.py
```

https://github.com/user-attachments/assets/d8b98af6-eaa9-4ed3-ad4b-f87a37057d7e


