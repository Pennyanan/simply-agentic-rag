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
