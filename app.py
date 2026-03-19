import os
import json
import time
import base64
from typing import Optional, Tuple, Dict, Any, Callable, List

import gradio as gr
import pandas as pd

from dotenv import load_dotenv
from openai import OpenAI

# RAG tools
from tools.rag_text_tool import TextRAGTool
from tools.rag_table_tool import TableRAGTool
from tools.chart_tool import ChartTool

# =========================================================
# OpenAI 初始化
# =========================================================

load_dotenv()
client = OpenAI()
CHAT_MODEL = "gpt-4.1-mini"

# =========================================================
# LlamaIndex 設定
# =========================================================

from llama_index.core import Settings
from llama_index.llms.openai import OpenAI as LlamaOpenAI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

Settings.llm = LlamaOpenAI(model="gpt-4.1-mini")
Settings.embed_model = HuggingFaceEmbedding("sentence-transformers/all-MiniLM-L6-v2")

# =========================================================
# AutoSales 建立
# =========================================================

def build_or_load_autosales():
    out_path = "data/structured/auto_sales.csv"

    if os.path.exists(out_path):
        return pd.read_csv(out_path)

    meta = "data/structured/amazon_books_metadata.csv"
    rev = "data/structured/amazon_books_reviews.csv"

    if not os.path.exists(meta) or not os.path.exists(rev):
        return None

    meta_df = pd.read_csv(meta)
    rev_df = pd.read_csv(rev)

    sales_df = rev_df.groupby("parent_asin").size().reset_index(name="review_count")
    merged = sales_df.merge(meta_df, on="parent_asin", how="left")
    merged = merged.sort_values("review_count", ascending=False)

    merged.to_csv(out_path, index=False)
    return merged

AUTO_SALES = build_or_load_autosales()

# =========================================================
# Tools 初始化
# =========================================================

text_rag_tool = TextRAGTool()
table_rag_tool = TableRAGTool()
chart_tool = ChartTool()

# =========================================================
# Web Search
# =========================================================

from duckduckgo_search import DDGS

def web_search(query: str, max_results: int = 5) -> str:
    try:
        results = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                title = r.get("title", "")
                url = r.get("href", "")
                desc = r.get("body", "")
                results.append(f"標題: {title}\n網址: {url}\n摘要: {desc}")
        return "\n\n---\n\n".join(results) if results else "找不到資料"
    except Exception as e:
        return f"[WebSearchError] {e}"

# =========================================================
# ChartTool 包裝
# =========================================================

def chart_generate(spec: dict) -> str:
    try:
        ts = int(time.time() * 1000)
        output_path = f"sales_chart_{ts}.png"

        result = chart_tool.generate(spec, output_path=output_path)
        obj = json.loads(result)

        if obj.get("status") == "ok":
            obj["path"] = output_path
            return json.dumps(obj, ensure_ascii=False)
        return result
    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)}, ensure_ascii=False)

# =========================================================
# AutoSales 查詢
# =========================================================

def autosales_query(query: str) -> str:
    if AUTO_SALES is None:
        return "[AutoSalesError] 缺少 auto_sales.csv"

    df = AUTO_SALES.copy()
    q = query.strip()

    detect_rank = any(w in q for w in ["排行", "前十", "Top 10", "前 10"])
    detect_sales = any(w in q for w in ["銷售量", "銷量", "熱銷", "暢銷"])

    if detect_rank or detect_sales:
        df = df.sort_values("review_count", ascending=False).head(10)
        lines = ["[AutoSales] Top 10（以評論數作為銷售量代理）\n"]
        for i, row in df.iterrows():
            lines.append(f"{len(lines)}. {row.get('title','未知')} — {int(row['review_count'])}")
        return "\n".join(lines)

    df = df.sort_values("review_count", ascending=False).head(20)
    return df.to_string()
# =========================================================
# Tools 映射表
# =========================================================

TOOL_IMPLS = {
    "text_rag_query": lambda query, **_: text_rag_tool.query(query),
    "table_rag_query": lambda query, **_: table_rag_tool.query(query),
    "autosales_query": lambda query, **_: autosales_query(query),
    "web_search": lambda query, max_results=5, **_: web_search(query, max_results),
    "chart_generate": lambda spec, **_: chart_generate(spec),
}

# =========================================================
# OpenAI tools schema（非常重要）
# =========================================================

TOOL_DEFS = [
    {
        "type": "function",
        "function": {
            "name": "text_rag_query",
            "description": "查詢 PDF/TXT 等非結構化文件內容。",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "table_rag_query",
            "description": "查詢 CSV 結構化資料。",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "autosales_query",
            "description": "查詢 AutoSales（review_count 作為銷售量代理）。",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "使用 DuckDuckGo 搜尋外部最新資訊。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "max_results": {"type": "integer", "default": 5},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "chart_generate",
            "description": "生成統計圖表。",
            "parameters": {
                "type": "object",
                "properties": {"spec": {"type": "object"}},
                "required": ["spec"],
            },
        },
    },
]

# =========================================================
# API retry（避免 429）
# =========================================================

def safe_chat(payload, retry=4, delay=1.2):
    for i in range(retry):
        try:
            return client.chat.completions.create(**payload)
        except Exception as e:
            if "429" in str(e):
                time.sleep(delay * (i + 1))
                continue
            raise
    raise Exception("多次 retry 後仍然 429 error")

# =========================================================
# System prompt
# =========================================================

def build_system_prompt():
    return """
你是一位友善、溫暖且擅長資料分析的 Amazon 書籍資料助理。
你會耐心理解使用者需求，並以自然、貼近聊天的語氣回覆，但內容仍然保持精準。

你具備以下工具：
1. text_rag_query（查 PDF/TXT）
2. table_rag_query（查 CSV）
3. autosales_query（以 review_count 作為銷售量代理）
4. chart_generate（可畫 bar / barh / pie / line / scatter / box / histogram）
5. web_search（搜尋外部最新資訊）

重要規則：
- 只要使用者提到「銷售量、前十名、排行、Top 10」等詞，就優先使用 autosales_query。
- 若同時提到「畫圖、圖表、可視化、長條圖、圓餅圖」，則需再呼叫 chart_generate。
- 回覆內容全部使用繁體中文。
- 回答風格可以更自然、有溫度，例如：
  「我來幫你比較一下～」、
  「這個結果有點有趣，我解釋一下給你聽」。

你不會顯示你的思考過程，只會呈現最終友善且清楚的回答。

【智能 Web Search 規則】
你具備 web_search 工具，可搜尋外部最新資訊。

請根據下列情況「自動決定」是否使用 web_search：

1. 當使用者詢問的是「即時性、最新、今年、目前」等需要新資訊的問題。
   例：今年的暢銷書是什麼？目前 Amazon 排行榜有哪些新變化？

2. 當資料庫中（CSV / RAG）沒有足夠資訊可以回答時。

3. 當使用者明確要求：
   - 「查網路」
   - 「找外部資料」
   - 「搜尋最新資訊」
   - 「網路上怎麼說？」

4. 若你不確定答案是否更新過（如出版趨勢、業界新聞），可「先查一次」再回答。

5. 使用網路搜尋後，你需要：
   - 做整理與摘要
   - 不要只 dump 原文
   - 不要杜撰不存在的網址
   - 一律使用自然、清楚的繁體中文說明

web_search 只在確定需要時才使用，不需要每題都查。

"""


# =========================================================
# 單階段工具推理（符合 OpenAI 新規範，不會 400）
# =========================================================
import re

def extract_top_n(user_query: str) -> int:
    """
    從使用者自然語言智能擷取「前 N 名」
    default = 10
    """
    # 移除雜訊詞
    q = user_query.replace("名", "").replace("前", "").replace("第", "").lower()

    # 尋找阿拉伯數字
    nums = re.findall(r"\d+", q)
    if nums:
        n = int(nums[0])
        return max(1, min(n, 100))  # 限制避免無限大

    # 中文數字補丁
    mapping = {
        "一": 1, "二": 2, "三": 3, "四": 4, "五": 5,
        "六": 6, "七": 7, "八": 8, "九": 9, "十": 10,
        "十五": 15, "二十": 20
    }
    for zh, num in mapping.items():
        if zh in user_query:
            return num

    return 10
def extract_chart_type(user_query: str) -> str:
    """
    從使用者語句判斷要畫哪種圖
    default = barh
    """
    q = user_query.lower()

    if any(w in q for w in ["圓餅", "pie"]):
        return "pie"
    if any(w in q for w in ["折線", "趨勢", "line"]):
        return "line"
    if any(w in q for w in ["散點", "scatter"]):
        return "scatter"
    if any(w in q for w in ["箱型", "箱線", "box"]):
        return "box"
    if any(w in q for w in ["直方", "hist"]):
        return "hist"
    if any(w in q for w in ["水平", "橫向", "barh"]):
        return "barh"
    if any(w in q for w in ["長條", "bar"]):
        return "bar"

    return "barh"

def agent_run(user_query: str) -> Tuple[str, Optional[str]]:
    # ---------------------------------------------------------
    # 智能 Web Search 啟動前置提示（不會影響工具呼叫格式）
    # ---------------------------------------------------------
    web_keywords = ["最新", "現在", "目前", "今年", "外部", "網路", "google", "搜尋", "news", "新聞"]
    needs_web = any(k in user_query.lower() for k in web_keywords)

    # 若偵測到問題具網路屬性 → 在 system prompt 中加強提示
    if needs_web:
        user_query = "[需要網路搜尋] " + user_query

    # ------------------------------
    # 第一次模型判斷要用哪些工具
    # ------------------------------
    first = safe_chat({
        "model": CHAT_MODEL,
        "messages": [
            {"role": "system", "content": build_system_prompt()},
            {"role": "user", "content": user_query},
        ],
        "tools": TOOL_DEFS,
        "tool_choice": "auto",
        "max_tokens": 500,
    })

    assistant_msg = first.choices[0].message
    tool_calls = assistant_msg.tool_calls
    assistant_text = assistant_msg.content or ""

    # ------------------------------
    # 若沒有工具 → 回覆文字（仍可能 fallback 圖）
    # ------------------------------
    if not tool_calls:
        final_text = assistant_text
        last_chart_path = None

    else:
        # ------------------------------
        # 執行工具
        # ------------------------------
        tool_messages = []
        last_chart_path: Optional[str] = None

        for tc in tool_calls:
            fn_name = tc.function.name
            try:
                args = json.loads(tc.function.arguments)
            except:
                args = {}

            impl = TOOL_IMPLS.get(fn_name)
            result = impl(**args) if impl else f"[ToolError] 未知工具：{fn_name}"

            # 若為圖片工具 → 解析圖片路徑
            if fn_name == "chart_generate":
                try:
                    obj = json.loads(result)
                    if obj.get("status") == "ok":
                        p = obj.get("path")
                        if p and os.path.exists(p):
                            last_chart_path = p
                except:
                    pass

            tool_messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "name": fn_name,
                "content": result,
            })

        # ------------------------------
        # 第二次模型：整理文字回覆
        # ------------------------------
        follow = safe_chat({
            "model": CHAT_MODEL,
            "messages": [
                {"role": "system","content": (
                "你現在要根據工具結果，提供一份友善且清晰的最終回答。\n"
                "請不要只重述工具輸出的內容，而要真的進行「資料分析」。\n\n"
                "分析時請做到：\n"
                "1. 解讀數據：指出最高、最低、明顯差距、前段與後段的落差。\n"
                "2. 若是排行榜，請描述整體趨勢，例如是否前幾名特別突出。\n"
                "3. 若有圖表，請說明圖表呈現的重點，例如集中程度、異常點。\n"
                "4. 用貼近對話、溫暖且自然的繁體中文來說明，例如：\n"
                "   -「可以看到前幾名的差距滿明顯的，我說明一下～」\n"
                "   -「有趣的是，從數字可以看出...」\n"
                "5. 不要編造不存在的欄位，也不要猜測未提供的資訊。\n"
                    )
                },
                {"role": "user", "content": user_query},
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in tool_calls
                    ],
                },
                *tool_messages,
            ],
            "max_tokens": 800,
        })

        final_text = follow.choices[0].message.content or ""

    # ---------------------------------------------------------
    # Auto Fallback：自動補圖（前 N 名 + 圖類型）
    # ---------------------------------------------------------
    need_chart = any(w in user_query for w in ["圖", "畫", "長條", "圓餅", "圖表", "視覺化"])
    need_sales = any(w in user_query for w in ["銷售", "銷量", "熱銷", "排行", "前", "top"])

    if last_chart_path is None and need_chart and need_sales and AUTO_SALES is not None:
        
        try:
            # 擷取 N 與圖表類型
            top_n = extract_top_n(user_query)
            chart_type = extract_chart_type(user_query)

            # 選資料 Top N
            df = AUTO_SALES.sort_values("review_count", ascending=False).head(top_n)

            # 建立資料摘要（讓模型能分析數據）
            summary = "本次前 {} 名的評論數如下：\n".format(top_n)
            summary += "\n".join([
                f"{i+1}. {row['title']} — {row['review_count']}"
                for i, row in df.iterrows()
            ])
            final_text += "\n\n" + summary + "\n"

            # 構造 fallback 專用 spec
            spec = {
                "csv_name": "auto_sales",
                "x": "title",
                "y": "review_count",
                "chart_type": chart_type
            }

            raw = chart_generate(spec)
            obj = json.loads(raw)


            if obj.get("status") == "ok":
                p = obj.get("path")
                if p and os.path.exists(p):
                    last_chart_path = p
                    final_text += f"\n\n（已根據你的需求，自動繪製前 {top_n} 名的 {chart_type} 圖。）"

        except Exception as e:
            final_text += f"\n\n[AutoChartFallbackError] {e}"

    # ------------------------------
    # 回傳文字 + 圖片路徑
    # ------------------------------
    return final_text, last_chart_path

# =========================================================
# Chatbot 內顯示縮圖；下方顯示完整大圖
# =========================================================

def format_with_image(text: str, img_path: Optional[str]) -> str:
    """
    Chatbot 內顯示縮圖（固定寬度、保持比例）
    下方的 gr.Image 會顯示完整大圖
    """
    if img_path and os.path.exists(img_path):
        with open(img_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        data_uri = f"data:image/png;base64,{b64}"

        return (
            text +
            f'\n\n<img src="{data_uri}" '
            f'style="width:100%; height:auto; border-radius:10px;">'
        )

    return text


def chat_respond(msg: str, history: List[List[str]]):
    """
    - Chatbot: 回傳文字 + 縮圖
    - big_img: 回傳完整 png 檔路徑
    """
    if history is None:
        history = []

    text, img_path = agent_run(msg)
    bot_output = format_with_image(text, img_path)

    big_img = img_path if img_path and os.path.exists(img_path) else None

    history.append([msg, bot_output])
    return history, "", big_img


def clear_all():
    return [], "", None


# =========================================================
# Gradio 主介面
# =========================================================

if __name__ == "__main__":
    with gr.Blocks(title="Amazon Book 助理") as demo:

        gr.Markdown("""
        # Amazon Book 資料助理

        功能：
        - RAG：查詢 PDF / TXT
        - CSV 查詢：以自然語言查欄位、條件
        - AutoSales：review_count 作為銷售量代理
        - 圖表：bar / barh / pie / line / scatter / box / histogram
        - Web Search：外部資訊搜尋

        顯示方式：
        - Chatbot 內：縮圖（不變形）
        - 下面區域：完整大圖（依原始比例，不會被壓縮）
        """)

        chatbot = gr.Chatbot(
            height=550,
            label="對話區"
        )

        # 下方完整大圖區域
        big_img = gr.Image(
            label="完整圖表預覽",
            interactive=False
        )

        msg = gr.Textbox(
            lines=2,
            show_label=False,
            placeholder="輸入問題（例如：銷售量前十名並畫長條圖）"
        )

        with gr.Row():
            send_btn = gr.Button("送出", variant="primary")
            clear_btn = gr.Button("清除")

        # Enter
        msg.submit(
            fn=chat_respond,
            inputs=[msg, chatbot],
            outputs=[chatbot, msg, big_img]
        )

        # 按鈕送出
        send_btn.click(
            fn=chat_respond,
            inputs=[msg, chatbot],
            outputs=[chatbot, msg, big_img]
        )

        # 清除
        clear_btn.click(
            fn=clear_all,
            outputs=[chatbot, msg, big_img]
        )

    demo.launch()