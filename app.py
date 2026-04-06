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

def web_search(query: str, max_results: int = 10) -> str:
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

    # 只回傳前 10 筆，避免 token 超限
    df = df.sort_values("review_count", ascending=False).head(10)
    lines = ["[AutoSales] Top 10\n"]
    for i, (_, row) in enumerate(df.iterrows(), 1):
        lines.append(f"{i}. {row.get('title', '未知')} — {int(row['review_count'])} reviews")
    return "\n".join(lines)

# =========================================================
# Tools 映射表
# =========================================================

TOOL_IMPLS = {
    "text_rag_query":  lambda query, **_: text_rag_tool.query(query),
    "table_rag_query": lambda query, **_: table_rag_tool.query(query),
    "autosales_query": lambda query, **_: autosales_query(query),
    "web_search":      lambda query, max_results=5, **_: web_search(query, max_results),
    "chart_generate":  lambda **kwargs: chart_generate(kwargs.get("spec", {})),
}

# =========================================================
# OpenAI tools schema
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
            "description": (
                "生成統計圖表。必須傳入 spec 物件，包含 csv_name、x、y、chart_type。"
                "內容定制： csv_name=auto_sales, x=title, y=review_count, "
                "chart_type 可選 bar/barh/pie/line/scatter/box/hist。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "spec": {
                        "type": "object",
                        "description": "圖表規格，必須包含 csv_name、x、y、chart_type。",
                        "properties": {
                            "csv_name": {
                                "type": "string",
                                "description": "固定填 auto_sales",
                                "enum": ["auto_sales"]
                            },
                            "x": {
                                "type": "string",
                                "description": "固定填 title",
                                "enum": ["title"]
                            },
                            "y": {
                                "type": "string",
                                "description": "固定填 review_count",
                                "enum": ["review_count"]
                            },
                            "chart_type": {
                                "type": "string",
                                "description": "圖表類型",
                                "enum": ["bar", "barh", "pie", "line", "scatter", "box", "hist"]
                            }
                        },
                        "required": ["csv_name", "x", "y", "chart_type"]
                    }
                },
                "required": ["spec"],
            },
        },
    },
]

# =========================================================
# API retry（避免 429，指數退避）
# =========================================================

def safe_chat(payload, retry=5, delay=3):
    for i in range(retry):
        try:
            return client.chat.completions.create(**payload)
        except Exception as e:
            print(f"[ERROR 完整訊息] {str(e)}")
            if "429" in str(e):
                wait = delay * (2 ** i)  # 3, 6, 12, 24, 48 秒
                print(f"[429] Rate limit，等待 {wait} 秒後重試（第 {i+1} 次）")
                time.sleep(wait)
                continue
            raise
    raise Exception("多次 retry 後仍然 429 error，請稍後再試")

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
- 回答風格可以更自然、有溫度。

【重要】當你已經取得足夠的資訊可以回答使用者問題時，請直接給出最終答案，不要再呼叫任何工具。
每個工具只需呼叫一次，不要重複呼叫同一個工具。
若工具回傳 status=error，請直接告知使用者錯誤原因，絕對不要重複呼叫同一個工具。

【chart_generate 使用規則 - 非常重要】
chart_generate 的 spec 必須嚴格使用以下格式：
{{
  "csv_name": "auto_sales",
  "x": "title",
  "y": "review_count",
  "chart_type": "pie"  // 可選: bar / barh / pie / line / scatter / box / hist
}}
- x 欄位固定使用 "title"（書名）
- y 欄位固定使用 "review_count"（評論數，作為銷售量代理）
- csv_name 固定使用 "auto_sales"
- 不要使用其他欄位名稱，否則會失敗
- 若 chart_generate 回傳 status=error，不要重試，直接告知使用者

【智能 Web Search 規則】
請根據下列情況「自動決定」是否使用 web_search：
1. 當使用者詢問的是「即時性、最新、今年、目前」等需要新資訊的問題。
2. 當資料庫中（CSV / RAG）沒有足夠資訊可以回答時。
3. 當使用者明確要求查網路。
web_search 只在確定需要時才使用，不需要每題都查。
"""

# =========================================================
# 工具函式
# =========================================================
import re

def extract_top_n(user_query: str) -> int:
    q = user_query.replace("名", "").replace("前", "").replace("第", "").lower()
    nums = re.findall(r"\d+", q)
    if nums:
        n = int(nums[0])
        return max(1, min(n, 100))
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

# =========================================================
# ReAct Agent Loop
# =========================================================

def agent_run(user_query: str) -> Tuple[str, Optional[str]]:

    MAX_ITERATIONS = 8

    web_keywords = ["最新", "現在", "目前", "今年", "外部", "網路", "google", "搜尋", "news", "新聞"]
    needs_web = any(k in user_query.lower() for k in web_keywords)
    if needs_web:
        user_query = "[需要網路搜尋] " + user_query

    messages = [
        {"role": "system", "content": build_system_prompt()},
        {"role": "user", "content": user_query},
    ]

    last_chart_path: Optional[str] = None
    tool_call_counts: Dict[str, int] = {}  # 記錄每個 tool 被呼叫幾次

    for iteration in range(MAX_ITERATIONS):
        print(f"\n[Loop {iteration+1}] 開始")
        time.sleep(1)  # 避免 RPM 超限

        response = safe_chat({
            "model": CHAT_MODEL,
            "messages": messages,
            "tools": TOOL_DEFS,
            "tool_choice": "auto",
            "max_tokens": 800,
        })

        assistant_msg = response.choices[0].message
        tool_calls = assistant_msg.tool_calls

        print(f"[Loop {iteration+1}] tool_calls: {[tc.function.name for tc in tool_calls] if tool_calls else '無，準備給答案'}")

        # 沒有 tool call → GPT 認為答案夠了，結束
        if not tool_calls:
            print(f"[Loop {iteration+1}] 正常結束，回傳答案")
            return assistant_msg.content or "", last_chart_path

        # 把 assistant 決定加進歷史
        messages.append({
            "role": "assistant",
            "content": assistant_msg.content,
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
        })

        # 執行 tools
        for tc in tool_calls:
            fn_name = tc.function.name
            try:
                args = json.loads(tc.function.arguments)
            except:
                args = {}

            # 累計呼叫次數
            tool_call_counts[fn_name] = tool_call_counts.get(fn_name, 0) + 1

            impl = TOOL_IMPLS.get(fn_name)
            if fn_name == "chart_generate":
                print(f"[ChartDebug] GPT 傳來的 args: {args}")
            result = impl(**args) if impl else f"[ToolError] 未知工具：{fn_name}"

            if fn_name == "chart_generate":
                try:
                    obj = json.loads(result)
                    print(f"[ChartDebug] status={obj.get('status')} path={obj.get('path')}")
                    if obj.get("status") == "ok":
                        p = obj.get("path")
                        print(f"[ChartDebug] 檔案存在={os.path.exists(p) if p else False}")
                        if p and os.path.exists(p):
                            last_chart_path = p
                            print(f"[ChartDebug] last_chart_path 設定成功: {last_chart_path}")
                        else:
                            # path 存在但檔案找不到，可能是相對路徑問題
                            # 嘗試在目前工作目錄尋找
                            alt_path = os.path.join(os.getcwd(), p) if p else None
                            print(f"[ChartDebug] 嘗試絕對路徑: {alt_path}")
                            if alt_path and os.path.exists(alt_path):
                                last_chart_path = alt_path
                                print(f"[ChartDebug] 用絕對路徑成功: {last_chart_path}")
                    else:
                        # 失敗時加強提示，阻止 GPT 重試
                        obj["note"] = "圖表生成失敗，請勿重複呼叫 chart_generate，直接用文字告知使用者錯誤原因。"
                        result = json.dumps(obj, ensure_ascii=False)
                except Exception as chart_err:
                    print(f"[ChartDebug] 解析 result 失敗: {chart_err}, raw={result[:200]}")

            # 防止單一 tool result 太大塞爆 token
            MAX_RESULT_CHARS = 3000
            if len(result) > MAX_RESULT_CHARS:
                result = result[:MAX_RESULT_CHARS] + "\n...[result truncated]"

            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "name": fn_name,
                "content": result,
            })

        # 若有任何 tool 被重複呼叫超過 2 次 → 強制整合
        if any(v > 2 for v in tool_call_counts.values()):
            print(f"[Loop {iteration+1}] 偵測到重複呼叫，強制整合答案")
            break

    # 超過 MAX_ITERATIONS 或重複呼叫 → 強制整合現有結果
    print("[Agent] 強制整合最終答案")
    final = safe_chat({
        "model": CHAT_MODEL,
        "messages": messages,
        "max_tokens": 800,
        # 不傳 tools，強制只給文字答案
    })
    return final.choices[0].message.content or "", last_chart_path

# =========================================================
# Chatbot 內顯示縮圖；下方顯示完整大圖
# =========================================================

def format_with_image(text: str, img_path: Optional[str]) -> str:
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

        msg.submit(
            fn=chat_respond,
            inputs=[msg, chatbot],
            outputs=[chatbot, msg, big_img]
        )

        send_btn.click(
            fn=chat_respond,
            inputs=[msg, chatbot],
            outputs=[chatbot, msg, big_img]
        )

        clear_btn.click(
            fn=clear_all,
            outputs=[chatbot, msg, big_img]
        )

    demo.launch()
