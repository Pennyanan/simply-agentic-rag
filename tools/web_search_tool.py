import json
from duckduckgo_search import DDGS


class WebSearchTool:
    """
    WebSearchTool: 使用 DuckDuckGo API 實現網路搜尋
    不需 API Key、可即時查詢、結果乾淨易解析。
    """

    def __init__(self):
        self.ddg = DDGS()

    def run(self, query: str, max_results: int = 5):
        try:
            results = self.ddg.text(query, max_results=max_results)
            out = []

            for r in results:
                out.append({
                    "title": r.get("title"),
                    "snippet": r.get("body"),
                    "url": r.get("href")
                })

            return json.dumps({
                "status": "ok",
                "query": query,
                "results": out
            }, ensure_ascii=False)

        except Exception as e:
            return json.dumps({
                "status": "error",
                "message": str(e)
            }, ensure_ascii=False)
