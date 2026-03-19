import os
import glob
import json
import pandas as pd
import matplotlib.pyplot as plt
import time


class ChartTool:
    """
    ChartTool Ultra 完整版（修正版）：
    - 自動依照圖表類型調整 figsize（不會扁、不會變形）
    - barh 自適應高度（大量分類也不會壓扁）
    - 高 DPI（文字清楚）
    
    支援 7 種圖表：
    - bar (直條圖)
    - barh (橫條圖)
    - pie (圓餅圖)
    - line (折線圖)
    - scatter (散佈圖)
    - box (箱型圖)
    - histogram (直方圖)
    """

    def __init__(self, data_root="data/structured"):
        self.data_root = data_root
        self.csv_map = self._load_all_csv()

    def _load_all_csv(self):
        csv_files = glob.glob(os.path.join(self.data_root, "*.csv"))
        csv_map = {}
        for path in csv_files:
            name = os.path.basename(path).replace(".csv", "")
            try:
                df = pd.read_csv(path)
                csv_map[name] = df
                print(f"[ChartToolUltra] 已載入 CSV：{name} ({df.shape[0]} rows)")
            except Exception as e:
                print(f"[ChartToolUltra] 無法載入 {path}：{e}")
        return csv_map

    def _auto_select_csv(self, x, y, spec_filter):
        for name, df in self.csv_map.items():
            if x in df.columns and y in df.columns:
                return name
        return None

    def _apply_filter(self, df, spec_filter):
        if not isinstance(spec_filter, dict):
            return df, None

        for col, condition in spec_filter.items():
            if col not in df.columns:
                return None, f"[ChartUltraError] 欄位 {col} 不存在"

            if isinstance(condition, dict) and "min" in condition and "max" in condition:
                df = df[(df[col] >= condition["min"]) & (df[col] <= condition["max"])]
            else:
                df = df[df[col] == condition]

        if df.empty:
            return None, "[ChartUltraError] 套用 filter 後沒有資料"

        return df, None

    def generate(self, spec, output_path=None):
        """
        支援圖表種類：
        bar / barh / pie / line / scatter / box / histogram
        """

        # ---------- 驗證參數 ----------
        try:
            x = spec["x"]
            y = spec["y"]
        except KeyError:
            return json.dumps({
                "status": "error",
                "message": "[ChartUltraError] spec 必須包含 x 和 y"
            }, ensure_ascii=False)

        csv_name = spec.get("csv_name")
        chart_type = spec.get("chart_type", "barh")  # 預設 barh
        spec_filter = spec.get("filter")

        # ---------- 自動選擇 CSV ----------
        if csv_name is None:
            csv_name = self._auto_select_csv(x, y, spec_filter)
            if csv_name is None:
                return json.dumps({
                    "status": "error",
                    "message": "[ChartUltraError] 找不到包含 x/y 的 CSV"
                }, ensure_ascii=False)

        df = self.csv_map[csv_name].copy()

        if x not in df.columns or y not in df.columns:
            return json.dumps({
                "status": "error",
                "message": f"[ChartUltraError] 找不到欄位 x={x} y={y}",
                "suggest_columns": list(df.columns)
            }, ensure_ascii=False)

        df, err = self._apply_filter(df, spec_filter)
        if err:
            return json.dumps({"status": "error", "message": err}, ensure_ascii=False)

        # ---------- 過多分類 → 取前 20 ----------
        if df[x].nunique() > 20:
            df = df.nlargest(20, y)

        df = df.sort_values(y, ascending=False)

        # ---------- 書名換行（防止超長） ----------
        df["__x_short"] = df[x].apply(
            lambda t: "\n".join([str(t)[i:i+22] for i in range(0, len(str(t)), 22)])
        )

        # ---------- 檔名 ----------
        if output_path is None:
            ts = int(time.time() * 1000)
            output_path = f"chart_{ts}.png"

        # ================================================================
        #                          圖片尺寸調整（重點!!!）
        # ================================================================

        plt.clf()

        if chart_type == "bar":
            plt.figure(figsize=(16, 10), dpi=200)

        elif chart_type == "barh":
            # 每個分類佔 0.5 高度，自動避免壓扁
            auto_height = max(8, len(df) * 0.5)
            plt.figure(figsize=(14, auto_height), dpi=200)

        elif chart_type == "pie":
            plt.figure(figsize=(12, 12), dpi=200)  # 正方形最漂亮

        elif chart_type in ["line", "scatter"]:
            plt.figure(figsize=(16, 10), dpi=200)

        elif chart_type in ["box", "histogram"]:
            plt.figure(figsize=(14, 10), dpi=200)

        else:
            plt.figure(figsize=(14, 10), dpi=200)

        # ================================================================
        #                         開始畫圖
        # ================================================================

        if chart_type == "bar":
            plt.bar(df["__x_short"], df[y], color="#5AB2FF")
            plt.xticks(rotation=30, ha="right")

        elif chart_type == "barh":
            plt.barh(df["__x_short"], df[y], color="#5AB2FF")
            plt.gca().invert_yaxis()

        elif chart_type == "pie":
            plt.pie(df[y], labels=df["__x_short"],
                    autopct="%1.1f%%", textprops={'fontsize': 8})
            plt.axis("equal")

        elif chart_type == "line":
            plt.plot(df["__x_short"], df[y], marker="o")
            plt.xticks(rotation=30, ha="right")

        elif chart_type == "scatter":
            plt.scatter(df["__x_short"], df[y], c="#5AB2FF")
            plt.xticks(rotation=30, ha="right")

        elif chart_type == "box":
            plt.boxplot(df[y], vert=True)
            plt.xticks([1], [y])

        elif chart_type == "histogram":
            plt.hist(df[y], bins=10, color="#5AB2FF")

        else:
            return json.dumps({
                "status": "error",
                "message": f"[ChartUltraError] 不支援的 chart_type: {chart_type}"
            })

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        return json.dumps({
            "status": "ok",
            "path": output_path,
            "csv": csv_name,
            "spec": spec
        }, ensure_ascii=False)
