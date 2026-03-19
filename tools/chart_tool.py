import os
import glob
import json
import time
import pandas as pd
import matplotlib.pyplot as plt


class ChartTool:
    """
    ChartTool Ultra Enhanced:
    - 支援可調整字體大小 font_size
    - 自動換行避免文字重疊
    - barh 自動高度
    - 高 DPI，清晰輸出
    """

    def __init__(self, data_root="data/structured"):
        self.data_root = data_root
        self.csv_map = self._load_all_csv()

    # ---------------------------------------------------------
    # Load all CSVs
    # ---------------------------------------------------------
    def _load_all_csv(self):
        csv_map = {}
        csv_files = glob.glob(os.path.join(self.data_root, "*.csv"))
        for p in csv_files:
            name = os.path.basename(p).replace(".csv", "")
            try:
                df = pd.read_csv(p)
                csv_map[name] = df
                print(f"[ChartTool] Loaded CSV: {name} ({df.shape[0]} rows)")
            except Exception as e:
                print(f"[ChartTool] Failed to load {p}: {e}")
        return csv_map

    # ---------------------------------------------------------
    # Auto select CSV
    # ---------------------------------------------------------
    def _auto_select_csv(self, x, y):
        for name, df in self.csv_map.items():
            if x in df.columns and y in df.columns:
                return name
        return None

    # ---------------------------------------------------------
    # Apply filter
    # ---------------------------------------------------------
    def _apply_filter(self, df, filters):
        if not isinstance(filters, dict):
            return df, None

        for col, cond in filters.items():
            if col not in df.columns:
                return None, f"[ChartError] 欄位 {col} 不存在"

            if isinstance(cond, dict) and "min" in cond and "max" in cond:
                df = df[(df[col] >= cond["min"]) & (df[col] <= cond["max"])]
            else:
                df = df[df[col] == cond]

        if df.empty:
            return None, "[ChartError] filter 套用後無資料"

        return df, None

    # ---------------------------------------------------------
    # Generate chart
    # ---------------------------------------------------------
    def generate(self, spec, output_path=None):
        # 必要參數
        try:
            x = spec["x"]
            y = spec["y"]
        except KeyError:
            return json.dumps({
                "status": "error",
                "message": "[ChartError] spec 必須包含 x, y"
            }, ensure_ascii=False)

        chart_type = spec.get("chart_type", "barh")
        csv_name = spec.get("csv_name")
        filters = spec.get("filter")
        font_size = spec.get("font_size", 7)

        # 自動尋找 CSV
        if csv_name is None:
            csv_name = self._auto_select_csv(x, y)
            if csv_name is None:
                return json.dumps({
                    "status": "error",
                    "message": "[ChartError] 找不到包含 x,y 欄位的 CSV"
                })

        df = self.csv_map[csv_name].copy()

        # 欄位驗證
        if x not in df.columns or y not in df.columns:
            return json.dumps({
                "status": "error",
                "message": f"[ChartError] x={x}, y={y} 其中之一不存在",
                "available": list(df.columns)
            })

        # filter
        df, err = self._apply_filter(df, filters)
        if err:
            return json.dumps({"status": "error", "message": err})

        # 避免太多分類，截前 20 筆最高
        if df[x].nunique() > 20:
            df = df.nlargest(20, y)

        df = df.sort_values(y, ascending=False)

        # ⭐自動換行避免重疊
        df["__x_short"] = df[x].apply(
            lambda t: "\n".join([str(t)[i:i+18] for i in range(0, len(str(t)), 18)])
        )

        # output path
        if output_path is None:
            output_path = f"chart_{int(time.time() * 1000)}.png"

        # ---------------------------------------------------------
        # 圖片尺寸設定（與類型自動調整）
        # ---------------------------------------------------------
        plt.clf()

        if chart_type == "barh":
            auto_h = max(6, df.shape[0] * 0.5)
            plt.figure(figsize=(14, auto_h), dpi=200)
        elif chart_type == "pie":
            plt.figure(figsize=(10, 10), dpi=200)
        else:
            plt.figure(figsize=(16, 10), dpi=200)

        # ---------------------------------------------------------
        # 畫圖
        # ---------------------------------------------------------
        if chart_type == "bar":
            plt.bar(df["__x_short"], df[y], color="#5AB2FF")
            plt.xticks(rotation=30, ha="right", fontsize=font_size)
            plt.yticks(fontsize=font_size)

        elif chart_type == "barh":
            plt.barh(df["__x_short"], df[y], color="#5AB2FF")
            plt.gca().invert_yaxis()
            plt.xticks(fontsize=font_size)
            plt.yticks(fontsize=font_size)

        elif chart_type == "pie":
            plt.pie(
                df[y],
                labels=df["__x_short"],
                autopct="%1.1f%%",
                textprops={"fontsize": font_size}
            )

        elif chart_type == "line":
            plt.plot(df["__x_short"], df[y], marker="o")
            plt.xticks(rotation=30, ha="right", fontsize=font_size)
            plt.yticks(fontsize=font_size)

        elif chart_type == "scatter":
            plt.scatter(df["__x_short"], df[y], c="#5AB2FF")
            plt.xticks(rotation=30, ha="right", fontsize=font_size)
            plt.yticks(fontsize=font_size)

        elif chart_type == "box":
            plt.boxplot(df[y])
            plt.xticks([1], [y], fontsize=font_size)
            plt.yticks(fontsize=font_size)

        elif chart_type in ["hist", "histogram"]:
            plt.hist(df[y], bins=10, color="#5AB2FF")
            plt.xticks(fontsize=font_size)
            plt.yticks(fontsize=font_size)

        else:
            return json.dumps({
                "status": "error",
                "message": f"[ChartError] 不支援的 chart_type: {chart_type}"
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
