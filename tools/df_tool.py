import os
import glob
import pandas as pd
import matplotlib.pyplot as plt


class ChartTool:
    """
    從 data/structured 底下的所有 CSV 合併成一個 DataFrame，
    然後根據 spec 繪製圖表。
    spec 格式範例：
    {
        "chart_type": "bar",        # bar / pie / line
        "x": "category",
        "y": "sales",
        "filter": {
            "year": 2024
        }
    }
    """

    def __init__(self, data_dir: str = "data/structured"):
        csv_paths = glob.glob(os.path.join(data_dir, "*.csv"))
        dfs = []
        for p in csv_paths:
            dfs.append(pd.read_csv(p))
        if len(dfs) > 0:
            self.df = pd.concat(dfs, ignore_index=True)
        else:
            self.df = pd.DataFrame()
        print(f"ChartTool 已載入 {len(self.df)} 列資料")

    def generate(self, spec: dict, output_path: str = "chart.png") -> str:
        if self.df.empty:
            raise ValueError("目前沒有任何結構化資料可以用來畫圖。")

        df = self.df.copy()

        # 處理 filter
        if "filter" in spec and isinstance(spec["filter"], dict):
            for col, val in spec["filter"].items():
                if col in df.columns:
                    df = df[df[col] == val]

        if df.empty:
            raise ValueError("經過過濾後，沒有資料可以繪圖。請調整條件。")

        chart_type = spec.get("chart_type", "bar")
        x = spec.get("x")
        y = spec.get("y")

        if x not in df.columns or y not in df.columns:
            raise ValueError(f"欄位 {x} 或 {y} 不存在於資料中。")

        plt.figure(figsize=(7, 5))

        if chart_type == "bar":
            df.plot(kind="bar", x=x, y=y, legend=False)
            plt.xlabel(x)
            plt.ylabel(y)
        elif chart_type == "pie":
            series = df.set_index(x)[y]
            series.plot(kind="pie", autopct="%1.1f%%")
            plt.ylabel("")
        elif chart_type == "line":
            df.plot(kind="line", x=x, y=y, legend=False)
            plt.xlabel(x)
            plt.ylabel(y)
        else:
            raise ValueError(f"不支援的圖表類型: {chart_type}")

        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

        print(f"已產生圖表: {output_path}")
        return output_path
