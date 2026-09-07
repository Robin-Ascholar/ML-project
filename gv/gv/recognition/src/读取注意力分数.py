#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pandas as pd

# ==================== 配置 ====================
INPUT_TABLE = r"C:\Users\r9000\Desktop\毕设（无监督聚类）\Table_4_9_Attention_Results\Table_4_9_ESM2_attention_score_statistics.csv"

TOP_N = 4


def main():
    if not os.path.exists(INPUT_TABLE):
        print(f"❌ 文件不存在：{INPUT_TABLE}")
        return

    df = pd.read_csv(INPUT_TABLE, encoding="utf-8-sig")

    required_cols = [
        "GVP类型",
        "片段位置",
        "注意力分",
        "注意力标准差",
        "平均百分位"
    ]

    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"表格缺少字段：{col}")

    result_list = []

    for gvp_type, sub_df in df.groupby("GVP类型"):
        sub_df = sub_df.copy()

        sub_df = sub_df.sort_values(
            by="注意力分",
            ascending=False
        ).reset_index(drop=True)

        top_df = sub_df.head(TOP_N)

        result_list.append(top_df)

    result_df = pd.concat(result_list, ignore_index=True)

    result_df = result_df.sort_values(
        by=["GVP类型", "注意力分"],
        ascending=[True, False]
    ).reset_index(drop=True)

    result_df = result_df[
        [
            "GVP类型",
            "片段位置",
            "注意力分",
            "注意力标准差",
            "平均百分位"
        ]
    ]

    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 160)
    pd.set_option("display.max_colwidth", 50)

    print("\n" + "=" * 120)
    print("表4-9 候选片段ESM-2注意力分数统计结果（每种GVP取注意力分前4个）")
    print("=" * 120)
    print(result_df.to_string(index=False))
    print("=" * 120)

    print(f"原始候选片段数：{len(df)}")
    print(f"输出片段数：{len(result_df)}")


if __name__ == "__main__":
    main()
