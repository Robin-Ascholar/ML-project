#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pandas as pd


# =========================================================
# 路径配置
# =========================================================
INPUT_TABLE = (
    r"C:\Users\r9000\Desktop\毕设（无监督聚类）"
    r"\Table_4_11_ESMFold_API_Structure_Results"
    r"\tables"
    r"\Table_4_11_ESMFold_PyMOL_secondary_structure_summary.csv"
)

OUTPUT_TABLE = (
    r"C:\Users\r9000\Desktop\毕设（无监督聚类）"
    r"\Table_4_11_ESMFold_API_Structure_Results"
    r"\tables"
    r"\Table_4_11_ESMFold_PyMOL_secondary_structure_summary_top4.csv"
)

TOP_N = 4


# =========================================================
# GVP排序
# =========================================================
GVP_ORDER = [
    "GvpA", "GvpC", "GvpN", "GvpO",
    "GvpG", "GvpJ", "GvpK", "GvpP"
]


# =========================================================
# 主函数
# =========================================================
def main():
    if not os.path.exists(INPUT_TABLE):
        print(f"❌ 文件不存在：{INPUT_TABLE}")
        return

    df = pd.read_csv(INPUT_TABLE, encoding="utf-8-sig")

    required_cols = [
        "GVP类型",
        "片段位置",
        "结构预测数",
        "α螺旋比例",
        "β折叠比例",
        "无规则卷曲比例",
        "含二级结构片段占比"
    ]

    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"表格缺少字段：{col}")

    # 转数值，防止读入为字符串
    numeric_cols = [
        "结构预测数",
        "α螺旋比例",
        "β折叠比例",
        "无规则卷曲比例",
        "含二级结构片段占比"
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(
        subset=[
            "α螺旋比例",
            "β折叠比例",
            "无规则卷曲比例",
            "含二级结构片段占比"
        ]
    ).copy()

    # α螺旋 + β折叠，表示有明确二级结构残基比例
    df["二级结构残基比例"] = df["α螺旋比例"] + df["β折叠比例"]

    # 提取片段起始位置，方便最终排序
    df["片段起始位置"] = (
        df["片段位置"]
        .astype(str)
        .str.replace("–", "-", regex=False)
        .str.replace("—", "-", regex=False)
        .str.split("-")
        .str[0]
        .astype(int)
    )

    result_list = []

    for gvp_type, sub_df in df.groupby("GVP类型", sort=False):
        sub_df = sub_df.copy()

        # 先按结构强度排序，再取前4
        sub_df = sub_df.sort_values(
            by=[
                "含二级结构片段占比",
                "二级结构残基比例",
                "结构预测数"
            ],
            ascending=[False, False, False]
        ).reset_index(drop=True)

        top_df = sub_df.head(TOP_N)

        result_list.append(top_df)

    if not result_list:
        print("❌ 没有生成任何结果")
        return

    result_df = pd.concat(result_list, ignore_index=True)

    # 按指定GVP顺序排列
    gvp_order_map = {
        gvp: i for i, gvp in enumerate(GVP_ORDER)
    }

    result_df["GVP排序"] = result_df["GVP类型"].map(gvp_order_map)

    result_df = result_df.sort_values(
        by=[
            "GVP排序",
            "含二级结构片段占比",
            "二级结构残基比例",
            "片段起始位置"
        ],
        ascending=[True, False, False, True]
    ).reset_index(drop=True)

    # 最终表格字段
    final_cols = [
        "GVP类型",
        "片段位置",
        "结构预测数",
        "α螺旋比例",
        "β折叠比例",
        "无规则卷曲比例",
        "含二级结构片段占比"
    ]

    result_df = result_df[final_cols]

    result_df.to_csv(
        OUTPUT_TABLE,
        index=False,
        encoding="utf-8-sig"
    )

    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 180)
    pd.set_option("display.max_colwidth", 50)

    print("\n" + "=" * 120)
    print("表4-11 候选片段ESMFold结构预测与PyMOL二级结构统计结果（每种GVP取前4个）")
    print("=" * 120)
    print(result_df.to_string(index=False))
    print("=" * 120)

    print(f"原始候选片段数：{len(df)}")
    print(f"输出候选片段数：{len(result_df)}")
    print(f"✅ 前4表格已保存至：{OUTPUT_TABLE}")


if __name__ == "__main__":
    main()
