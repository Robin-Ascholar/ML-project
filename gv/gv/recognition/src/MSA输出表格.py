#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pandas as pd

# ==================== 配置 ====================
INPUT_TABLE = r"C:\Users\r9000\Desktop\毕设（无监督聚类）\Table_4_10_MSA_Results\Table_4_10_MAFFT_conservation_analysis.csv"

OUTPUT_TABLE = r"C:\Users\r9000\Desktop\毕设（无监督聚类）\Table_4_10_MSA_Results\Table_4_10_MAFFT_conservation_analysis_top4_nonclose.csv"

TOP_N = 4

# 相邻片段起始位置距离小于5 aa时，保留后面那个
CLOSE_DISTANCE_THRESHOLD = 5


def parse_start(pos_str):
    """
    从 '25-55' 这种片段位置中提取起始位置。
    """
    return int(str(pos_str).split("-")[0])


def remove_close_fragments(sub_df):
    """
    对同一种GVP内部的片段进行距离去重。
    如果两个片段起始位置距离 < 5 aa，则保留后面那个。
    """
    sub_df = sub_df.copy()
    sub_df["start_pos"] = sub_df["片段位置"].apply(parse_start)

    sub_df = sub_df.sort_values("start_pos").reset_index(drop=True)

    filtered_rows = []
    removed_rows = []

    for _, row in sub_df.iterrows():
        row_dict = row.to_dict()

        if not filtered_rows:
            filtered_rows.append(row_dict)
            continue

        last = filtered_rows[-1]

        last_start = int(last["start_pos"])
        current_start = int(row_dict["start_pos"])

        if current_start - last_start < CLOSE_DISTANCE_THRESHOLD:
            removed_rows.append({
                "GVP类型": row_dict["GVP类型"],
                "被舍弃片段": last["片段位置"],
                "保留片段": row_dict["片段位置"],
                "距离": current_start - last_start,
                "原因": f"起始位置距离小于{CLOSE_DISTANCE_THRESHOLD}aa，保留后面片段"
            })

            filtered_rows[-1] = row_dict

        else:
            filtered_rows.append(row_dict)

    filtered_df = pd.DataFrame(filtered_rows)

    if "start_pos" in filtered_df.columns:
        filtered_df = filtered_df.drop(columns=["start_pos"])

    removed_df = pd.DataFrame(removed_rows)

    return filtered_df, removed_df


def main():
    if not os.path.exists(INPUT_TABLE):
        print(f"❌ 文件不存在：{INPUT_TABLE}")
        return

    df = pd.read_csv(INPUT_TABLE, encoding="utf-8-sig")

    required_cols = [
        "GVP类型",
        "片段位置",
        "参与比对序列数",
        "平均保守性分数",
        "保守性标准差",
        "平均百分位"
    ]

    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"表格缺少字段：{col}")

    result_list = []
    removed_list = []

    for gvp_type, sub_df in df.groupby("GVP类型"):
        sub_df = sub_df.copy()

        # 1. 先去掉相邻太近的片段
        nonclose_df, removed_df = remove_close_fragments(sub_df)

        if not removed_df.empty:
            removed_list.append(removed_df)

        # 2. 再按平均保守性分数排序
        nonclose_df = nonclose_df.sort_values(
            by="平均保守性分数",
            ascending=False
        ).reset_index(drop=True)

        # 3. 每种GVP取前4个
        top_df = nonclose_df.head(TOP_N)

        result_list.append(top_df)

    result_df = pd.concat(result_list, ignore_index=True)

    result_df = result_df.sort_values(
        by=["GVP类型", "平均保守性分数"],
        ascending=[True, False]
    ).reset_index(drop=True)

    result_df = result_df[
        [
            "GVP类型",
            "片段位置",
            "参与比对序列数",
            "平均保守性分数",
            "保守性标准差",
            "平均百分位"
        ]
    ]

    result_df.to_csv(
        OUTPUT_TABLE,
        index=False,
        encoding="utf-8-sig"
    )

    removed_output = OUTPUT_TABLE.replace(".csv", "_removed_close_fragments.csv")

    if removed_list:
        removed_all = pd.concat(removed_list, ignore_index=True)
        removed_all.to_csv(
            removed_output,
            index=False,
            encoding="utf-8-sig"
        )
    else:
        removed_all = pd.DataFrame()

    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 180)
    pd.set_option("display.max_colwidth", 50)

    print("\n" + "=" * 120)
    print("表4-10 候选片段MAFFT保守性分析结果（去除相邻<5aa后，每种GVP取前4个）")
    print("=" * 120)
    print(result_df.to_string(index=False))
    print("=" * 120)

    print(f"原始候选片段数：{len(df)}")
    print(f"去除相邻过近后输出候选片段数：{len(result_df)}")
    print(f"✅ 前4表格已保存至：{OUTPUT_TABLE}")

    if not removed_all.empty:
        print(f"✅ 被舍弃的相邻过近片段记录已保存至：{removed_output}")


if __name__ == "__main__":
    main()
