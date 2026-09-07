#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np


# =========================================================
# 路径配置
# =========================================================
OUTPUT_ROOT = r"C:\Users\r9000\Desktop\毕设（无监督聚类）\GVP_Four_Type_Clustering_Results"

FINAL_OUTPUT_DIR = os.path.join(
    OUTPUT_ROOT,
    "Final_Passed_NonClose_Candidates"
)

os.makedirs(FINAL_OUTPUT_DIR, exist_ok=True)


# =========================================================
# GVP配置
# =========================================================
TARGET_GVPS = [
    "gvpa", "gvpc", "gvpn", "gvpo",
    "gvpg", "gvpj", "gvpk", "gvpp"
]

GVP_WINDOW_CONFIG = {
    "gvpa": 30,
    "gvpc": 35,
    "gvpn": 40,
    "gvpo": 50,
    "gvpg": 25,
    "gvpj": 35,
    "gvpk": 25,
    "gvpp": 40
}


# =========================================================
# 筛选参数
# =========================================================
CV_PASS_THRESHOLD = 0.6
SPECIES_PERCENT_THRESHOLD = 20.0

# 距离 < 5 aa 才删除后面的；距离 = 5 aa 不删
CLOSE_DISTANCE_THRESHOLD = 5

# 每种 GVP 最终展示前4个
TOP_N = 4


# =========================================================
# 工具函数
# =========================================================
def format_gvp_name(gvp_type):
    return f"Gvp{gvp_type[-1].upper()}"


def normalize_species_percent(series):
    """
    兼容两种跨物种占比格式：
    0.55 表示 55%
    55 表示 55%

    最终统一为百分数形式，例如 55.09。
    """
    series = pd.to_numeric(series, errors="coerce")

    if series.dropna().empty:
        return series

    if series.dropna().max() <= 1.5:
        return series * 100

    return series


def get_candidate_file(gvp_type):
    """
    优先读取原始 candidate_cluster_position_stats.csv。
    """
    candidate_files = [
        os.path.join(
            OUTPUT_ROOT,
            gvp_type,
            f"{gvp_type}_candidate_cluster_position_stats.csv"
        ),
        os.path.join(
            OUTPUT_ROOT,
            gvp_type,
            f"{gvp_type}_filtered_candidate_cluster_position_stats.csv"
        ),
        os.path.join(
            OUTPUT_ROOT,
            gvp_type,
            f"{gvp_type}_cluster_position_stats.csv"
        ),
        os.path.join(
            OUTPUT_ROOT,
            gvp_type,
            f"{gvp_type}_all_cluster_position_stats.csv"
        )
    ]

    for path in candidate_files:
        if os.path.exists(path):
            return path

    return None


def region_to_string(start, end):
    return f"{int(start)}-{int(end)}"


# =========================================================
# 核心：从前往后删除距离过近的后一个
# =========================================================
def remove_close_candidates_forward(df, gvp_type):
    """
    从前往后删除距离过近的候选区域。

    规则：
    1. 删除前先对 mean_start_position 向下取整，得到 start_floor；
    2. 按 start_floor 从小到大排序；
    3. 第一个候选区域默认保留；
    4. 当前候选区域和上一个已保留候选区域比较；
    5. 如果 start_floor 距离 < 5 aa，删除当前候选区域；
    6. 如果 start_floor 距离 >= 5 aa，保留当前候选区域；
    7. 不考虑窗口重叠。
    """

    window = GVP_WINDOW_CONFIG[gvp_type]

    df = df.copy()

    df["start_floor"] = np.floor(
        pd.to_numeric(df["mean_start_position"], errors="coerce")
    ).astype(int)

    df["end_floor"] = df["start_floor"] + window

    df = df.sort_values(
        by=["start_floor", "mean_start_position"]
    ).reset_index(drop=True)

    kept_rows = []
    removed_rows = []

    for _, row in df.iterrows():
        current = row.to_dict()

        current_start = int(current["start_floor"])
        current_end = int(current["end_floor"])

        if not kept_rows:
            kept_rows.append(current)
            continue

        last_kept = kept_rows[-1]

        last_start = int(last_kept["start_floor"])
        last_end = int(last_kept["end_floor"])

        distance = current_start - last_start

        # 关键：小于5才删，等于5不删
        if distance < CLOSE_DISTANCE_THRESHOLD:
            removed_rows.append({
                "原始GVP类型": gvp_type,
                "GVP类型": format_gvp_name(gvp_type),

                "被删除cluster_id": current["cluster_id"],
                "被删除片段位置": region_to_string(current_start, current_end),
                "被删除平均起始位置": round(float(current["mean_start_position"]), 4),
                "被删除取整起始位置": current_start,
                "被删除CV": round(float(current["cv_start_position"]), 4),
                "被删除跨物种占比": round(float(current["species_percent"]) / 100, 4),

                "保留cluster_id": last_kept["cluster_id"],
                "保留片段位置": region_to_string(last_start, last_end),
                "保留平均起始位置": round(float(last_kept["mean_start_position"]), 4),
                "保留取整起始位置": last_start,
                "保留CV": round(float(last_kept["cv_start_position"]), 4),
                "保留跨物种占比": round(float(last_kept["species_percent"]) / 100, 4),

                "取整后起始位置距离": distance,
                "删除原因": "先向下取整；从前往后比较；与上一个已保留候选区域距离<5aa，删除当前后面的候选区域"
            })
        else:
            kept_rows.append(current)

    kept_df = pd.DataFrame(kept_rows)
    removed_df = pd.DataFrame(removed_rows)

    if not kept_df.empty:
        kept_df = kept_df.sort_values(
            by=["start_floor", "mean_start_position"]
        ).reset_index(drop=True)

    return kept_df, removed_df


# =========================================================
# 生成最终候选区域表
# =========================================================
def generate_final_candidate_tables():
    all_final_rows = []
    all_removed_rows = []

    gvp_order = {
        gvp: idx
        for idx, gvp in enumerate(TARGET_GVPS)
    }

    for gvp_type in TARGET_GVPS:
        file_path = get_candidate_file(gvp_type)

        if file_path is None:
            print(f"跳过 {gvp_type}，未找到候选簇文件")
            continue

        print(f"\n读取 {gvp_type}: {file_path}")

        df = pd.read_csv(file_path, encoding="utf-8-sig")

        if df.empty:
            print(f"跳过 {gvp_type}，文件为空")
            continue

        required_cols = [
            "cluster_id",
            "mean_start_position",
            "cv_start_position"
        ]

        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"{gvp_type} 文件缺少字段：{col}")

        if "species_percent" not in df.columns:
            if "species_count" in df.columns and "total_species_in_dataset" in df.columns:
                df["species_percent"] = (
                    df["species_count"].astype(float)
                    / df["total_species_in_dataset"].astype(float)
                    * 100
                )
            else:
                raise ValueError(
                    f"{gvp_type} 缺少 species_percent，"
                    "且无法通过 species_count / total_species_in_dataset 计算"
                )

        df["mean_start_position"] = pd.to_numeric(
            df["mean_start_position"],
            errors="coerce"
        )

        df["cv_start_position"] = pd.to_numeric(
            df["cv_start_position"],
            errors="coerce"
        )

        df["species_percent"] = normalize_species_percent(
            df["species_percent"]
        )

        if "sample_count" not in df.columns:
            df["sample_count"] = 0

        df = df.dropna(
            subset=[
                "cluster_id",
                "mean_start_position",
                "cv_start_position",
                "species_percent"
            ]
        ).copy()

        if df.empty:
            print(f"{gvp_type} 关键字段为空")
            continue

        # =================================================
        # 统一重新筛选“通过”候选区域
        # 不依赖旧文件里的 candidate_result
        # =================================================
        passed_df = df[
            (df["cv_start_position"] < CV_PASS_THRESHOLD)
            & (df["species_percent"] >= SPECIES_PERCENT_THRESHOLD)
        ].copy()

        if passed_df.empty:
            print(f"{gvp_type} 没有通过候选区域")
            continue

        # =================================================
        # 先删太近，再排序取 top4
        # =================================================
        final_df, removed_df = remove_close_candidates_forward(
            passed_df,
            gvp_type
        )

        if not removed_df.empty:
            all_removed_rows.append(removed_df)

        # 每个GVP保存一份详细最终候选区域
        per_gvp_detail_path = os.path.join(
            FINAL_OUTPUT_DIR,
            f"{gvp_type}_passed_nonclose_candidates_detail.csv"
        )

        final_df.to_csv(
            per_gvp_detail_path,
            index=False,
            encoding="utf-8-sig"
        )

        print(
            f"{format_gvp_name(gvp_type)} 通过候选数：{len(passed_df)}，"
            f"距离<5去重后：{len(final_df)}"
        )

        # 调试输出：方便你检查 GvpG 是否正确
        debug_cols = [
            "cluster_id",
            "mean_start_position",
            "start_floor",
            "end_floor",
            "cv_start_position",
            "species_percent"
        ]

        print("去重后保留候选：")
        print(final_df[debug_cols].to_string(index=False))

        for _, row in final_df.iterrows():
            start = int(row["start_floor"])
            end = int(row["end_floor"])

            all_final_rows.append({
                "GVP排序": gvp_order[gvp_type],
                "原始GVP类型": gvp_type,
                "GVP类型": format_gvp_name(gvp_type),
                "cluster_id": int(row["cluster_id"]),
                "片段位置": region_to_string(start, end),
                "平均起始位置": round(float(row["mean_start_position"]), 2),
                "窗口长度": GVP_WINDOW_CONFIG[gvp_type],
                "CV": round(float(row["cv_start_position"]), 4),
                "跨物种占比": round(float(row["species_percent"]) / 100, 4),
                "start_floor": start,
                "end_floor": end,
                "sample_count": int(row.get("sample_count", 0))
            })

    if not all_final_rows:
        print("未生成任何最终候选区域")
        return None, None

    all_df = pd.DataFrame(all_final_rows)

    # =====================================================
    # 删完后，再按每个GVP内跨物种占比排序
    # =====================================================
    all_df = all_df.sort_values(
        by=["GVP排序", "跨物种占比"],
        ascending=[True, False]
    ).reset_index(drop=True)

    top4_df = (
        all_df
        .groupby("原始GVP类型", sort=False)
        .head(TOP_N)
        .reset_index(drop=True)
    )

    display_cols = [
        "GVP类型",
        "片段位置",
        "平均起始位置",
        "窗口长度",
        "CV",
        "跨物种占比"
    ]

    all_display_df = all_df[display_cols]
    top4_display_df = top4_df[display_cols]

    # =====================================================
    # 保存文件
    # =====================================================
    all_output_path = os.path.join(
        FINAL_OUTPUT_DIR,
        "Passed_Candidate_NonClose_All.csv"
    )

    top4_output_path = os.path.join(
        FINAL_OUTPUT_DIR,
        "Passed_Candidate_NonClose_Top4_By_Species_Ratio.csv"
    )

    detail_output_path = os.path.join(
        FINAL_OUTPUT_DIR,
        "Passed_Candidate_NonClose_All_Detail_For_Downstream.csv"
    )

    removed_output_path = os.path.join(
        FINAL_OUTPUT_DIR,
        "Removed_Close_Candidates.csv"
    )

    all_display_df.to_csv(
        all_output_path,
        index=False,
        encoding="utf-8-sig"
    )

    top4_display_df.to_csv(
        top4_output_path,
        index=False,
        encoding="utf-8-sig"
    )

    all_df.to_csv(
        detail_output_path,
        index=False,
        encoding="utf-8-sig"
    )

    if all_removed_rows:
        removed_all_df = pd.concat(
            all_removed_rows,
            ignore_index=True
        )
    else:
        removed_all_df = pd.DataFrame()

    removed_all_df.to_csv(
        removed_output_path,
        index=False,
        encoding="utf-8-sig"
    )

    # =====================================================
    # 终端输出
    # =====================================================
    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 180)
    pd.set_option("display.max_colwidth", 50)

    print("\n" + "=" * 120)
    print("最终通过候选区域完整表：距离<5aa已删除后面的")
    print("=" * 120)
    print(all_display_df.to_string(index=False))

    print("\n" + "=" * 120)
    print("最终通过候选区域 Top4 表：删完后按跨物种占比排序")
    print("=" * 120)
    print(top4_display_df.to_string(index=False))

    print("\n保存文件：")
    print(f"完整最终候选表：{all_output_path}")
    print(f"Top4最终候选表：{top4_output_path}")
    print(f"下游脚本用详细表：{detail_output_path}")
    print(f"被删除候选记录：{removed_output_path}")

    return top4_display_df, all_display_df


if __name__ == "__main__":
    generate_final_candidate_tables()
