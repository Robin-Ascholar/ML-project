#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =========================================================
# 路径配置
# =========================================================

# 35M最终去重候选区域表
BASELINE_35M_FINAL_TABLE = (
    r"C:\Users\r9000\Desktop\毕设（无监督聚类）"
    r"\GVP_Four_Type_Clustering_Results"
    r"\Final_Passed_NonClose_Candidates"
    r"\Passed_Candidate_NonClose_All_Detail_For_Downstream.csv"
)

# 150M / 650M 已经跑完并保存的结果根目录
MULTI_MODEL_RESULT_ROOT = (
    r"C:\Users\r9000\Desktop\毕设（无监督聚类）"
    r"\GVP_ESM2_Multi_Model_Comparison_All8"
)

# 本统计脚本输出目录
OUTPUT_DIR = os.path.join(
    MULTI_MODEL_RESULT_ROOT,
    "saved_result_comparison_summary"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# =========================================================
# GVP配置
# =========================================================
GVP_TYPES = [
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

MODEL_LABELS = ["35M", "150M", "650M"]

# 匹配阈值：起始位置距离 <= 5 aa 认为是同一候选区域
MATCH_DISTANCE_THRESHOLD = 10

# 是否允许区间重叠也算匹配
# 如果你只想严格按起始位置判断，保持 False
# 如果你想“窗口有交集就算匹配”，改为 True
USE_INTERVAL_OVERLAP_FOR_MATCH = False


# =========================================================
# 工具函数
# =========================================================
def format_gvp_name(gvp_type):
    return f"Gvp{gvp_type[-1].upper()}"


def gvp_name_to_raw(gvp_name):
    return "gvp" + str(gvp_name)[-1].lower()


def parse_region(region):
    """
    解析 83-113 这种片段位置。
    """
    parts = str(region).strip().split("-")
    if len(parts) != 2:
        raise ValueError(f"片段位置格式错误：{region}")

    return int(float(parts[0])), int(float(parts[1]))


def region_to_string(start, end):
    return f"{int(start)}-{int(end)}"


def interval_overlap(a_start, a_end, b_start, b_end):
    return max(a_start, b_start) < min(a_end, b_end)


def normalize_species_percent(series):
    """
    兼容 0.55 和 55 两种格式。
    返回百分数形式，例如 55.09。
    """
    series = pd.to_numeric(series, errors="coerce")

    if series.dropna().empty:
        return series

    if series.dropna().max() <= 1.5:
        return series * 100

    return series


def read_csv_safely(path):
    try:
        return pd.read_csv(path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="utf-8")


# =========================================================
# 读取35M最终去重候选区域
# =========================================================
def load_35m_final_candidates(gvp_type):
    if not os.path.exists(BASELINE_35M_FINAL_TABLE):
        raise FileNotFoundError(
            f"35M最终去重候选区域表不存在：{BASELINE_35M_FINAL_TABLE}"
        )

    df = read_csv_safely(BASELINE_35M_FINAL_TABLE)

    if df.empty:
        return []

    if "原始GVP类型" in df.columns:
        sub_df = df[df["原始GVP类型"] == gvp_type].copy()
    elif "GVP类型" in df.columns:
        sub_df = df[df["GVP类型"] == format_gvp_name(gvp_type)].copy()
    else:
        raise ValueError("35M最终表缺少 原始GVP类型 或 GVP类型 字段")

    if sub_df.empty:
        print(f"⚠️ {format_gvp_name(gvp_type)} 在35M最终去重表中没有候选区域")
        return []

    if "start_floor" not in sub_df.columns or "end_floor" not in sub_df.columns:
        if "片段位置" not in sub_df.columns:
            raise ValueError("35M最终表缺少 start_floor/end_floor，也缺少 片段位置")

        starts = []
        ends = []
        for region in sub_df["片段位置"]:
            s, e = parse_region(region)
            starts.append(s)
            ends.append(e)

        sub_df["start_floor"] = starts
        sub_df["end_floor"] = ends

    if "cluster_id" not in sub_df.columns:
        sub_df["cluster_id"] = [
            f"{gvp_type}_{region}"
            for region in sub_df["片段位置"]
        ]

    if "平均起始位置" in sub_df.columns:
        sub_df["mean_start_position"] = pd.to_numeric(
            sub_df["平均起始位置"],
            errors="coerce"
        )
    elif "mean_start_position" in sub_df.columns:
        sub_df["mean_start_position"] = pd.to_numeric(
            sub_df["mean_start_position"],
            errors="coerce"
        )
    else:
        sub_df["mean_start_position"] = pd.to_numeric(
            sub_df["start_floor"],
            errors="coerce"
        )

    candidates = []

    for _, row in sub_df.iterrows():
        start = int(row["start_floor"])
        end = int(row["end_floor"])
        region = row["片段位置"] if "片段位置" in row else region_to_string(start, end)

        candidates.append({
            "model": "35M",
            "gvp_raw": gvp_type,
            "GVP类型": format_gvp_name(gvp_type),
            "cluster_id": row["cluster_id"],
            "region": region,
            "start": start,
            "end": end,
            "mean_start_position": float(row["mean_start_position"]),
            "source": BASELINE_35M_FINAL_TABLE
        })

    candidates = sorted(candidates, key=lambda x: x["start"])

    print(f"{format_gvp_name(gvp_type)} 35M最终候选数：{len(candidates)}")

    return candidates


# =========================================================
# 读取150M/650M已经保存的候选区域
# =========================================================
def get_model_candidate_file(model_label, gvp_type):
    """
    读取之前多模型脚本保存的候选区域文件。
    """

    candidate_files = [
        os.path.join(
            MULTI_MODEL_RESULT_ROOT,
            model_label,
            gvp_type,
            f"{gvp_type}_{model_label}_passed_candidate_clusters.csv"
        ),
        os.path.join(
            MULTI_MODEL_RESULT_ROOT,
            model_label,
            gvp_type,
            f"{gvp_type}_{model_label}_filtered_candidate_cluster_position_stats.csv"
        ),
        os.path.join(
            MULTI_MODEL_RESULT_ROOT,
            model_label,
            gvp_type,
            f"{gvp_type}_{model_label}_passed_candidate_cluster_position_stats.csv"
        ),
        os.path.join(
            MULTI_MODEL_RESULT_ROOT,
            model_label,
            gvp_type,
            f"{gvp_type}_{model_label}_all_cluster_position_stats.csv"
        )
    ]

    for path in candidate_files:
        if os.path.exists(path):
            return path

    return None


def load_model_saved_candidates(model_label, gvp_type):
    """
    读取150M/650M已经保存的候选区域。

    注意：
    这里不重新跑模型，也不重新聚类。
    默认认为这些文件已经是该模型对应的候选结果。
    """

    if model_label == "35M":
        return load_35m_final_candidates(gvp_type)

    path = get_model_candidate_file(model_label, gvp_type)

    if path is None:
        print(f"⚠️ 未找到 {format_gvp_name(gvp_type)} {model_label} 候选区域文件")
        return []

    print(f"读取 {format_gvp_name(gvp_type)} {model_label}: {path}")

    df = read_csv_safely(path)

    if df.empty:
        return []

    # 如果读到的是 all_cluster_position_stats，则只保留通过候选
    if "candidate_result" in df.columns:
        df = df[df["candidate_result"].astype(str) == "通过"].copy()

    if df.empty:
        return []

    if "species_percent" in df.columns:
        df["species_percent"] = normalize_species_percent(df["species_percent"])

    if "start_floor" in df.columns and "end_floor" in df.columns:
        df["start"] = pd.to_numeric(df["start_floor"], errors="coerce")
        df["end"] = pd.to_numeric(df["end_floor"], errors="coerce")

    elif "片段位置" in df.columns:
        starts = []
        ends = []
        for region in df["片段位置"]:
            s, e = parse_region(region)
            starts.append(s)
            ends.append(e)

        df["start"] = starts
        df["end"] = ends

    elif "mean_start_position" in df.columns:
        df["mean_start_position"] = pd.to_numeric(
            df["mean_start_position"],
            errors="coerce"
        )
        df["start"] = np.floor(df["mean_start_position"]).astype(int)
        df["end"] = df["start"] + GVP_WINDOW_CONFIG[gvp_type]

    else:
        raise ValueError(
            f"{model_label} {gvp_type} 候选文件缺少 start_floor/end_floor、片段位置 或 mean_start_position"
        )

    if "mean_start_position" not in df.columns:
        df["mean_start_position"] = df["start"]

    if "cluster_id" not in df.columns:
        df["cluster_id"] = [
            f"{model_label}_{gvp_type}_{i}"
            for i in range(len(df))
        ]

    df = df.dropna(
        subset=["start", "end", "mean_start_position"]
    ).copy()

    candidates = []

    for _, row in df.iterrows():
        start = int(row["start"])
        end = int(row["end"])

        if "片段位置" in row and not pd.isna(row["片段位置"]):
            region = str(row["片段位置"])
        else:
            region = region_to_string(start, end)

        candidates.append({
            "model": model_label,
            "gvp_raw": gvp_type,
            "GVP类型": format_gvp_name(gvp_type),
            "cluster_id": row["cluster_id"],
            "region": region,
            "start": start,
            "end": end,
            "mean_start_position": float(row["mean_start_position"]),
            "source": path
        })

    candidates = sorted(candidates, key=lambda x: x["start"])

    print(f"{format_gvp_name(gvp_type)} {model_label} 候选数：{len(candidates)}")

    return candidates


# =========================================================
# 匹配逻辑
# =========================================================
def is_candidate_match(base, other):
    """
    默认匹配标准：
    1. 起始位置距离 <= 5 aa；
    2. 如果 USE_INTERVAL_OVERLAP_FOR_MATCH=True，则区间重叠也算匹配。
    """

    start_distance = abs(int(other["start"]) - int(base["start"]))

    if start_distance <= MATCH_DISTANCE_THRESHOLD:
        return True, "start_distance<=5", start_distance

    if USE_INTERVAL_OVERLAP_FOR_MATCH:
        if interval_overlap(base["start"], base["end"], other["start"], other["end"]):
            return True, "interval_overlap", start_distance

    return False, "", start_distance


def match_to_baseline(base_candidates, other_candidates):
    """
    将150M/650M候选区域一对一匹配到35M候选区域。
    一个35M候选区域最多匹配一个新模型候选区域；
    一个新模型候选区域也最多使用一次。
    """

    if len(base_candidates) == 0:
        return {
            "matched_count": 0,
            "ratio": np.nan,
            "matched_pairs": []
        }

    possible_pairs = []

    for i, base in enumerate(base_candidates):
        for j, other in enumerate(other_candidates):
            matched, match_type, start_distance = is_candidate_match(base, other)

            if matched:
                possible_pairs.append({
                    "base_index": i,
                    "other_index": j,
                    "start_distance": start_distance,
                    "match_type": match_type,
                    "base": base,
                    "other": other
                })

    possible_pairs = sorted(
        possible_pairs,
        key=lambda x: x["start_distance"]
    )

    used_base = set()
    used_other = set()
    matched_pairs = []

    for pair in possible_pairs:
        i = pair["base_index"]
        j = pair["other_index"]

        if i in used_base or j in used_other:
            continue

        used_base.add(i)
        used_other.add(j)

        base = pair["base"]
        other = pair["other"]

        matched_pairs.append({
            "GVP类型": base["GVP类型"],
            "比较模型": f"{other['model']} vs 35M",

            "35M_cluster_id": base["cluster_id"],
            "35M_region": base["region"],
            "35M_start": base["start"],
            "35M_end": base["end"],

            "other_cluster_id": other["cluster_id"],
            "other_region": other["region"],
            "other_start": other["start"],
            "other_end": other["end"],

            "start_distance": pair["start_distance"],
            "match_type": pair["match_type"]
        })

    matched_count = len(matched_pairs)
    ratio = matched_count / len(base_candidates) * 100

    return {
        "matched_count": matched_count,
        "ratio": ratio,
        "matched_pairs": matched_pairs
    }


# =========================================================
# 统计汇总
# =========================================================
def generate_summary_and_detail():
    all_candidates = {}
    summary_rows = []
    detail_rows = []

    for gvp_type in GVP_TYPES:
        all_candidates.setdefault(gvp_type, {})

        base_35m = load_model_saved_candidates("35M", gvp_type)
        cand_150m = load_model_saved_candidates("150M", gvp_type)
        cand_650m = load_model_saved_candidates("650M", gvp_type)

        all_candidates[gvp_type]["35M"] = base_35m
        all_candidates[gvp_type]["150M"] = cand_150m
        all_candidates[gvp_type]["650M"] = cand_650m

        res_150m = match_to_baseline(base_35m, cand_150m)
        res_650m = match_to_baseline(base_35m, cand_650m)

        base_count = len(base_35m)

        summary_rows.append({
            "GVP类型": format_gvp_name(gvp_type),

            "35M候选区域数": base_count,
            "150M候选区域数": len(cand_150m),
            "650M候选区域数": len(cand_650m),

            "35M相对比例(%)": 100.0 if base_count > 0 else np.nan,

            "150M匹配35M数量": res_150m["matched_count"],
            "150M相对35M比例(%)": (
                round(res_150m["ratio"], 2)
                if not pd.isna(res_150m["ratio"])
                else np.nan
            ),

            "650M匹配35M数量": res_650m["matched_count"],
            "650M相对35M比例(%)": (
                round(res_650m["ratio"], 2)
                if not pd.isna(res_650m["ratio"])
                else np.nan
            )
        })

        detail_rows.extend(res_150m["matched_pairs"])
        detail_rows.extend(res_650m["matched_pairs"])

    summary_df = pd.DataFrame(summary_rows)
    detail_df = pd.DataFrame(detail_rows)

    summary_path = os.path.join(
        OUTPUT_DIR,
        "Three_Model_Candidate_Comparison_From_Saved_Summary.csv"
    )

    detail_path = os.path.join(
        OUTPUT_DIR,
        "Three_Model_Candidate_Comparison_From_Saved_Details.csv"
    )

    summary_df.to_csv(
        summary_path,
        index=False,
        encoding="utf-8-sig"
    )

    detail_df.to_csv(
        detail_path,
        index=False,
        encoding="utf-8-sig"
    )

    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_colwidth", 50)

    print("\n" + "=" * 140)
    print("三种ESM-2模型候选区域一致性统计结果")
    print("=" * 140)
    print(summary_df.to_string(index=False))
    print("=" * 140)

    print(f"✅ 汇总表已保存：{summary_path}")
    print(f"✅ 匹配明细已保存：{detail_path}")

    return summary_df, detail_df


# =========================================================
# 出图
# =========================================================
def plot_grouped_bar(summary_df):
    if summary_df.empty:
        print("⚠️ 汇总表为空，跳过绘图")
        return

    x_labels = summary_df["GVP类型"].tolist()

    y_35m = summary_df["35M相对比例(%)"].fillna(0).astype(float).values
    y_150m = summary_df["150M相对35M比例(%)"].fillna(0).astype(float).values
    y_650m = summary_df["650M相对35M比例(%)"].fillna(0).astype(float).values

    x = np.arange(len(x_labels))
    width = 0.24

    colors = {
        "35M": "#3B5B92",
        "150M": "#4E8F5A",
        "650M": "#B44745"
    }

    plt.figure(figsize=(14, 7))
    ax = plt.gca()

    bars_35m = ax.bar(
        x - width,
        y_35m,
        width,
        label="ESM-2 35M",
        color=colors["35M"],
        edgecolor="black",
        linewidth=0.4
    )

    bars_150m = ax.bar(
        x,
        y_150m,
        width,
        label="ESM-2 150M",
        color=colors["150M"],
        edgecolor="black",
        linewidth=0.4
    )

    bars_650m = ax.bar(
        x + width,
        y_650m,
        width,
        label="ESM-2 650M",
        color=colors["650M"],
        edgecolor="black",
        linewidth=0.4
    )

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=11)

    ax.set_ylabel(
        "Matched candidate regions relative to ESM-2 35M (%)",
        fontsize=12
    )

    ax.set_xlabel(
        "GVP type",
        fontsize=12
    )

    ax.set_title(
        "Consistency of Candidate Regions Across ESM-2 Models",
        fontsize=15,
        fontweight="bold"
    )

    ax.set_ylim(0, 115)

    ax.legend(
        frameon=False,
        fontsize=10,
        loc="upper right"
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.spines["left"].set_color("#4D4D4D")
    ax.spines["bottom"].set_color("#4D4D4D")

    ax.tick_params(
        axis="both",
        labelsize=10,
        colors="#333333",
        direction="out",
        length=4,
        width=0.8
    )

    def add_labels(bars):
        for bar in bars:
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + 1.5,
                f"{h:.1f}",
                ha="center",
                va="bottom",
                fontsize=8
            )

    add_labels(bars_35m)
    add_labels(bars_150m)
    add_labels(bars_650m)

    plt.tight_layout()

    png_path = os.path.join(
        OUTPUT_DIR,
        "All_GVPs_Three_Model_Candidate_Comparison_From_Saved.png"
    )

    pdf_path = os.path.join(
        OUTPUT_DIR,
        "All_GVPs_Three_Model_Candidate_Comparison_From_Saved.pdf"
    )

    plt.savefig(
        png_path,
        dpi=600,
        bbox_inches="tight"
    )

    plt.savefig(
        pdf_path,
        bbox_inches="tight"
    )

    plt.close()

    print(f"✅ 图片已保存：{png_path}")
    print(f"✅ PDF已保存：{pdf_path}")


# =========================================================
# 主函数
# =========================================================
def main():
    print("开始读取已保存候选区域并统计三模型一致性")
    print(f"35M基准文件：{BASELINE_35M_FINAL_TABLE}")
    print(f"150M/650M结果目录：{MULTI_MODEL_RESULT_ROOT}")
    print(f"输出目录：{OUTPUT_DIR}")

    summary_df, detail_df = generate_summary_and_detail()

    plot_grouped_bar(summary_df)

    print("\n🎉 全部完成！")


if __name__ == "__main__":
    main()
