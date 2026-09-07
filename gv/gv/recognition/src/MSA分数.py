#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import subprocess
import warnings
from collections import Counter

import numpy as np
import pandas as pd


warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


# =========================================================
# 配置区域
# =========================================================
MAFFT_PATH = r"C:\Users\r9000\Desktop\mafft-win\mafft.bat"

DATA_ROOT = r"C:\Users\r9000\Desktop\毕设（无监督聚类）\新gvp"

# =========================================================
# 直接读取已经去重后的候选区域表
# 后续不再自己筛选、不再自己去重
# =========================================================
FINAL_CANDIDATE_TABLE = (
    r"C:\Users\r9000\Desktop\毕设（无监督聚类）"
    r"\GVP_Four_Type_Clustering_Results"
    r"\Final_Passed_NonClose_Candidates"
    r"\Passed_Candidate_NonClose_All_Detail_For_Downstream.csv"
)

OUTPUT_ROOT = r"C:\Users\r9000\Desktop\毕设（无监督聚类）\Table_4_10_MSA_Results_From_Final_Candidates"

MSA_DIR = os.path.join(OUTPUT_ROOT, "full_sequence_msa")
TABLE_DIR = os.path.join(OUTPUT_ROOT, "tables")

os.makedirs(OUTPUT_ROOT, exist_ok=True)
os.makedirs(MSA_DIR, exist_ok=True)
os.makedirs(TABLE_DIR, exist_ok=True)


GVP_TYPES = [
    "gvpa", "gvpc", "gvpn", "gvpo",
    "gvpg", "gvpj", "gvpk", "gvpp"
]


# =========================================================
# 基础工具函数
# =========================================================
def format_gvp_name(gvp_type):
    return f"Gvp{gvp_type[-1].upper()}"


def gvp_name_to_raw(gvp_name):
    """
    GvpA -> gvpa
    """
    return "gvp" + str(gvp_name)[-1].lower()


def clean_sequence(seq):
    valid_aa = set("ACDEFGHIKLMNPQRSTVWY")
    return "".join([
        c for c in str(seq).strip().upper()
        if c in valid_aa
    ])


def get_sequence_id(item, idx):
    return (
        item.get("unique_sequence_id")
        or item.get("id")
        or item.get("accession")
        or item.get("protein_id")
        or f"seq_{idx + 1}"
    )


def parse_region(region):
    """
    从 83-113 解析 start=83, end=113。
    """
    parts = str(region).strip().split("-")

    if len(parts) != 2:
        raise ValueError(f"片段位置格式错误：{region}")

    start = int(float(parts[0]))
    end = int(float(parts[1]))

    return start, end


def region_to_string(start, end):
    return f"{int(start)}-{int(end)}"


# =========================================================
# 读取最终去重候选区域
# =========================================================
def load_final_candidates_for_gvp(gvp):
    """
    直接读取最终去重候选区域表。

    不再执行：
    1. CV筛选
    2. 跨物种阈值筛选
    3. 5 aa去重
    4. 窗口重叠去重

    如果表中没有 start_floor / end_floor，则从“片段位置”解析。
    如果表中没有 cluster_id，则用片段位置生成 candidate_id。
    """

    if not os.path.exists(FINAL_CANDIDATE_TABLE):
        raise FileNotFoundError(
            f"最终去重候选区域表不存在：{FINAL_CANDIDATE_TABLE}"
        )

    df = pd.read_csv(FINAL_CANDIDATE_TABLE, encoding="utf-8-sig")

    if df.empty:
        return df

    if "原始GVP类型" in df.columns:
        sub_df = df[df["原始GVP类型"] == gvp].copy()

    elif "GVP类型" in df.columns:
        sub_df = df[df["GVP类型"] == format_gvp_name(gvp)].copy()

    else:
        raise ValueError("最终候选区域表缺少字段：原始GVP类型 或 GVP类型")

    if sub_df.empty:
        print(f"⚠️ {gvp.upper()} 在最终候选区域表中没有记录")
        return sub_df

    if "GVP类型" not in sub_df.columns:
        sub_df["GVP类型"] = format_gvp_name(gvp)

    if "原始GVP类型" not in sub_df.columns:
        sub_df["原始GVP类型"] = gvp

    if "片段位置" not in sub_df.columns:
        if "start_floor" in sub_df.columns and "end_floor" in sub_df.columns:
            sub_df["片段位置"] = sub_df.apply(
                lambda x: region_to_string(x["start_floor"], x["end_floor"]),
                axis=1
            )
        else:
            raise ValueError("最终候选区域表缺少字段：片段位置")

    if "start_floor" not in sub_df.columns or "end_floor" not in sub_df.columns:
        starts = []
        ends = []

        for region in sub_df["片段位置"]:
            s, e = parse_region(region)
            starts.append(s)
            ends.append(e)

        sub_df["start_floor"] = starts
        sub_df["end_floor"] = ends

    sub_df["start_floor"] = pd.to_numeric(
        sub_df["start_floor"],
        errors="coerce"
    )

    sub_df["end_floor"] = pd.to_numeric(
        sub_df["end_floor"],
        errors="coerce"
    )

    sub_df = sub_df.dropna(
        subset=["start_floor", "end_floor"]
    ).copy()

    sub_df["start_floor"] = sub_df["start_floor"].astype(int)
    sub_df["end_floor"] = sub_df["end_floor"].astype(int)

    if "cluster_id" not in sub_df.columns:
        sub_df["cluster_id"] = [
            f"{gvp}_{region}"
            for region in sub_df["片段位置"]
        ]

    sub_df = sub_df.sort_values(
        by=["start_floor", "end_floor"]
    ).reset_index(drop=True)

    return sub_df


# =========================================================
# MAFFT调用与FASTA读取
# =========================================================
def run_my_mafft(input_fasta, out_aln):
    cmd = f'"{MAFFT_PATH}" --auto "{input_fasta}"'

    with open(out_aln, "w", encoding="utf-8") as f:
        subprocess.run(
            cmd,
            stdout=f,
            stderr=subprocess.DEVNULL,
            shell=True,
            check=True
        )


def write_fasta(seq_records, fasta_path):
    with open(fasta_path, "w", encoding="utf-8") as f:
        for rec in seq_records:
            f.write(f">{rec['id']}\n")
            f.write(f"{rec['seq']}\n")


def read_fasta_alignment(aln_file):
    names = []
    seqs = []

    current_name = None
    current_seq = []

    with open(aln_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            if line.startswith(">"):
                if current_name is not None:
                    names.append(current_name)
                    seqs.append("".join(current_seq))

                current_name = line[1:]
                current_seq = []

            else:
                current_seq.append(line)

        if current_name is not None:
            names.append(current_name)
            seqs.append("".join(current_seq))

    return names, seqs


# =========================================================
# 读取原始序列
# =========================================================
def load_gvp_sequences(gvp):
    seq_path = os.path.join(
        DATA_ROOT,
        gvp,
        f"Gvp{gvp[-1].upper()}_sequences.json"
    )

    if not os.path.exists(seq_path):
        print(f"❌ 未找到序列文件：{seq_path}")
        return []

    with open(seq_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    records = []

    for idx, item in enumerate(data):
        seq = clean_sequence(item.get("sequence", ""))

        if len(seq) < 20:
            continue

        records.append({
            "id": get_sequence_id(item, idx),
            "seq": seq,
            "length": len(seq)
        })

    return records


# =========================================================
# 保守性分数计算
# =========================================================
def calc_msa_column_scores(aligned_seqs):
    """
    对MSA每一列计算保守性分数。
    分数 = 1 - Shannon entropy / log2(20)
    """
    valid_aa = set("ACDEFGHIKLMNPQRSTVWY")

    if not aligned_seqs:
        return []

    L = len(aligned_seqs[0])
    col_scores = []

    for i in range(L):
        col = [
            seq[i]
            for seq in aligned_seqs
            if seq[i] in valid_aa
        ]

        if not col:
            col_scores.append(0.0)
            continue

        cnt = Counter(col)

        p = np.array([
            v / len(col)
            for v in cnt.values()
        ])

        ent = -np.sum(p * np.log2(p))
        score = 1 - ent / np.log2(20)

        col_scores.append(float(score))

    return col_scores


def map_scores_to_reference(ref_aligned_seq, msa_col_scores):
    """
    将MSA列保守性分数映射回某条参考序列原始坐标。
    key: 原始序列0-based位置
    value: 保守性分数
    """
    valid_aa = set("ACDEFGHIKLMNPQRSTVWY")

    score_map = {}
    orig_pos = 0

    for msa_pos, aa in enumerate(ref_aligned_seq):
        if aa in valid_aa:
            score_map[orig_pos] = msa_col_scores[msa_pos]
            orig_pos += 1

    return score_map


# =========================================================
# 单个GVP执行MSA并统计候选区域保守性
# =========================================================
def run_one_gvp_msa(gvp):
    print("\n" + "=" * 90)
    print(f"正在进行 MAFFT 保守性分析：{gvp.upper()}")
    print("=" * 90)

    seq_records = load_gvp_sequences(gvp)

    if len(seq_records) < 2:
        print(f"⚠️ {gvp.upper()} 有效序列不足")
        return pd.DataFrame()

    candidates = load_final_candidates_for_gvp(gvp)

    if candidates.empty:
        print(f"⚠️ {gvp.upper()} 无最终候选区域")
        return pd.DataFrame()

    print(f"{gvp.upper()} 读取最终去重候选区域数：{len(candidates)}")

    fasta_path = os.path.join(
        MSA_DIR,
        f"{gvp}_full_sequences.fasta"
    )

    aln_path = os.path.join(
        MSA_DIR,
        f"{gvp}_full_sequences.aln"
    )

    write_fasta(seq_records, fasta_path)

    print(f"运行 MAFFT：{gvp.upper()} 全序列")
    run_my_mafft(fasta_path, aln_path)

    aln_names, aligned = read_fasta_alignment(aln_path)

    if not aligned:
        print(f"⚠️ {gvp.upper()} MSA结果为空")
        return pd.DataFrame()

    if len(aligned) != len(seq_records):
        print(
            f"⚠️ {gvp.upper()} MSA序列数与原始序列数不一致："
            f"{len(aligned)} vs {len(seq_records)}"
        )

    msa_col_scores = calc_msa_column_scores(aligned)

    all_results = []

    for seq_idx, rec in enumerate(seq_records):
        if seq_idx >= len(aligned):
            continue

        cons_map = map_scores_to_reference(
            aligned[seq_idx],
            msa_col_scores
        )

        all_scores = list(cons_map.values())

        if not all_scores:
            continue

        all_scores_arr = np.array(all_scores)
        seq_len = rec["length"]

        for _, row in candidates.iterrows():
            cluster_id = row["cluster_id"]

            start = int(row["start_floor"])
            end = int(row["end_floor"])

            if start < 0:
                start = 0

            if end > seq_len:
                end = seq_len

            original_len = int(row["end_floor"]) - int(row["start_floor"])

            if end - start < original_len // 2:
                continue

            fragment_scores = []

            for pos in range(start, end):
                if pos in cons_map:
                    fragment_scores.append(cons_map[pos])

            if not fragment_scores:
                continue

            avg_fragment_score = float(np.mean(fragment_scores))

            percentile = (
                np.sum(all_scores_arr <= avg_fragment_score)
                / len(all_scores_arr)
                * 100
            )

            all_results.append({
                "GVP类型": row["GVP类型"],
                "原始GVP类型": gvp,
                "cluster_id": cluster_id,
                "seq_id": rec["id"],

                "片段位置": row["片段位置"],
                "fragment_start": start,
                "fragment_end": end,

                "fragment_avg_score": round(avg_fragment_score, 4),
                "percentile_rank": round(float(percentile), 2)
            })

    detail_df = pd.DataFrame(all_results)

    detail_path = os.path.join(
        TABLE_DIR,
        f"{gvp}_all_sequence_fragment_msa_scores.csv"
    )

    detail_df.to_csv(
        detail_path,
        index=False,
        encoding="utf-8-sig"
    )

    if detail_df.empty:
        print(f"⚠️ {gvp.upper()} 未生成有效MSA统计结果")
        return detail_df

    # =====================================================
    # 候选区域汇总
    # =====================================================
    summary_rows = []

    for (cluster_id, region), sub_df in detail_df.groupby(["cluster_id", "片段位置"]):
        avg_score = float(sub_df["fragment_avg_score"].mean())
        std_score = float(sub_df["fragment_avg_score"].std())
        avg_percentile = float(sub_df["percentile_rank"].mean())

        summary_rows.append({
            "GVP类型": sub_df["GVP类型"].iloc[0],
            "片段位置": region,
            "参与比对序列数": int(sub_df["seq_id"].nunique()),
            "平均保守性分数": round(avg_score, 4),
            "保守性标准差": round(std_score, 4) if not np.isnan(std_score) else "",
            "平均百分位": round(avg_percentile, 2)
        })

    summary_df = pd.DataFrame(summary_rows)

    summary_df["start_sort"] = (
        summary_df["片段位置"]
        .astype(str)
        .str.split("-")
        .str[0]
        .astype(int)
    )

    summary_df = summary_df.sort_values(
        by=["GVP类型", "start_sort"]
    ).drop(columns=["start_sort"]).reset_index(drop=True)

    summary_path = os.path.join(
        TABLE_DIR,
        f"{gvp}_table_4_10_msa_summary.csv"
    )

    summary_df.to_csv(
        summary_path,
        index=False,
        encoding="utf-8-sig"
    )

    print(f"✅ {gvp.upper()} MSA表格已保存：{summary_path}")

    return summary_df


# =========================================================
# 主函数
# =========================================================
def main():
    print("🧬 开始进行表4-10 MAFFT保守性分析")
    print(f"读取最终去重候选区域表：{FINAL_CANDIDATE_TABLE}")

    all_summary_tables = []

    for gvp in GVP_TYPES:
        try:
            summary_df = run_one_gvp_msa(gvp)

            if not summary_df.empty:
                all_summary_tables.append(summary_df)

        except Exception as e:
            print(f"❌ {gvp.upper()} 处理失败：{e}")

    if not all_summary_tables:
        print("❌ 未生成任何MSA统计结果")
        return

    table_4_10 = pd.concat(
        all_summary_tables,
        ignore_index=True
    )

    table_4_10["start_sort"] = (
        table_4_10["片段位置"]
        .astype(str)
        .str.split("-")
        .str[0]
        .astype(int)
    )

    table_4_10 = table_4_10.sort_values(
        by=["GVP类型", "start_sort"]
    ).drop(columns=["start_sort"]).reset_index(drop=True)

    table_4_10_path = os.path.join(
        OUTPUT_ROOT,
        "Table_4_10_MAFFT_conservation_analysis.csv"
    )

    table_4_10.to_csv(
        table_4_10_path,
        index=False,
        encoding="utf-8-sig"
    )

    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 180)
    pd.set_option("display.max_colwidth", 50)

    print("\n" + "=" * 120)
    print("表4-10 候选片段MAFFT保守性分析结果")
    print("=" * 120)
    print(table_4_10.to_string(index=False))
    print("=" * 120)

    print(f"✅ 表4-10 已保存：{table_4_10_path}")

    print("\n🎉 全部完成！")
    print(f"结果目录：{OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
