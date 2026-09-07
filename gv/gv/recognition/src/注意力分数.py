#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import traceback
import warnings

import numpy as np
import pandas as pd
import torch
import esm
from tqdm import tqdm


warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


# =========================================================
# 基础配置
# =========================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_ROOT = r"C:\Users\r9000\Desktop\毕设（无监督聚类）\新gvp"

# =========================================================
# 直接读取上一步已经去重后的最终候选区域表
# 注意：后面不再自己筛选、不再自己去重
# =========================================================
FINAL_CANDIDATE_TABLE = (
    r"C:\Users\r9000\Desktop\毕设（无监督聚类）"
    r"\GVP_Four_Type_Clustering_Results"
    r"\Final_Passed_NonClose_Candidates"
    r"\Passed_Candidate_NonClose_All_Detail_For_Downstream.csv"
)

# 输出目录
OUTPUT_ROOT = r"C:\Users\r9000\Desktop\毕设（无监督聚类）\Table_4_9_Attention_Results"

DETAIL_DIR = os.path.join(OUTPUT_ROOT, "per_sequence_attention")
STAT_DIR = os.path.join(OUTPUT_ROOT, "cluster_attention_stats")

os.makedirs(OUTPUT_ROOT, exist_ok=True)
os.makedirs(DETAIL_DIR, exist_ok=True)
os.makedirs(STAT_DIR, exist_ok=True)


# =========================================================
# GVP类型
# =========================================================
GVP_TYPES = [
    "gvpa", "gvpc", "gvpn", "gvpo",
    "gvpg", "gvpj", "gvpk", "gvpp"
]


# =========================================================
# 基础工具函数
# =========================================================
def format_gvp_name(gvp_type):
    return f"Gvp{gvp_type[-1].upper()}"


def clean_sequence(seq):
    return "".join([
        c for c in str(seq).strip().upper()
        if c in "ACDEFGHIKLMNPQRSTVWY"
    ])


def get_sequence_id(item, idx):
    return (
        item.get("unique_sequence_id")
        or item.get("id")
        or item.get("accession")
        or item.get("protein_id")
        or f"seq_{idx + 1}"
    )


# =========================================================
# 直接读取最终去重候选区域
# =========================================================
def load_final_candidates_for_gvp(gvp_type):
    """
    直接读取已经去重后的最终候选区域表。

    不再做：
        1. CV筛选
        2. 跨物种覆盖筛选
        3. 5 aa距离去重
        4. 起始位置重新计算

    下游所有候选区域完全以 FINAL_CANDIDATE_TABLE 为准。
    """

    if not os.path.exists(FINAL_CANDIDATE_TABLE):
        raise FileNotFoundError(
            f"最终去重候选区域表不存在：{FINAL_CANDIDATE_TABLE}"
        )

    df = pd.read_csv(FINAL_CANDIDATE_TABLE, encoding="utf-8-sig")

    required_cols = [
        "原始GVP类型",
        "GVP类型",
        "cluster_id",
        "片段位置",
        "start_floor",
        "end_floor"
    ]

    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"最终候选区域表缺少字段：{col}")

    sub_df = df[df["原始GVP类型"] == gvp_type].copy()

    if sub_df.empty:
        print(f"⚠️ {gvp_type.upper()} 在最终候选区域表中没有记录")
        return sub_df

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

    sub_df = sub_df.sort_values(
        by=["start_floor", "end_floor"]
    ).reset_index(drop=True)

    return sub_df


# =========================================================
# ESM-2 原生 attention 模型
# =========================================================
class ESM2NativeAttention:
    def __init__(self):
        print("\n加载 ESM-2 35M 原生模型：esm2_t12_35M_UR50D")
        self.model, self.alphabet = esm.pretrained.esm2_t12_35M_UR50D()
        self.model.to(DEVICE)
        self.model.eval()
        self.batch_converter = self.alphabet.get_batch_converter()
        print(f"模型加载完成，当前设备：{DEVICE}")

    def get_attention(self, sequence):
        """
        返回每个残基的 attention 分数，归一化到 0-1。
        """
        sequence = clean_sequence(sequence)
        L = len(sequence)

        if L < 10:
            return None

        _, _, tokens = self.batch_converter([("seq", sequence)])
        tokens = tokens.to(DEVICE)

        with torch.no_grad():
            out = self.model(
                tokens,
                repr_layers=[12],
                need_head_weights=True
            )

        # attentions shape:
        # [batch, layers, heads, tokens, tokens]
        att = out["attentions"][0]

        # 平均所有层和所有注意力头
        att_mean = att.mean(dim=(0, 1))

        # 对每个残基被关注程度求和
        residue_att = att_mean.sum(dim=0).cpu().numpy()

        # 去掉 <cls> 和 <eos>
        final_att = residue_att[1:1 + L]

        # 每条序列内部归一化到 0-1
        att_range = final_att.max() - final_att.min()

        if att_range > 1e-8:
            final_att = (final_att - final_att.min()) / att_range
        else:
            final_att = np.zeros_like(final_att)

        return final_att


# =========================================================
# 计算单个 GVP 的候选片段 attention
# =========================================================
def compute_attention_for_gvp(gvp, model):
    print("\n" + "=" * 90)
    print(f"正在计算 {gvp.upper()} 候选片段 attention")
    print("=" * 90)

    out_detail_path = os.path.join(
        DETAIL_DIR,
        f"{gvp}_all_seqs_attention.csv"
    )

    seq_path = os.path.join(
        DATA_ROOT,
        gvp,
        f"Gvp{gvp[-1].upper()}_sequences.json"
    )

    if not os.path.exists(seq_path):
        print(f"❌ 未找到序列文件：{seq_path}")
        return None

    with open(seq_path, "r", encoding="utf-8") as f:
        all_seqs_data = json.load(f)

    # =====================================================
    # 直接读取最终去重候选区域
    # =====================================================
    candidate_df = load_final_candidates_for_gvp(gvp)

    if candidate_df.empty:
        print(f"⚠️ {gvp.upper()} 无最终候选区域")
        return None

    print(f"{gvp.upper()} 读取最终去重候选区域数：{len(candidate_df)}")

    all_results = []

    for idx, item in enumerate(
        tqdm(all_seqs_data, desc=f"{gvp.upper()} sequences")
    ):
        seq_id = get_sequence_id(item, idx)
        raw_seq = item.get("sequence", "")
        clean_seq = clean_sequence(raw_seq)
        L = len(clean_seq)

        if L < 10:
            continue

        try:
            att_scores = model.get_attention(clean_seq)

            if att_scores is None:
                continue

        except RuntimeError as e:
            print(f"⚠️ {gvp.upper()} 序列 {seq_id} attention 计算失败：{e}")

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            continue

        except Exception as e:
            print(f"⚠️ {gvp.upper()} 序列 {seq_id} 出错：{e}")
            continue

        for _, row in candidate_df.iterrows():
            cluster_id = int(row["cluster_id"])

            # =================================================
            # 直接使用最终候选表中的 start_floor / end_floor
            # 不再重新取整、不再重新筛选、不再去重
            # =================================================
            start = int(row["start_floor"])
            end = int(row["end_floor"])

            if start < 0:
                start = 0

            if end > L:
                end = L

            original_len = int(row["end_floor"]) - int(row["start_floor"])

            # 如果该序列太短，导致片段截断超过一半，则跳过
            if end - start < original_len // 2:
                continue

            fragment_att = att_scores[start:end]

            if len(fragment_att) == 0:
                continue

            score = float(np.mean(fragment_att))

            percentile = float(
                np.sum(att_scores <= score) / len(att_scores) * 100
            )

            all_results.append({
                "GVP类型": row["GVP类型"],
                "原始GVP类型": gvp,
                "seq_id": seq_id,
                "cluster_id": cluster_id,

                "片段位置": row["片段位置"],
                "fragment_start": start,
                "fragment_end": end,

                "attention_score": round(score, 4),
                "percentile": round(percentile, 2)
            })

    if not all_results:
        print(f"⚠️ {gvp.upper()} 无有效 attention 结果")
        return None

    df_out = pd.DataFrame(all_results)

    # 每次重新覆盖明细文件
    df_out.to_csv(
        out_detail_path,
        index=False,
        encoding="utf-8-sig"
    )

    print(f"✅ {gvp.upper()} attention 明细已保存：{out_detail_path}")
    print(f"有效记录数：{len(df_out)}")

    return out_detail_path


# =========================================================
# 汇总生成表4-9
# =========================================================
def generate_table_4_9():
    table_rows = []

    for gvp in GVP_TYPES:
        file_path = os.path.join(
            DETAIL_DIR,
            f"{gvp}_all_seqs_attention.csv"
        )

        if not os.path.exists(file_path):
            print(f"⚠️ 跳过 {gvp.upper()}，未找到 attention 明细文件")
            continue

        df = pd.read_csv(file_path, encoding="utf-8-sig")

        if df.empty:
            print(f"⚠️ 跳过 {gvp.upper()}，attention 明细为空")
            continue

        stat = df.groupby(["cluster_id", "片段位置"]).agg(
            gvp_type=("GVP类型", "first"),
            avg_attn=("attention_score", "mean"),
            std_attn=("attention_score", "std"),
            avg_percentile=("percentile", "mean")
        ).reset_index()

        stat = stat.round(4)

        stat["start_sort"] = (
            stat["片段位置"]
            .astype(str)
            .str.split("-")
            .str[0]
            .astype(int)
        )

        stat = stat.sort_values(
            by=["gvp_type", "start_sort"]
        ).reset_index(drop=True)

        stat_out_path = os.path.join(
            STAT_DIR,
            f"{gvp}_cluster_attention_statistics.csv"
        )

        stat.to_csv(
            stat_out_path,
            index=False,
            encoding="utf-8-sig"
        )

        for _, row in stat.iterrows():
            table_rows.append({
                "GVP类型": row["gvp_type"],
                "片段位置": row["片段位置"],
                "注意力分": round(float(row["avg_attn"]), 4),
                "注意力标准差": (
                    round(float(row["std_attn"]), 4)
                    if not pd.isna(row["std_attn"])
                    else ""
                ),
                "平均百分位": round(float(row["avg_percentile"]), 2)
            })

    table_4_9 = pd.DataFrame(table_rows)

    if table_4_9.empty:
        print("\n❌ 未生成表4-9，可能没有有效 attention 结果")
        return table_4_9

    table_4_9["start_sort"] = (
        table_4_9["片段位置"]
        .astype(str)
        .str.split("-")
        .str[0]
        .astype(int)
    )

    table_4_9 = table_4_9.sort_values(
        by=["GVP类型", "start_sort"]
    ).drop(columns=["start_sort"]).reset_index(drop=True)

    out_path = os.path.join(
        OUTPUT_ROOT,
        "Table_4_9_ESM2_attention_score_statistics.csv"
    )

    table_4_9.to_csv(
        out_path,
        index=False,
        encoding="utf-8-sig"
    )

    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 180)
    pd.set_option("display.max_colwidth", 50)

    print("\n" + "=" * 120)
    print("表4-9 候选片段ESM-2注意力分数统计结果")
    print("=" * 120)
    print(table_4_9.to_string(index=False))
    print("=" * 120)

    print(f"表4-9总候选片段数：{len(table_4_9)}")
    print(f"✅ 表4-9 CSV已保存至：{out_path}")

    return table_4_9


# =========================================================
# 主函数
# =========================================================
def main():
    print(f"当前设备：{DEVICE}")
    print(f"读取最终去重候选区域表：{FINAL_CANDIDATE_TABLE}")

    att_model = ESM2NativeAttention()

    for gvp in GVP_TYPES:
        try:
            compute_attention_for_gvp(gvp, att_model)

        except Exception as e:
            print(f"❌ {gvp.upper()} 处理失败：{e}")
            traceback.print_exc()

    generate_table_4_9()

    print("\n🎉 全部完成！")
    print(f"输出目录：{OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
