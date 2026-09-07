#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import re
import pandas as pd

# ================= 基础路径配置 =================
DATA_ROOT = r"C:\Users\r9000\Desktop\毕设（无监督聚类）\新gvp"
OUTPUT_ROOT = r"C:\Users\r9000\Desktop\毕设（无监督聚类）\GVP_Four_Type_Clustering_Results"
os.makedirs(OUTPUT_ROOT, exist_ok=True)

# 要统计的 GVP 类型
TARGET_GVPS = ["gvpa", "gvpc", "gvpn", "gvpo", "gvpg", "gvpj", "gvpk", "gvpp"]

# ================= 表4-3所需参数 =================
GVP_WINDOW_CONFIG = {
    "gvpa": {"window": 30, "step": 5},
    "gvpc": {"window": 35, "step": 8},
    "gvpn": {"window": 40, "step": 8},
    "gvpo": {"window": 50, "step": 5},
    "gvpg": {"window": 25, "step": 5},
    "gvpj": {"window": 35, "step": 10},
    "gvpk": {"window": 25, "step": 4},
    "gvpp": {"window": 40, "step": 10},
}


def normalize_sequence(seq):
    """
    只做格式标准化：
    1. 去除空白字符；
    2. 转为大写。

    注意：
    这里不做非标准氨基酸过滤；
    这里不做长度过滤；
    这里不剔除序列。
    """
    if seq is None:
        return ""
    return re.sub(r"\s+", "", str(seq)).upper()


def format_gvp_name(gvp_type):
    """
    将 gvpa 转为 GvpA。
    """
    return f"Gvp{gvp_type[-1].upper()}"


def get_sequence_from_item(item):
    """
    从 JSON 记录中读取序列。
    正常字段应为 sequence。
    兼容之前可能写错的 sequenc 字段。
    """
    seq = item.get("sequence", "")

    if not seq:
        seq = item.get("sequenc", "")

    return normalize_sequence(seq)


def get_accession_from_item(item, gvp_type, index):
    """
    获取序列 ID。
    优先使用 unique_sequence_id。
    如果没有，则尝试 representative_annotation.id。
    如果仍然没有，则自动生成一个 ID。
    """
    acc = item.get("unique_sequence_id")

    representative_annotation = item.get("representative_annotation", {}) or {}

    if not acc:
        acc = representative_annotation.get("id")

    if not acc:
        acc = f"{gvp_type}_{index}"

    return acc


def get_species_from_item(item):
    """
    获取物种信息。
    """
    representative_annotation = item.get("representative_annotation", {}) or {}
    return representative_annotation.get("organism")


def load_filtered_gvp_json(json_path, gvp_type):
    """
    读取已经过滤后的 GVP JSON 文件。

    重要：
    这里假设 JSON 已经是最终数据：
    - 已经 CD-HIT 去重；
    - 已经完成非标准字符过滤；
    - 已经完成长度异常过滤。

    所以本函数不再做任何剔除。
    """
    if not os.path.exists(json_path):
        print(f"❌ File not found: {json_path}")
        return []

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    records = []

    for idx, item in enumerate(data):
        seq = get_sequence_from_item(item)
        acc = get_accession_from_item(item, gvp_type, idx)
        species = get_species_from_item(item)

        records.append({
            "seq": seq,
            "acc": acc,
            "species": species,
            "gvp_type": gvp_type,
            "full_seq_length": len(seq),
            "raw_item": item
        })

    return records


def slide_window(records, window, step):
    """
    对最终过滤后的蛋白序列进行滑动窗口切分。

    注意：
    短于窗口长度的序列不算“剔除”，只是无法生成片段。
    所以仍然计入最终序列数，只是在核对表中单独记录。
    """
    fragments = []
    meta = []

    skipped_short_seq = 0
    generated_seq_count = 0

    for rec in records:
        seq = rec["seq"]
        full_len = rec["full_seq_length"]

        if full_len < window:
            skipped_short_seq += 1
            continue

        generated_seq_count += 1

        for start in range(0, full_len - window + 1, step):
            end = start + window

            fragment_id = f"{rec['acc']}_{start}-{end}"
            fragment_seq = seq[start:end]

            fragments.append({
                "fragment_id": fragment_id,
                "fragment_seq": fragment_seq
            })

            meta.append({
                "fragment_id": fragment_id,
                "acc": rec["acc"],
                "start": start,
                "end": end,
                "center": (start + end) / 2,
                "window_size": window,
                "step": step,
                "full_seq_length": full_len,
                "species": rec["species"],
                "gvp_type": rec["gvp_type"]
            })

    return fragments, meta, generated_seq_count, skipped_short_seq


def calculate_length_stats(records):
    """
    计算序列长度统计信息。
    """
    lengths = [rec["full_seq_length"] for rec in records]

    if not lengths:
        return {
            "min_len": 0,
            "max_len": 0,
            "mean_len": 0
        }

    return {
        "min_len": min(lengths),
        "max_len": max(lengths),
        "mean_len": sum(lengths) / len(lengths)
    }


def generate_table_4_3():
    """
    生成表4-3：不同 GVP 类型滑动窗口片段生成结果。

    输出两个文件：
    1. Table_4_3_sliding_window_fragment_statistics.csv
       用于论文表4-3。

    2. Table_4_3_sliding_window_check_statistics.csv
       用于核对长度、短序列、实际参与切片序列数等。
    """
    table_rows = []
    check_rows = []

    for gvp_type in TARGET_GVPS:
        if gvp_type not in GVP_WINDOW_CONFIG:
            print(f"⚠️ Skip {gvp_type}: no window config")
            continue

        config = GVP_WINDOW_CONFIG[gvp_type]
        gvp_suffix = gvp_type[-1].upper()

        json_path = os.path.join(
            DATA_ROOT,
            gvp_type,
            f"Gvp{gvp_suffix}_sequences.json"
        )

        print("\n" + "=" * 90)
        print(f"🧬 Processing {format_gvp_name(gvp_type)}")
        print(f"📄 JSON: {json_path}")

        records = load_filtered_gvp_json(json_path, gvp_type)

        final_seq_num = len(records)

        length_stats = calculate_length_stats(records)
        min_len = length_stats["min_len"]
        max_len = length_stats["max_len"]
        mean_len = length_stats["mean_len"]

        fragments, meta, generated_seq_count, skipped_short_seq = slide_window(
            records,
            config["window"],
            config["step"]
        )

        fragment_total = len(fragments)

        avg_frag_per_final_seq = (
            fragment_total / final_seq_num
            if final_seq_num > 0 else 0
        )

        avg_frag_per_generated_seq = (
            fragment_total / generated_seq_count
            if generated_seq_count > 0 else 0
        )

        print(f"最终序列数：{final_seq_num}")
        print(f"平均序列长度：{mean_len:.2f} aa")
        print(f"窗口长度：{config['window']}")
        print(f"步长：{config['step']}")
        print(f"最短序列长度：{min_len}")
        print(f"最长序列长度：{max_len}")
        print(f"短于窗口未切片序列数：{skipped_short_seq}")
        print(f"实际参与切片序列数：{generated_seq_count}")
        print(f"生成片段总数：{fragment_total}")
        print(f"平均每条最终序列片段数：{avg_frag_per_final_seq:.2f}")
        print(f"平均每条参与切片序列片段数：{avg_frag_per_generated_seq:.2f}")

        # 论文表4-3
        table_rows.append({
            "GVP类型": format_gvp_name(gvp_type),
            "最终序列数": final_seq_num,
            "平均序列长度/aa": round(mean_len, 2),
            "窗口长度": config["window"],
            "步长": config["step"],
            "生成片段总数": fragment_total,
            "平均每条序列片段数": round(avg_frag_per_final_seq, 2)
        })

        # 核对表
        check_rows.append({
            "GVP类型": format_gvp_name(gvp_type),
            "最终序列数": final_seq_num,
            "最短序列长度": min_len,
            "最长序列长度": max_len,
            "平均序列长度/aa": round(mean_len, 2),
            "窗口长度": config["window"],
            "步长": config["step"],
            "短于窗口未切片序列数": skipped_short_seq,
            "实际参与切片序列数": generated_seq_count,
            "生成片段总数": fragment_total,
            "平均每条最终序列片段数": round(avg_frag_per_final_seq, 2),
            "平均每条参与切片序列片段数": round(avg_frag_per_generated_seq, 2)
        })

    df_table_4_3 = pd.DataFrame(table_rows)
    df_check = pd.DataFrame(check_rows)

    table_4_3_path = os.path.join(
        OUTPUT_ROOT,
        "Table_4_3_sliding_window_fragment_statistics.csv"
    )

    check_path = os.path.join(
        OUTPUT_ROOT,
        "Table_4_3_sliding_window_check_statistics.csv"
    )

    df_table_4_3.to_csv(table_4_3_path, index=False, encoding="utf-8-sig")
    df_check.to_csv(check_path, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 90)
    print("表4-3 不同GVP类型滑动窗口片段生成结果")
    print(df_table_4_3)

    print(f"\n✅ 表4-3已保存至：{table_4_3_path}")
    print(f"✅ 核对表已保存至：{check_path}")

    return df_table_4_3, df_check


if __name__ == "__main__":
    generate_table_4_3()
