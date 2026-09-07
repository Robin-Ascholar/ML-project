#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import json
import numpy as np
import pandas as pd

# ================= 配置区域 =================

# 原始 FASTA 分类目录，用于统计“原始序列数”
RAW_FASTA_DIR = r"C:\Users\r9000\Desktop\毕业设计\去重后\classified_by_gene_type_multi"

# CD-HIT 去冗余后的 JSON 根目录
JSON_ROOT_DIR = r"C:\Users\r9000\Desktop\毕设（无监督聚类）\新gvp"

# 输出目录
OUTPUT_DIR = r"C:\Users\r9000\Desktop\毕设（无监督聚类）\GVP_Four_Type_Clustering_Results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 输出文件
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "Table_4_2_sequence_cleaning_summary.csv")
OUTPUT_XLSX = os.path.join(OUTPUT_DIR, "Table_4_2_sequence_cleaning_summary.xlsx")

# GVP 类型顺序
TARGET_GVPS = ["gvpa", "gvpc", "gvpg", "gvpj", "gvpk", "gvpn", "gvpo", "gvpp"]

# 是否允许 X/B/Z
# 如果你的论文写“允许 X/B/Z 模糊氨基酸”，保持 True
# 如果论文写“只保留20种标准氨基酸”，改成 False
ALLOW_AMBIGUOUS = True

STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")
if ALLOW_AMBIGUOUS:
    STANDARD_AA.update(set("XBZ"))

# ===========================================


def format_gvp_name(gvp_type):
    """
    gvpa -> GvpA
    """
    return f"Gvp{gvp_type[-1].upper()}"


def normalize_gvp_name(name):
    """
    将文件夹名或文件名中的 GVP 类型统一转成 gvpa/gvpc/gvpg 等格式。
    """
    name_lower = name.lower()

    match = re.search(r"gvp[a-z]", name_lower)
    if match:
        return match.group(0)

    return None


def clean_sequence(seq):
    """
    去除空白并转为大写。
    """
    if seq is None:
        return ""
    return re.sub(r"\s+", "", str(seq)).upper()


def has_only_standard_aa(seq):
    """
    判断是否只包含允许的氨基酸字符。
    """
    seq = clean_sequence(seq)

    if not seq:
        return False, "-"

    for c in seq:
        if c not in STANDARD_AA:
            return False, c

    return True, None


def parse_fasta_content(content):
    """
    解析 FASTA 内容。
    返回 [(header, sequence), ...]
    """
    sequences = []
    current_header = None
    current_seq_lines = []

    for line in content.splitlines():
        line = line.strip()

        if not line:
            continue

        if line.startswith(">"):
            if current_header is not None:
                sequences.append((current_header, "".join(current_seq_lines)))

            current_header = line
            current_seq_lines = []
        else:
            current_seq_lines.append(line)

    if current_header is not None:
        sequences.append((current_header, "".join(current_seq_lines)))

    return sequences


def count_raw_fasta_sequences(raw_fasta_dir):
    """
    统计原始 FASTA 中每个 GVP 类型的序列数。
    """
    raw_counts = {gvp: 0 for gvp in TARGET_GVPS}

    for root, dirs, files in os.walk(raw_fasta_dir):
        folder_name = os.path.basename(root)
        gvp_type = normalize_gvp_name(folder_name)

        if gvp_type not in raw_counts:
            continue

        for file_name in files:
            if not (file_name.lower().endswith(".fasta") or file_name.lower().endswith(".fa")):
                continue

            if file_name.lower().endswith(".backup"):
                continue

            fasta_path = os.path.join(root, file_name)

            try:
                with open(fasta_path, "r", encoding="utf-8") as f:
                    content = f.read()

                seqs = parse_fasta_content(content)
                raw_counts[gvp_type] += len(seqs)

            except Exception as e:
                print(f"⚠️ FASTA读取失败：{fasta_path}，原因：{e}")

    return raw_counts


def get_sequence_from_json_item(item):
    """
    从 JSON 记录中读取 sequence。
    兼容 sequenc 拼写。
    """
    seq = item.get("sequence", "")

    if not seq:
        seq = item.get("sequenc", "")

    return clean_sequence(seq)


def find_json_for_gvp(json_root_dir, gvp_type):
    """
    查找某个 GVP 类型对应的 JSON 文件。
    例如：
    gvpa -> GvpA_sequences.json
    """
    target_name = f"Gvp{gvp_type[-1].upper()}_sequences.json"

    for root, dirs, files in os.walk(json_root_dir):
        for file_name in files:
            if file_name == target_name:
                return os.path.join(root, file_name)

    return None


def calculate_3sigma_threshold(length_list):
    """
    按你之前 JSON 脚本的逻辑计算 3-Sigma 阈值：

    mean ± 3 * std
    std 使用样本标准差 ddof=1
    lower / upper 使用 round() 取整
    """
    if not length_list:
        return 0, 0, 0, 0

    if len(length_list) < 2:
        value = length_list[0]
        return value, value, round(value, 2), 0

    arr = np.array(length_list, dtype=np.float32)

    mean_val = np.mean(arr)
    std_val = np.std(arr, ddof=1)

    lower = mean_val - 3 * std_val
    upper = mean_val + 3 * std_val

    return round(lower), round(upper), round(mean_val, 2), round(std_val, 2)


def analyze_json_cleaning(json_path):
    """
    分析一个 GVP JSON 文件：

    1. CD-HIT 去冗余后序列数 = JSON 记录数
    2. 非标准字符过滤数 = 不符合 STANDARD_AA 的序列数
    3. 长度异常过滤数 = 通过氨基酸过滤后的序列，再按 3-Sigma 过滤
    4. 最终保留序列数
    5. 最终保留序列平均长度
    """
    if json_path is None or not os.path.exists(json_path):
        return {
            "cdhit_count": 0,
            "invalid_aa_count": 0,
            "length_removed_count": 0,
            "final_count": 0,
            "mean_length": 0,
            "lower_limit": 0,
            "upper_limit": 0,
            "length_mean_before_filter": 0,
            "length_std_before_filter": 0
        }

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        data = [data]

    cdhit_count = len(data)

    aa_valid_records = []
    invalid_aa_count = 0

    # 第一步：非标准字符过滤
    for item in data:
        if not isinstance(item, dict):
            continue

        seq = get_sequence_from_json_item(item)

        ok_aa, invalid_char = has_only_standard_aa(seq)

        if ok_aa:
            aa_valid_records.append({
                "item": item,
                "seq": seq,
                "length": len(seq)
            })
        else:
            invalid_aa_count += 1

    lengths_after_aa = [rec["length"] for rec in aa_valid_records]

    if not lengths_after_aa:
        return {
            "cdhit_count": cdhit_count,
            "invalid_aa_count": invalid_aa_count,
            "length_removed_count": 0,
            "final_count": 0,
            "mean_length": 0,
            "lower_limit": 0,
            "upper_limit": 0,
            "length_mean_before_filter": 0,
            "length_std_before_filter": 0
        }

    # 第二步：按之前 JSON 脚本逻辑计算 3-Sigma
    lower, upper, mean_before, std_before = calculate_3sigma_threshold(lengths_after_aa)

    final_records = []
    length_removed_count = 0

    for rec in aa_valid_records:
        seq_len = rec["length"]

        if lower <= seq_len <= upper:
            final_records.append(rec)
        else:
            length_removed_count += 1

    final_count = len(final_records)

    if final_records:
        mean_length = sum(rec["length"] for rec in final_records) / final_count
    else:
        mean_length = 0

    return {
        "cdhit_count": cdhit_count,
        "invalid_aa_count": invalid_aa_count,
        "length_removed_count": length_removed_count,
        "final_count": final_count,
        "mean_length": mean_length,
        "lower_limit": lower,
        "upper_limit": upper,
        "length_mean_before_filter": mean_before,
        "length_std_before_filter": std_before
    }


def generate_summary_table():
    """
    生成最终表格。
    """
    raw_counts = count_raw_fasta_sequences(RAW_FASTA_DIR)

    rows = []

    for gvp_type in TARGET_GVPS:
        gvp_name = format_gvp_name(gvp_type)

        json_path = find_json_for_gvp(JSON_ROOT_DIR, gvp_type)

        if json_path is None:
            print(f"⚠️ 未找到 {gvp_name} 对应的 JSON 文件")
        else:
            print(f"正在统计 {gvp_name}: {json_path}")

        json_stats = analyze_json_cleaning(json_path)

        rows.append({
            "Gvp": gvp_name,
            "原始序列数": raw_counts.get(gvp_type, 0),
            "CD-HIT去冗余后序列数": json_stats["cdhit_count"],
            "非标准字符过滤数": json_stats["invalid_aa_count"],
            "长度异常过滤数": json_stats["length_removed_count"],
            "最终保留序列数": json_stats["final_count"],
            "平均长度": f"{json_stats['mean_length']:.2f} aa"
        })

    df = pd.DataFrame(rows)

    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    df.to_excel(OUTPUT_XLSX, index=False)

    print("\n表格生成完成：")
    print(df)

    print(f"\nCSV文件：{OUTPUT_CSV}")
    print(f"Excel文件：{OUTPUT_XLSX}")

    return df


if __name__ == "__main__":
    generate_summary_table()
