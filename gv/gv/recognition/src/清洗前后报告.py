#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import math
import csv
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from collections import defaultdict

# ================= 配置区域 =================
INPUT_DIR = r"C:\Users\r9000\Desktop\毕业设计\去重后\classified_by_gene_type_multi"
OUTPUT_DIR = INPUT_DIR.rstrip(os.sep) + "_clean_by_type"

REPORT_FILE = r"C:\Users\r9000\Desktop\毕业设计\去重后\cleaning_report_before_after_per_type.txt"

RAW_LENGTH_TABLE_FILE = r"C:\Users\r9000\Desktop\毕业设计\去重后\sequence_length_raw_by_type.tsv"
CLEAN_LENGTH_TABLE_FILE = r"C:\Users\r9000\Desktop\毕业设计\去重后\sequence_length_after_aa_filter_by_type.tsv"

FIG_OUTPUT_PNG = r"C:\Users\r9000\Desktop\毕业设计\去重后\fig_4_2_gvp_length_3std_after_removed.png"
FIG_OUTPUT_PDF = r"C:\Users\r9000\Desktop\毕业设计\去重后\fig_4_2_gvp_length_3std_after_removed.pdf"
FIG_DATA_TSV = r"C:\Users\r9000\Desktop\毕业设计\去重后\fig_4_2_gvp_length_3std_after_removed_data.tsv"

GVP_PLOT_ORDER = ["GvpA", "GvpC", "GvpG", "GvpJ", "GvpK", "GvpN", "GvpO", "GvpP"]

ALLOW_AMBIGUOUS = True
# ===========================================

STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")
if ALLOW_AMBIGUOUS:
    STANDARD_AA.update(set("XBZ"))


def parse_fasta_content(content):
    sequences = []
    lines = content.splitlines()
    current_header = None
    current_seq_lines = []

    for line in lines:
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


def clean_sequence(seq):
    return re.sub(r"\s+", "", seq).upper()


def has_only_standard_aa(seq):
    """
    非标准氨基酸过滤逻辑：
    如果序列中存在不在 STANDARD_AA 中的字符，则整条序列剔除。

    注意：
    ALLOW_AMBIGUOUS = True 时，X/B/Z 会被允许。
    """
    seq = clean_sequence(seq)

    for c in seq:
        if c not in STANDARD_AA:
            return False, c

    return True, None


def calculate_stats(lengths):
    """
    用于第一次长度异常过滤的 3-Sigma 逻辑：

    1. 使用 numpy 计算均值；
    2. 使用样本标准差 ddof=1；
    3. 阈值为 mean ± 3 * std；
    4. lower / upper 使用 round() 取整；
    5. 不强制 lower >= 10。

    返回：
    mean, std_dev, lower, upper, min_len, max_len
    """
    if not lengths:
        return 0, 0, 0, 0, 0, 0

    min_len = min(lengths)
    max_len = max(lengths)

    if len(lengths) < 2:
        mean_val = float(lengths[0])
        std_val = 0.0
        lower = round(mean_val)
        upper = round(mean_val)
        return mean_val, std_val, lower, upper, min_len, max_len

    arr = np.array(lengths, dtype=float)

    mean_val = np.mean(arr)
    std_val = np.std(arr, ddof=1)

    lower = mean_val - 3 * std_val
    upper = mean_val + 3 * std_val

    lower = round(lower)
    upper = round(upper)

    return float(mean_val), float(std_val), lower, upper, min_len, max_len


def calculate_final_stats(lengths):
    """
    用于最终报告和画图的统计逻辑。

    重点：
    这里输入的是已经剔除长度异常后的 Kept 序列长度。
    所以最终 mean/std/±3SD 都是基于最终保留序列计算。
    """
    if not lengths:
        return {
            "count": 0,
            "min_len": 0,
            "max_len": 0,
            "mean": 0.0,
            "std": 0.0,
            "minus_1sigma": 0.0,
            "plus_1sigma": 0.0,
            "minus_2sigma": 0.0,
            "plus_2sigma": 0.0,
            "minus_3sigma": 0.0,
            "plus_3sigma": 0.0,
        }

    arr = np.array(lengths, dtype=float)

    min_len = float(np.min(arr))
    max_len = float(np.max(arr))
    mean_val = float(np.mean(arr))

    if len(arr) < 2:
        std_val = 0.0
    else:
        std_val = float(np.std(arr, ddof=1))

    return {
        "count": int(len(arr)),
        "min_len": min_len,
        "max_len": max_len,
        "mean": mean_val,
        "std": std_val,
        "minus_1sigma": mean_val - std_val,
        "plus_1sigma": mean_val + std_val,
        "minus_2sigma": mean_val - 2 * std_val,
        "plus_2sigma": mean_val + 2 * std_val,
        "minus_3sigma": mean_val - 3 * std_val,
        "plus_3sigma": mean_val + 3 * std_val,
    }


def is_length_valid(seq, min_l, max_l):
    """
    长度判断逻辑：
    按第一次 round(mean ± 3*std) 得到的整数阈值判断。
    """
    seq = clean_sequence(seq)
    length = len(seq)

    if length < min_l:
        return False, length, f"Length Too Short ({length})"

    if length > max_l:
        return False, length, f"Length Too Long ({length})"

    return True, length, "OK"


def collect_group_files(current_input_path):
    group_files_data = {}

    for root, dirs, files in os.walk(current_input_path):
        for file in files:
            if (file.endswith(".fasta") or file.endswith(".fa")) and not file.endswith(".backup"):
                fpath = os.path.join(root, file)

                try:
                    with open(fpath, "r", encoding="utf-8") as f:
                        content = f.read()

                    seqs = parse_fasta_content(content)
                    group_files_data[fpath] = seqs

                except Exception as e:
                    print(f"   ⚠️ 读取失败 {fpath}: {e}")

    return group_files_data


def update_raw_length_final_status(raw_length_records, group_name, rel_path, header, final_status, final_reason):
    for item in reversed(raw_length_records):
        if item["type"] == group_name and item["file"] == rel_path and item["header"] == header:
            item["final_status"] = final_status
            item["final_reason"] = final_reason
            return


def write_raw_length_table(raw_length_records):
    report_dir = os.path.dirname(RAW_LENGTH_TABLE_FILE)
    if report_dir and not os.path.exists(report_dir):
        os.makedirs(report_dir)

    headers = [
        "type", "file", "header", "length",
        "type_min_length", "type_max_length",
        "aa_status", "invalid_char", "final_status", "final_reason"
    ]

    with open(RAW_LENGTH_TABLE_FILE, "w", encoding="utf-8") as f:
        f.write("\t".join(headers) + "\n")

        for item in raw_length_records:
            row = []
            for h in headers:
                row.append(str(item[h]).replace("\t", " ").replace("\n", " "))
            f.write("\t".join(row) + "\n")


def write_clean_length_table(clean_length_records):
    report_dir = os.path.dirname(CLEAN_LENGTH_TABLE_FILE)
    if report_dir and not os.path.exists(report_dir):
        os.makedirs(report_dir)

    headers = [
        "type", "file", "header", "length",

        # 最终保留序列的最短/最长，也就是剔除异常之后的范围
        "type_min_length", "type_max_length",

        "status", "reason",

        # 最终保留序列重新计算的统计量
        "mean", "std",
        "minus_1sigma", "plus_1sigma",
        "minus_2sigma", "plus_2sigma",
        "minus_3sigma", "plus_3sigma",

        # 第一次过滤前的统计量，用于说明异常过滤阈值来源
        "pre_filter_type_min_length",
        "pre_filter_type_max_length",
        "pre_filter_mean",
        "pre_filter_std",
        "pre_filter_minus_3sigma",
        "pre_filter_plus_3sigma",
        "filter_min",
        "filter_max",
    ]

    with open(CLEAN_LENGTH_TABLE_FILE, "w", encoding="utf-8") as f:
        f.write("\t".join(headers) + "\n")

        for item in clean_length_records:
            row = []
            for h in headers:
                v = item.get(h, "")
                if isinstance(v, float):
                    row.append(f"{v:.4f}")
                else:
                    row.append(str(v).replace("\t", " ").replace("\n", " "))
            f.write("\t".join(row) + "\n")


def normalize_gvp_for_plot(name):
    if name is None:
        return "Unknown"

    text = str(name).strip()
    text_lower = text.lower()

    m = re.search(r"gvp[\s_\-]*([acgjknop])", text_lower)
    if m:
        return "Gvp" + m.group(1).upper()

    if len(text_lower) == 1 and text_lower in ["a", "c", "g", "j", "k", "n", "o", "p"]:
        return "Gvp" + text_lower.upper()

    return text


def plot_gvp8_length_3std_from_reports(type_reports):
    """
    画图只使用最终保留序列的统计量：
    min/max/mean/std/mean±3SD 全部来自剔除长度异常后的 Kept 序列。
    """
    if not type_reports:
        print("⚠️ 没有统计结果，无法画图。")
        return

    report_map = {}

    for item in type_reports:
        gvp = normalize_gvp_for_plot(item.get("type"))
        if gvp in GVP_PLOT_ORDER:
            report_map[gvp] = item

    plot_items = []

    for gvp in GVP_PLOT_ORDER:
        if gvp not in report_map:
            print(f"⚠️ {gvp} 没有数据，跳过。")
            continue

        item = report_map[gvp]

        plot_items.append({
            "gvp": gvp,
            "after_aa_count": int(item["after_aa"]),
            "final_count": int(item["final"]),
            "length_removed": int(item["length_removed"]),
            "min_len": float(item["clean_min_len"]),
            "max_len": float(item["clean_max_len"]),
            "mean": float(item["mean"]),
            "std": float(item["std"]),
            "minus_3sigma": float(item["minus_3sigma"]),
            "plus_3sigma": float(item["plus_3sigma"]),
        })

    if not plot_items:
        print("⚠️ 八种 GVP 都没有可画的数据。")
        return

    os.makedirs(os.path.dirname(FIG_DATA_TSV), exist_ok=True)

    with open(FIG_DATA_TSV, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow([
            "GVP类型",
            "去非标准氨基酸后序列数",
            "最终保留序列数",
            "长度异常剔除数",
            "最终保留序列最短长度",
            "最终保留序列最长长度",
            "最终保留序列平均长度",
            "最终保留序列标准差",
            "最终保留序列Mean-3SD",
            "最终保留序列Mean+3SD",
        ])

        for item in plot_items:
            writer.writerow([
                item["gvp"],
                item["after_aa_count"],
                item["final_count"],
                item["length_removed"],
                f"{item['min_len']:.2f}",
                f"{item['max_len']:.2f}",
                f"{item['mean']:.2f}",
                f"{item['std']:.2f}",
                f"{item['minus_3sigma']:.2f}",
                f"{item['plus_3sigma']:.2f}",
            ])

    x = np.arange(len(plot_items))
    labels = [item["gvp"] for item in plot_items]

    min_vals = np.array([item["min_len"] for item in plot_items], dtype=float)
    max_vals = np.array([item["max_len"] for item in plot_items], dtype=float)
    mean_vals = np.array([item["mean"] for item in plot_items], dtype=float)
    minus_3_vals_raw = np.array([item["minus_3sigma"] for item in plot_items], dtype=float)
    plus_3_vals = np.array([item["plus_3sigma"] for item in plot_items], dtype=float)

    minus_3_vals_plot = np.maximum(0, minus_3_vals_raw)

    fig, ax = plt.subplots(figsize=(13, 7), dpi=300)

    for i, item in enumerate(plot_items):
        ax.vlines(
            x=i,
            ymin=min_vals[i],
            ymax=max_vals[i],
            linewidth=2.8,
            label="Min-max range" if i == 0 else None,
        )

        ax.hlines(
            y=mean_vals[i],
            xmin=i - 0.16,
            xmax=i + 0.16,
            linewidth=3.2,
            label="Mean" if i == 0 else None,
        )

        ax.hlines(
            y=minus_3_vals_plot[i],
            xmin=i - 0.16,
            xmax=i + 0.16,
            linestyles="dashed",
            linewidth=2.2,
            label="Mean ± 3SD" if i == 0 else None,
        )

        ax.hlines(
            y=plus_3_vals[i],
            xmin=i - 0.16,
            xmax=i + 0.16,
            linestyles="dashed",
            linewidth=2.2,
        )

        ax.text(
            i + 0.20,
            mean_vals[i],
            f"{mean_vals[i]:.1f}",
            va="center",
            ha="left",
            fontsize=9,
        )

        ax.text(
            i + 0.20,
            plus_3_vals[i],
            f"+3SD {plus_3_vals[i]:.1f}",
            va="center",
            ha="left",
            fontsize=8,
        )

        ax.text(
            i + 0.20,
            minus_3_vals_plot[i],
            f"-3SD {minus_3_vals_raw[i]:.1f}",
            va="center",
            ha="left",
            fontsize=8,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)

    ax.set_xlabel("GVP type", fontsize=13)
    ax.set_ylabel("Sequence length (aa)", fontsize=13)
    ax.set_title(
        "Sequence Length Ranges and Mean ± 3SD After Removing Outliers",
        fontsize=15,
        pad=15,
    )

    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.legend(frameon=False, loc="upper right")

    y_top = max(max_vals.max(), plus_3_vals.max())
    ax.set_ylim(bottom=0, top=y_top * 1.10)

    plt.tight_layout()

    os.makedirs(os.path.dirname(FIG_OUTPUT_PNG), exist_ok=True)

    plt.savefig(FIG_OUTPUT_PNG, dpi=300, bbox_inches="tight")
    plt.savefig(FIG_OUTPUT_PDF, bbox_inches="tight")
    plt.close()

    print("📊 八种 GVP 的最终保留序列长度 3SD 图已生成：")
    print(f"PNG：{FIG_OUTPUT_PNG}")
    print(f"PDF：{FIG_OUTPUT_PDF}")
    print(f"画图数据：{FIG_DATA_TSV}")


def process_directory_by_type(base_dir, output_dir):
    if not os.path.exists(base_dir):
        print(f"❌ 错误：找不到目录 '{base_dir}'")
        return

    print("🔍 模式：按【子文件夹/基因类型】独立统计并清洗")
    print("清洗顺序：原始序列 -> 去除非标准氨基酸 -> 按类型独立 3-Sigma 长度过滤")
    print("说明：过滤阈值用去非标准氨基酸后的序列计算；最终均值和3SD用剔除长度异常后的 Kept 序列重新计算。")
    print("-" * 90)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    total_original_all = 0
    total_after_aa_all = 0
    total_final_all = 0
    total_invalid_aa_removed_all = 0
    total_length_removed_all = 0

    all_reports = []
    file_reports = []

    raw_length_records = []
    clean_length_records = []

    sub_dirs = [
        d for d in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, d))
    ]

    if not sub_dirs:
        sub_dirs = [""]

    print(f"📂 发现 {len(sub_dirs)} 个基因类型组。")

    for sub_name in sub_dirs:
        group_name = sub_name or "Root"
        current_input_path = os.path.join(base_dir, sub_name)
        current_output_path = os.path.join(output_dir, sub_name)

        print(f"\n🧬 处理组别：{group_name}")

        group_files_data = collect_group_files(current_input_path)

        if not group_files_data:
            print("   ⚠️ 未找到 FASTA 文件，跳过。")
            continue

        group_original_count = 0
        group_after_aa_count = 0
        group_final_count = 0
        group_invalid_aa_removed = 0
        group_length_removed = 0

        valid_aa_files_data = {}
        valid_aa_lengths = []
        raw_lengths = []

        invalid_char_counter = defaultdict(int)

        for fpath, seqs in group_files_data.items():
            valid_aa_seqs = []
            rel_path = os.path.relpath(fpath, base_dir)

            for h, s in seqs:
                seq_clean = clean_sequence(s)

                if not seq_clean:
                    continue

                seq_len = len(seq_clean)
                raw_lengths.append(seq_len)
                group_original_count += 1

                ok_aa, invalid_char = has_only_standard_aa(seq_clean)

                raw_length_records.append({
                    "type": group_name,
                    "file": rel_path,
                    "header": h,
                    "length": seq_len,
                    "type_min_length": "",
                    "type_max_length": "",
                    "aa_status": "AA_OK" if ok_aa else "Invalid_AA",
                    "invalid_char": invalid_char if invalid_char else "-",
                    "final_status": "Pending",
                    "final_reason": "Pending",
                })

                if ok_aa:
                    valid_aa_seqs.append((h, seq_clean))
                    valid_aa_lengths.append(seq_len)
                    group_after_aa_count += 1
                else:
                    group_invalid_aa_removed += 1
                    invalid_char_counter[invalid_char] += 1
                    update_raw_length_final_status(
                        raw_length_records,
                        group_name,
                        rel_path,
                        h,
                        "Removed",
                        f"Invalid AA ({invalid_char})"
                    )

            valid_aa_files_data[fpath] = valid_aa_seqs

        raw_min_len = min(raw_lengths) if raw_lengths else 0
        raw_max_len = max(raw_lengths) if raw_lengths else 0

        for item in raw_length_records:
            if item["type"] == group_name:
                item["type_min_length"] = raw_min_len
                item["type_max_length"] = raw_max_len

        if not valid_aa_lengths:
            print("   ⚠️ 去除非标准氨基酸后无有效序列，跳过。")
            continue

        pre_mean, pre_std, filter_min, filter_max, pre_clean_min_len, pre_clean_max_len = calculate_stats(valid_aa_lengths)

        pre_minus_3sigma = pre_mean - 3 * pre_std
        pre_plus_3sigma = pre_mean + 3 * pre_std

        print(f"   原始序列数：{group_original_count}")
        print(f"   去除非标准氨基酸后：{group_after_aa_count}")
        print(f"   非标准氨基酸剔除：{group_invalid_aa_removed}")
        print(f"   原始最短长度：{raw_min_len} aa")
        print(f"   原始最长长度：{raw_max_len} aa")
        print(f"   去非标准后最短长度：{pre_clean_min_len} aa")
        print(f"   去非标准后最长长度：{pre_clean_max_len} aa")
        print(f"   用于过滤的均值 μ：{pre_mean:.2f} aa")
        print(f"   用于过滤的标准差 σ：{pre_std:.2f} aa")
        print(f"   用于过滤的 μ ± 3σ：[{pre_minus_3sigma:.2f}, {pre_plus_3sigma:.2f}] aa")
        print(f"   实际 3-Sigma 过滤范围：[{filter_min}, {filter_max}] aa")

        if not os.path.exists(current_output_path):
            os.makedirs(current_output_path)

        group_clean_records = []
        kept_lengths_for_final_stats = []

        for fpath, seqs in valid_aa_files_data.items():
            rel_path = os.path.relpath(fpath, base_dir)
            new_fpath = os.path.join(output_dir, rel_path)
            os.makedirs(os.path.dirname(new_fpath), exist_ok=True)

            kept_seqs = []

            file_original = len(group_files_data[fpath])
            file_after_aa = len(seqs)
            file_final = 0
            file_length_removed = 0
            file_reasons = defaultdict(int)

            for h, s in seqs:
                ok_len, length, reason = is_length_valid(s, filter_min, filter_max)

                group_clean_records.append({
                    "type": group_name,
                    "file": rel_path,
                    "header": h,
                    "length": length,
                    "status": "Kept" if ok_len else "Removed",
                    "reason": reason,

                    "pre_filter_type_min_length": pre_clean_min_len,
                    "pre_filter_type_max_length": pre_clean_max_len,
                    "pre_filter_mean": pre_mean,
                    "pre_filter_std": pre_std,
                    "pre_filter_minus_3sigma": pre_minus_3sigma,
                    "pre_filter_plus_3sigma": pre_plus_3sigma,
                    "filter_min": filter_min,
                    "filter_max": filter_max,
                })

                update_raw_length_final_status(
                    raw_length_records,
                    group_name,
                    rel_path,
                    h,
                    "Kept" if ok_len else "Removed",
                    reason
                )

                if ok_len:
                    kept_seqs.append((h, s))
                    kept_lengths_for_final_stats.append(length)
                    file_final += 1
                    group_final_count += 1
                else:
                    file_length_removed += 1
                    group_length_removed += 1
                    file_reasons[reason] += 1

            try:
                with open(new_fpath, "w", encoding="utf-8") as out:
                    for h, s in kept_seqs:
                        out.write(h + "\n")
                        for i in range(0, len(s), 60):
                            out.write(s[i:i + 60] + "\n")

            except Exception as e:
                print(f"   ❌ 写入失败 {new_fpath}: {e}")

            file_reports.append({
                "type": group_name,
                "file": rel_path,
                "original": file_original,
                "after_aa": file_after_aa,
                "final": file_final,
                "invalid_aa_removed": file_original - file_after_aa,
                "length_removed": file_length_removed,
                "total_removed": file_original - file_final,
                "reasons": dict(file_reasons)
            })

        final_stats = calculate_final_stats(kept_lengths_for_final_stats)

        for rec in group_clean_records:
            rec["type_min_length"] = final_stats["min_len"]
            rec["type_max_length"] = final_stats["max_len"]
            rec["mean"] = final_stats["mean"]
            rec["std"] = final_stats["std"]
            rec["minus_1sigma"] = final_stats["minus_1sigma"]
            rec["plus_1sigma"] = final_stats["plus_1sigma"]
            rec["minus_2sigma"] = final_stats["minus_2sigma"]
            rec["plus_2sigma"] = final_stats["plus_2sigma"]
            rec["minus_3sigma"] = final_stats["minus_3sigma"]
            rec["plus_3sigma"] = final_stats["plus_3sigma"]

            clean_length_records.append(rec)

        total_removed = group_original_count - group_final_count
        keep_rate = group_final_count / group_original_count * 100 if group_original_count else 0

        print(f"   长度异常剔除：{group_length_removed}")
        print(f"   最终保留：{group_final_count}")
        print(f"   剔除异常后最短长度：{final_stats['min_len']:.0f} aa")
        print(f"   剔除异常后最长长度：{final_stats['max_len']:.0f} aa")
        print(f"   剔除异常后平均长度 μ：{final_stats['mean']:.2f} aa")
        print(f"   剔除异常后标准差 σ：{final_stats['std']:.2f} aa")
        print(f"   剔除异常后 μ ± 3σ：[{final_stats['minus_3sigma']:.2f}, {final_stats['plus_3sigma']:.2f}] aa")
        print(f"   总剔除：{total_removed}")
        print(f"   保留率：{keep_rate:.2f}%")

        all_reports.append({
            "type": group_name,
            "raw_min_len": raw_min_len,
            "raw_max_len": raw_max_len,

            "pre_clean_min_len": pre_clean_min_len,
            "pre_clean_max_len": pre_clean_max_len,
            "pre_mean": pre_mean,
            "pre_std": pre_std,
            "pre_minus_3sigma": pre_minus_3sigma,
            "pre_plus_3sigma": pre_plus_3sigma,
            "filter_min": filter_min,
            "filter_max": filter_max,

            "clean_min_len": final_stats["min_len"],
            "clean_max_len": final_stats["max_len"],
            "mean": final_stats["mean"],
            "std": final_stats["std"],
            "minus_1sigma": final_stats["minus_1sigma"],
            "plus_1sigma": final_stats["plus_1sigma"],
            "minus_2sigma": final_stats["minus_2sigma"],
            "plus_2sigma": final_stats["plus_2sigma"],
            "minus_3sigma": final_stats["minus_3sigma"],
            "plus_3sigma": final_stats["plus_3sigma"],

            "original": group_original_count,
            "after_aa": group_after_aa_count,
            "final": group_final_count,
            "invalid_aa_removed": group_invalid_aa_removed,
            "length_removed": group_length_removed,
            "total_removed": total_removed,
            "keep_rate": keep_rate,
            "invalid_chars": dict(invalid_char_counter)
        })

        total_original_all += group_original_count
        total_after_aa_all += group_after_aa_count
        total_final_all += group_final_count
        total_invalid_aa_removed_all += group_invalid_aa_removed
        total_length_removed_all += group_length_removed

    write_raw_length_table(raw_length_records)
    write_clean_length_table(clean_length_records)

    generate_report(
        all_reports,
        file_reports,
        total_original_all,
        total_after_aa_all,
        total_final_all,
        total_invalid_aa_removed_all,
        total_length_removed_all,
        output_dir
    )

    plot_gvp8_length_3std_from_reports(all_reports)

    print("\n" + "=" * 90)
    print("🎉 全部完成！")
    print(f"📂 输出目录：{output_dir}")
    print(f"📄 统计报告：{REPORT_FILE}")
    print(f"📈 原始长度分布表：{RAW_LENGTH_TABLE_FILE}")
    print(f"📈 去非标准氨基酸后长度分布表：{CLEAN_LENGTH_TABLE_FILE}")
    print(f"📊 长度范围图 PNG：{FIG_OUTPUT_PNG}")
    print(f"📊 长度范围图 PDF：{FIG_OUTPUT_PDF}")
    print(f"📄 画图数据表：{FIG_DATA_TSV}")
    print(f"📊 总计：原始 {total_original_all:,} 条")
    print(f"📊 去除非标准氨基酸后：{total_after_aa_all:,} 条")
    print(f"📊 最终保留：{total_final_all:,} 条")
    print(f"📊 非标准氨基酸剔除：{total_invalid_aa_removed_all:,} 条")
    print(f"📊 三标准差长度剔除：{total_length_removed_all:,} 条")
    print(f"📊 总剔除：{total_original_all - total_final_all:,} 条")

    if total_original_all:
        print(f"📊 总保留率：{total_final_all / total_original_all * 100:.2f}%")

    print("=" * 90)


def generate_report(
    type_reports,
    file_reports,
    total_original,
    total_after_aa,
    total_final,
    total_invalid_aa_removed,
    total_length_removed,
    out_dir
):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    report_dir = os.path.dirname(REPORT_FILE)
    if report_dir and not os.path.exists(report_dir):
        os.makedirs(report_dir)

    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write("序列清洗前后对比统计报告\n")
        f.write("清洗方式：先去除非标准氨基酸，再按每个基因类型独立进行 3-Sigma 长度过滤。\n")
        f.write("注意：实际过滤阈值基于去非标准氨基酸后的序列计算；报告中的 μ、σ、μ±3σ 基于剔除长度异常后的最终保留序列重新计算。\n")
        f.write(f"时间：{ts}\n")
        f.write(f"输入目录：{INPUT_DIR}\n")
        f.write(f"输出目录：{out_dir}\n")
        f.write("=" * 150 + "\n\n")

        f.write("一、总体统计\n")
        f.write("-" * 150 + "\n")
        f.write(f"原始序列总数：{total_original:,}\n")
        f.write(f"去除非标准氨基酸后序列数：{total_after_aa:,}\n")
        f.write(f"最终保留序列数：{total_final:,}\n")
        f.write(f"非标准氨基酸剔除数：{total_invalid_aa_removed:,}\n")
        f.write(f"三标准差长度异常剔除数：{total_length_removed:,}\n")
        f.write(f"总剔除数：{total_original - total_final:,}\n")

        if total_original:
            f.write(f"非标准氨基酸剔除率：{total_invalid_aa_removed / total_original * 100:.2f}%\n")
            f.write(f"长度异常剔除率：{total_length_removed / total_original * 100:.2f}%\n")
            f.write(f"总保留率：{total_final / total_original * 100:.2f}%\n")

        f.write("\n\n")

        f.write("二、按基因类型统计\n")
        f.write("-" * 300 + "\n")
        f.write(
            f"{'类型':<12} | "
            f"{'原始数':>8} | "
            f"{'去非标准后':>10} | "
            f"{'最终保留':>8} | "
            f"{'非标准剔除':>10} | "
            f"{'长度剔除':>8} | "
            f"{'最终最短':>8} | "
            f"{'最终最长':>8} | "
            f"{'最终μ':>8} | "
            f"{'最终σ':>8} | "
            f"{'最终μ-3σ':>10} | "
            f"{'最终μ+3σ':>10} | "
            f"{'过滤阈值':>18} | "
            f"{'保留率':>8} | "
            f"非标准字符\n"
        )
        f.write("-" * 300 + "\n")

        for item in type_reports:
            invalid_chars = "-"
            if item["invalid_chars"]:
                invalid_chars = ", ".join(
                    [f"{k}:{v}" for k, v in sorted(item["invalid_chars"].items())]
                )

            f.write(
                f"{item['type']:<12} | "
                f"{item['original']:>8} | "
                f"{item['after_aa']:>10} | "
                f"{item['final']:>8} | "
                f"{item['invalid_aa_removed']:>10} | "
                f"{item['length_removed']:>8} | "
                f"{item['clean_min_len']:>8.0f} | "
                f"{item['clean_max_len']:>8.0f} | "
                f"{item['mean']:>8.2f} | "
                f"{item['std']:>8.2f} | "
                f"{item['minus_3sigma']:>10.2f} | "
                f"{item['plus_3sigma']:>10.2f} | "
                f"{item['filter_min']:>8}-{item['filter_max']:<8} | "
                f"{item['keep_rate']:>7.2f}% | "
                f"{invalid_chars}\n"
            )

        f.write("-" * 300 + "\n\n")

        f.write("三、按文件统计\n")
        f.write("-" * 150 + "\n")
        f.write(
            f"{'类型':<12} | "
            f"{'文件':<50} | "
            f"{'原始数':>8} | "
            f"{'去非标准后':>10} | "
            f"{'最终保留':>8} | "
            f"{'非标准剔除':>10} | "
            f"{'长度剔除':>8} | "
            f"{'总剔除':>8}\n"
        )
        f.write("-" * 150 + "\n")

        for item in file_reports:
            fname = item["file"]
            if len(fname) > 48:
                fname = fname[:45] + "..."

            f.write(
                f"{item['type']:<12} | "
                f"{fname:<50} | "
                f"{item['original']:>8} | "
                f"{item['after_aa']:>10} | "
                f"{item['final']:>8} | "
                f"{item['invalid_aa_removed']:>10} | "
                f"{item['length_removed']:>8} | "
                f"{item['total_removed']:>8}\n"
            )

        f.write("-" * 150 + "\n\n")

        f.write("四、说明\n")
        f.write("-" * 150 + "\n")
        f.write("1. 过滤阈值：基于去除非标准氨基酸后的序列长度，使用样本标准差 ddof=1，按 round(mean ± 3*std) 计算。\n")
        f.write("2. 最终最短/最长、最终 μ、最终 σ、最终 μ±3σ：均基于剔除长度异常后的 Kept 序列重新计算。\n")
        f.write("3. 原始长度分布表包含所有原始序列，包括后续因非标准氨基酸被剔除的序列。\n")
        f.write("4. 去非标准氨基酸后长度分布表包含每条进入长度判断的序列，同时记录最终统计量和过滤阈值来源。\n")
        f.write("5. 当前允许的氨基酸字符集合为：")
        f.write("".join(sorted(STANDARD_AA)) + "\n")


if __name__ == "__main__":
    print("即将进行序列清洗统计。")
    print("清洗顺序：原始序列 -> 去除非标准氨基酸 -> 每个类型独立 3-Sigma 长度过滤")
    print("注意：最终平均长度和3SD位置将在剔除长度异常后重新计算。")
    print(f"源目录：{INPUT_DIR}")
    print(f"目标目录：{OUTPUT_DIR}")
    print(f"报告文件：{REPORT_FILE}")
    print(f"原始长度分布表：{RAW_LENGTH_TABLE_FILE}")
    print(f"去非标准氨基酸后长度分布表：{CLEAN_LENGTH_TABLE_FILE}")
    print(f"长度图 PNG：{FIG_OUTPUT_PNG}")
    print(f"长度图 PDF：{FIG_OUTPUT_PDF}")

    conf = input("确认执行？(y/n): ")

    if conf.lower() == "y":
        process_directory_by_type(INPUT_DIR, OUTPUT_DIR)
    else:
        print("已取消。")
