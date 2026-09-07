#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
表4-11 候选片段 ESMFold API 结构预测与 PyMOL 二级结构统计结果

流程：
1. 读取之前已经去重后的候选片段表；
2. 每个候选片段选择最多5条代表序列；
   - organism 不能为空；
   - 序列长度必须覆盖该片段；
   - 不允许重复物种 organism；
   - 优先选择不同属 genus；
   - 如果不同属不足5条，再允许同属但不同物种 organism；
   - 如果不同物种不足5条，不补齐，选多少是多少；
   - 不考虑 full_length 接近平均长度；
3. 调用 ESM Atlas API 进行 ESMFold 结构预测；
4. 保存 PDB；
5. 使用 PyMOL dss 识别二级结构；
6. 使用 CA 原子数近似残基数，统计：
   - α螺旋比例
   - β折叠比例
   - 无规则卷曲比例
   - 含二级结构片段占比
7. 输出表4-11和结构图片。

额外输出：
1. all_selected_fragment_records.csv
   严格物种去重后的代表片段明细。

2. strict_species_selection_summary_for_structure.csv
   每个候选片段的选择摘要。

3. strict_species_all_available_species_for_structure.csv
   每个候选片段全部可用物种名，一行一个物种。

4. all_predicted_pdb_records.csv
   ESMFold API 预测得到的 PDB 记录。

5. ESMFold_PyMOL_secondary_structure_detail.csv
   每个 PDB 的 PyMOL 二级结构明细。

6. Table_4_11_ESMFold_PyMOL_secondary_structure_summary.csv
   表4-11区域汇总。
"""

import os
import re
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm
import pymol2


warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


# =========================================================
# 1. 路径配置
# =========================================================

# 原始序列 JSON 根目录
DATA_ROOT = r"C:\Users\r9000\Desktop\毕设（无监督聚类）\新gvp"

# 直接读取之前已经去重后的候选片段表
DEDUP_REGION_TABLE = (
    r"C:\Users\r9000\Desktop\毕设（无监督聚类）"
    r"\GVP_Four_Type_Clustering_Results"
    r"\Final_Passed_NonClose_Candidates"
    r"\Passed_Candidate_NonClose_All_Detail_For_Downstream.csv"
)

# 输出目录：API版本单独保存，避免覆盖本地 ESMFold 版本
OUTPUT_ROOT = r"C:\Users\r9000\Desktop\毕设（无监督聚类）\Table_4_11_ESMFold_API_Structure_Results"

FRAGMENT_DIR = os.path.join(OUTPUT_ROOT, "selected_fragments")
PDB_DIR = os.path.join(OUTPUT_ROOT, "predicted_pdb")
PNG_DIR = os.path.join(OUTPUT_ROOT, "rendered_png")
TABLE_DIR = os.path.join(OUTPUT_ROOT, "tables")

os.makedirs(OUTPUT_ROOT, exist_ok=True)
os.makedirs(FRAGMENT_DIR, exist_ok=True)
os.makedirs(PDB_DIR, exist_ok=True)
os.makedirs(PNG_DIR, exist_ok=True)
os.makedirs(TABLE_DIR, exist_ok=True)


# =========================================================
# 2. 参数配置
# =========================================================

GVP_TYPES = [
    "gvpa", "gvpc", "gvpn", "gvpo",
    "gvpg", "gvpj", "gvpk", "gvpp"
]

WINDOW_CONFIG = {
    "gvpa": 30,
    "gvpc": 35,
    "gvpn": 40,
    "gvpo": 50,
    "gvpg": 25,
    "gvpj": 35,
    "gvpk": 25,
    "gvpp": 40
}

# 每个候选片段最多选择5条代表序列
SELECT_NUM_PER_REGION = 5

# API 配置
API_URL = "https://api.esmatlas.com/foldSequence/v1/pdb"
TIMEOUT = 1200
RETRY_TIMES = 3
RETRY_SLEEP = 5

# 防止API请求太密集；不想等待可设为0
API_SLEEP_BETWEEN_REQUESTS = 0.5

# 如果PDB已存在，是否跳过重新预测
SKIP_EXISTING_PDB = True

# 如果PNG已存在，是否跳过重新渲染
SKIP_EXISTING_PNG = False


# =========================================================
# 3. 工具函数
# =========================================================

def format_gvp_name(gvp_type):
    """
    gvpa -> GvpA
    """
    return f"Gvp{gvp_type[-1].upper()}"


def gvp_display_to_key(name):
    """
    GvpA -> gvpa
    gvpa -> gvpa
    """
    s = str(name).strip()

    if re.fullmatch(r"Gvp[A-Z]", s, flags=re.IGNORECASE):
        return "gvp" + s[-1].lower()

    return s.lower()


def clean_sequence(seq):
    """
    只保留20种标准氨基酸。
    """
    return "".join([
        c for c in str(seq).strip().upper()
        if c in "ACDEFGHIKLMNPQRSTVWY"
    ])


def safe_str(x):
    if x is None:
        return ""

    s = str(x).strip()

    if s.lower() in ["nan", "none", "null"]:
        return ""

    return s


def get_sequence_id(item, idx):
    return (
        item.get("unique_sequence_id")
        or item.get("id")
        or item.get("accession")
        or item.get("protein_id")
        or f"seq_{idx + 1}"
    )


def get_genus(organism):
    organism = safe_str(organism)

    if not organism:
        return ""

    parts = organism.split()

    if len(parts) == 0:
        return ""

    return parts[0]


def parse_region_str(region_str):
    """
    解析片段位置：
    83-113
    83–113
    83—113
    """
    if pd.isna(region_str):
        return None, None

    s = str(region_str).strip()
    s = s.replace("–", "-").replace("—", "-")

    m = re.match(r"^\s*(\d+)\s*-\s*(\d+)\s*$", s)

    if not m:
        return None, None

    start = int(m.group(1))
    end = int(m.group(2))

    if end < start:
        start, end = end, start

    return start, end


def region_to_string(start, end):
    return f"{int(start)}-{int(end)}"


def read_csv_safely(path):
    try:
        return pd.read_csv(path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="utf-8")


def sanitize_filename(text, max_len=80):
    """
    清理文件名，防止路径过长和特殊符号报错。
    """
    text = str(text)

    for ch in ['\\', '/', ':', '*', '?', '"', '<', '>', '|', ';', '[', ']']:
        text = text.replace(ch, "_")

    text = text.replace(" ", "_")
    text = re.sub(r"_+", "_", text)

    return text[:max_len]


# =========================================================
# 4. 读取去重后的候选片段表
# =========================================================

def load_deduplicated_regions(csv_path):
    """
    直接读取已经去重后的候选片段表。

    支持字段：
    1. 原始GVP类型 + start_floor + end_floor
    2. GVP类型 + 片段位置
    3. GVP类型 + 平均起始位置 + 窗口长度
    """

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"未找到去重候选片段表：{csv_path}")

    df = read_csv_safely(csv_path)

    if df.empty:
        raise ValueError("去重候选片段表为空")

    records = []

    for _, row in df.iterrows():
        # GVP类型
        if "原始GVP类型" in df.columns and not pd.isna(row.get("原始GVP类型")):
            gvp_key = str(row["原始GVP类型"]).strip().lower()
            gvp_name = format_gvp_name(gvp_key)

        elif "GVP类型" in df.columns and not pd.isna(row.get("GVP类型")):
            gvp_name_raw = str(row["GVP类型"]).strip()
            gvp_key = gvp_display_to_key(gvp_name_raw)
            gvp_name = format_gvp_name(gvp_key)

        else:
            raise ValueError("去重候选片段表缺少 GVP类型 或 原始GVP类型 字段")

        # 位置优先级1：start_floor / end_floor
        if "start_floor" in df.columns and "end_floor" in df.columns:
            if pd.isna(row["start_floor"]) or pd.isna(row["end_floor"]):
                continue

            start = int(row["start_floor"])
            end = int(row["end_floor"])

        # 位置优先级2：片段位置
        elif "片段位置" in df.columns:
            start, end = parse_region_str(row["片段位置"])

            if start is None or end is None:
                continue

        # 位置优先级3：平均起始位置 + 窗口长度
        elif "平均起始位置" in df.columns and "窗口长度" in df.columns:
            start = int(np.floor(float(row["平均起始位置"])))
            end = start + int(row["窗口长度"])

        else:
            raise ValueError(
                "候选片段表必须包含以下字段之一：\n"
                "1. start_floor + end_floor\n"
                "2. 片段位置\n"
                "3. 平均起始位置 + 窗口长度"
            )

        if end <= start:
            continue

        region_str = region_to_string(start, end)

        if "cluster_id" in df.columns and not pd.isna(row.get("cluster_id")):
            cluster_id = row["cluster_id"]
        else:
            cluster_id = f"{gvp_key}_{region_str}"

        records.append({
            "GVP类型": gvp_name,
            "gvp_key": gvp_key,
            "cluster_id": cluster_id,
            "片段位置": region_str,
            "start": int(start),
            "end": int(end),
            "window_length": int(end - start)
        })

    region_df = pd.DataFrame(records)

    if region_df.empty:
        raise ValueError("未从去重候选片段表中解析到有效片段")

    region_df = region_df.drop_duplicates(
        subset=["gvp_key", "start", "end"]
    ).reset_index(drop=True)

    gvp_order = {gvp: i for i, gvp in enumerate(GVP_TYPES)}
    region_df["sort_order"] = region_df["gvp_key"].map(gvp_order)

    region_df = region_df.sort_values(
        by=["sort_order", "start", "end"]
    ).drop(columns=["sort_order"]).reset_index(drop=True)

    out_path = os.path.join(
        TABLE_DIR,
        "normalized_deduplicated_regions_for_structure.csv"
    )

    region_df.to_csv(
        out_path,
        index=False,
        encoding="utf-8-sig"
    )

    print(f"✅ 已读取去重候选片段表：{csv_path}")
    print(f"✅ 标准化片段表已保存：{out_path}")
    print(f"候选片段总数：{len(region_df)}")

    return region_df


# =========================================================
# 5. 读取原始序列
# =========================================================

def load_gvp_sequences(gvp_key):
    """
    读取某个GVP的所有序列。
    要求 organism 非空。
    """

    json_path = os.path.join(
        DATA_ROOT,
        gvp_key,
        f"Gvp{gvp_key[-1].upper()}_sequences.json"
    )

    if not os.path.exists(json_path):
        print(f"❌ 未找到序列文件：{json_path}")
        return []

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    records = []

    for idx, item in enumerate(data):
        seq = clean_sequence(item.get("sequence", ""))

        if len(seq) < 10:
            continue

        annotation = item.get("representative_annotation", {}) or {}
        organism = safe_str(annotation.get("organism"))

        # 物种名不能为空
        if not organism:
            continue

        genus = get_genus(organism)

        if not genus:
            continue

        records.append({
            "seq_id": get_sequence_id(item, idx),
            "sequence": seq,
            "full_length": len(seq),
            "annotation": annotation,
            "organism": organism,
            "genus": genus
        })

    print(
        f"{format_gvp_name(gvp_key)} 有效序列数：{len(records)} "
        f"(organism非空)"
    )

    return records


# =========================================================
# 6. 选择每个片段的代表序列
# =========================================================

def select_representative_sequences_for_region(
    seq_records,
    start,
    end,
    select_num=5
):
    """
    严格选择规则：
    1. 序列长度必须覆盖该片段；
    2. organism 非空，已在读取阶段保证；
    3. 不允许重复物种 organism；
    4. 第一轮优先选择不同属 genus；
    5. 第二轮如果不同属不足 select_num，则允许同属但不同物种 organism；
    6. 如果不同物种不足 select_num，则不补齐，最终数量可以少于 select_num；
    7. 不考虑 full_length 接近平均长度。
    """

    valid = []

    for rec in seq_records:
        if len(rec["sequence"]) >= end:
            valid.append(rec)

    if len(valid) == 0:
        return [], []

    selected = []
    selected_ids = set()
    used_genera = set()
    used_organisms = set()

    # 第一轮：优先不同属
    for rec in valid:
        if len(selected) >= select_num:
            break

        seq_id = rec["seq_id"]
        genus = rec["genus"]
        organism = rec["organism"]

        if seq_id in selected_ids:
            continue

        if organism in used_organisms:
            continue

        if genus not in used_genera:
            selected.append(rec)
            selected_ids.add(seq_id)
            used_genera.add(genus)
            used_organisms.add(organism)

    # 第二轮：属不够时，补同属但不同物种
    # 仍然不允许重复 organism
    if len(selected) < select_num:
        for rec in valid:
            if len(selected) >= select_num:
                break

            seq_id = rec["seq_id"]
            organism = rec["organism"]

            if seq_id in selected_ids:
                continue

            if organism in used_organisms:
                continue

            selected.append(rec)
            selected_ids.add(seq_id)
            used_organisms.add(organism)

    return selected, valid


def extract_fragment(sequence, start, end):
    """
    与前面注意力/MSA一致：
    start 为 0-based，end 为右开区间。
    """

    if start < 0:
        start = 0

    if end > len(sequence):
        end = len(sequence)

    if start >= end:
        return None

    frag = sequence[start:end]

    if len(frag) < 5:
        return None

    return frag


def save_fasta(fragment_records, fasta_path):
    with open(fasta_path, "w", encoding="utf-8") as f:
        for rec in fragment_records:
            header = (
                f">{rec['seq_id']}"
                f"|organism={rec['organism']}"
                f"|genus={rec['genus']}"
                f"|full_length={rec['full_length']}"
                f"|region={rec['片段位置']}"
            )

            f.write(header + "\n")
            f.write(rec["fragment"] + "\n")


def build_fragment_dataset(region_df):
    """
    为每个候选片段构建最多5条代表片段。

    严格规则：
    - 不允许重复物种；
    - 优先不同属；
    - 不同属不足时可同属不同物种；
    - 不同物种不足5条时不补齐。
    """

    all_fragment_rows = []
    selection_summary_rows = []
    all_species_rows = []

    for gvp_key, sub_df in region_df.groupby("gvp_key", sort=False):
        print("\n" + "=" * 90)
        print(f"构建 {format_gvp_name(gvp_key)} 代表候选片段")
        print("=" * 90)

        seq_records = load_gvp_sequences(gvp_key)

        if len(seq_records) == 0:
            print(f"⚠️ {format_gvp_name(gvp_key)} 没有可用序列")
            continue

        gvp_out_dir = os.path.join(
            FRAGMENT_DIR,
            format_gvp_name(gvp_key)
        )

        os.makedirs(gvp_out_dir, exist_ok=True)

        for _, region in sub_df.iterrows():
            gvp_name = region["GVP类型"]
            cluster_id = region["cluster_id"]
            start = int(region["start"])
            end = int(region["end"])
            region_str = region["片段位置"]

            selected, valid = select_representative_sequences_for_region(
                seq_records,
                start=start,
                end=end,
                select_num=SELECT_NUM_PER_REGION
            )

            valid_organisms = sorted(set([x["organism"] for x in valid]))
            valid_genera = sorted(set([x["genus"] for x in valid]))

            selected_organisms = [x["organism"] for x in selected]
            selected_genera = [x["genus"] for x in selected]
            selected_seq_ids = [str(x["seq_id"]) for x in selected]
            selected_organism_set = set(selected_organisms)

            # 输出所有可用物种，一行一个物种
            for species_rank, organism in enumerate(valid_organisms, 1):
                species_records = [
                    x for x in valid
                    if x["organism"] == organism
                ]

                genus = species_records[0]["genus"] if species_records else ""
                seq_ids_for_species = [
                    str(x["seq_id"])
                    for x in species_records
                ]

                all_species_rows.append({
                    "GVP类型": gvp_name,
                    "gvp_key": gvp_key,
                    "cluster_id": cluster_id,
                    "片段位置": region_str,
                    "start": start,
                    "end": end,
                    "species_rank": species_rank,
                    "organism": organism,
                    "genus": genus,
                    "该物种覆盖序列数": len(species_records),
                    "该物种覆盖seq_id列表": ";".join(seq_ids_for_species),
                    "是否被严格规则选中": "是" if organism in selected_organism_set else "否"
                })

            if len(selected) == 0:
                print(f"⚠️ {gvp_name} {region_str} 没有可用代表序列")

                selection_summary_rows.append({
                    "GVP类型": gvp_name,
                    "gvp_key": gvp_key,
                    "cluster_id": cluster_id,
                    "片段位置": region_str,
                    "start": start,
                    "end": end,
                    "覆盖该片段的序列数": len(valid),
                    "可用不同属数": len(valid_genera),
                    "可用不同物种数": len(valid_organisms),
                    "全部可用属列表": ";".join(valid_genera),
                    "全部可用物种列表": ";".join(valid_organisms),
                    "selected_count": 0,
                    "selected_unique_genus_count": 0,
                    "selected_unique_organism_count": 0,
                    "selected_seq_ids": "",
                    "selected_genera": "",
                    "selected_organisms": "",
                    "selection_status": "无可用覆盖序列"
                })

                continue

            fragment_records = []

            for rec in selected:
                frag = extract_fragment(
                    rec["sequence"],
                    start,
                    end
                )

                if frag is None:
                    continue

                fragment_records.append({
                    "GVP类型": gvp_name,
                    "gvp_key": gvp_key,
                    "cluster_id": cluster_id,
                    "片段位置": region_str,
                    "start": start,
                    "end": end,
                    "seq_id": rec["seq_id"],
                    "organism": rec["organism"],
                    "genus": rec["genus"],
                    "full_length": rec["full_length"],
                    "fragment": frag,
                    "fragment_length": len(frag)
                })

            if len(fragment_records) == 0:
                print(f"⚠️ {gvp_name} {region_str} 片段提取失败")

                selection_summary_rows.append({
                    "GVP类型": gvp_name,
                    "gvp_key": gvp_key,
                    "cluster_id": cluster_id,
                    "片段位置": region_str,
                    "start": start,
                    "end": end,
                    "覆盖该片段的序列数": len(valid),
                    "可用不同属数": len(valid_genera),
                    "可用不同物种数": len(valid_organisms),
                    "全部可用属列表": ";".join(valid_genera),
                    "全部可用物种列表": ";".join(valid_organisms),
                    "selected_count": 0,
                    "selected_unique_genus_count": 0,
                    "selected_unique_organism_count": 0,
                    "selected_seq_ids": "",
                    "selected_genera": "",
                    "selected_organisms": "",
                    "selection_status": "片段提取失败"
                })

                continue

            selected_genera_real = [x["genus"] for x in fragment_records]
            selected_organisms_real = [x["organism"] for x in fragment_records]
            selected_seq_ids_real = [str(x["seq_id"]) for x in fragment_records]

            selected_unique_genus_count = len(set(selected_genera_real))
            selected_unique_organism_count = len(set(selected_organisms_real))

            if len(fragment_records) >= SELECT_NUM_PER_REGION:
                if selected_unique_genus_count >= SELECT_NUM_PER_REGION:
                    selection_status = "已选满5条，且5条不同属"
                else:
                    selection_status = "已选满5条，含同属不同物种"
            else:
                selection_status = f"不同物种不足5条，仅选择{len(fragment_records)}条"

            selection_summary_rows.append({
                "GVP类型": gvp_name,
                "gvp_key": gvp_key,
                "cluster_id": cluster_id,
                "片段位置": region_str,
                "start": start,
                "end": end,
                "覆盖该片段的序列数": len(valid),
                "可用不同属数": len(valid_genera),
                "可用不同物种数": len(valid_organisms),
                "全部可用属列表": ";".join(valid_genera),
                "全部可用物种列表": ";".join(valid_organisms),
                "selected_count": len(fragment_records),
                "selected_unique_genus_count": selected_unique_genus_count,
                "selected_unique_organism_count": selected_unique_organism_count,
                "selected_seq_ids": ";".join(selected_seq_ids_real),
                "selected_genera": ";".join(selected_genera_real),
                "selected_organisms": ";".join(selected_organisms_real),
                "selection_status": selection_status
            })

            fasta_path = os.path.join(
                gvp_out_dir,
                f"{gvp_name}_region_{start}_{end}.fasta"
            )

            save_fasta(fragment_records, fasta_path)

            region_detail_path = os.path.join(
                gvp_out_dir,
                f"{gvp_name}_region_{start}_{end}_selected_sequences.csv"
            )

            pd.DataFrame(fragment_records).to_csv(
                region_detail_path,
                index=False,
                encoding="utf-8-sig"
            )

            print(
                f"✅ {gvp_name} {region_str} "
                f"选择代表片段数：{len(fragment_records)} | {selection_status}"
            )

            all_fragment_rows.extend(fragment_records)

    fragment_df = pd.DataFrame(all_fragment_rows)
    selection_summary_df = pd.DataFrame(selection_summary_rows)
    all_species_df = pd.DataFrame(all_species_rows)

    selection_summary_path = os.path.join(
        TABLE_DIR,
        "strict_species_selection_summary_for_structure.csv"
    )

    all_species_path = os.path.join(
        TABLE_DIR,
        "strict_species_all_available_species_for_structure.csv"
    )

    selection_summary_df.to_csv(
        selection_summary_path,
        index=False,
        encoding="utf-8-sig"
    )

    all_species_df.to_csv(
        all_species_path,
        index=False,
        encoding="utf-8-sig"
    )

    print(f"\n✅ 严格不同物种选择摘要已保存：{selection_summary_path}")
    print(f"✅ 每个候选片段全部可用物种名已保存：{all_species_path}")

    if fragment_df.empty:
        raise ValueError("未生成任何代表候选片段")

    fragment_path = os.path.join(
        TABLE_DIR,
        "all_selected_fragment_records.csv"
    )

    fragment_df.to_csv(
        fragment_path,
        index=False,
        encoding="utf-8-sig"
    )

    print(f"✅ 所有代表片段记录已保存：{fragment_path}")
    print(f"✅ 代表片段总数：{len(fragment_df)}")

    return fragment_df


# =========================================================
# 7. ESMFold API 结构预测
# =========================================================

class ESMFoldAPIPredictor:
    """
    使用 ESM Atlas API 进行 ESMFold 结构预测。
    不需要本地安装 ESMFold/openfold/omegaconf。
    """

    def __init__(self):
        print("\n🌐 使用 ESM Atlas API 进行 ESMFold 结构预测")
        print(f"API: {API_URL}")

    def predict_pdb(self, sequence):
        seq = clean_sequence(sequence)

        if len(seq) < 5:
            return None

        for attempt in range(1, RETRY_TIMES + 1):
            try:
                res = requests.post(
                    API_URL,
                    data=seq,
                    timeout=TIMEOUT
                )

                res.raise_for_status()
                pdb_data = res.text

                if not pdb_data or "ATOM" not in pdb_data:
                    print(f"⚠️ API返回内容异常，长度={len(pdb_data)}")
                    return None

                return pdb_data

            except Exception as e:
                print(
                    f"⚠️ ESMFold API预测失败，第 {attempt}/{RETRY_TIMES} 次：{e}"
                )

                if attempt < RETRY_TIMES:
                    time.sleep(RETRY_SLEEP)

        return None


def predict_all_structures(fragment_df, predictor):
    """
    对所有代表候选片段进行ESMFold API结构预测。
    """

    pdb_rows = []

    for _, row in tqdm(
        fragment_df.iterrows(),
        total=len(fragment_df),
        desc="ESMFold API结构预测"
    ):
        gvp_name = row["GVP类型"]
        start = int(row["start"])
        end = int(row["end"])
        region_str = row["片段位置"]
        seq_id = str(row["seq_id"])
        fragment = row["fragment"]

        safe_seq_id = sanitize_filename(seq_id)

        out_dir = os.path.join(
            PDB_DIR,
            gvp_name,
            f"region_{start}_{end}"
        )

        os.makedirs(out_dir, exist_ok=True)

        pdb_path = os.path.join(
            out_dir,
            f"{safe_seq_id}_region_{start}_{end}.pdb"
        )

        if SKIP_EXISTING_PDB and os.path.exists(pdb_path):
            pass

        else:
            try:
                pdb_str = predictor.predict_pdb(fragment)

                if API_SLEEP_BETWEEN_REQUESTS and API_SLEEP_BETWEEN_REQUESTS > 0:
                    time.sleep(API_SLEEP_BETWEEN_REQUESTS)

                if pdb_str is None:
                    continue

                with open(pdb_path, "w", encoding="utf-8") as f:
                    f.write(pdb_str)

            except Exception as e:
                print(f"⚠️ ESMFold API预测失败：{seq_id} {region_str} -> {e}")
                continue

        pdb_rows.append({
            "GVP类型": gvp_name,
            "gvp_key": row["gvp_key"],
            "cluster_id": row["cluster_id"],
            "片段位置": region_str,
            "start": start,
            "end": end,
            "seq_id": seq_id,
            "organism": row["organism"],
            "genus": row["genus"],
            "fragment_length": int(row["fragment_length"]),
            "pdb_path": pdb_path
        })

    pdb_df = pd.DataFrame(pdb_rows)

    if pdb_df.empty:
        raise ValueError("未生成任何PDB文件")

    pdb_record_path = os.path.join(
        TABLE_DIR,
        "all_predicted_pdb_records.csv"
    )

    pdb_df.to_csv(
        pdb_record_path,
        index=False,
        encoding="utf-8-sig"
    )

    print(f"\n✅ PDB预测记录已保存：{pdb_record_path}")

    return pdb_df


# =========================================================
# 8. PyMOL二级结构统计与渲染
# =========================================================

def render_and_calc_ss(cmd, pdb_path, out_png):
    """
    PyMOL渲染并统计二级结构。

    使用 CA 原子数近似残基数：
        total_ca
        helix_ca: ss H
        sheet_ca: ss S
        coil_ca: total_ca - helix_ca - sheet_ca

    含二级结构：
        helix_ca + sheet_ca > 0
    """

    cmd.reinitialize()

    cmd.bg_color("white")
    cmd.set("ray_opaque_background", 1)
    cmd.set("antialias", 2)
    cmd.set("orthoscopic", 1)
    cmd.set("cartoon_fancy_helices", 1)
    cmd.set("ray_shadows", 0)
    cmd.set("specular", 0.2)
    cmd.set("ambient", 0.55)
    cmd.set("direct", 0.35)

    cmd.load(str(pdb_path), "obj")
    cmd.remove("solvent")
    cmd.hide("everything", "all")
    cmd.show("cartoon", "obj and polymer.protein")

    # 重新识别二级结构
    cmd.dss("obj")

    # Nature风格配色
    # helix: 红色，sheet: 蓝色，coil: 灰色
    cmd.color("gray85", "obj and polymer.protein")
    cmd.color("firebrick", "obj and polymer.protein and ss H")
    cmd.color("marine", "obj and polymer.protein and ss S")
    cmd.color("gray75", "obj and polymer.protein and not ss H+S")

    cmd.orient("obj")
    cmd.zoom("obj", 2.0)

    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)

    if not (SKIP_EXISTING_PNG and out_png.exists()):
        cmd.ray(1800, 1400)
        cmd.png(str(out_png), dpi=300)

    total_ca = cmd.count_atoms(
        "obj and polymer.protein and name CA"
    )

    helix_ca = cmd.count_atoms(
        "obj and polymer.protein and name CA and ss H"
    )

    sheet_ca = cmd.count_atoms(
        "obj and polymer.protein and name CA and ss S"
    )

    coil_ca = max(total_ca - helix_ca - sheet_ca, 0)

    has_secondary_structure = 1 if (helix_ca + sheet_ca) > 0 else 0

    return {
        "total_ca": int(total_ca),
        "helix_ca": int(helix_ca),
        "sheet_ca": int(sheet_ca),
        "coil_ca": int(coil_ca),
        "has_secondary_structure": int(has_secondary_structure)
    }


def render_all_and_summarize_ss(pdb_df):
    """
    生成：
    1. 每个PDB二级结构明细表；
    2. 表4-11区域汇总表。

    表4-11字段：
        GVP类型
        片段位置
        结构预测数
        α螺旋比例
        β折叠比例
        无规则卷曲比例
        含二级结构片段占比
    """

    detail_rows = []

    with pymol2.PyMOL() as pm:
        cmd = pm.cmd

        for _, row in tqdm(
            pdb_df.iterrows(),
            total=len(pdb_df),
            desc="PyMOL二级结构统计与渲染"
        ):
            gvp_name = row["GVP类型"]
            start = int(row["start"])
            end = int(row["end"])
            region_str = row["片段位置"]
            seq_id = str(row["seq_id"])
            safe_seq_id = sanitize_filename(seq_id)

            pdb_path = Path(row["pdb_path"])

            out_png = (
                Path(PNG_DIR)
                / gvp_name
                / f"region_{start}_{end}"
                / f"{safe_seq_id}_region_{start}_{end}.png"
            )

            try:
                ss = render_and_calc_ss(
                    cmd,
                    pdb_path,
                    out_png
                )

            except Exception as e:
                print(f"⚠️ PyMOL处理失败：{pdb_path.name} -> {e}")
                continue

            detail_rows.append({
                "GVP类型": gvp_name,
                "gvp_key": row["gvp_key"],
                "cluster_id": row["cluster_id"],
                "片段位置": region_str,
                "start": start,
                "end": end,
                "seq_id": seq_id,
                "organism": row["organism"],
                "genus": row["genus"],
                "pdb_path": str(pdb_path),
                "png_path": str(out_png),

                "total_ca": ss["total_ca"],
                "helix_ca": ss["helix_ca"],
                "sheet_ca": ss["sheet_ca"],
                "coil_ca": ss["coil_ca"],
                "has_secondary_structure": ss["has_secondary_structure"]
            })

    detail_df = pd.DataFrame(detail_rows)

    if detail_df.empty:
        raise ValueError("未生成任何二级结构统计结果")

    detail_path = os.path.join(
        TABLE_DIR,
        "ESMFold_PyMOL_secondary_structure_detail.csv"
    )

    detail_df.to_csv(
        detail_path,
        index=False,
        encoding="utf-8-sig"
    )

    print(f"\n✅ 二级结构明细表已保存：{detail_path}")

    # =====================================================
    # 表4-11：按候选区域汇总
    # =====================================================

    summary_rows = []

    grouped = detail_df.groupby(
        ["GVP类型", "片段位置"],
        sort=False
    )

    for (gvp_name, region_str), sub_df in grouped:
        total_ca_sum = int(sub_df["total_ca"].sum())
        helix_ca_sum = int(sub_df["helix_ca"].sum())
        sheet_ca_sum = int(sub_df["sheet_ca"].sum())
        coil_ca_sum = int(sub_df["coil_ca"].sum())

        structure_count = int(len(sub_df))

        if total_ca_sum > 0:
            helix_ratio = helix_ca_sum / total_ca_sum
            sheet_ratio = sheet_ca_sum / total_ca_sum
            coil_ratio = coil_ca_sum / total_ca_sum
        else:
            helix_ratio = 0.0
            sheet_ratio = 0.0
            coil_ratio = 0.0

        ss_fragment_ratio = (
            sub_df["has_secondary_structure"].sum() / structure_count
            if structure_count > 0
            else 0.0
        )

        start_sort = int(str(region_str).split("-")[0])

        summary_rows.append({
            "GVP类型": gvp_name,
            "片段位置": region_str,
            "结构预测数": structure_count,
            "α螺旋比例": round(float(helix_ratio), 4),
            "β折叠比例": round(float(sheet_ratio), 4),
            "无规则卷曲比例": round(float(coil_ratio), 4),
            "含二级结构片段占比": round(float(ss_fragment_ratio), 4),
            "排序_start": start_sort
        })

    summary_df = pd.DataFrame(summary_rows)

    summary_df = summary_df.sort_values(
        by=["GVP类型", "排序_start"]
    ).drop(columns=["排序_start"]).reset_index(drop=True)

    table_4_11_path = os.path.join(
        TABLE_DIR,
        "Table_4_11_ESMFold_PyMOL_secondary_structure_summary.csv"
    )

    summary_df.to_csv(
        table_4_11_path,
        index=False,
        encoding="utf-8-sig"
    )

    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 180)
    pd.set_option("display.max_colwidth", 50)

    print("\n" + "=" * 140)
    print("表4-11 候选片段ESMFold API结构预测与PyMOL二级结构统计结果")
    print("=" * 140)
    print(summary_df.to_string(index=False))
    print("=" * 140)

    print(f"✅ 表4-11 已保存：{table_4_11_path}")
    print(f"✅ PDB目录：{PDB_DIR}")
    print(f"✅ PNG图片目录：{PNG_DIR}")

    return detail_df, summary_df


# =========================================================
# 9. 主函数
# =========================================================

def main():
    print(f"读取去重候选片段表：{DEDUP_REGION_TABLE}")

    # 1. 读取已去重候选片段
    region_df = load_deduplicated_regions(
        DEDUP_REGION_TABLE
    )

    # 2. 每个片段选择最多5条代表序列
    fragment_df = build_fragment_dataset(
        region_df
    )

    # 3. 使用ESM Atlas API预测结构
    predictor = ESMFoldAPIPredictor()

    pdb_df = predict_all_structures(
        fragment_df,
        predictor
    )

    # 4. PyMOL统计二级结构并渲染图片
    render_all_and_summarize_ss(
        pdb_df
    )

    print("\n🎉 全部完成！")
    print(f"结果目录：{OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
