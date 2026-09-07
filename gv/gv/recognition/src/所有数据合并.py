#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
构建 GVP 序列级候选片段数据库：干净宽表 + 旧数据库字段兼容版

一行 = 一个 GVP 候选片段中的一条代表序列 / 一个结构预测记录。

重要规则：
1. 数据库主表是序列级：
   - seq_id
   - selected_fragment
   - full_sequence
   - pdb_path
   - image
   - PyMOL 单条结构统计

2. 但是 CV/SP、ATT、MSA 全部使用片段级汇总结果：
   - cluster_species_percent
   - cluster_cv_start_position
   - attn_avg_attn
   - attn_avg_percentile
   - msa_avg_score
   - msa_avg_percentile

3. 因此同一个 GVP + 片段位置下的多条序列记录，共享相同的：
   - CV/SP
   - ATT 汇总
   - MSA 汇总

4. JSON 匹配只匹配主记录：
   - unique_sequence_id
   - 顶层 id / accession / protein_id
   - representative_annotation.id

5. 不读取、不索引、不展开 redundant_members_details。

6. full_sequence 真实写入主表和 SQLite DB。

7. selected_fragment 从 all_selected_fragment_records.csv 按 seq_id 精确/别名补齐。

8. 二级结构占比拆分：
   - ss_alpha_ratio
   - ss_beta_ratio
   - ss_flexible_ratio

9. 兼容旧字段：
   - alpha_helix_ratio
   - beta_sheet_ratio
   - flexible_region_ratio

10. ss_ratio 保留，定义为 alpha + beta。

输出：
    GVP_Sequence_Level_Candidate_Database_API_CLEAN/GVP_sequence_level_candidate_database_clean_wide.csv
    GVP_Sequence_Level_Candidate_Database_API_CLEAN/GVP_sequence_level_candidate_database_clean_wide.tsv
    GVP_Sequence_Level_Candidate_Database_API_CLEAN/gvp_web.db
    GVP_Sequence_Level_Candidate_Database_API_CLEAN/GVP_sequence_level_candidate_database_missing_report.csv
"""

import csv
import json
import re
import sqlite3
import traceback
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple


# =========================================================
# 0. 路径配置
# =========================================================

PROJECT_ROOT = Path(r"C:\Users\r9000\Desktop\毕设（无监督聚类）")

DATA_ROOT = PROJECT_ROOT / "新gvp"

DEDUP_REGION_TABLE = (
    PROJECT_ROOT
    / "GVP_Four_Type_Clustering_Results"
    / "Final_Passed_NonClose_Candidates"
    / "Passed_Candidate_NonClose_All_Detail_For_Downstream.csv"
)

# 表4-9 Attention：使用片段级汇总表
ATTN_ROOT = PROJECT_ROOT / "Table_4_9_Attention_Results"
ATTN_DETAIL_DIR = ATTN_ROOT / "per_sequence_attention"
ATTN_STAT_DIR = ATTN_ROOT / "cluster_attention_stats"
ATTN_MAIN_TABLE = ATTN_ROOT / "Table_4_9_ESM2_attention_score_statistics.csv"

# 表4-10 MSA：使用片段级汇总表
MSA_ROOT = PROJECT_ROOT / "Table_4_10_MSA_Results_From_Final_Candidates"
MSA_TABLE_DIR = MSA_ROOT / "tables"
MSA_MAIN_TABLE = MSA_ROOT / "Table_4_10_MAFFT_conservation_analysis.csv"

# 表4-11 结构结果：优先 API 版，然后回退本地版
STRUCTURE_OUTPUT_CANDIDATES = [
    PROJECT_ROOT / "Table_4_11_ESMFold_API_Structure_Results",
    PROJECT_ROOT / "Table_4_11_ESMFold_Structure_Results",
]

DATABASE_DIR = PROJECT_ROOT / "GVP_Sequence_Level_Candidate_Database_API_CLEAN"
DATABASE_DIR.mkdir(parents=True, exist_ok=True)

OUT_DATABASE_CSV = DATABASE_DIR / "GVP_sequence_level_candidate_database_clean_wide.csv"
OUT_DATABASE_TSV = DATABASE_DIR / "GVP_sequence_level_candidate_database_clean_wide.tsv"
OUT_DATABASE_DB = DATABASE_DIR / "gvp_web.db"
OUT_MISSING_REPORT_CSV = DATABASE_DIR / "GVP_sequence_level_candidate_database_missing_report.csv"
OUT_LOG = DATABASE_DIR / "GVP_sequence_level_candidate_database_build.log"

DB_TABLE = "sequence_records"

WRITE_FULL_SEQUENCE_TO_MAIN = True


# =========================================================
# 1. GVP 配置
# =========================================================

GVP_TYPES = [
    "gvpa", "gvpc", "gvpn", "gvpo",
    "gvpg", "gvpj", "gvpk", "gvpp"
]

GVP_RAW_TO_DISPLAY = {
    "gvpa": "GvpA",
    "gvpc": "GvpC",
    "gvpn": "GvpN",
    "gvpo": "GvpO",
    "gvpg": "GvpG",
    "gvpj": "GvpJ",
    "gvpk": "GvpK",
    "gvpp": "GvpP",
}

GVP_DISPLAY_TO_RAW = {v: k for k, v in GVP_RAW_TO_DISPLAY.items()}
GVP_ORDER = {g: i for i, g in enumerate(GVP_TYPES)}


# =========================================================
# 2. 日志
# =========================================================

STEP_NO = 0

def log(msg: str):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(OUT_LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")

def step(msg: str):
    global STEP_NO
    STEP_NO += 1
    log("STEP " + str(STEP_NO).zfill(3) + " | " + msg)


# =========================================================
# 3. 基础工具函数
# =========================================================

def normalize_text(x: Any) -> str:
    if x is None:
        return ""
    s = str(x).strip()
    if s.lower() in {"nan", "none", "null", "<na>"}:
        return ""
    return s

def to_int(x: Any) -> Optional[int]:
    try:
        s = normalize_text(x)
        if not s:
            return None
        return int(float(s))
    except Exception:
        return None

def to_float(x: Any) -> Optional[float]:
    try:
        s = normalize_text(x)
        if not s:
            return None
        return float(s)
    except Exception:
        return None

def round_float(x: Any, ndigits: int) -> Optional[float]:
    v = to_float(x)
    if v is None:
        return None
    return round(v, ndigits)

def normalize_cluster_id(x: Any) -> str:
    s = normalize_text(x)
    if not s:
        return ""
    try:
        v = float(s)
        if v.is_integer():
            return str(int(v))
    except Exception:
        pass
    return s

def format_gvp_name(x: Any) -> str:
    s = normalize_text(x)
    if not s:
        return ""
    low = s.lower()
    if low in GVP_RAW_TO_DISPLAY:
        return GVP_RAW_TO_DISPLAY[low]
    if re.fullmatch(r"gvp[a-z]", low):
        return "Gvp" + low[-1].upper()
    if re.fullmatch(r"Gvp[A-Z]", s):
        return s
    return s

def raw_gvp_name(x: Any) -> str:
    s = normalize_text(x)
    if not s:
        return ""

    low = s.lower()

    if low in GVP_RAW_TO_DISPLAY:
        return low

    display = format_gvp_name(s)
    if display in GVP_DISPLAY_TO_RAW:
        return GVP_DISPLAY_TO_RAW[display]

    if re.fullmatch(r"Gvp[A-Z]", display):
        return "gvp" + display[-1].lower()

    return low

def parse_region(region: Any) -> Tuple[Optional[int], Optional[int]]:
    s = normalize_text(region).replace("–", "-").replace("—", "-")
    m = re.match(r"^\s*(\d+)\s*-\s*(\d+)\s*$", s)
    if not m:
        return None, None

    a = int(m.group(1))
    b = int(m.group(2))

    if b < a:
        a, b = b, a

    return a, b

def region_to_string(start: Any, end: Any) -> str:
    s = to_int(start)
    e = to_int(end)

    if s is None or e is None:
        return ""

    return f"{s}-{e}"

def sanitize_filename(text: Any, max_len: int = 80) -> str:
    s = str(text)

    for ch in ['\\', '/', ':', '*', '?', '"', '<', '>', '|', ';', '[', ']']:
        s = s.replace(ch, "_")

    s = s.replace(" ", "_")
    s = re.sub(r"_+", "_", s)

    return s[:max_len]

def clean_sequence(seq: Any) -> str:
    return "".join([
        c for c in normalize_text(seq).upper()
        if c in "ACDEFGHIKLMNPQRSTVWY"
    ])

def safe_slice_0_based(seq: str, start: Optional[int], end: Optional[int]) -> Optional[str]:
    if not seq or start is None or end is None:
        return None

    if start < 0:
        start = 0

    if end > len(seq):
        end = len(seq)

    if end <= start:
        return None

    return seq[start:end]

def to_file_uri(p: Any) -> str:
    s = normalize_text(p)

    if not s:
        return ""

    try:
        return Path(s).resolve().as_uri()
    except Exception:
        ss = s.replace("\\", "/")
        if re.match(r"^[A-Za-z]:/", ss):
            return "file:///" + ss
        return s

def sanitize_cell_value(x: Any) -> str:
    if x is None:
        return ""

    if isinstance(x, (list, dict, tuple, set)):
        try:
            s = json.dumps(x, ensure_ascii=False)
        except Exception:
            s = str(x)
    else:
        s = str(x)

    s = s.replace("\r\n", " ").replace("\n", " ").replace("\r", " ").replace("\t", " ")
    s = "".join(ch if ord(ch) >= 32 else " " for ch in s)
    s = re.sub(r" +", " ", s).strip()

    return s

def load_csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        log("  [WARN] CSV 不存在: " + str(path))
        return []

    for enc in ["utf-8-sig", "utf-8", "gbk"]:
        try:
            with open(path, "r", encoding=enc, newline="") as f:
                reader = csv.DictReader(f)
                rows = [dict(r) for r in reader]

            log("  CSV 读取成功: " + str(path) + f" | enc={enc} | rows={len(rows)}")
            return rows

        except UnicodeDecodeError:
            continue

        except Exception as e:
            log("  [WARN] CSV 读取失败: " + str(path) + " | " + str(e))
            return []

    log("  [WARN] CSV 编码不匹配: " + str(path))
    return []

def write_table(
    path: Path,
    rows: List[Dict[str, Any]],
    fields: List[str],
    delimiter: str,
    quote_all: bool
):
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fields,
            delimiter=delimiter,
            extrasaction="ignore",
            quoting=csv.QUOTE_ALL if quote_all else csv.QUOTE_MINIMAL,
            lineterminator="\n"
        )

        writer.writeheader()

        for r in rows:
            writer.writerow({
                k: sanitize_cell_value(r.get(k, ""))
                for k in fields
            })

    log(
        "  表格写出完成: "
        + str(path)
        + f" | rows={len(rows)} | cols={len(fields)} | delimiter={repr(delimiter)}"
    )

def load_json_records(json_path: Path) -> List[Dict[str, Any]]:
    if not json_path.exists():
        log("  [WARN] JSON 不存在: " + str(json_path))
        return []

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            obj = json.load(f)
    except Exception as e:
        log("  [WARN] JSON 读取失败: " + str(json_path) + " | " + str(e))
        return []

    if isinstance(obj, list):
        return obj

    if isinstance(obj, dict):
        if isinstance(obj.get("data"), list):
            return obj["data"]
        if isinstance(obj.get("records"), list):
            return obj["records"]
        return [obj]

    return []

def first_text(*vals) -> str:
    for v in vals:
        s = normalize_text(v)
        if s:
            return s
    return ""

def first_int(*vals):
    for v in vals:
        iv = to_int(v)
        if iv is not None:
            return iv
    return ""

def first_float(*vals):
    for v in vals:
        fv = to_float(v)
        if fv is not None:
            return fv
    return ""

def calc_overlap_len(a_start, a_end, b_start, b_end):
    a1 = to_int(a_start)
    a2 = to_int(a_end)
    b1 = to_int(b_start)
    b2 = to_int(b_end)

    if a1 is None or a2 is None or b1 is None or b2 is None:
        return ""

    left = max(a1, b1)
    right = min(a2, b2)

    if right <= left:
        return 0

    return right - left


# =========================================================
# 4. seq_id alias 与 JSON 主记录索引
# =========================================================

def make_aliases(raw_id: Any) -> set:
    """
    只生成安全别名。

    不按 "_" 拆分，因此 WP_146646520.1 不会生成 WP。
    """
    s = normalize_text(raw_id)

    if not s:
        return set()

    out = set()

    out.add(s)
    out.add(s.upper())

    safe = sanitize_filename(s)
    if safe:
        out.add(safe)
        out.add(safe.upper())

    if "." in s:
        base = s.rsplit(".", 1)[0]

        if base:
            out.add(base)
            out.add(base.upper())

            safe_base = sanitize_filename(base)
            if safe_base:
                out.add(safe_base)
                out.add(safe_base.upper())

    return {x for x in out if x}

def build_json_index(records: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, List[int]]]:
    """
    只索引 JSON 主记录，不使用冗余成员。
    """
    seq_items = []
    alias_map: Dict[str, List[int]] = {}

    for idx, r in enumerate(records):
        ann = r.get("representative_annotation", {}) or {}

        uid = normalize_text(r.get("unique_sequence_id"))
        rec_id = normalize_text(r.get("id"))
        accession = normalize_text(r.get("accession"))
        protein_id = normalize_text(r.get("protein_id"))
        ann_id = normalize_text(ann.get("id"))

        primary_id = uid or ann_id or rec_id or accession or protein_id or f"seq_{idx + 1}"
        sequence = clean_sequence(r.get("sequence"))

        item = {
            "json_primary_id": primary_id,
            "json_unique_sequence_id": uid,
            "json_annotation_id": ann_id,
            "json_record_id": rec_id,
            "json_accession": accession,
            "json_protein_id": protein_id,
            "json_organism": normalize_text(ann.get("organism")),
            "json_gene": normalize_text(ann.get("gene")),
            "json_source_file": normalize_text(ann.get("source_file")),
            "json_gvp_types": (
                ",".join(ann.get("gvp_types", []))
                if isinstance(ann.get("gvp_types"), list)
                else normalize_text(ann.get("gvp_types"))
            ),
            "sequence": sequence,
        }

        item_idx = len(seq_items)
        seq_items.append(item)

        aliases = set()

        for idv in [
            primary_id,
            uid,
            ann_id,
            rec_id,
            accession,
            protein_id,
        ]:
            aliases |= make_aliases(idv)

        for a in aliases:
            alias_map.setdefault(a.upper(), []).append(item_idx)

    return seq_items, alias_map

def match_json_sequence(
    raw: str,
    seq_id: Any,
    json_indices: Dict[str, Dict[str, Any]]
) -> Tuple[Optional[Dict[str, Any]], Optional[str], int]:

    if raw not in json_indices:
        return None, None, 0

    alias_map = json_indices[raw]["alias_map"]
    seq_items = json_indices[raw]["seq_items"]

    for a in make_aliases(seq_id):
        idxs = alias_map.get(a.upper(), [])
        uniq = sorted(set(idxs))

        if len(uniq) == 1:
            return seq_items[uniq[0]], a, 1

        if len(uniq) > 1:
            return None, f"AMBIGUOUS_ALIAS:{a}", len(uniq)

    return None, None, 0

def load_all_json_indices() -> Dict[str, Dict[str, Any]]:
    step("加载并索引所有 GVP JSON 主记录")

    out = {}

    for raw in GVP_TYPES:
        display = format_gvp_name(raw)
        json_path = DATA_ROOT / raw / f"{display}_sequences.json"

        records = load_json_records(json_path)
        seq_items, alias_map = build_json_index(records)

        out[raw] = {
            "json_path": str(json_path),
            "seq_items": seq_items,
            "alias_map": alias_map,
        }

        log(
            f"  {display}: json_records={len(records)}, "
            f"main_seq_items={len(seq_items)}, alias_keys={len(alias_map)}"
        )

    return out


# =========================================================
# 5. 选择结构结果目录
# =========================================================

def choose_structure_root() -> Dict[str, Path]:
    step("选择表4-11结构结果目录")

    chosen = None

    for root in STRUCTURE_OUTPUT_CANDIDATES:
        table_dir = root / "tables"
        frag = table_dir / "all_selected_fragment_records.csv"
        pdb = table_dir / "all_predicted_pdb_records.csv"
        detail = table_dir / "ESMFold_PyMOL_secondary_structure_detail.csv"

        log(
            "  检查: "
            + str(root)
            + f" | selected={frag.exists()} pdb_record={pdb.exists()} pymol_detail={detail.exists()}"
        )

        if frag.exists() or pdb.exists() or detail.exists():
            chosen = root
            break

    if chosen is None:
        chosen = STRUCTURE_OUTPUT_CANDIDATES[0]
        log("  [WARN] 未发现结构结果表，默认使用: " + str(chosen))
    else:
        log("  使用结构结果目录: " + str(chosen))

    return {
        "OUTPUT_ROOT": chosen,
        "FRAGMENT_DIR": chosen / "selected_fragments",
        "PDB_DIR": chosen / "predicted_pdb",
        "PNG_DIR": chosen / "rendered_png",
        "TABLE_DIR": chosen / "tables",
        "FRAGMENT_RECORD_TABLE": chosen / "tables" / "all_selected_fragment_records.csv",
        "PDB_RECORD_TABLE": chosen / "tables" / "all_predicted_pdb_records.csv",
        "PYMOL_DETAIL_TABLE": chosen / "tables" / "ESMFold_PyMOL_secondary_structure_detail.csv",
        "PYMOL_SUMMARY_TABLE": chosen / "tables" / "Table_4_11_ESMFold_PyMOL_secondary_structure_summary.csv",
    }


# =========================================================
# 6. 索引 CV/SP、ATT、MSA、selected_fragment、PyMOL
# =========================================================

def load_candidate_index() -> Dict[str, Dict[Tuple[str, str], Dict[str, Any]]]:
    """
    CV/SP 是片段级。
    """
    step("读取并索引去重候选区域 CV/SP 表")

    rows = load_csv_rows(DEDUP_REGION_TABLE)

    by_cluster = {}
    by_region = {}

    for r in rows:
        raw = raw_gvp_name(r.get("原始GVP类型") or r.get("GVP类型"))
        display = format_gvp_name(raw)

        region = normalize_text(r.get("片段位置"))

        if not region:
            region = region_to_string(r.get("start_floor"), r.get("end_floor"))

        s, e = parse_region(region)

        cluster_id = normalize_cluster_id(r.get("cluster_id"))

        sp = to_float(r.get("跨物种占比") or r.get("species_percent"))

        if sp is not None and sp > 1.5:
            sp = sp / 100.0

        item = {
            "candidate_gvp_type": display,
            "candidate_raw_gvp_type": raw,
            "candidate_cluster_id": cluster_id,
            "candidate_segment_position": region,
            "candidate_start_pos": s,
            "candidate_end_pos": e,
            "candidate_mean_start_position": round_float(
                r.get("平均起始位置") or r.get("mean_start_position"),
                2
            ),
            "candidate_window_length": (
                to_int(r.get("窗口长度") or r.get("window_length"))
                or ((e - s) if s is not None and e is not None else None)
            ),
            "candidate_cv": round_float(
                r.get("CV") or r.get("cv_start_position") or r.get("cv"),
                4
            ),
            "candidate_species_percent": round(sp, 4) if sp is not None else None,
            "candidate_start_floor": to_int(r.get("start_floor")) if normalize_text(r.get("start_floor")) else s,
            "candidate_end_floor": to_int(r.get("end_floor")) if normalize_text(r.get("end_floor")) else e,
            "candidate_sample_count": to_int(r.get("sample_count")),
        }

        if raw and cluster_id:
            by_cluster[(raw, cluster_id)] = item

        if raw and region:
            by_region[(raw, region)] = item

    log(f"  CV/SP 片段级索引完成: by_cluster={len(by_cluster)}, by_region={len(by_region)}")

    return {
        "by_cluster": by_cluster,
        "by_region": by_region,
    }

def add_seq_alias_keys(
    index: Dict[Tuple[str, str, str], Dict[str, Any]],
    raw: str,
    key2: str,
    seq_id: Any,
    item: Dict[str, Any]
):
    for a in make_aliases(seq_id):
        index[(raw, key2, a.upper())] = item

def load_attention_index() -> Dict[str, Dict]:
    """
    ATT 最终字段使用片段级汇总。
    序列级明细只作为审计字段，不参与 attn_avg_attn / attn_avg_percentile。
    """
    step("读取并索引 ATTENTION 明细与片段级汇总")

    detail_by_cluster_seq = {}
    detail_by_region_seq = {}

    for raw in GVP_TYPES:
        path = ATTN_DETAIL_DIR / f"{raw}_all_seqs_attention.csv"

        for r in load_csv_rows(path):
            rr = raw_gvp_name(r.get("原始GVP类型") or r.get("GVP类型") or raw)
            region = normalize_text(r.get("片段位置"))
            cluster_id = normalize_cluster_id(r.get("cluster_id"))
            seq_id = normalize_text(r.get("seq_id"))

            item = {
                "attn_seq_id": seq_id,
                "attn_cluster_id": cluster_id,
                "attn_segment_position": region,
                "attn_fragment_start": to_int(r.get("fragment_start")),
                "attn_fragment_end": to_int(r.get("fragment_end")),

                # 审计用字段，不进入最终核心分数
                "attn_sequence_attention_score": round_float(r.get("attention_score"), 4),
                "attn_sequence_percentile": round_float(r.get("percentile"), 2),
            }

            if cluster_id:
                add_seq_alias_keys(detail_by_cluster_seq, rr, cluster_id, seq_id, item)

            if region:
                add_seq_alias_keys(detail_by_region_seq, rr, region, seq_id, item)

    stat_by_cluster = {}
    stat_by_region = {}

    stat_rows = []

    for raw in GVP_TYPES:
        stat_rows.extend(load_csv_rows(ATTN_STAT_DIR / f"{raw}_cluster_attention_statistics.csv"))

    if not stat_rows:
        stat_rows.extend(load_csv_rows(ATTN_MAIN_TABLE))

    for r in stat_rows:
        raw = raw_gvp_name(r.get("原始GVP类型") or r.get("GVP类型") or r.get("gvp_type"))
        display = format_gvp_name(raw)

        region = normalize_text(r.get("片段位置"))
        cluster_id = normalize_cluster_id(r.get("cluster_id"))

        item = {
            "attn_summary_gvp_type": display,
            "attn_summary_cluster_id": cluster_id,
            "attn_summary_segment_position": region,
            "attn_summary_avg_attn": round_float(r.get("注意力分") or r.get("avg_attn"), 4),
            "attn_summary_std_attn": round_float(r.get("注意力标准差") or r.get("std_attn"), 4),
            "attn_summary_avg_percentile": round_float(r.get("平均百分位") or r.get("avg_percentile"), 2),
        }

        if raw and cluster_id:
            stat_by_cluster[(raw, cluster_id)] = item

        if raw and region:
            stat_by_region[(raw, region)] = item

    log(
        "  ATT 索引完成: "
        f"detail_cluster_seq={len(detail_by_cluster_seq)}, "
        f"detail_region_seq={len(detail_by_region_seq)}, "
        f"summary_cluster={len(stat_by_cluster)}, summary_region={len(stat_by_region)}"
    )

    return {
        "detail_by_cluster_seq": detail_by_cluster_seq,
        "detail_by_region_seq": detail_by_region_seq,
        "stat_by_cluster": stat_by_cluster,
        "stat_by_region": stat_by_region,
    }

def load_msa_index() -> Dict[str, Dict]:
    """
    MSA 最终字段使用片段级汇总。
    序列级明细只作为审计字段，不参与 msa_avg_score / msa_avg_percentile。
    """
    step("读取并索引 MSA 明细与片段级汇总")

    detail_by_cluster_seq = {}
    detail_by_region_seq = {}

    for raw in GVP_TYPES:
        path = MSA_TABLE_DIR / f"{raw}_all_sequence_fragment_msa_scores.csv"

        for r in load_csv_rows(path):
            rr = raw_gvp_name(r.get("原始GVP类型") or r.get("GVP类型") or raw)
            region = normalize_text(r.get("片段位置"))
            cluster_id = normalize_cluster_id(r.get("cluster_id"))
            seq_id = normalize_text(r.get("seq_id"))

            item = {
                "msa_seq_id": seq_id,
                "msa_cluster_id": cluster_id,
                "msa_segment_position": region,
                "msa_fragment_start": to_int(r.get("fragment_start")),
                "msa_fragment_end": to_int(r.get("fragment_end")),

                # 审计用字段，不进入最终核心分数
                "msa_sequence_fragment_avg_score": round_float(r.get("fragment_avg_score"), 4),
                "msa_sequence_percentile_rank": round_float(r.get("percentile_rank"), 2),
            }

            if cluster_id:
                add_seq_alias_keys(detail_by_cluster_seq, rr, cluster_id, seq_id, item)

            if region:
                add_seq_alias_keys(detail_by_region_seq, rr, region, seq_id, item)

    stat_by_cluster = {}
    stat_by_region = {}

    stat_rows = []

    for raw in GVP_TYPES:
        stat_rows.extend(load_csv_rows(MSA_TABLE_DIR / f"{raw}_table_4_10_msa_summary.csv"))

    if not stat_rows:
        stat_rows.extend(load_csv_rows(MSA_MAIN_TABLE))

    for r in stat_rows:
        raw = raw_gvp_name(r.get("原始GVP类型") or r.get("GVP类型") or r.get("gvp_type"))
        display = format_gvp_name(raw)

        region = normalize_text(r.get("片段位置"))
        cluster_id = normalize_cluster_id(r.get("cluster_id"))

        item = {
            "msa_summary_gvp_type": display,
            "msa_summary_cluster_id": cluster_id,
            "msa_summary_segment_position": region,
            "msa_summary_sequence_count": to_int(r.get("参与比对序列数") or r.get("sequence_count")),
            "msa_summary_avg_conservation_score": round_float(r.get("平均保守性分数") or r.get("avg_score"), 4),
            "msa_summary_conservation_std": round_float(r.get("保守性标准差") or r.get("std_score"), 4),
            "msa_summary_avg_percentile": round_float(r.get("平均百分位") or r.get("avg_percentile"), 2),
        }

        if raw and cluster_id:
            stat_by_cluster[(raw, cluster_id)] = item

        if raw and region:
            stat_by_region[(raw, region)] = item

    log(
        "  MSA 索引完成: "
        f"detail_cluster_seq={len(detail_by_cluster_seq)}, "
        f"detail_region_seq={len(detail_by_region_seq)}, "
        f"summary_cluster={len(stat_by_cluster)}, summary_region={len(stat_by_region)}"
    )

    return {
        "detail_by_cluster_seq": detail_by_cluster_seq,
        "detail_by_region_seq": detail_by_region_seq,
        "stat_by_cluster": stat_by_cluster,
        "stat_by_region": stat_by_region,
    }

def load_selected_fragment_indexes(fragment_record_table: Path) -> Dict[str, Dict]:
    step("读取并索引 selected_fragment 明细与分组信息")

    rows = load_csv_rows(fragment_record_table)

    seq_by_cluster = {}
    seq_by_region = {}
    groups: Dict[Tuple[str, str], List[Dict[str, str]]] = {}

    for r in rows:
        raw = raw_gvp_name(r.get("gvp_key") or r.get("GVP类型"))
        region = normalize_text(r.get("片段位置"))
        cluster_id = normalize_cluster_id(r.get("cluster_id"))
        seq_id = normalize_text(r.get("seq_id"))

        item = {
            "selected_seq_id": seq_id,
            "selected_organism": normalize_text(r.get("organism")),
            "selected_genus": normalize_text(r.get("genus")),
            "selected_full_length": to_int(r.get("full_length")),
            "selected_fragment": normalize_text(r.get("fragment")),
            "selected_fragment_length": to_int(r.get("fragment_length")),
        }

        if raw and cluster_id and seq_id:
            add_seq_alias_keys(seq_by_cluster, raw, cluster_id, seq_id, item)

        if raw and region and seq_id:
            add_seq_alias_keys(seq_by_region, raw, region, seq_id, item)

        if raw and region:
            groups.setdefault((raw, region), []).append(r)

    group_index = {}

    for (raw, region), arr in groups.items():
        seq_ids = [normalize_text(x.get("seq_id")) for x in arr if normalize_text(x.get("seq_id"))]
        organisms = [normalize_text(x.get("organism")) for x in arr if normalize_text(x.get("organism"))]
        genera = [normalize_text(x.get("genus")) for x in arr if normalize_text(x.get("genus"))]

        group_index[(raw, region)] = {
            "selected_region_sequence_count": len(seq_ids),
            "selected_region_seq_ids": " | ".join(seq_ids),
            "selected_region_organisms": " | ".join(organisms),
            "selected_region_genera": " | ".join(genera),
            "selected_region_unique_species_count": len(set(organisms)),
            "selected_region_unique_genus_count": len(set(genera)),
        }

    log(
        "  selected_fragment 索引完成: "
        f"seq_by_cluster={len(seq_by_cluster)}, "
        f"seq_by_region={len(seq_by_region)}, "
        f"regions={len(group_index)}"
    )

    return {
        "seq_by_cluster": seq_by_cluster,
        "seq_by_region": seq_by_region,
        "group_by_region": group_index,
    }

def load_pymol_summary_index(pymol_summary_table: Path) -> Dict[str, Dict]:
    step("读取并索引 PyMOL 区域汇总")

    by_cluster = {}
    by_region = {}

    for r in load_csv_rows(pymol_summary_table):
        raw = raw_gvp_name(r.get("GVP类型") or r.get("gvp_type"))
        display = format_gvp_name(raw)

        region = normalize_text(r.get("片段位置"))
        cluster_id = normalize_cluster_id(r.get("cluster_id"))

        item = {
            "pymol_summary_gvp_type": display,
            "pymol_summary_cluster_id": cluster_id,
            "pymol_summary_segment_position": region,
            "pymol_structure_prediction_count": to_int(r.get("结构预测数")),
            "pymol_alpha_helix_ratio": round_float(r.get("α螺旋比例"), 4),
            "pymol_beta_sheet_ratio": round_float(r.get("β折叠比例"), 4),
            "pymol_coil_ratio": round_float(r.get("无规则卷曲比例"), 4),
            "pymol_secondary_structure_fragment_ratio": round_float(r.get("含二级结构片段占比"), 4),
        }

        if raw and cluster_id:
            by_cluster[(raw, cluster_id)] = item

        if raw and region:
            by_region[(raw, region)] = item

    log(f"  PyMOL 汇总索引完成: by_cluster={len(by_cluster)}, by_region={len(by_region)}")

    return {
        "by_cluster": by_cluster,
        "by_region": by_region,
    }


# =========================================================
# 7. 构建结构主表
# =========================================================

def infer_paths(
    structure_paths: Dict[str, Path],
    gvp_type: str,
    start: Any,
    end: Any,
    seq_id: Any
) -> Tuple[str, str]:

    display = format_gvp_name(gvp_type)
    s = to_int(start)
    e = to_int(end)
    safe_seq_id = sanitize_filename(seq_id)

    if not display or s is None or e is None or not safe_seq_id:
        return "", ""

    pdb_path = (
        structure_paths["PDB_DIR"]
        / display
        / f"region_{s}_{e}"
        / f"{safe_seq_id}_region_{s}_{e}.pdb"
    )

    png_path = (
        structure_paths["PNG_DIR"]
        / display
        / f"region_{s}_{e}"
        / f"{safe_seq_id}_region_{s}_{e}.png"
    )

    return str(pdb_path), str(png_path)

def base_row_common(
    source: str,
    r: Dict[str, Any],
    structure_paths: Dict[str, Path]
) -> Dict[str, Any]:

    raw = raw_gvp_name(r.get("gvp_key") or r.get("GVP类型") or r.get("gvp_type"))
    display = format_gvp_name(raw)

    region = normalize_text(r.get("片段位置") or r.get("segment_position"))

    start = to_int(r.get("start"))
    end = to_int(r.get("end"))

    if not region and start is not None and end is not None:
        region = region_to_string(start, end)

    if (start is None or end is None) and region:
        start, end = parse_region(region)

    seq_id = normalize_text(r.get("seq_id"))

    inferred_pdb, inferred_png = infer_paths(
        structure_paths,
        display,
        start,
        end,
        seq_id
    )

    pdb_path = normalize_text(r.get("pdb_path")) or inferred_pdb
    png_path = normalize_text(r.get("png_path")) or inferred_png

    return {
        "base_source": source,

        "gvp_type": display,
        "raw_gvp_type": raw,
        "cluster_id": normalize_cluster_id(r.get("cluster_id")),
        "segment_position": region,
        "start_pos": start,
        "end_pos": end,
        "window_length": (end - start) if start is not None and end is not None else None,

        "seq_id": seq_id,

        "structure_organism": normalize_text(r.get("organism")),
        "structure_genus": normalize_text(r.get("genus")),
        "selected_fragment": normalize_text(r.get("fragment")),
        "selected_fragment_length": to_int(r.get("fragment_length")),
        "selected_full_length": to_int(r.get("full_length")),

        "pdb_path": pdb_path,
        "pdb_uri": to_file_uri(pdb_path),
        "pdb_exists": 1 if pdb_path and Path(pdb_path).exists() else 0,

        "image": png_path,
        "image_url": to_file_uri(png_path),
        "image_exists": 1 if png_path and Path(png_path).exists() else 0,
        "image_bind_rule": "table_4_11_api_path_rule",
    }

def load_structure_base_rows(structure_paths: Dict[str, Path]) -> List[Dict[str, Any]]:
    step("构建序列级结构主表")

    rows = []

    detail_rows = load_csv_rows(structure_paths["PYMOL_DETAIL_TABLE"])

    if detail_rows:
        log("  使用 ESMFold_PyMOL_secondary_structure_detail.csv 作为主表")

        for r in detail_rows:
            row = base_row_common("pymol_secondary_structure_detail", r, structure_paths)

            row.update({
                "pymol_total_ca": to_int(r.get("total_ca")),
                "pymol_helix_ca": to_int(r.get("helix_ca")),
                "pymol_sheet_ca": to_int(r.get("sheet_ca")),
                "pymol_coil_ca": to_int(r.get("coil_ca")),
                "pymol_has_secondary_structure": to_int(r.get("has_secondary_structure")),
                "pymol_detail_error": "",
            })

            rows.append(row)

    if not rows:
        pdb_rows = load_csv_rows(structure_paths["PDB_RECORD_TABLE"])

        if pdb_rows:
            log("  使用 all_predicted_pdb_records.csv 作为主表")

            for r in pdb_rows:
                row = base_row_common("all_predicted_pdb_records", r, structure_paths)

                row.update({
                    "pymol_total_ca": None,
                    "pymol_helix_ca": None,
                    "pymol_sheet_ca": None,
                    "pymol_coil_ca": None,
                    "pymol_has_secondary_structure": None,
                    "pymol_detail_error": "no_pymol_detail_table",
                })

                rows.append(row)

    if not rows:
        selected_rows = load_csv_rows(structure_paths["FRAGMENT_RECORD_TABLE"])

        if selected_rows:
            log("  使用 all_selected_fragment_records.csv 作为主表，按规则推断 PDB/PNG 路径")

            for r in selected_rows:
                row = base_row_common("all_selected_fragment_records", r, structure_paths)

                row.update({
                    "pymol_total_ca": None,
                    "pymol_helix_ca": None,
                    "pymol_sheet_ca": None,
                    "pymol_coil_ca": None,
                    "pymol_has_secondary_structure": None,
                    "pymol_detail_error": "no_pdb_or_pymol_detail_table_yet",
                })

                rows.append(row)

    seen = set()
    dedup = []

    for r in rows:
        key = (
            r.get("raw_gvp_type"),
            r.get("segment_position"),
            r.get("seq_id"),
            r.get("pdb_path"),
        )

        if key in seen:
            continue

        seen.add(key)
        dedup.append(r)

    log(f"  结构主表完成: raw_rows={len(rows)}, dedup_rows={len(dedup)}")

    return dedup


# =========================================================
# 8. 合并辅助
# =========================================================

def lookup_region(
    index: Dict[str, Dict],
    raw: str,
    cluster_id: str,
    region: str,
    cluster_key: str,
    region_key: str
) -> Dict[str, Any]:

    if cluster_id:
        hit = index.get(cluster_key, {}).get((raw, cluster_id))
        if hit:
            return hit

    if region:
        hit = index.get(region_key, {}).get((raw, region))
        if hit:
            return hit

    return {}

def lookup_seq_detail(
    index: Dict[str, Dict],
    raw: str,
    cluster_id: str,
    region: str,
    seq_id: str,
    cluster_key: str,
    region_key: str
) -> Tuple[Dict[str, Any], str]:

    aliases = make_aliases(seq_id)

    if cluster_id:
        for a in aliases:
            hit = index.get(cluster_key, {}).get((raw, cluster_id, a.upper()))
            if hit:
                return hit, "cluster_id+seq_id"

    if region:
        for a in aliases:
            hit = index.get(region_key, {}).get((raw, region, a.upper()))
            if hit:
                return hit, "region+seq_id"

    return {}, ""


# =========================================================
# 9. 数据库平台字段映射
# =========================================================

def add_required_database_fields(row: Dict[str, Any]) -> Dict[str, Any]:
    start = to_int(row.get("start_pos"))
    end = to_int(row.get("end_pos"))

    window_expected_len = first_int(
        row.get("window_expected_len"),
        row.get("selected_fragment_length"),
        row.get("candidate_window_length"),
        row.get("window_length"),
    )

    if window_expected_len == "" and start is not None and end is not None:
        window_expected_len = end - start

    accession = first_text(
        row.get("accession"),
        row.get("seq_id"),
        row.get("json_accession"),
        row.get("json_primary_id"),
        row.get("json_unique_sequence_id"),
        row.get("json_protein_id"),
    )

    window_seq = first_text(
        row.get("window_seq"),
        row.get("selected_fragment"),
    )

    full_sequence = first_text(row.get("full_sequence"))

    row["accession"] = accession
    row["raw_name"] = first_text(row.get("raw_name"), accession)

    row["parse_ok"] = 1 if (
        normalize_text(row.get("gvp_type"))
        and accession
        and start is not None
        and end is not None
    ) else 0

    row["window_expected_len"] = window_expected_len
    row["window_seq"] = window_seq

    row["window_len"] = first_int(
        row.get("window_len"),
        len(window_seq) if window_seq else "",
    )

    row["full_sequence"] = full_sequence

    row["full_sequence_len"] = first_int(
        row.get("full_sequence_len"),
        len(full_sequence) if full_sequence else "",
    )

    row["json_description"] = ""

    row["name_organism"] = first_text(
        row.get("name_organism"),
        row.get("json_organism"),
        row.get("structure_organism"),
    )

    row["name_gene"] = first_text(
        row.get("name_gene"),
        row.get("json_gene"),
    )

    # =====================================================
    # CV / SP：只使用片段级去重候选表
    # =====================================================
    row["cluster_species_percent"] = first_float(
        row.get("candidate_species_percent"),
        row.get("cluster_species_percent"),
    )

    row["cluster_cv_start_position"] = first_float(
        row.get("candidate_cv"),
        row.get("cluster_cv_start_position"),
    )

    row["cluster_id_matched"] = first_text(
        row.get("candidate_cluster_id"),
        row.get("cluster_id_matched"),
        row.get("cluster_id"),
    )

    row["cv_sp_score_level"] = "fragment_summary"

    cand_start = to_int(row.get("candidate_start_floor"))

    if start is not None and cand_start is not None:
        row["cluster_start_delta"] = abs(start - cand_start)
    else:
        row["cluster_start_delta"] = first_float(row.get("cluster_start_delta"))

    # =====================================================
    # MSA：最终字段只使用片段级汇总
    # 不使用序列级 msa_sequence_fragment_avg_score / msa_sequence_percentile_rank
    # =====================================================
    row["msa_start_matched"] = first_int(
        row.get("msa_start_matched"),
        row.get("start_pos"),
    )

    row["msa_end_matched"] = first_int(
        row.get("msa_end_matched"),
        row.get("end_pos"),
    )

    row["msa_avg_score"] = first_float(
        row.get("msa_summary_avg_conservation_score"),
        row.get("msa_avg_score"),
    )

    row["msa_avg_percentile"] = first_float(
        row.get("msa_summary_avg_percentile"),
        row.get("msa_avg_percentile"),
    )

    row["msa_overlap_len"] = window_expected_len
    row["msa_score_level"] = "fragment_summary"

    # =====================================================
    # ATTENTION：最终字段只使用片段级汇总
    # 不使用序列级 attn_sequence_attention_score / attn_sequence_percentile
    # =====================================================
    row["attn_start_matched"] = first_int(
        row.get("attn_start_matched"),
        row.get("start_pos"),
    )

    row["attn_end_matched"] = first_int(
        row.get("attn_end_matched"),
        row.get("end_pos"),
    )

    row["attn_avg_attn"] = first_float(
        row.get("attn_summary_avg_attn"),
        row.get("attn_avg_attn"),
    )

    row["attn_avg_percentile"] = first_float(
        row.get("attn_summary_avg_percentile"),
        row.get("attn_avg_percentile"),
    )

    row["attn_overlap_len"] = window_expected_len
    row["attn_score_level"] = "fragment_summary"

    # =====================================================
    # 二级结构
    # 序列级 PyMOL 明细优先；没有时用片段级 PyMOL 汇总
    # =====================================================
    alpha = first_float(
        row.get("ss_alpha_ratio"),
        row.get("alpha_helix_ratio"),
        row.get("pymol_detail_helix_ratio"),
        row.get("pymol_alpha_helix_ratio"),
    )

    beta = first_float(
        row.get("ss_beta_ratio"),
        row.get("beta_sheet_ratio"),
        row.get("pymol_detail_sheet_ratio"),
        row.get("pymol_beta_sheet_ratio"),
    )

    flexible = first_float(
        row.get("ss_flexible_ratio"),
        row.get("flexible_region_ratio"),
        row.get("pymol_detail_coil_ratio"),
        row.get("pymol_coil_ratio"),
    )

    row["ss_alpha_ratio"] = alpha
    row["ss_beta_ratio"] = beta
    row["ss_flexible_ratio"] = flexible

    row["alpha_helix_ratio"] = alpha
    row["beta_sheet_ratio"] = beta
    row["flexible_region_ratio"] = flexible

    if alpha != "" or beta != "":
        row["ss_ratio"] = round((to_float(alpha) or 0.0) + (to_float(beta) or 0.0), 6)
    else:
        row["ss_ratio"] = first_float(row.get("ss_ratio"))

    total_ca = to_int(row.get("pymol_total_ca"))
    helix_ca = to_int(row.get("pymol_helix_ca"))
    sheet_ca = to_int(row.get("pymol_sheet_ca"))

    if helix_ca is not None or sheet_ca is not None:
        hs_ca = (helix_ca or 0) + (sheet_ca or 0)
    else:
        hs_ca = ""

    row["ss_total_ca"] = first_int(row.get("ss_total_ca"), total_ca)
    row["ss_hs_ca"] = first_int(row.get("ss_hs_ca"), hs_ca)

    if total_ca is not None and total_ca > 0 and hs_ca != "":
        row["ss_residue_ratio"] = round(float(hs_ca) / float(total_ca), 6)
    else:
        row["ss_residue_ratio"] = first_float(
            row.get("ss_residue_ratio"),
            row.get("ss_ratio"),
        )

    ss_has = first_int(row.get("ss_has_01"), row.get("pymol_has_secondary_structure"))

    if ss_has == "" and row.get("ss_ratio") != "":
        ss_has = 1 if (to_float(row.get("ss_ratio")) or 0.0) > 0 else 0

    row["ss_has_01"] = ss_has

    region_pdb_count = first_int(
        row.get("region_pdb_count"),
        row.get("pymol_structure_prediction_count"),
        row.get("selected_region_sequence_count"),
    )

    row["region_pdb_count"] = region_pdb_count

    row["region_scored_count"] = first_int(
        row.get("region_scored_count"),
        row.get("pymol_structure_prediction_count"),
        region_pdb_count,
    )

    row["is_complete_3"] = (
        1 if region_pdb_count != "" and int(region_pdb_count) >= 3 else 0
    )

    row["ss_total_score"] = first_float(
        row.get("ss_total_score"),
        row.get("ss_ratio"),
    )

    row["image"] = first_text(row.get("image"))
    row["image_url"] = first_text(row.get("image_url"), to_file_uri(row.get("image")))

    return row

def deduplicate_by_database_key(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out_map = {}
    order = []

    for r in rows:
        key = (
            normalize_text(r.get("gvp_type")),
            normalize_text(r.get("accession")),
            normalize_text(r.get("start_pos")),
            normalize_text(r.get("end_pos")),
        )

        if key not in out_map:
            order.append(key)

        out_map[key] = r

    return [out_map[k] for k in order]


# =========================================================
# 10. SQLite DB 写出
# =========================================================

def write_sqlite_database(
    db_path: Path,
    table_name: str,
    rows: List[Dict[str, Any]],
    fields: List[str]
):
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()

    cur.execute(f'DROP TABLE IF EXISTS "{table_name}"')

    col_sql = ", ".join([f'"{c}" TEXT' for c in fields])
    cur.execute(f'CREATE TABLE "{table_name}" ({col_sql})')

    if rows:
        placeholders = ", ".join(["?"] * len(fields))
        col_names = ", ".join([f'"{c}"' for c in fields])

        data = []

        for r in rows:
            data.append([
                sanitize_cell_value(r.get(c, ""))
                for c in fields
            ])

        cur.executemany(
            f'INSERT INTO "{table_name}" ({col_names}) VALUES ({placeholders})',
            data
        )

    cur.execute(
        f'CREATE UNIQUE INDEX IF NOT EXISTS "uq_{table_name}_key" '
        f'ON "{table_name}"("gvp_type","accession","start_pos","end_pos")'
    )

    cur.execute(
        f'CREATE INDEX IF NOT EXISTS "idx_{table_name}_gvp" '
        f'ON "{table_name}"("gvp_type")'
    )

    cur.execute(
        f'CREATE INDEX IF NOT EXISTS "idx_{table_name}_acc" '
        f'ON "{table_name}"("accession")'
    )

    cur.execute(
        f'CREATE INDEX IF NOT EXISTS "idx_{table_name}_start" '
        f'ON "{table_name}"("start_pos")'
    )

    conn.commit()
    conn.close()

    log(
        "  SQLite DB 写出完成: "
        + str(db_path)
        + f" | table={table_name} | rows={len(rows)} | cols={len(fields)}"
    )


# =========================================================
# 11. 主流程
# =========================================================

def build_database():
    if OUT_LOG.exists():
        OUT_LOG.unlink()

    step("脚本启动")
    log("PROJECT_ROOT: " + str(PROJECT_ROOT))
    log("DATA_ROOT exists? " + str(DATA_ROOT.exists()) + " -> " + str(DATA_ROOT))
    log("DEDUP_REGION_TABLE exists? " + str(DEDUP_REGION_TABLE.exists()) + " -> " + str(DEDUP_REGION_TABLE))

    structure_paths = choose_structure_root()

    for k, p in structure_paths.items():
        log(f"{k}: exists={p.exists()} -> {p}")

    json_indices = load_all_json_indices()
    candidate_index = load_candidate_index()
    attn_index = load_attention_index()
    msa_index = load_msa_index()
    selected_indexes = load_selected_fragment_indexes(structure_paths["FRAGMENT_RECORD_TABLE"])
    pymol_summary_index = load_pymol_summary_index(structure_paths["PYMOL_SUMMARY_TABLE"])

    base_rows = load_structure_base_rows(structure_paths)

    if not base_rows:
        log("[WARN] 没有可用结构/代表片段主表，无法构建数据库")

        write_table(OUT_DATABASE_CSV, [], CLEAN_WIDE_FIELDS, delimiter=",", quote_all=True)
        write_table(OUT_DATABASE_TSV, [], CLEAN_WIDE_FIELDS, delimiter="\t", quote_all=False)
        write_sqlite_database(OUT_DATABASE_DB, DB_TABLE, [], CLEAN_WIDE_FIELDS)

        return []

    final_rows = []
    missing_rows = []

    step("逐行合并 JSON 主记录、selected_fragment、片段级 CV/SP、片段级 ATT、片段级 MSA、PyMOL、PDB/PNG")

    for idx, base in enumerate(base_rows, 1):
        row = dict(base)

        raw = row.get("raw_gvp_type")
        display = row.get("gvp_type")
        cluster_id = normalize_cluster_id(row.get("cluster_id"))
        region = normalize_text(row.get("segment_position"))
        seq_id = normalize_text(row.get("seq_id"))

        row["database_row_id"] = f"{display}|{region}|{seq_id}|{idx}"
        row["cluster_id"] = cluster_id

        # =================================================
        # JSON 主记录匹配
        # =================================================
        hit, hit_alias, hit_count = match_json_sequence(raw, seq_id, json_indices)

        row["match_found"] = 1 if hit else 0
        row["hit_alias"] = hit_alias or ""
        row["match_candidate_count"] = hit_count

        if hit:
            full_seq = hit.get("sequence")
            win_seq = safe_slice_0_based(
                full_seq,
                to_int(row.get("start_pos")),
                to_int(row.get("end_pos"))
            )

            row.update({
                "json_primary_id": hit.get("json_primary_id"),
                "json_unique_sequence_id": hit.get("json_unique_sequence_id"),
                "json_annotation_id": hit.get("json_annotation_id"),
                "json_record_id": hit.get("json_record_id"),
                "json_accession": hit.get("json_accession"),
                "json_protein_id": hit.get("json_protein_id"),
                "json_organism": hit.get("json_organism"),
                "json_gene": hit.get("json_gene"),
                "json_source_file": hit.get("json_source_file"),
                "json_gvp_types": hit.get("json_gvp_types"),

                "full_sequence": full_seq if WRITE_FULL_SEQUENCE_TO_MAIN else "",
                "full_sequence_len": len(full_seq) if full_seq else "",

                "window_seq": win_seq or "",
                "window_len": len(win_seq) if win_seq else "",
                "window_valid": 1 if win_seq else 0,
            })

        else:
            row.update({
                "json_primary_id": "",
                "json_unique_sequence_id": "",
                "json_annotation_id": "",
                "json_record_id": "",
                "json_accession": "",
                "json_protein_id": "",
                "json_organism": "",
                "json_gene": "",
                "json_source_file": "",
                "json_gvp_types": "",
                "full_sequence": "",
                "full_sequence_len": "",
                "window_seq": "",
                "window_len": "",
                "window_valid": 0,
            })

        # =================================================
        # selected_fragment：序列级
        # =================================================
        selected_detail, selected_rule = lookup_seq_detail(
            selected_indexes,
            raw,
            cluster_id,
            region,
            seq_id,
            "seq_by_cluster",
            "seq_by_region"
        )

        if selected_detail:
            for k, v in selected_detail.items():
                if normalize_text(row.get(k)) == "":
                    row[k] = v

            row["selected_detail_match_rule"] = selected_rule
            row["has_selected_fragment_record"] = 1

        else:
            row["selected_detail_match_rule"] = ""
            row["has_selected_fragment_record"] = 1 if normalize_text(row.get("selected_fragment")) else 0

        selected_group = selected_indexes["group_by_region"].get((raw, region), {})
        row.update(selected_group)
        row["has_selected_region_record"] = 1 if selected_group else 0

        # =================================================
        # CV/SP：片段级，只按 cluster_id 或 region 合并
        # =================================================
        cand = lookup_region(
            candidate_index,
            raw,
            cluster_id,
            region,
            "by_cluster",
            "by_region"
        )

        row.update(cand)
        row["has_cv_sp"] = 1 if cand else 0

        if not cluster_id and cand.get("candidate_cluster_id"):
            row["cluster_id"] = cand.get("candidate_cluster_id")
            cluster_id = row["cluster_id"]

        # =================================================
        # ATTENTION 明细：只作为审计，不用于最终分数字段
        # =================================================
        att_detail, att_rule = lookup_seq_detail(
            attn_index,
            raw,
            cluster_id,
            region,
            seq_id,
            "detail_by_cluster_seq",
            "detail_by_region_seq"
        )

        row.update(att_detail)
        row["attn_detail_match_rule"] = att_rule
        row["has_attn_detail"] = 1 if att_detail else 0

        # ATTENTION 汇总：片段级，最终 attn_avg_* 来源
        att_sum = lookup_region(
            attn_index,
            raw,
            cluster_id,
            region,
            "stat_by_cluster",
            "stat_by_region"
        )

        row.update(att_sum)
        row["has_attn_summary"] = 1 if att_sum else 0

        # =================================================
        # MSA 明细：只作为审计，不用于最终分数字段
        # =================================================
        msa_detail, msa_rule = lookup_seq_detail(
            msa_index,
            raw,
            cluster_id,
            region,
            seq_id,
            "detail_by_cluster_seq",
            "detail_by_region_seq"
        )

        row.update(msa_detail)
        row["msa_detail_match_rule"] = msa_rule
        row["has_msa_detail"] = 1 if msa_detail else 0

        # MSA 汇总：片段级，最终 msa_avg_* 来源
        msa_sum = lookup_region(
            msa_index,
            raw,
            cluster_id,
            region,
            "stat_by_cluster",
            "stat_by_region"
        )

        row.update(msa_sum)
        row["has_msa_summary"] = 1 if msa_sum else 0

        # =================================================
        # PyMOL 区域汇总
        # =================================================
        pymol_sum = lookup_region(
            pymol_summary_index,
            raw,
            cluster_id,
            region,
            "by_cluster",
            "by_region"
        )

        row.update(pymol_sum)
        row["has_pymol_summary"] = 1 if pymol_sum else 0

        # =================================================
        # PyMOL 单条结构比例
        # =================================================
        total_ca = to_float(row.get("pymol_total_ca"))
        helix_ca = to_float(row.get("pymol_helix_ca"))
        sheet_ca = to_float(row.get("pymol_sheet_ca"))
        coil_ca = to_float(row.get("pymol_coil_ca"))

        if total_ca and total_ca > 0:
            row["pymol_detail_helix_ratio"] = round((helix_ca or 0) / total_ca, 6)
            row["pymol_detail_sheet_ratio"] = round((sheet_ca or 0) / total_ca, 6)
            row["pymol_detail_coil_ratio"] = round((coil_ca or 0) / total_ca, 6)
        else:
            row["pymol_detail_helix_ratio"] = ""
            row["pymol_detail_sheet_ratio"] = ""
            row["pymol_detail_coil_ratio"] = ""

        # =================================================
        # 文件存在性
        # =================================================
        row["pdb_exists"] = (
            1 if normalize_text(row.get("pdb_path")) and Path(normalize_text(row.get("pdb_path"))).exists()
            else 0
        )

        row["image_exists"] = (
            1 if normalize_text(row.get("image")) and Path(normalize_text(row.get("image"))).exists()
            else 0
        )

        # =================================================
        # 合并数据库平台字段
        # 这里会强制 ATT/MSA/CVSP 使用片段级汇总
        # =================================================
        row = add_required_database_fields(row)

        # =================================================
        # 缺失报告
        # =================================================
        for flag, label in [
            ("match_found", "json_main_sequence_match"),
            ("has_selected_fragment_record", "selected_fragment"),
            ("has_cv_sp", "cv_sp_fragment_summary"),
            ("has_attn_summary", "attn_fragment_summary"),
            ("has_msa_summary", "msa_fragment_summary"),
            ("has_pymol_summary", "pymol_region_summary"),
            ("pdb_exists", "pdb_file_exists"),
            ("image_exists", "png_file_exists"),
        ]:
            if row.get(flag) != 1:
                missing_rows.append({
                    "missing_type": label,
                    "database_row_id": row.get("database_row_id"),
                    "gvp_type": row.get("gvp_type"),
                    "raw_gvp_type": row.get("raw_gvp_type"),
                    "cluster_id": row.get("cluster_id"),
                    "segment_position": row.get("segment_position"),
                    "seq_id": row.get("seq_id"),
                    "accession": row.get("accession"),
                    "base_source": row.get("base_source"),
                    "pdb_path": row.get("pdb_path"),
                    "image": row.get("image"),
                })

        final_rows.append(row)

    # 排序
    final_rows.sort(
        key=lambda r: (
            GVP_ORDER.get(r.get("raw_gvp_type"), 99),
            to_int(r.get("start_pos")) if to_int(r.get("start_pos")) is not None else 999999,
            to_int(r.get("end_pos")) if to_int(r.get("end_pos")) is not None else 999999,
            normalize_text(r.get("accession")),
        )
    )

    # 数据库 key 去重
    before_dedup = len(final_rows)
    final_rows = deduplicate_by_database_key(final_rows)
    after_dedup = len(final_rows)

    log(f"数据库 key 去重: before={before_dedup}, after={after_dedup}")

    step("写出干净宽表 CSV / TSV / SQLite DB")

    write_table(OUT_DATABASE_CSV, final_rows, CLEAN_WIDE_FIELDS, delimiter=",", quote_all=True)
    write_table(OUT_DATABASE_TSV, final_rows, CLEAN_WIDE_FIELDS, delimiter="\t", quote_all=False)
    write_sqlite_database(OUT_DATABASE_DB, DB_TABLE, final_rows, CLEAN_WIDE_FIELDS)

    step("写出缺失报告")

    write_table(
        OUT_MISSING_REPORT_CSV,
        missing_rows,
        MISSING_FIELDS,
        delimiter=",",
        quote_all=True
    )

    def count_flag(flag):
        return sum(1 for r in final_rows if r.get(flag) == 1)

    def count_nonempty(field):
        return sum(1 for r in final_rows if normalize_text(r.get(field)))

    step("完成")

    log("数据库输出 CSV: " + str(OUT_DATABASE_CSV))
    log("数据库输出 TSV: " + str(OUT_DATABASE_TSV))
    log("数据库输出 DB: " + str(OUT_DATABASE_DB))
    log("缺失报告: " + str(OUT_MISSING_REPORT_CSV))
    log("数据库行数: " + str(len(final_rows)))

    log("match_found: " + str(count_flag("match_found")) + "/" + str(len(final_rows)))
    log("has_selected_fragment_record: " + str(count_flag("has_selected_fragment_record")) + "/" + str(len(final_rows)))
    log("has_cv_sp: " + str(count_flag("has_cv_sp")) + "/" + str(len(final_rows)))
    log("has_attn_summary: " + str(count_flag("has_attn_summary")) + "/" + str(len(final_rows)))
    log("has_msa_summary: " + str(count_flag("has_msa_summary")) + "/" + str(len(final_rows)))
    log("has_pymol_summary: " + str(count_flag("has_pymol_summary")) + "/" + str(len(final_rows)))
    log("pdb_exists: " + str(count_flag("pdb_exists")) + "/" + str(len(final_rows)))
    log("image_exists: " + str(count_flag("image_exists")) + "/" + str(len(final_rows)))

    log("full_sequence 非空: " + str(count_nonempty("full_sequence")) + "/" + str(len(final_rows)))
    log("window_seq 非空: " + str(count_nonempty("window_seq")) + "/" + str(len(final_rows)))
    log("selected_fragment 非空: " + str(count_nonempty("selected_fragment")) + "/" + str(len(final_rows)))

    log("说明: cluster_species_percent / cluster_cv_start_position / attn_avg_* / msa_avg_* 均为片段级汇总值")
    log("说明: 每个片段下的多条序列结构记录共享相同 CV/SP、ATT、MSA 片段级指标")

    return final_rows


# =========================================================
# 12. 固定宽表字段
# =========================================================

DB_REQUIRED_FIELDS = [
    # KEY
    "gvp_type",
    "accession",
    "start_pos",
    "end_pos",

    # 基础解析字段
    "parse_ok",
    "match_found",
    "match_candidate_count",

    # 序列字段
    "full_sequence",
    "window_seq",
    "full_sequence_len",
    "window_len",
    "window_expected_len",

    # 注释字段
    "json_organism",
    "json_gene",
    "json_description",
    "name_organism",
    "name_gene",
    "raw_name",

    # 二级结构旧字段
    "ss_has_01",
    "ss_residue_ratio",
    "ss_total_ca",
    "ss_hs_ca",

    # 二级结构新字段
    "ss_alpha_ratio",
    "ss_beta_ratio",
    "ss_flexible_ratio",

    # 二级结构兼容字段
    "alpha_helix_ratio",
    "beta_sheet_ratio",
    "flexible_region_ratio",

    # ss_ratio = alpha + beta
    "ss_ratio",
    "ss_total_score",

    # CV / SP：片段级
    "cluster_species_percent",
    "cluster_cv_start_position",
    "cluster_id_matched",
    "cluster_start_delta",
    "cv_sp_score_level",

    # MSA：片段级汇总
    "msa_start_matched",
    "msa_end_matched",
    "msa_avg_score",
    "msa_avg_percentile",
    "msa_overlap_len",
    "msa_score_level",

    # Attention：片段级汇总
    "attn_start_matched",
    "attn_end_matched",
    "attn_avg_attn",
    "attn_avg_percentile",
    "attn_overlap_len",
    "attn_score_level",

    # 区域结构汇总
    "region_pdb_count",
    "region_scored_count",
    "is_complete_3",

    # 图片
    "image",
    "image_url",
]

EXTRA_CLEAN_FIELDS = [
    # 行身份
    "database_row_id",
    "base_source",

    # 候选片段身份
    "raw_gvp_type",
    "cluster_id",
    "segment_position",
    "window_length",

    # 结构代表序列身份
    "seq_id",
    "structure_organism",
    "structure_genus",

    # selected fragment 单条明细
    "selected_seq_id",
    "selected_organism",
    "selected_genus",
    "selected_full_length",
    "selected_fragment",
    "selected_fragment_length",
    "selected_detail_match_rule",
    "has_selected_fragment_record",

    # JSON 主记录匹配短字段
    "hit_alias",
    "json_primary_id",
    "json_unique_sequence_id",
    "json_annotation_id",
    "json_record_id",
    "json_accession",
    "json_protein_id",
    "json_source_file",
    "json_gvp_types",
    "window_valid",

    # CV / SP 片段级原始字段
    "candidate_cluster_id",
    "candidate_segment_position",
    "candidate_mean_start_position",
    "candidate_window_length",
    "candidate_cv",
    "candidate_species_percent",
    "candidate_start_floor",
    "candidate_end_floor",
    "candidate_sample_count",
    "has_cv_sp",

    # selected_fragment 分组
    "selected_region_sequence_count",
    "selected_region_seq_ids",
    "selected_region_organisms",
    "selected_region_genera",
    "selected_region_unique_species_count",
    "selected_region_unique_genus_count",
    "has_selected_region_record",

    # ATT 序列级审计字段
    "attn_sequence_attention_score",
    "attn_sequence_percentile",
    "attn_fragment_start",
    "attn_fragment_end",
    "attn_detail_match_rule",
    "has_attn_detail",

    # ATT 片段级汇总字段
    "attn_summary_avg_attn",
    "attn_summary_std_attn",
    "attn_summary_avg_percentile",
    "has_attn_summary",

    # MSA 序列级审计字段
    "msa_sequence_fragment_avg_score",
    "msa_sequence_percentile_rank",
    "msa_fragment_start",
    "msa_fragment_end",
    "msa_detail_match_rule",
    "has_msa_detail",

    # MSA 片段级汇总字段
    "msa_summary_sequence_count",
    "msa_summary_avg_conservation_score",
    "msa_summary_conservation_std",
    "msa_summary_avg_percentile",
    "has_msa_summary",

    # PDB / PNG
    "pdb_path",
    "pdb_uri",
    "pdb_exists",
    "image_exists",
    "image_bind_rule",

    # PyMOL 单条结构明细
    "pymol_total_ca",
    "pymol_helix_ca",
    "pymol_sheet_ca",
    "pymol_coil_ca",
    "pymol_has_secondary_structure",
    "pymol_detail_helix_ratio",
    "pymol_detail_sheet_ratio",
    "pymol_detail_coil_ratio",
    "pymol_detail_error",

    # PyMOL 区域汇总
    "pymol_structure_prediction_count",
    "pymol_alpha_helix_ratio",
    "pymol_beta_sheet_ratio",
    "pymol_coil_ratio",
    "pymol_secondary_structure_fragment_ratio",
    "has_pymol_summary",
]

CLEAN_WIDE_FIELDS = []
_seen_fields = set()

for _c in DB_REQUIRED_FIELDS + EXTRA_CLEAN_FIELDS:
    if _c not in _seen_fields:
        CLEAN_WIDE_FIELDS.append(_c)
        _seen_fields.add(_c)

MISSING_FIELDS = [
    "missing_type",
    "database_row_id",
    "gvp_type",
    "raw_gvp_type",
    "cluster_id",
    "segment_position",
    "seq_id",
    "accession",
    "base_source",
    "pdb_path",
    "image",
]


# =========================================================
# 13. 入口
# =========================================================

if __name__ == "__main__":
    try:
        build_database()
    except Exception as e:
        log("[FATAL] 程序异常退出: " + str(e))
        log(traceback.format_exc())
        raise
