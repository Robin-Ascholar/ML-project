#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
把 GVP clean wide 宽表转换为 SQLite DB。

输入优先级：
1. GVP_sequence_level_candidate_database_clean_wide.tsv
2. GVP_sequence_level_candidate_database_clean_wide.csv

输出：
C:\\Users\\r9000\\Desktop\\毕设（无监督聚类）\\GVP_Sequence_Level_Candidate_Database_API_CLEAN\\gvp_web.db

DB 表：
sequence_records

兼容旧代码字段：
gvp_type
accession
start_pos
end_pos
full_sequence
window_seq
full_sequence_len
window_len
window_expected_len

说明：
1. accession 默认使用 seq_id，因为现在是一行一个代表序列/结构记录；
2. 如果 full_sequence 为空，会按 seq_id 从 新gvp/*.json 中匹配补齐；
3. 如果 window_seq 为空，会用 full_sequence[start_pos:end_pos] 补齐；
4. 使用 0-based、end 右开区间，和你前面的 Attention/MSA/ESMFold 脚本一致；
5. 默认重建 DB，避免旧表字段残留或脏数据污染。
"""

import json
import re
import sqlite3
import traceback
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


# =========================================================
# 0) 路径配置
# =========================================================

PROJECT_ROOT = Path(r"C:\Users\r9000\Desktop\毕设（无监督聚类）")

DATA_ROOT = PROJECT_ROOT / "新gvp"

DATABASE_DIR = PROJECT_ROOT / "GVP_Sequence_Level_Candidate_Database_API_CLEAN"

INPUT_TSV = DATABASE_DIR / "GVP_sequence_level_candidate_database_clean_wide.tsv"
INPUT_CSV = DATABASE_DIR / "GVP_sequence_level_candidate_database_clean_wide.csv"

DB_PATH = DATABASE_DIR / "gvp_web.db"

TABLE = "sequence_records"

LOG_PATH = DATABASE_DIR / "build_gvp_web_db_from_clean_wide.log"

# True：每次重新创建 DB
# False：保留旧 DB，只重建 sequence_records 表
RECREATE_DB = True

# 是否把 full_sequence 写入 DB。
# 如果前端不需要全长序列、DB 太大，可以改成 False。
WRITE_FULL_SEQUENCE = True


# =========================================================
# 1) GVP 配置
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
# 2) 兼容旧代码字段
# =========================================================

KEY_COLS = [
    "gvp_type",
    "accession",
    "start_pos",
    "end_pos",
]

SEQ_COLS = [
    "full_sequence",
    "window_seq",
    "full_sequence_len",
    "window_len",
    "window_expected_len",
]

COMPAT_FIRST_FIELDS = KEY_COLS + SEQ_COLS


# =========================================================
# 3) 日志
# =========================================================

STEP_NO = 0

def log(msg: str):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")

def step(msg: str):
    global STEP_NO
    STEP_NO += 1
    log("STEP " + str(STEP_NO).zfill(3) + " | " + msg)


# =========================================================
# 4) 基础工具函数
# =========================================================

def normalize_text(x: Any) -> str:
    if x is None:
        return ""
    try:
        if pd.isna(x):
            return ""
    except Exception:
        pass
    s = str(x).strip()
    if s.lower() in {"nan", "none", "null", "<na>"}:
        return ""
    return s

def to_int(x: Any) -> Optional[int]:
    s = normalize_text(x)
    if not s:
        return None
    try:
        return int(float(s))
    except Exception:
        return None

def to_float(x: Any) -> Optional[float]:
    s = normalize_text(x)
    if not s:
        return None
    try:
        return float(s)
    except Exception:
        return None

def norm_num_for_key(x: Any) -> str:
    s = normalize_text(x)
    if not s:
        return ""
    try:
        return f"{float(s):g}"
    except Exception:
        return s

def norm_txt_for_key(x: Any) -> str:
    return normalize_text(x)

def clean_sequence(seq: Any) -> str:
    return "".join([
        c for c in normalize_text(seq).upper()
        if c in "ACDEFGHIKLMNPQRSTVWY"
    ])

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
    if re.fullmatch(r"gvp[A-Z]", s):
        return "Gvp" + s[-1].upper()
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
    if len(display) == 4 and display.startswith("Gvp"):
        return "gvp" + display[-1].lower()
    return low

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

def build_key_series(df: pd.DataFrame) -> pd.Series:
    return (
        df["gvp_type"].map(norm_txt_for_key) + "||" +
        df["accession"].map(norm_txt_for_key) + "||" +
        df["start_pos"].map(norm_num_for_key) + "||" +
        df["end_pos"].map(norm_num_for_key)
    )

def quote_ident(name: str) -> str:
    return '"' + str(name).replace('"', '""') + '"'


# =========================================================
# 5) 读取 clean wide
# =========================================================

def read_table_auto() -> pd.DataFrame:
    """
    优先读取 TSV，避免 CSV 里长字段/逗号导致 Excel/WPS 误读。
    """
    if INPUT_TSV.exists():
        log(f"  使用 TSV: {INPUT_TSV}")
        return pd.read_csv(
            INPUT_TSV,
            sep="\t",
            dtype=str,
            keep_default_na=False,
            encoding="utf-8-sig",
            engine="python"
        )

    if INPUT_CSV.exists():
        log(f"  使用 CSV: {INPUT_CSV}")

        for enc in ["utf-8-sig", "utf-8", "gbk"]:
            try:
                df = pd.read_csv(
                    INPUT_CSV,
                    dtype=str,
                    keep_default_na=False,
                    encoding=enc,
                    engine="python"
                )

                # 如果错误读成一列，尝试 TSV 方式
                if len(df.columns) == 1:
                    df2 = pd.read_csv(
                        INPUT_CSV,
                        sep="\t",
                        dtype=str,
                        keep_default_na=False,
                        encoding=enc,
                        engine="python"
                    )
                    if len(df2.columns) > len(df.columns):
                        df = df2

                log(f"  CSV读取成功: enc={enc}, rows={len(df)}, cols={len(df.columns)}")
                return df

            except Exception:
                continue

    raise FileNotFoundError(
        "未找到 clean wide 输入文件：\n"
        f"TSV: {INPUT_TSV}\n"
        f"CSV: {INPUT_CSV}"
    )


# =========================================================
# 6) JSON 序列索引，用于补 full_sequence / window_seq
# =========================================================

def make_aliases(raw_id: Any) -> set:
    s = normalize_text(raw_id)
    if not s:
        return set()

    out = {s, s.upper()}

    norm = re.sub(r"[|/\\\s:;]+", "_", s)
    out.add(norm)
    out.add(norm.upper())

    for x in list(out):
        if "." in x:
            b = x.rsplit(".", 1)[0]
            out.add(b)
            out.add(b.upper())

    parts = re.split(r"[|_/\\\s:;]+", s)
    for p in parts:
        p = p.strip()
        if not p:
            continue
        out.add(p)
        out.add(p.upper())
        if "." in p:
            b = p.rsplit(".", 1)[0]
            out.add(b)
            out.add(b.upper())

    for w in re.findall(r"WP_\d+(?:\.\d+)?", s, flags=re.I):
        out.add(w)
        out.add(w.upper())
        if "." in w:
            b = w.rsplit(".", 1)[0]
            out.add(b)
            out.add(b.upper())

    for u in re.findall(r"\b[A-NR-Z][0-9][A-Z0-9]{3}[0-9](?:\.\d+)?\b", s, flags=re.I):
        out.add(u)
        out.add(u.upper())
        if "." in u:
            b = u.rsplit(".", 1)[0]
            out.add(b)
            out.add(b.upper())

    return {x for x in out if x}

def load_json_records(json_path: Path) -> List[Dict[str, Any]]:
    if not json_path.exists():
        log(f"  [WARN] JSON不存在: {json_path}")
        return []

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            obj = json.load(f)
    except Exception as e:
        log(f"  [WARN] JSON读取失败: {json_path} | {e}")
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

def build_json_indices() -> Dict[str, Dict[str, Any]]:
    out = {}

    for raw in GVP_TYPES:
        display = format_gvp_name(raw)
        json_path = DATA_ROOT / raw / f"{display}_sequences.json"
        records = load_json_records(json_path)

        seq_items = []
        alias_map: Dict[str, List[int]] = {}

        for idx, item in enumerate(records):
            ann = item.get("representative_annotation", {}) or {}

            uid = normalize_text(item.get("unique_sequence_id"))
            rid = normalize_text(item.get("id"))
            accession = normalize_text(item.get("accession"))
            protein_id = normalize_text(item.get("protein_id"))
            ann_id = normalize_text(ann.get("id"))

            primary_id = uid or ann_id or rid or accession or protein_id or f"seq_{idx + 1}"
            sequence = clean_sequence(item.get("sequence"))

            seq_item = {
                "primary_id": primary_id,
                "unique_sequence_id": uid,
                "annotation_id": ann_id,
                "record_id": rid,
                "accession": accession,
                "protein_id": protein_id,
                "sequence": sequence,
            }

            seq_idx = len(seq_items)
            seq_items.append(seq_item)

            aliases = set()
            for val in [
                primary_id,
                uid,
                rid,
                accession,
                protein_id,
                ann_id,
                ann.get("full_header"),
                ann.get("description"),
            ]:
                aliases |= make_aliases(val)

            for a in aliases:
                alias_map.setdefault(a.upper(), []).append(seq_idx)

        out[raw] = {
            "json_path": str(json_path),
            "seq_items": seq_items,
            "alias_map": alias_map,
        }

        log(
            f"  {display}: json_records={len(records)}, "
            f"seq_items={len(seq_items)}, alias_keys={len(alias_map)}"
        )

    return out

def match_sequence(raw: str, seq_id: Any, json_indices: Dict[str, Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if raw not in json_indices:
        return None

    aliases = make_aliases(seq_id)
    alias_map = json_indices[raw]["alias_map"]
    seq_items = json_indices[raw]["seq_items"]

    candidates = []
    for a in aliases:
        candidates.extend(alias_map.get(a.upper(), []))

    if candidates:
        uniq = sorted(set(candidates))
        return seq_items[uniq[0]]

    sid = normalize_text(seq_id).upper()
    if sid:
        for k, idxs in alias_map.items():
            if sid in k or k in sid:
                return seq_items[idxs[0]]

    return None


# =========================================================
# 7) 清洗并补齐兼容字段
# =========================================================

def ensure_column(df: pd.DataFrame, col: str, default: Any = "") -> pd.DataFrame:
    if col not in df.columns:
        df[col] = default
    return df

def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # 去掉 Excel 常见 Unnamed 空列
    keep_cols = []
    for c in df.columns:
        name = normalize_text(c)
        if not name:
            continue
        if name.lower().startswith("unnamed:"):
            continue
        keep_cols.append(c)
    df = df[keep_cols].copy()

    # 基础字段
    ensure_column(df, "gvp_type", "")
    ensure_column(df, "raw_gvp_type", "")
    ensure_column(df, "seq_id", "")
    ensure_column(df, "segment_position", "")
    ensure_column(df, "start_pos", "")
    ensure_column(df, "end_pos", "")

    # 统一 GVP
    df["gvp_type"] = df.apply(
        lambda r: format_gvp_name(r.get("gvp_type") or r.get("raw_gvp_type")),
        axis=1
    )
    df["raw_gvp_type"] = df.apply(
        lambda r: raw_gvp_name(r.get("raw_gvp_type") or r.get("gvp_type")),
        axis=1
    )

    # start/end 兜底从 segment_position 解析
    starts = []
    ends = []
    for _, r in df.iterrows():
        s = to_int(r.get("start_pos"))
        e = to_int(r.get("end_pos"))

        if (s is None or e is None) and normalize_text(r.get("segment_position")):
            ps, pe = parse_region(r.get("segment_position"))
            if s is None:
                s = ps
            if e is None:
                e = pe

        starts.append("" if s is None else s)
        ends.append("" if e is None else e)

    df["start_pos"] = starts
    df["end_pos"] = ends

    df["segment_position"] = df.apply(
        lambda r: normalize_text(r.get("segment_position")) or region_to_string(r.get("start_pos"), r.get("end_pos")),
        axis=1
    )

    # accession 按旧代码字段补齐
    # 现在一行是一个代表序列，所以 accession 使用 seq_id 最稳。
    if "accession" not in df.columns:
        df["accession"] = ""

    df["accession"] = df.apply(
        lambda r: (
            normalize_text(r.get("accession"))
            or normalize_text(r.get("seq_id"))
            or normalize_text(r.get("json_accession"))
            or normalize_text(r.get("json_primary_id"))
            or normalize_text(r.get("json_unique_sequence_id"))
        ),
        axis=1
    )

    # 兼容序列字段
    for c in SEQ_COLS:
        ensure_column(df, c, "")

    # window_expected_len
    df["window_expected_len"] = df.apply(
        lambda r: (
            to_int(r.get("window_expected_len"))
            if to_int(r.get("window_expected_len")) is not None
            else (
                to_int(r.get("window_length"))
                if to_int(r.get("window_length")) is not None
                else (
                    to_int(r.get("end_pos")) - to_int(r.get("start_pos"))
                    if to_int(r.get("start_pos")) is not None and to_int(r.get("end_pos")) is not None
                    else ""
                )
            )
        ),
        axis=1
    )

    return df

def fill_sequences_from_json(df: pd.DataFrame) -> pd.DataFrame:
    step("按 seq_id 从 JSON 补齐 full_sequence / window_seq")

    json_indices = build_json_indices()

    df = df.copy()

    filled_full = 0
    filled_window = 0
    matched = 0

    full_values = []
    win_values = []
    full_len_values = []
    win_len_values = []

    for _, r in df.iterrows():
        raw = raw_gvp_name(r.get("raw_gvp_type") or r.get("gvp_type"))
        seq_id = normalize_text(r.get("seq_id") or r.get("accession"))

        full_seq = clean_sequence(r.get("full_sequence"))
        win_seq = clean_sequence(r.get("window_seq"))

        if not full_seq and seq_id:
            hit = match_sequence(raw, seq_id, json_indices)
            if hit:
                matched += 1
                full_seq = clean_sequence(hit.get("sequence"))
                if full_seq:
                    filled_full += 1

        s = to_int(r.get("start_pos"))
        e = to_int(r.get("end_pos"))

        if not win_seq and full_seq and s is not None and e is not None:
            # 与 Attention / MSA / ESMFold 脚本一致：0-based，end 右开
            s2 = max(s, 0)
            e2 = min(e, len(full_seq))
            if e2 > s2:
                win_seq = full_seq[s2:e2]
                filled_window += 1

        if not WRITE_FULL_SEQUENCE:
            full_values.append("")
        else:
            full_values.append(full_seq)

        win_values.append(win_seq)
        full_len_values.append(len(full_seq) if full_seq else "")
        win_len_values.append(len(win_seq) if win_seq else "")

    df["full_sequence"] = full_values
    df["window_seq"] = win_values
    df["full_sequence_len"] = full_len_values
    df["window_len"] = win_len_values

    log(f"  JSON匹配行数: {matched}")
    log(f"  full_sequence 补齐行数: {filled_full}")
    log(f"  window_seq 补齐行数: {filled_window}")

    return df

def reorder_fields(df: pd.DataFrame) -> pd.DataFrame:
    """
    旧代码字段放前面，其余宽表字段跟在后面。
    """
    cols = list(df.columns)

    first = []
    seen = set()

    for c in COMPAT_FIRST_FIELDS:
        if c not in df.columns:
            df[c] = ""
        if c not in seen:
            first.append(c)
            seen.add(c)

    # 推荐常用字段紧跟旧兼容字段后面
    preferred = [
        "database_row_id",
        "base_source",
        "raw_gvp_type",
        "cluster_id",
        "segment_position",
        "window_length",
        "seq_id",
        "structure_organism",
        "structure_genus",
        "selected_full_length",
        "selected_fragment",
        "selected_fragment_length",
        "match_found",
        "json_primary_id",
        "json_unique_sequence_id",
        "json_annotation_id",
        "json_accession",
        "json_protein_id",
        "json_organism",
        "json_gene",
        "candidate_cv",
        "candidate_species_percent",
        "attn_attention_score",
        "attn_percentile",
        "attn_summary_avg_attn",
        "attn_summary_avg_percentile",
        "msa_fragment_avg_score",
        "msa_percentile_rank",
        "msa_summary_avg_conservation_score",
        "msa_summary_avg_percentile",
        "pdb_path",
        "pdb_uri",
        "pdb_exists",
        "image",
        "image_url",
        "image_exists",
        "pymol_total_ca",
        "pymol_helix_ca",
        "pymol_sheet_ca",
        "pymol_coil_ca",
        "pymol_has_secondary_structure",
        "pymol_detail_helix_ratio",
        "pymol_detail_sheet_ratio",
        "pymol_detail_coil_ratio",
        "pymol_alpha_helix_ratio",
        "pymol_beta_sheet_ratio",
        "pymol_coil_ratio",
        "pymol_secondary_structure_fragment_ratio",
    ]

    ordered = first[:]

    for c in preferred:
        if c in df.columns and c not in seen:
            ordered.append(c)
            seen.add(c)

    for c in cols:
        if c not in seen:
            ordered.append(c)
            seen.add(c)

    return df[ordered].copy()

def deduplicate_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    step("按旧代码 KEY_COLS 去重")

    for c in KEY_COLS:
        if c not in df.columns:
            df[c] = ""

    before = len(df)
    df = df.copy()
    df["_db_key"] = build_key_series(df)

    # 空 key 不参与正常去重，但仍保留
    df = df.drop_duplicates(subset=["_db_key"], keep="last").reset_index(drop=True)
    df = df.drop(columns=["_db_key"])

    after = len(df)
    log(f"  去重前: {before}")
    log(f"  去重后: {after}")
    return df


# =========================================================
# 8) SQLite 类型与写入
# =========================================================

TEXT_FORCE_COLS = {
    "gvp_type", "raw_gvp_type", "accession", "seq_id", "cluster_id",
    "segment_position", "database_row_id", "base_source",
    "full_sequence", "window_seq", "selected_fragment",
    "pdb_path", "pdb_uri", "image", "image_url",
    "structure_organism", "structure_genus",
    "selected_region_seq_ids", "selected_region_organisms", "selected_region_genera",
    "json_primary_id", "json_unique_sequence_id", "json_annotation_id",
    "json_record_id", "json_accession", "json_protein_id",
    "json_organism", "json_gene", "json_source_file", "json_gvp_types",
    "hit_alias", "attn_detail_match_rule", "msa_detail_match_rule",
    "image_bind_rule", "pymol_detail_error",
}

REAL_HINTS = [
    "cv", "percent", "percentile", "score", "ratio",
    "attn", "conservation", "mean_start",
]

INTEGER_HINTS = [
    "start", "end", "len", "length", "count", "exists",
    "has_", "match_found", "valid", "ca",
]

def infer_sql_type(col: str, series: pd.Series) -> str:
    if col in TEXT_FORCE_COLS:
        return "TEXT"

    c = col.lower()

    if any(h in c for h in REAL_HINTS):
        return "REAL"

    if any(h in c for h in INTEGER_HINTS):
        return "INTEGER"

    # 尝试自动判断
    nonempty = [normalize_text(x) for x in series.tolist() if normalize_text(x)]
    if not nonempty:
        return "TEXT"

    sample = nonempty[:100]
    all_num = True
    has_float = False

    for v in sample:
        try:
            fv = float(v)
            if not fv.is_integer():
                has_float = True
        except Exception:
            all_num = False
            break

    if all_num:
        return "REAL" if has_float else "INTEGER"

    return "TEXT"

def sql_value(col: str, value: Any, col_type: str):
    s = normalize_text(value)

    if s == "":
        return None

    if col_type == "INTEGER":
        try:
            return int(float(s))
        except Exception:
            return None

    if col_type == "REAL":
        try:
            return float(s)
        except Exception:
            return None

    return s

def create_and_insert_db(df: pd.DataFrame):
    step("写入 SQLite DB")

    DATABASE_DIR.mkdir(parents=True, exist_ok=True)

    if RECREATE_DB and DB_PATH.exists():
        DB_PATH.unlink()
        log(f"  已删除旧 DB: {DB_PATH}")

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute(f"DROP TABLE IF EXISTS {quote_ident(TABLE)}")

    col_types = {
        col: infer_sql_type(col, df[col])
        for col in df.columns
    }

    col_defs = [
        f"{quote_ident(col)} {col_types[col]}"
        for col in df.columns
    ]

    create_sql = (
        f"CREATE TABLE {quote_ident(TABLE)} "
        f"({', '.join(col_defs)})"
    )
    cur.execute(create_sql)

    cols_sql = ", ".join(quote_ident(c) for c in df.columns)
    placeholders = ", ".join(["?"] * len(df.columns))

    insert_sql = (
        f"INSERT INTO {quote_ident(TABLE)} "
        f"({cols_sql}) VALUES ({placeholders})"
    )

    records = []
    for _, row in df.iterrows():
        records.append(tuple(
            sql_value(col, row[col], col_types[col])
            for col in df.columns
        ))

    cur.executemany(insert_sql, records)

    # 索引
    index_specs = [
        ("idx_sequence_records_key", ["gvp_type", "accession", "start_pos", "end_pos"]),
        ("idx_sequence_records_gvp", ["gvp_type"]),
        ("idx_sequence_records_region", ["gvp_type", "segment_position"]),
        ("idx_sequence_records_seq", ["seq_id"]),
        ("idx_sequence_records_cluster", ["gvp_type", "cluster_id"]),
    ]

    for idx_name, cols in index_specs:
        if all(c in df.columns for c in cols):
            cur.execute(
                f"CREATE INDEX IF NOT EXISTS {quote_ident(idx_name)} "
                f"ON {quote_ident(TABLE)} ({', '.join(quote_ident(c) for c in cols)})"
            )

    # metadata
    cur.execute('DROP TABLE IF EXISTS "metadata"')
    cur.execute(
        'CREATE TABLE "metadata" ('
        'key TEXT PRIMARY KEY, '
        'value TEXT'
        ')'
    )

    meta = {
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "source_tsv": str(INPUT_TSV),
        "source_csv": str(INPUT_CSV),
        "db_path": str(DB_PATH),
        "table": TABLE,
        "rows": str(len(df)),
        "columns": str(len(df.columns)),
        "write_full_sequence": str(WRITE_FULL_SEQUENCE),
    }

    cur.executemany(
        'INSERT INTO "metadata" (key, value) VALUES (?, ?)',
        list(meta.items())
    )

    conn.commit()

    # 校验
    n_rows = cur.execute(f"SELECT COUNT(*) FROM {quote_ident(TABLE)}").fetchone()[0]

    n_window = cur.execute(
        f"SELECT COUNT(*) FROM {quote_ident(TABLE)} "
        f"WHERE TRIM(COALESCE(window_seq, '')) <> ''"
    ).fetchone()[0] if "window_seq" in df.columns else 0

    n_full = cur.execute(
        f"SELECT COUNT(*) FROM {quote_ident(TABLE)} "
        f"WHERE TRIM(COALESCE(full_sequence, '')) <> ''"
    ).fetchone()[0] if "full_sequence" in df.columns else 0

    n_img = cur.execute(
        f"SELECT COUNT(*) FROM {quote_ident(TABLE)} "
        f"WHERE COALESCE(image_exists, 0) = 1"
    ).fetchone()[0] if "image_exists" in df.columns else 0

    n_pdb = cur.execute(
        f"SELECT COUNT(*) FROM {quote_ident(TABLE)} "
        f"WHERE COALESCE(pdb_exists, 0) = 1"
    ).fetchone()[0] if "pdb_exists" in df.columns else 0

    conn.close()

    log(f"  DB写入完成: {DB_PATH}")
    log(f"  表名: {TABLE}")
    log(f"  行数: {n_rows}")
    log(f"  列数: {len(df.columns)}")
    log(f"  full_sequence 非空: {n_full}/{n_rows}")
    log(f"  window_seq 非空: {n_window}/{n_rows}")
    log(f"  pdb_exists=1: {n_pdb}/{n_rows}")
    log(f"  image_exists=1: {n_img}/{n_rows}")


# =========================================================
# 9) 主流程
# =========================================================

def main():
    if LOG_PATH.exists():
        LOG_PATH.unlink()

    step("脚本启动")
    log(f"PROJECT_ROOT: {PROJECT_ROOT}")
    log(f"DATA_ROOT exists? {DATA_ROOT.exists()} -> {DATA_ROOT}")
    log(f"INPUT_TSV exists? {INPUT_TSV.exists()} -> {INPUT_TSV}")
    log(f"INPUT_CSV exists? {INPUT_CSV.exists()} -> {INPUT_CSV}")
    log(f"DB_PATH: {DB_PATH}")
    log(f"TABLE: {TABLE}")

    step("读取 clean wide 宽表")
    df = read_table_auto()
    log(f"  原始读取: rows={len(df)}, cols={len(df.columns)}")

    step("清洗字段并补齐旧兼容字段")
    df = prepare_dataframe(df)
    df = fill_sequences_from_json(df)
    df = reorder_fields(df)
    df = deduplicate_dataframe(df)

    step("最终字段检查")
    for c in COMPAT_FIRST_FIELDS:
        if c not in df.columns:
            raise ValueError(f"缺少兼容字段: {c}")

    log("  前置字段: " + ", ".join(df.columns[:len(COMPAT_FIRST_FIELDS)]))
    log(f"  最终宽表: rows={len(df)}, cols={len(df.columns)}")

    create_and_insert_db(df)

    step("完成")
    log("[OK] clean wide CSV/TSV 已转换为 SQLite DB")
    log(f"[OK] DB: {DB_PATH}")
    log(f"[OK] table: {TABLE}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log("[FATAL] 程序异常退出: " + str(e))
        log(traceback.format_exc())
        raise
