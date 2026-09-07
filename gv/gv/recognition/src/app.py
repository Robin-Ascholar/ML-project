# -*- coding: utf-8 -*-

import io
import re
import json
import zipfile
import sqlite3
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List
from urllib.parse import urlparse, unquote

import pandas as pd
import streamlit as st


# =========================
# 0) 配置
# =========================

PROJECT_ROOT = Path(r"C:\Users\r9000\Desktop\毕设（无监督聚类）")
DATABASE_DIR = PROJECT_ROOT / "GVP_Sequence_Level_Candidate_Database_API_CLEAN"

DEFAULT_CSV_PATH = DATABASE_DIR / "GVP_sequence_level_candidate_database_clean_wide.csv"
DB_PATH = DATABASE_DIR / "gvp_web.db"
DEFAULT_TABLE = "sequence_records"

KEY_COLS = ["gvp_type", "accession", "start_pos", "end_pos"]

GVP_SORT_ORDER = {
    "GvpA": 1,
    "GvpC": 2,
    "GvpN": 3,
    "GvpO": 4,
    "GvpG": 5,
    "GvpJ": 6,
    "GvpK": 7,
    "GvpP": 8,
    "gvpa": 1,
    "gvpc": 2,
    "gvpn": 3,
    "gvpo": 4,
    "gvpg": 5,
    "gvpj": 6,
    "gvpk": 7,
    "gvpp": 8,
}

DROP_DISPLAY_COLS = [
    "json_organism",
    "structure_organism",
    "selected_fragment",
    "seq_id",
    "selected_seq_id",
    "selected_organism",
    "selected_genus",
]

DB_FIELD_ORDER = [
    "gvp_type",
    "accession",
    "start_pos",
    "end_pos",

    "name_organism",
    "name_gene",
    "json_gene",

    "full_sequence",
    "window_seq",
    "full_sequence_len",
    "window_len",
    "window_expected_len",

    "parse_ok",
    "match_found",
    "match_candidate_count",

    "cluster_species_percent",
    "cluster_cv_start_position",
    "cluster_id_matched",
    "cluster_start_delta",

    "msa_start_matched",
    "msa_end_matched",
    "msa_avg_score",
    "msa_avg_percentile",
    "msa_overlap_len",

    "attn_start_matched",
    "attn_end_matched",
    "attn_avg_attn",
    "attn_avg_percentile",
    "attn_overlap_len",

    "ss_has_01",

    # 新增：有二级结构序列占比
    "ss_sequence_ratio",

    "ss_total_ca",
    "ss_hs_ca",
    "ss_alpha_ratio",
    "ss_beta_ratio",
    "ss_flexible_ratio",
    "ss_residue_ratio",
    "ss_ratio",
    "ss_total_score",

    "region_pdb_count",
    "region_scored_count",
    "is_complete_3",

    "image",
    "image_url",
    "image_exists",
    "pdb_path",
    "pdb_exists",

    "window_length",
    "pymol_total_ca",
    "pymol_helix_ca",
    "pymol_sheet_ca",
    "pymol_coil_ca",
    "pymol_has_secondary_structure",
    "pymol_structure_prediction_count",
    "pymol_alpha_helix_ratio",
    "pymol_beta_sheet_ratio",
    "pymol_coil_ratio",
    "pymol_secondary_structure_fragment_ratio",
]

NUM_COLS = [
    "start_pos",
    "end_pos",
    "window_length",
    "match_candidate_count",

    "full_sequence_len",
    "window_len",
    "window_expected_len",

    "cluster_species_percent",
    "cluster_cv_start_position",
    "cluster_id_matched",
    "cluster_start_delta",

    "msa_start_matched",
    "msa_end_matched",
    "msa_avg_score",
    "msa_avg_percentile",
    "msa_overlap_len",

    "attn_start_matched",
    "attn_end_matched",
    "attn_avg_attn",
    "attn_avg_percentile",
    "attn_overlap_len",

    "ss_has_01",
    "ss_sequence_ratio",
    "ss_total_ca",
    "ss_hs_ca",
    "ss_alpha_ratio",
    "ss_beta_ratio",
    "ss_flexible_ratio",
    "ss_residue_ratio",
    "ss_ratio",
    "ss_total_score",

    "region_pdb_count",
    "region_scored_count",
    "is_complete_3",

    "image_exists",
    "pdb_exists",

    "pymol_total_ca",
    "pymol_helix_ca",
    "pymol_sheet_ca",
    "pymol_coil_ca",
    "pymol_has_secondary_structure",
    "pymol_structure_prediction_count",
    "pymol_alpha_helix_ratio",
    "pymol_beta_sheet_ratio",
    "pymol_coil_ratio",
    "pymol_secondary_structure_fragment_ratio",
]

BOOL_COLS = [
    "parse_ok",
    "match_found",
    "ss_has_01",
    "is_complete_3",
    "image_exists",
    "pdb_exists",
    "pymol_has_secondary_structure",
]


st.set_page_config(
    page_title="GVP 数据库平台",
    layout="wide"
)

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
    }
    .small-muted {
        color: #666;
        font-size: 12px;
    }
    .metric-note {
        color: #777;
        font-size: 12px;
        margin-top: -10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# =========================
# 1) DB 工具
# =========================

def get_conn():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(DB_PATH)


def list_tables(conn) -> List[str]:
    return [
        r[0]
        for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()
    ]


def table_exists(conn, name: str) -> bool:
    return conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
        (name,)
    ).fetchone() is not None


def table_columns(conn, table_name: str) -> List[str]:
    rows = conn.execute(f'PRAGMA table_info("{table_name}")').fetchall()
    return [r[1] for r in rows]


def create_indexes(conn, table_name: str):
    cur = conn.cursor()

    cur.execute(
        f'''
        CREATE UNIQUE INDEX IF NOT EXISTS "uq_{table_name}_key"
        ON "{table_name}"("gvp_type","accession","start_pos","end_pos")
        '''
    )

    cur.execute(
        f'''
        CREATE INDEX IF NOT EXISTS "idx_{table_name}_gvp"
        ON "{table_name}"("gvp_type")
        '''
    )

    cur.execute(
        f'''
        CREATE INDEX IF NOT EXISTS "idx_{table_name}_acc"
        ON "{table_name}"("accession")
        '''
    )

    cur.execute(
        f'''
        CREATE INDEX IF NOT EXISTS "idx_{table_name}_pos"
        ON "{table_name}"("start_pos","end_pos")
        '''
    )

    conn.commit()


def ensure_db_schema(conn, table_name: str):
    """
    页面启动时自动补齐数据库字段。

    重点：
    - 自动添加 ss_sequence_ratio 列
    - 如果原表有 pymol_secondary_structure_fragment_ratio，
      则把它回填到 ss_sequence_ratio。
    """
    if not table_exists(conn, table_name):
        return

    cols = table_columns(conn, table_name)

    for c in DB_FIELD_ORDER:
        if c not in cols:
            conn.execute(f'ALTER TABLE "{table_name}" ADD COLUMN "{c}" TEXT')
            cols.append(c)

    if (
        "ss_sequence_ratio" in cols
        and "pymol_secondary_structure_fragment_ratio" in cols
    ):
        conn.execute(
            f'''
            UPDATE "{table_name}"
            SET "ss_sequence_ratio" = "pymol_secondary_structure_fragment_ratio"
            WHERE
                ("ss_sequence_ratio" IS NULL OR TRIM("ss_sequence_ratio") = "")
                AND "pymol_secondary_structure_fragment_ratio" IS NOT NULL
                AND TRIM("pymol_secondary_structure_fragment_ratio") != ""
            '''
        )

    conn.commit()
    create_indexes(conn, table_name)


def keep_database_fields(df: pd.DataFrame, add_missing: bool = False) -> pd.DataFrame:
    out = df.copy()

    for c in DROP_DISPLAY_COLS:
        if c in out.columns:
            out = out.drop(columns=[c])

    # 如果没有 ss_sequence_ratio，但有 PyMOL 区域汇总字段，就自动补
    if "ss_sequence_ratio" not in out.columns:
        if "pymol_secondary_structure_fragment_ratio" in out.columns:
            out["ss_sequence_ratio"] = out["pymol_secondary_structure_fragment_ratio"]

    if add_missing:
        for c in DB_FIELD_ORDER:
            if c not in out.columns:
                out[c] = None

    cols = [c for c in DB_FIELD_ORDER if c in out.columns]
    out = out[cols]

    return out


@st.cache_data(show_spinner=False)
def load_data(table_name: str, cache_version: int) -> pd.DataFrame:
    if not DB_PATH.exists():
        return pd.DataFrame()

    conn = get_conn()

    try:
        if not table_exists(conn, table_name):
            return pd.DataFrame()

        ensure_db_schema(conn, table_name)

        df = pd.read_sql_query(
            f'SELECT * FROM "{table_name}"',
            conn
        )

    finally:
        conn.close()

    if "name_organism" not in df.columns:
        df["name_organism"] = ""

    if "full_sequence" not in df.columns:
        df["full_sequence"] = ""

    if "ss_sequence_ratio" not in df.columns:
        if "pymol_secondary_structure_fragment_ratio" in df.columns:
            df["ss_sequence_ratio"] = df["pymol_secondary_structure_fragment_ratio"]
        else:
            df["ss_sequence_ratio"] = None

    df = keep_database_fields(df, add_missing=False)

    for c in NUM_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


# =========================
# 2) 导入处理
# =========================

def read_csv_auto(file_or_path):
    for enc in ["utf-8-sig", "utf-8", "gbk"]:
        try:
            if hasattr(file_or_path, "seek"):
                file_or_path.seek(0)

            return pd.read_csv(file_or_path, encoding=enc)

        except Exception:
            pass

    raise RuntimeError("CSV读取失败，请检查编码")


def bool_to_int(s: pd.Series) -> pd.Series:
    m = s.astype(str).str.strip().str.lower()

    return m.map({
        "true": 1,
        "false": 0,
        "1": 1,
        "0": 0,
        "yes": 1,
        "no": 0,
        "": pd.NA,
        "nan": pd.NA,
        "none": pd.NA,
    }).astype("Int64")


def normalize_organism_col(s: pd.Series) -> pd.Series:
    x = s.fillna("").astype(str).str.strip()

    mask_unc = x.str.lower().str.contains("uncultur", na=False)
    mask_empty = x.eq("")

    x.loc[mask_unc | mask_empty] = "un"

    return x


def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    for k in KEY_COLS:
        if k not in out.columns:
            out[k] = None

    if "name_organism" not in out.columns:
        out["name_organism"] = ""

    if "full_sequence" not in out.columns:
        out["full_sequence"] = ""

    if "image" not in out.columns:
        out["image"] = ""

    if "image_url" not in out.columns:
        out["image_url"] = ""

    if "ss_sequence_ratio" not in out.columns:
        if "pymol_secondary_structure_fragment_ratio" in out.columns:
            out["ss_sequence_ratio"] = out["pymol_secondary_structure_fragment_ratio"]
        else:
            out["ss_sequence_ratio"] = None

    out = keep_database_fields(out, add_missing=True)

    out["image"] = out["image"].fillna("").astype(str).str.strip()
    out["image_url"] = out["image_url"].fillna("").astype(str).str.strip()
    out.loc[out["image_url"] == "", "image_url"] = out["image"]

    if "full_sequence" in out.columns and "full_sequence_len" in out.columns:
        empty_len = (
            out["full_sequence_len"].isna()
            | out["full_sequence_len"].astype(str).str.strip().isin(["", "nan", "None"])
        )
        out.loc[empty_len, "full_sequence_len"] = (
            out.loc[empty_len, "full_sequence"]
            .fillna("")
            .astype(str)
            .str.len()
        )

    if "window_seq" in out.columns and "window_len" in out.columns:
        empty_len = (
            out["window_len"].isna()
            | out["window_len"].astype(str).str.strip().isin(["", "nan", "None"])
        )
        out.loc[empty_len, "window_len"] = (
            out.loc[empty_len, "window_seq"]
            .fillna("")
            .astype(str)
            .str.len()
        )

    for c in NUM_COLS:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    for c in BOOL_COLS:
        if c in out.columns:
            out[c] = bool_to_int(out[c])

    if "name_organism" in out.columns:
        out["name_organism"] = normalize_organism_col(out["name_organism"])

    out = out.drop_duplicates(
        subset=KEY_COLS,
        keep="last"
    ).reset_index(drop=True)

    return out


def save_data(new_df: pd.DataFrame, table_name: str, mode: str):
    conn = get_conn()

    try:
        new_df = keep_database_fields(new_df, add_missing=True)

        if mode == "覆盖全部" or not table_exists(conn, table_name):
            new_df.to_sql(
                table_name,
                conn,
                if_exists="replace",
                index=False
            )
            ensure_db_schema(conn, table_name)
            return

        ensure_db_schema(conn, table_name)

        old = pd.read_sql_query(
            f'SELECT * FROM "{table_name}"',
            conn
        )

        old = keep_database_fields(old, add_missing=True)

        all_cols = [c for c in DB_FIELD_ORDER if c in old.columns or c in new_df.columns]

        for c in all_cols:
            if c not in old.columns:
                old[c] = None
            if c not in new_df.columns:
                new_df[c] = None

        old = old[all_cols]
        new_df = new_df[all_cols]

        merged = pd.concat(
            [old, new_df],
            ignore_index=True
        )

        if mode == "追加":
            merged = merged.drop_duplicates(
                subset=KEY_COLS,
                keep="first"
            )
        else:
            merged = merged.drop_duplicates(
                subset=KEY_COLS,
                keep="last"
            )

        merged.to_sql(
            table_name,
            conn,
            if_exists="replace",
            index=False
        )

        ensure_db_schema(conn, table_name)

    finally:
        conn.close()


# =========================
# 3) 图片与导出
# =========================

def uri_to_local_path(s: str) -> str:
    if not s:
        return ""

    s = str(s).strip()

    if not s:
        return ""

    if s.lower().startswith("file://"):
        try:
            p = unquote(urlparse(s).path)

            if len(p) >= 3 and p[0] == "/" and p[2] == ":":
                p = p[1:]

            return p

        except Exception:
            return ""

    return s


def pick_image_path(row: pd.Series) -> str:
    for col in ["image", "image_url"]:
        if col in row.index:
            v = str(row.get(col, "") or "").strip()

            if not v:
                continue

            local = uri_to_local_path(v)
            p = Path(local)

            if p.exists():
                return str(p)

    return ""


def get_export_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    return keep_database_fields(df, add_missing=False)


def df_to_fasta(df: pd.DataFrame, seq_col: str, tag: str) -> str:
    lines = []

    for _, r in df.iterrows():
        seq = str(r.get(seq_col, "") or "").strip()

        if not seq or seq.lower() == "nan":
            continue

        header = (
            f">{tag}|{r.get('gvp_type', '')}|{r.get('accession', '')}|"
            f"{r.get('start_pos', '')}-{r.get('end_pos', '')} "
            f"organism={r.get('name_organism', '')} "
            f"gene={r.get('json_gene', '')}"
        )

        lines.append(header)
        lines.append(seq)

    return "\n".join(lines) + ("\n" if lines else "")


def build_zip(df: pd.DataFrame) -> bytes:
    export_df = get_export_dataframe(df)

    mem = io.BytesIO()

    with zipfile.ZipFile(mem, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(
            "gvp_filtered.csv",
            export_df.to_csv(index=False).encode("utf-8-sig")
        )

        if "window_seq" in export_df.columns:
            zf.writestr(
                "gvp_window.fasta",
                df_to_fasta(export_df, "window_seq", "WINDOW").encode("utf-8")
            )

        if "full_sequence" in export_df.columns:
            zf.writestr(
                "gvp_full_length.fasta",
                df_to_fasta(export_df, "full_sequence", "FULL").encode("utf-8")
            )

        manifest_rows = []
        img_added = 0

        for i, (_, row) in enumerate(export_df.iterrows(), 1):
            p = pick_image_path(row)
            in_zip = ""

            if p:
                pp = Path(p)
                ext = pp.suffix if pp.suffix else ".png"

                safe_name = re.sub(
                    r'[\\/:*?"<>|\s]+',
                    "_",
                    pp.stem
                )

                in_zip = f"images/{i:06d}_{safe_name}{ext}"

                try:
                    zf.write(pp, arcname=in_zip)
                    img_added += 1
                except Exception:
                    in_zip = ""

            manifest_rows.append({
                "idx": i,
                "gvp_type": row.get("gvp_type", ""),
                "accession": row.get("accession", ""),
                "start_pos": row.get("start_pos", ""),
                "end_pos": row.get("end_pos", ""),
                "image_in_zip": in_zip,
            })

        zf.writestr(
            "image_manifest.csv",
            pd.DataFrame(manifest_rows)
            .to_csv(index=False)
            .encode("utf-8-sig")
        )

        zf.writestr(
            "manifest.json",
            json.dumps(
                {
                    "rows": int(len(export_df)),
                    "images_added": int(img_added),
                    "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                },
                ensure_ascii=False,
                indent=2
            ).encode("utf-8")
        )

    mem.seek(0)

    return mem.read()


# =========================
# 4) 多级排序
# =========================

def sort_dataframe(
    df: pd.DataFrame,
    sort_cols: List[str],
    ascending_list: List[bool]
) -> pd.DataFrame:
    if df.empty or not sort_cols:
        return df.copy()

    out = df.copy()

    real_sort_cols = []
    real_ascending = []
    temp_cols = []

    for col, asc in zip(sort_cols, ascending_list):
        if col not in out.columns:
            continue

        if col == "gvp_type":
            tmp = "__gvp_sort_order__"
            out[tmp] = out["gvp_type"].map(GVP_SORT_ORDER).fillna(9999)
            real_sort_cols.append(tmp)
            real_ascending.append(asc)
            temp_cols.append(tmp)

        else:
            real_sort_cols.append(col)
            real_ascending.append(asc)

    if real_sort_cols:
        out = out.sort_values(
            by=real_sort_cols,
            ascending=real_ascending,
            na_position="last",
            kind="mergesort"
        )

    if temp_cols:
        out = out.drop(columns=temp_cols)

    return out


def build_multisort_controls(df: pd.DataFrame, key_prefix: str) -> Dict[str, Any]:
    st.markdown("### 多级排序")

    sortable_cols = [c for c in DB_FIELD_ORDER if c in df.columns]

    default_order = [
        "gvp_type",
        "start_pos",
        "end_pos",
    ]

    options = ["不使用"] + sortable_cols

    c1, c2, c3 = st.columns(3)

    cols_ui = [c1, c2, c3]

    sort_cols = []
    ascending_list = []

    for i, col_ui in enumerate(cols_ui):
        default_col = (
            default_order[i]
            if i < len(default_order) and default_order[i] in sortable_cols
            else "不使用"
        )

        default_idx = options.index(default_col) if default_col in options else 0

        with col_ui:
            selected = st.selectbox(
                f"排序 {i + 1}",
                options,
                index=default_idx,
                key=f"{key_prefix}_sort_col_{i + 1}"
            )

            order_text = st.radio(
                f"顺序 {i + 1}",
                ["升序", "降序"],
                index=0,
                horizontal=True,
                key=f"{key_prefix}_sort_order_{i + 1}"
            )

        if selected != "不使用" and selected not in sort_cols:
            sort_cols.append(selected)
            ascending_list.append(order_text == "升序")

    return {
        "sort_cols": sort_cols,
        "ascending_list": ascending_list,
    }


# =========================
# 5) 筛选 + 便捷查找
# =========================

def build_sidebar_filters(df: pd.DataFrame) -> Dict[str, Any]:
    st.sidebar.header("筛选 / 便捷查找")

    cfg: Dict[str, Any] = {}

    if "gvp_type" in df.columns:
        gvp_vals = sorted(
            df["gvp_type"].dropna().astype(str).unique(),
            key=lambda x: GVP_SORT_ORDER.get(x, 999)
        )
    else:
        gvp_vals = []

    cfg["sel_gvp"] = st.sidebar.multiselect(
        "GVP类型",
        gvp_vals,
        default=gvp_vals
    )

    cfg["quick_all"] = st.sidebar.text_input("全局关键词（Acc/Gene/Org）")
    cfg["quick_acc_exact"] = st.sidebar.text_input("Accession 精确查找")
    cfg["quick_seq"] = st.sidebar.text_input("序列片段查找（full/window）")

    cfg["q_acc"] = st.sidebar.text_input("Accession 包含")
    cfg["q_org"] = st.sidebar.text_input("Organism 包含")
    cfg["q_gene"] = st.sidebar.text_input("Gene 包含")

    cfg["only_ss"] = st.sidebar.checkbox("仅 ss_has_01=1", False)
    cfg["only_img"] = st.sidebar.checkbox("仅有图片", False)
    cfg["only_complete"] = st.sidebar.checkbox("仅 is_complete_3=1", False)

    def range_widget(col: str, label: str):
        if col not in df.columns:
            return None, None

        valid = pd.to_numeric(df[col], errors="coerce").dropna()

        if valid.empty:
            return None, None

        mn = float(valid.min())
        mx = float(valid.max())

        if mn == mx:
            mx = mn + 1e-9

        return st.sidebar.slider(label, mn, mx, (mn, mx))

    cfg["sp_lo"], cfg["sp_hi"] = range_widget(
        "cluster_species_percent",
        "跨物种占比"
    )

    cfg["cv_lo"], cfg["cv_hi"] = range_widget(
        "cluster_cv_start_position",
        "CV"
    )

    cfg["msa_lo"], cfg["msa_hi"] = range_widget(
        "msa_avg_score",
        "MSA分"
    )

    cfg["att_lo"], cfg["att_hi"] = range_widget(
        "attn_avg_attn",
        "ATTN分"
    )

    cfg["ss_seq_lo"], cfg["ss_seq_hi"] = range_widget(
        "ss_sequence_ratio",
        "含二级结构序列占比"
    )

    cfg["ss_alpha_lo"], cfg["ss_alpha_hi"] = range_widget(
        "ss_alpha_ratio",
        "Alpha螺旋占比"
    )

    cfg["ss_beta_lo"], cfg["ss_beta_hi"] = range_widget(
        "ss_beta_ratio",
        "Beta折叠占比"
    )

    cfg["ss_flex_lo"], cfg["ss_flex_hi"] = range_widget(
        "ss_flexible_ratio",
        "柔性区域占比"
    )

    cfg["ssr_lo"], cfg["ssr_hi"] = range_widget(
        "ss_ratio",
        "有序二级结构占比"
    )

    return cfg


def apply_filters(df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    sub = df.copy()

    def contains(series: pd.Series, text: str):
        if not text:
            return pd.Series([True] * len(series), index=series.index)

        return series.fillna("").astype(str).str.contains(
            re.escape(text),
            case=False,
            na=False
        )

    if cfg["sel_gvp"] and "gvp_type" in sub.columns:
        sub = sub[sub["gvp_type"].astype(str).isin(cfg["sel_gvp"])]

    if cfg["quick_all"]:
        cols = [
            c for c in [
                "accession",
                "name_organism",
                "json_gene",
                "name_gene",
            ]
            if c in sub.columns
        ]

        if cols:
            mask = pd.Series(False, index=sub.index)

            for c in cols:
                mask = mask | contains(sub[c], cfg["quick_all"])

            sub = sub[mask]

    if cfg["quick_acc_exact"] and "accession" in sub.columns:
        sub = sub[
            sub["accession"]
            .fillna("")
            .astype(str)
            .str.strip()
            == cfg["quick_acc_exact"].strip()
        ]

    if cfg["quick_seq"]:
        mask = pd.Series(False, index=sub.index)

        if "full_sequence" in sub.columns:
            mask = mask | contains(sub["full_sequence"], cfg["quick_seq"])

        if "window_seq" in sub.columns:
            mask = mask | contains(sub["window_seq"], cfg["quick_seq"])

        sub = sub[mask]

    if "accession" in sub.columns:
        sub = sub[contains(sub["accession"], cfg["q_acc"])]

    if "name_organism" in sub.columns:
        sub = sub[contains(sub["name_organism"], cfg["q_org"])]

    if "json_gene" in sub.columns:
        sub = sub[contains(sub["json_gene"], cfg["q_gene"])]

    def apply_range(sub_df, col, lo, hi):
        if lo is None or hi is None or col not in sub_df.columns:
            return sub_df

        values = pd.to_numeric(sub_df[col], errors="coerce")

        return sub_df[
            (values.fillna(-1e18) >= lo)
            & (values.fillna(1e18) <= hi)
        ]

    sub = apply_range(sub, "cluster_species_percent", cfg["sp_lo"], cfg["sp_hi"])
    sub = apply_range(sub, "cluster_cv_start_position", cfg["cv_lo"], cfg["cv_hi"])
    sub = apply_range(sub, "msa_avg_score", cfg["msa_lo"], cfg["msa_hi"])
    sub = apply_range(sub, "attn_avg_attn", cfg["att_lo"], cfg["att_hi"])

    sub = apply_range(sub, "ss_sequence_ratio", cfg["ss_seq_lo"], cfg["ss_seq_hi"])
    sub = apply_range(sub, "ss_alpha_ratio", cfg["ss_alpha_lo"], cfg["ss_alpha_hi"])
    sub = apply_range(sub, "ss_beta_ratio", cfg["ss_beta_lo"], cfg["ss_beta_hi"])
    sub = apply_range(sub, "ss_flexible_ratio", cfg["ss_flex_lo"], cfg["ss_flex_hi"])
    sub = apply_range(sub, "ss_ratio", cfg["ssr_lo"], cfg["ssr_hi"])

    if cfg["only_ss"] and "ss_has_01" in sub.columns:
        sub = sub[sub["ss_has_01"] == 1]

    if cfg["only_img"] and "image_exists" in sub.columns:
        sub = sub[sub["image_exists"] == 1]

    if cfg["only_complete"] and "is_complete_3" in sub.columns:
        sub = sub[sub["is_complete_3"] == 1]

    return sub


# =========================
# 6) SQL 查询
# =========================

def is_safe_read_sql(sql: str) -> bool:
    s = sql.strip().lower()

    if not s:
        return False

    allow = (
        s.startswith("select")
        or s.startswith("with")
        or s.startswith("pragma")
    )

    deny_keywords = [
        "insert ",
        "update ",
        "delete ",
        "drop ",
        "alter ",
        "create ",
        "replace ",
        "attach ",
        "vacuum ",
    ]

    if any(k in s for k in deny_keywords):
        return False

    return allow


def run_sql_readonly(sql: str) -> pd.DataFrame:
    if not is_safe_read_sql(sql):
        raise ValueError("仅允许只读SQL：SELECT / WITH / PRAGMA")

    conn = get_conn()

    try:
        out = pd.read_sql_query(sql, conn)
        return keep_database_fields(out, add_missing=False) if not out.empty else out

    finally:
        conn.close()


# =========================
# 7) 页面主体
# =========================

if "cache_version" not in st.session_state:
    st.session_state.cache_version = 0


st.title("GVP 数据库平台")
st.caption(f"当前数据库：{DB_PATH}")

if not DB_PATH.exists():
    st.warning("DB文件不存在。请先运行合并建库脚本，或在导入页导入CSV。")


conn_tmp = get_conn()

try:
    tables = list_tables(conn_tmp)

finally:
    conn_tmp.close()


if not tables:
    tables = [DEFAULT_TABLE]


default_idx = tables.index(DEFAULT_TABLE) if DEFAULT_TABLE in tables else 0

table_name = st.selectbox(
    "选择数据表",
    tables,
    index=default_idx
)

if DB_PATH.exists():
    conn_migrate = get_conn()
    try:
        if table_exists(conn_migrate, table_name):
            ensure_db_schema(conn_migrate, table_name)
    finally:
        conn_migrate.close()


if st.button("刷新缓存"):
    st.session_state.cache_version += 1
    st.cache_data.clear()
    st.rerun()


tabs = st.tabs([
    "浏览",
    "导入",
    "导出",
    "统计",
    "SQL查询",
])


df = load_data(
    table_name,
    st.session_state.cache_version
) if DB_PATH.exists() else pd.DataFrame()


filtered = df.copy()

if not df.empty:
    cfg = build_sidebar_filters(df)
    filtered = apply_filters(df, cfg)


# =========================
# 浏览
# =========================

with tabs[0]:
    if df.empty:
        st.info("当前表暂无数据，请先导入或生成数据库。")

    else:
        st.success(f"筛选后：{len(filtered)} / {len(df)} 条")

        sort_cfg = build_multisort_controls(filtered, key_prefix="browse")

        show = sort_dataframe(
            filtered,
            sort_cfg["sort_cols"],
            sort_cfg["ascending_list"]
        )

        c1, c2 = st.columns([1, 1])

        with c1:
            page_size = st.selectbox(
                "每页显示",
                [20, 50, 100, 200],
                index=1
            )

        total = len(show)
        pages = max((total + page_size - 1) // page_size, 1)

        with c2:
            page = st.number_input(
                "页码",
                min_value=1,
                max_value=pages,
                value=1,
                step=1
            )

        part = show.iloc[
            (page - 1) * page_size:
            page * page_size
        ].copy()

        default_cols = [
            c for c in [
                "gvp_type",
                "accession",
                "name_organism",
                "json_gene",
                "start_pos",
                "end_pos",
                "window_seq",
                "full_sequence_len",
                "window_len",
                "cluster_species_percent",
                "cluster_cv_start_position",
                "msa_avg_score",
                "msa_avg_percentile",
                "attn_avg_attn",
                "attn_avg_percentile",
                "ss_has_01",
                "ss_sequence_ratio",
                "ss_alpha_ratio",
                "ss_beta_ratio",
                "ss_flexible_ratio",
                "ss_ratio",
                "image_exists",
            ]
            if c in part.columns
        ]

        cols_show = st.multiselect(
            "显示列",
            list(part.columns),
            default=default_cols
        )

        display_part = part[cols_show].copy() if cols_show else part.copy()

        st.caption("点击表格中的一行查看单条详情。")

        selected_event = st.dataframe(
            display_part,
            use_container_width=True,
            height=500,
            on_select="rerun",
            selection_mode="single-row",
            key="browse_table_selection"
        )

        selected_rows = []

        try:
            selected_rows = selected_event.selection.rows
        except Exception:
            selected_rows = []

        st.markdown("### 单条详情")

        if not selected_rows:
            st.info("请在上方表格中点击一行。")

        else:
            selected_pos = selected_rows[0]
            row = part.iloc[int(selected_pos)]

            l, r = st.columns([1, 2.6])

            with l:
                img_path = pick_image_path(row)

                if img_path:
                    st.image(
                        img_path,
                        caption=Path(img_path).name,
                        use_container_width=True
                    )
                else:
                    st.info("无可用图片")

            with r:
                st.markdown("#### 基础信息")
                basic_keys = [
                    ("gvp_type", "GVP 类型"),
                    ("accession", "Accession"),
                    ("name_organism", "物种"),
                    ("json_gene", "基因"),
                    ("start_pos", "片段起始位置"),
                    ("end_pos", "片段结束位置"),
                    ("window_expected_len", "理论片段长度"),
                    ("full_sequence_len", "全长序列长度"),
                    ("window_len", "窗口序列长度"),
                ]

                for k, label in basic_keys:
                    if k in row.index:
                        st.write(f"**{label}**: {row[k]}")

                st.markdown("#### 保守性 / 注意力 / 聚类")
                score_keys = [
                    ("cluster_species_percent", "跨物种占比"),
                    ("cluster_cv_start_position", "起始位置 CV"),
                    ("msa_avg_score", "MSA 平均保守性分数"),
                    ("msa_avg_percentile", "MSA 平均百分位"),
                    ("msa_overlap_len", "MSA 与片段重叠长度"),
                    ("attn_avg_attn", "Attention 平均分"),
                    ("attn_avg_percentile", "Attention 平均百分位"),
                    ("attn_overlap_len", "Attention 与片段重叠长度"),
                ]

                for k, label in score_keys:
                    if k in row.index:
                        st.write(f"**{label}**: {row[k]}")

                st.markdown("#### 二级结构")
                ss_keys = [
                    ("ss_has_01", "是否含二级结构"),
                    ("ss_sequence_ratio", "含二级结构序列占比"),
                    ("ss_total_ca", "总 CA 数"),
                    ("ss_hs_ca", "Alpha + Beta CA 数"),
                    ("ss_alpha_ratio", "Alpha 螺旋占比"),
                    ("ss_beta_ratio", "Beta 折叠占比"),
                    ("ss_flexible_ratio", "柔性区域占比"),
                    ("ss_ratio", "有序二级结构占比 Alpha + Beta"),
                ]

                for k, label in ss_keys:
                    if k in row.index:
                        st.write(f"**{label}**: {row[k]}")

                st.markdown("#### 文件")
                file_keys = [
                    ("image", "图片路径"),
                    ("image_url", "图片 URI"),
                    ("pdb_path", "PDB 路径"),
                ]

                for k, label in file_keys:
                    if k in row.index:
                        st.write(f"**{label}**: {row[k]}")

                if "window_seq" in row.index:
                    st.text_area(
                        "window_seq",
                        str(row.get("window_seq", "")),
                        height=90
                    )

                if "full_sequence" in row.index:
                    st.text_area(
                        "full_sequence",
                        str(row.get("full_sequence", "")),
                        height=160
                    )


# =========================
# 导入
# =========================

with tabs[1]:
    st.markdown("### 导入 CSV")
    st.write(f"默认CSV：`{DEFAULT_CSV_PATH}`")

    mode = st.radio(
        "导入模式",
        ["追加", "覆盖全部", "智能更新"],
        horizontal=True
    )

    c1, c2 = st.columns([1, 1])

    with c1:
        if st.button("一键导入默认CSV"):
            try:
                if not DEFAULT_CSV_PATH.exists():
                    st.error("默认CSV不存在")
                else:
                    raw = read_csv_auto(DEFAULT_CSV_PATH)
                    new_df = normalize_df(raw)

                    save_data(new_df, table_name, mode)

                    st.session_state.cache_version += 1
                    st.cache_data.clear()

                    st.success(f"导入成功：{len(new_df)} 条（{mode}）")
                    st.rerun()

            except Exception as e:
                st.error(f"导入失败：{e}")

    with c2:
        up = st.file_uploader("上传CSV", type=["csv"])

        if up is not None:
            try:
                preview = read_csv_auto(up)
                preview = keep_database_fields(preview, add_missing=False)
                st.dataframe(
                    preview.head(),
                    use_container_width=True
                )

            except Exception as e:
                st.error(f"预览失败：{e}")

    if st.button("开始上传导入"):
        if up is None:
            st.error("请先上传CSV")
        else:
            try:
                raw = read_csv_auto(up)
                new_df = normalize_df(raw)

                save_data(new_df, table_name, mode)

                st.session_state.cache_version += 1
                st.cache_data.clear()

                st.success(f"导入成功：{len(new_df)} 条（{mode}）")
                st.rerun()

            except Exception as e:
                st.error(f"导入失败：{e}")


# =========================
# 导出
# =========================

with tabs[2]:
    st.markdown("### 导出当前筛选结果")

    if df.empty:
        st.info("暂无数据")

    else:
        export_sort_cfg = build_multisort_controls(filtered, key_prefix="export")

        export_df = sort_dataframe(
            filtered,
            export_sort_cfg["sort_cols"],
            export_sort_cfg["ascending_list"]
        )

        export_df = get_export_dataframe(export_df)

        st.write(f"当前筛选：{len(export_df)} 条")

        st.download_button(
            "导出 CSV",
            export_df.to_csv(index=False).encode("utf-8-sig"),
            "gvp_filtered.csv",
            "text/csv"
        )

        if "window_seq" in export_df.columns:
            st.download_button(
                "导出 WINDOW FASTA",
                df_to_fasta(export_df, "window_seq", "WINDOW").encode("utf-8"),
                "gvp_window.fasta",
                "text/plain"
            )

        if "full_sequence" in export_df.columns:
            st.download_button(
                "导出 FULL FASTA",
                df_to_fasta(export_df, "full_sequence", "FULL").encode("utf-8"),
                "gvp_full_length.fasta",
                "text/plain"
            )

        st.download_button(
            "导出 ZIP（CSV+FASTA+图片）",
            build_zip(export_df),
            "gvp_bundle.zip",
            "application/zip"
        )


# =========================
# 统计
# =========================

with tabs[3]:
    st.markdown("### 统计")

    if df.empty:
        st.info("暂无数据")

    else:
        a, b, c, d, e = st.columns(5)

        a.metric("总记录", len(df))

        b.metric(
            "GVP类型数",
            int(df["gvp_type"].nunique()) if "gvp_type" in df.columns else 0
        )

        c.metric(
            "含二级结构记录",
            int((df["ss_has_01"] == 1).sum()) if "ss_has_01" in df.columns else 0
        )

        d.metric(
            "有图片",
            int((df["image_exists"] == 1).sum()) if "image_exists" in df.columns else 0
        )

        e.metric(
            "完整记录",
            int((df["is_complete_3"] == 1).sum()) if "is_complete_3" in df.columns else 0
        )

        if "gvp_type" in df.columns:
            st.markdown("**按 GVP 类型计数**")

            vc = df["gvp_type"].value_counts()
            vc = vc.sort_index(
                key=lambda idx: idx.map(lambda x: GVP_SORT_ORDER.get(x, 999))
            )

            st.bar_chart(vc)

        num_cols = [
            x for x in [
                "cluster_species_percent",
                "cluster_cv_start_position",
                "msa_avg_score",
                "msa_avg_percentile",
                "attn_avg_attn",
                "attn_avg_percentile",
                "ss_sequence_ratio",
                "ss_alpha_ratio",
                "ss_beta_ratio",
                "ss_flexible_ratio",
                "ss_ratio",
            ]
            if x in df.columns
        ]

        if num_cols:
            st.markdown("**关键分数描述统计**")
            st.dataframe(
                df[num_cols].describe().T,
                use_container_width=True
            )

        if all(c in df.columns for c in ["gvp_type", "ss_sequence_ratio", "ss_alpha_ratio", "ss_beta_ratio", "ss_flexible_ratio"]):
            st.markdown("**按 GVP 类型的平均二级结构指标**")

            ss_group = (
                df.groupby("gvp_type")[
                    [
                        "ss_sequence_ratio",
                        "ss_alpha_ratio",
                        "ss_beta_ratio",
                        "ss_flexible_ratio",
                    ]
                ]
                .mean(numeric_only=True)
                .reset_index()
            )

            ss_group["__order"] = ss_group["gvp_type"].map(GVP_SORT_ORDER).fillna(999)
            ss_group = ss_group.sort_values("__order").drop(columns="__order")

            st.dataframe(
                ss_group,
                use_container_width=True
            )


# =========================
# SQL 查询
# =========================

with tabs[4]:
    st.markdown("### SQL 查询（只读）")
    st.caption("支持 SELECT / WITH / PRAGMA；禁止修改语句。")

    sample_sql = (
        f'SELECT accession, gvp_type, start_pos, end_pos, '
        f'name_organism, json_gene, '
        f'full_sequence_len, window_len, '
        f'cluster_species_percent, cluster_cv_start_position, '
        f'msa_avg_score, attn_avg_attn, '
        f'ss_sequence_ratio, '
        f'ss_alpha_ratio, ss_beta_ratio, ss_flexible_ratio, ss_ratio '
        f'FROM "{table_name}" '
        f'ORDER BY '
        f'CASE gvp_type '
        f'WHEN "GvpA" THEN 1 '
        f'WHEN "GvpC" THEN 2 '
        f'WHEN "GvpN" THEN 3 '
        f'WHEN "GvpO" THEN 4 '
        f'WHEN "GvpG" THEN 5 '
        f'WHEN "GvpJ" THEN 6 '
        f'WHEN "GvpK" THEN 7 '
        f'WHEN "GvpP" THEN 8 '
        f'ELSE 999 END, '
        f'start_pos, end_pos '
        f'LIMIT 100;'
    )

    sql_text = st.text_area(
        "输入SQL",
        value=sample_sql,
        height=180
    )

    c1, c2 = st.columns([1, 5])

    run_btn = c1.button("执行SQL")

    if run_btn:
        try:
            out = run_sql_readonly(sql_text)

            st.success(f"返回 {len(out)} 行")

            st.dataframe(
                out,
                use_container_width=True,
                height=420
            )

            export_sql_df = get_export_dataframe(out)

            st.download_button(
                "下载SQL结果CSV",
                export_sql_df.to_csv(index=False).encode("utf-8-sig"),
                "sql_result.csv",
                "text/csv"
            )

        except Exception as e:
            st.error(f"SQL执行失败：{e}")
