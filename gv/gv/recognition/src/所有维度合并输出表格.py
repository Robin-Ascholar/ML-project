#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
表4-11 候选片段综合排序结果：每种 GVP 取前4

输出字段只包含：
    gvp_type
    region
    cluster_species_percent
    cluster_cv_start_position
    attn_avg_percentile
    msa_avg_percentile
    ss_ratio
    ss_sequence_ratio

排序逻辑：
    每种 GVP 内部单独排序。

    四大类：
    1. CV + SP
       - cluster_species_percent 越大越好
       - cluster_cv_start_position 越小越好
       - 二者各占 0.5

    2. Attention
       - attn_avg_percentile 越大越好

    3. MSA
       - msa_avg_percentile 越大越好

    4. 二级结构
       - ss_ratio 越大越好
       - ss_sequence_ratio 越大越好
       - 二者各占 0.5

    最终综合分：
       total_score =
           0.25 * cv_sp_score
         + 0.25 * attn_score
         + 0.25 * msa_score
         + 0.25 * ss_score

说明：
    ss_sequence_ratio 表示该候选片段中“含二级结构的序列数 / 该片段结构预测序列总数”。
    如果数据库中已经有 pymol_secondary_structure_fragment_ratio 或 ss_sequence_ratio，
    优先读取已有字段；如果没有，则按同一片段内 ss_has_01 的均值重新计算。

输入：
    优先读取：
        GVP_Sequence_Level_Candidate_Database_API_CLEAN/gvp_web.db
        表：sequence_records

    如果 DB 不存在，则读取：
        GVP_Sequence_Level_Candidate_Database_API_CLEAN/GVP_sequence_level_candidate_database_clean_wide.csv

输出：
    Table_4_11_top4_candidate_fragments_rank_fields.csv
    Table_4_11_top4_candidate_fragments_rank_fields.tsv
"""

import sqlite3
from pathlib import Path
from typing import Any, List, Optional

import pandas as pd


# =========================================================
# 0. 路径配置
# =========================================================

PROJECT_ROOT = Path(r"C:\Users\r9000\Desktop\毕设（无监督聚类）")

DATABASE_DIR = PROJECT_ROOT / "GVP_Sequence_Level_Candidate_Database_API_CLEAN"

INPUT_DB = DATABASE_DIR / "gvp_web.db"
INPUT_TABLE = "sequence_records"

INPUT_CSV = DATABASE_DIR / "GVP_sequence_level_candidate_database_clean_wide.csv"

OUT_CSV = DATABASE_DIR / "Table_4_11_top4_candidate_fragments_rank_fields.csv"
OUT_TSV = DATABASE_DIR / "Table_4_11_top4_candidate_fragments_rank_fields.tsv"


# =========================================================
# 1. GVP 排序
# =========================================================

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


# =========================================================
# 2. 最终输出字段
# =========================================================

OUTPUT_FIELDS = [
    "gvp_type",
    "region",

    "cluster_species_percent",
    "cluster_cv_start_position",

    "attn_avg_percentile",
    "msa_avg_percentile",

    "ss_ratio",
    "ss_sequence_ratio",
]


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


def first_existing_col(df: pd.DataFrame, cols: List[str]) -> Optional[str]:
    for c in cols:
        if c in df.columns:
            return c

    return None


def first_non_null(series: pd.Series):
    x = series.dropna()

    if x.empty:
        return pd.NA

    return x.iloc[0]


def mean_non_null(series: pd.Series):
    x = pd.to_numeric(series, errors="coerce").dropna()

    if x.empty:
        return pd.NA

    return float(x.mean())


def round_value(x, ndigits: int):
    try:
        if pd.isna(x):
            return pd.NA

        return round(float(x), ndigits)

    except Exception:
        return pd.NA


def ensure_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    out = df.copy()

    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    return out


def rank_score(s: pd.Series, higher_is_better: bool = True) -> pd.Series:
    """
    在同一个 GVP 类型内部计算 0-1 排名分。

    第 1 名 = 1
    最后 1 名 = 0
    缺失值 = 0
    """

    values = pd.to_numeric(s, errors="coerce")
    valid = values.notna()

    scores = pd.Series(0.0, index=s.index)

    n = int(valid.sum())

    if n == 0:
        return scores

    if n == 1:
        scores.loc[valid] = 1.0
        return scores

    ranks = values.loc[valid].rank(
        ascending=not higher_is_better,
        method="average"
    )

    scores.loc[valid] = 1.0 - (ranks - 1.0) / (n - 1.0)

    return scores


# =========================================================
# 4. 读取输入数据
# =========================================================

def read_input_data() -> pd.DataFrame:
    """
    优先读取 SQLite DB。
    如果 DB 不存在或表不存在，则读取 clean_wide CSV。
    """

    if INPUT_DB.exists():
        conn = sqlite3.connect(str(INPUT_DB))

        try:
            tables = pd.read_sql_query(
                "SELECT name FROM sqlite_master WHERE type='table'",
                conn
            )["name"].tolist()

            if INPUT_TABLE in tables:
                df = pd.read_sql_query(
                    f'SELECT * FROM "{INPUT_TABLE}"',
                    conn
                )

                print(f"[INFO] 已读取 SQLite DB: {INPUT_DB}")
                print(f"[INFO] 表名: {INPUT_TABLE}")
                print(f"[INFO] 原始记录数: {len(df)}")

                return df

        finally:
            conn.close()

    if INPUT_CSV.exists():
        for enc in ["utf-8-sig", "utf-8", "gbk"]:
            try:
                df = pd.read_csv(INPUT_CSV, encoding=enc)

                print(f"[INFO] 已读取 CSV: {INPUT_CSV}")
                print(f"[INFO] 编码: {enc}")
                print(f"[INFO] 原始记录数: {len(df)}")

                return df

            except UnicodeDecodeError:
                continue

    raise FileNotFoundError(
        "没有找到可用输入数据。请先生成 gvp_web.db 或 clean_wide CSV。\n"
        f"DB: {INPUT_DB}\n"
        f"CSV: {INPUT_CSV}"
    )


# =========================================================
# 5. 字段标准化
# =========================================================

def normalize_source_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    把不同版本数据库字段统一成本脚本需要的字段。
    """

    out = df.copy()

    # 必须字段
    for c in ["gvp_type", "start_pos", "end_pos"]:
        if c not in out.columns:
            out[c] = pd.NA

    # 跨物种占比
    if "cluster_species_percent" not in out.columns:
        c = first_existing_col(out, [
            "candidate_species_percent",
            "species_percent",
        ])
        out["cluster_species_percent"] = out[c] if c else pd.NA

    # 起始位置 CV
    if "cluster_cv_start_position" not in out.columns:
        c = first_existing_col(out, [
            "candidate_cv",
            "cv_start_position",
            "cv",
        ])
        out["cluster_cv_start_position"] = out[c] if c else pd.NA

    # Attention 平均百分位
    if "attn_avg_percentile" not in out.columns:
        c = first_existing_col(out, [
            "attn_summary_avg_percentile",
            "attn_percentile",
            "percentile",
        ])
        out["attn_avg_percentile"] = out[c] if c else pd.NA

    # MSA 平均百分位
    if "msa_avg_percentile" not in out.columns:
        c = first_existing_col(out, [
            "msa_summary_avg_percentile",
            "msa_percentile_rank",
        ])
        out["msa_avg_percentile"] = out[c] if c else pd.NA

    # ss_ratio
    if "ss_ratio" not in out.columns:
        c = first_existing_col(out, [
            "alpha_helix_ratio",
            "ss_alpha_ratio",
            "pymol_alpha_helix_ratio",
        ])

        b = first_existing_col(out, [
            "beta_sheet_ratio",
            "ss_beta_ratio",
            "pymol_beta_sheet_ratio",
        ])

        if c and b:
            out["ss_ratio"] = (
                pd.to_numeric(out[c], errors="coerce").fillna(0)
                + pd.to_numeric(out[b], errors="coerce").fillna(0)
            )
        else:
            out["ss_ratio"] = pd.NA

    # ss_has_01
    if "ss_has_01" not in out.columns:
        c = first_existing_col(out, [
            "pymol_has_secondary_structure",
            "has_secondary_structure",
        ])

        if c:
            out["ss_has_01"] = out[c]
        else:
            out["ss_has_01"] = pd.NA

    # ss_sequence_ratio
    if "ss_sequence_ratio" not in out.columns:
        c = first_existing_col(out, [
            "pymol_secondary_structure_fragment_ratio",
            "secondary_structure_fragment_ratio",
            "ss_fragment_ratio",
        ])

        if c:
            out["ss_sequence_ratio"] = out[c]
        else:
            out["ss_sequence_ratio"] = pd.NA

    numeric_cols = [
        "start_pos",
        "end_pos",
        "cluster_species_percent",
        "cluster_cv_start_position",
        "attn_avg_percentile",
        "msa_avg_percentile",
        "ss_ratio",
        "ss_has_01",
        "ss_sequence_ratio",
    ]

    out = ensure_numeric(out, numeric_cols)

    # 如果 ss_has_01 缺失，用 ss_ratio > 0 判断
    mask_has_missing = out["ss_has_01"].isna()

    out.loc[mask_has_missing, "ss_has_01"] = (
        out.loc[mask_has_missing, "ss_ratio"].fillna(0) > 0
    ).astype(int)

    return out


# =========================================================
# 6. 按候选片段聚合
# =========================================================

def build_fragment_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    一行 = 一个候选片段。
    分组键 = gvp_type + start_pos + end_pos。
    """

    df = normalize_source_columns(df)

    df = df.dropna(subset=["gvp_type", "start_pos", "end_pos"]).copy()

    df["gvp_type"] = df["gvp_type"].astype(str).str.strip()
    df["start_pos"] = pd.to_numeric(df["start_pos"], errors="coerce")
    df["end_pos"] = pd.to_numeric(df["end_pos"], errors="coerce")

    df = df.dropna(subset=["start_pos", "end_pos"]).copy()

    df["start_pos"] = df["start_pos"].astype(int)
    df["end_pos"] = df["end_pos"].astype(int)

    rows = []

    group_cols = ["gvp_type", "start_pos", "end_pos"]

    for (gvp_type, start_pos, end_pos), g in df.groupby(group_cols, dropna=False):
        # ss_sequence_ratio 优先使用已有片段级字段。
        # 如果没有，则按同片段 ss_has_01 均值计算。
        ss_sequence_ratio = first_non_null(g["ss_sequence_ratio"])

        if pd.isna(ss_sequence_ratio):
            ss_sequence_ratio = mean_non_null(g["ss_has_01"])

        row = {
            "gvp_type": gvp_type,
            "start_pos": int(start_pos),
            "end_pos": int(end_pos),
            "region": f"{int(start_pos)}-{int(end_pos)}",

            "cluster_species_percent": first_non_null(g["cluster_species_percent"]),
            "cluster_cv_start_position": first_non_null(g["cluster_cv_start_position"]),

            "attn_avg_percentile": first_non_null(g["attn_avg_percentile"]),
            "msa_avg_percentile": first_non_null(g["msa_avg_percentile"]),

            "ss_ratio": mean_non_null(g["ss_ratio"]),
            "ss_sequence_ratio": ss_sequence_ratio,
        }

        rows.append(row)

    out = pd.DataFrame(rows)

    if out.empty:
        return out

    numeric_cols = [
        "cluster_species_percent",
        "cluster_cv_start_position",
        "attn_avg_percentile",
        "msa_avg_percentile",
        "ss_ratio",
        "ss_sequence_ratio",
    ]

    out = ensure_numeric(out, numeric_cols)

    return out


# =========================================================
# 7. 每种 GVP 计算综合分并取前4
# =========================================================

def add_scores_one_gvp(g: pd.DataFrame) -> pd.DataFrame:
    """
    对单个 GVP 类型内部计算排名分和综合分。
    """

    out = g.copy()

    # 1. CV + SP 类
    out["species_rank_score"] = rank_score(
        out["cluster_species_percent"],
        higher_is_better=True
    )

    out["cv_rank_score"] = rank_score(
        out["cluster_cv_start_position"],
        higher_is_better=False
    )

    out["cv_sp_score"] = (
        0.5 * out["species_rank_score"]
        + 0.5 * out["cv_rank_score"]
    )

    # 2. Attention 类
    out["attn_score"] = rank_score(
        out["attn_avg_percentile"],
        higher_is_better=True
    )

    # 3. MSA 类
    out["msa_score"] = rank_score(
        out["msa_avg_percentile"],
        higher_is_better=True
    )

    # 4. 二级结构类
    out["ss_ratio_rank_score"] = rank_score(
        out["ss_ratio"],
        higher_is_better=True
    )

    out["ss_sequence_rank_score"] = rank_score(
        out["ss_sequence_ratio"],
        higher_is_better=True
    )

    out["ss_score"] = (
        0.5 * out["ss_ratio_rank_score"]
        + 0.5 * out["ss_sequence_rank_score"]
    )

    # 总分：四大类等权
    out["total_score"] = (
        0.25 * out["cv_sp_score"]
        + 0.25 * out["attn_score"]
        + 0.25 * out["msa_score"]
        + 0.25 * out["ss_score"]
    )

    return out


def select_top4_by_gvp(fragment_df: pd.DataFrame) -> pd.DataFrame:
    """
    每种 GVP 内部综合排序，取前 4。
    """

    if fragment_df.empty:
        return pd.DataFrame(columns=OUTPUT_FIELDS)

    scored_parts = []

    for gvp_type, g in fragment_df.groupby("gvp_type", dropna=False):
        scored = add_scores_one_gvp(g)

        scored = scored.sort_values(
            by=[
                "total_score",
                "cv_sp_score",
                "attn_score",
                "msa_score",
                "ss_score",
                "start_pos",
                "end_pos",
            ],
            ascending=[
                False,
                False,
                False,
                False,
                False,
                True,
                True,
            ],
            kind="mergesort"
        )

        scored_parts.append(scored.head(4))

    out = pd.concat(scored_parts, ignore_index=True)

    out["__gvp_order"] = out["gvp_type"].map(GVP_SORT_ORDER).fillna(9999)

    out = out.sort_values(
        by=[
            "__gvp_order",
            "total_score",
            "start_pos",
            "end_pos",
        ],
        ascending=[
            True,
            False,
            True,
            True,
        ],
        kind="mergesort"
    ).drop(columns=["__gvp_order"])

    # 四舍五入
    round_4_cols = [
        "cluster_species_percent",
        "cluster_cv_start_position",
        "ss_ratio",
        "ss_sequence_ratio",
    ]

    round_2_cols = [
        "attn_avg_percentile",
        "msa_avg_percentile",
    ]

    for c in round_4_cols:
        if c in out.columns:
            out[c] = out[c].apply(lambda x: round_value(x, 4))

    for c in round_2_cols:
        if c in out.columns:
            out[c] = out[c].apply(lambda x: round_value(x, 2))

    out = out[OUTPUT_FIELDS].reset_index(drop=True)

    return out


# =========================================================
# 8. 输出
# =========================================================

def print_terminal_table(df: pd.DataFrame):
    pd.set_option("display.max_rows", 1000)
    pd.set_option("display.max_columns", 100)
    pd.set_option("display.width", 220)
    pd.set_option("display.max_colwidth", 80)

    print("\n" + "=" * 140)
    print("表4-11 候选片段综合排序结果：每种 GVP 前4")
    print("=" * 140)

    if df.empty:
        print("[WARN] 没有可输出记录。")
    else:
        print(df.to_string(index=False))

    print("=" * 140)
    print(f"输出候选片段数: {len(df)}")
    print("=" * 140 + "\n")


def save_outputs(df: pd.DataFrame):
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    df.to_csv(OUT_TSV, index=False, encoding="utf-8-sig", sep="\t")

    print(f"[INFO] CSV 已输出: {OUT_CSV}")
    print(f"[INFO] TSV 已输出: {OUT_TSV}")


# =========================================================
# 9. 主程序
# =========================================================

def main():
    source_df = read_input_data()

    fragment_df = build_fragment_table(source_df)

    print(f"[INFO] 聚合后候选片段数: {len(fragment_df)}")

    top4_df = select_top4_by_gvp(fragment_df)

    print_terminal_table(top4_df)

    save_outputs(top4_df)


if __name__ == "__main__":
    main()
