#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
输出表4-11：每种 GVP 综合排序前4的候选片段验证结果

一行 = 一个候选片段，不是单条序列。

输出字段：
    gvp_type
    region
    cluster_species_percent
    cluster_cv_start_position
    msa_avg_score
    msa_std_score
    msa_avg_percentile
    attn_avg_attn
    attn_std_attn
    attn_avg_percentile
    ss_alpha_ratio
    ss_beta_ratio
    ss_flexible_ratio
    ss_ratio

排序逻辑：
    每种 GVP 内部单独排序，取综合得分前4。

综合排序分四大类：
    1. CV + SP 类：
       - cluster_species_percent 越高越好
       - cluster_cv_start_position 越低越好
    2. Attention 类：
       - attn_avg_percentile 越高越好
    3. MSA 类：
       - msa_avg_percentile 越高越好
    4. 二级结构类：
       - ss_ratio 越高越好
       - ss_sequence_ratio 越高越好

说明：
    ss_sequence_ratio 用于排序，但不输出到最终表格。
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

OUT_CSV = DATABASE_DIR / "Table_4_11_top4_candidate_fragment_validation_summary.csv"
OUT_TSV = DATABASE_DIR / "Table_4_11_top4_candidate_fragment_validation_summary.tsv"


# =========================================================
# 1. GVP 排序顺序
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

    "msa_avg_score",
    "msa_std_score",
    "msa_avg_percentile",

    "attn_avg_attn",
    "attn_std_attn",
    "attn_avg_percentile",

    "ss_alpha_ratio",
    "ss_beta_ratio",
    "ss_flexible_ratio",
    "ss_ratio",
]


# =========================================================
# 3. 工具函数
# =========================================================

def normalize_text(x: Any) -> str:
    if x is None:
        return ""

    s = str(x).strip()

    if s.lower() in {"", "nan", "none", "null", "<na>"}:
        return ""

    return s


def first_existing_col(df: pd.DataFrame, cols: List[str]) -> Optional[str]:
    for c in cols:
        if c in df.columns:
            return c
    return None


def to_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    out = df.copy()

    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    return out


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


def std_non_null(series: pd.Series):
    x = pd.to_numeric(series, errors="coerce").dropna()

    if len(x) <= 1:
        return pd.NA

    return float(x.std(ddof=1))


def round_value(x, ndigits: int):
    try:
        if pd.isna(x):
            return pd.NA

        return round(float(x), ndigits)

    except Exception:
        return pd.NA


def percentile_score(series: pd.Series, higher_is_better: bool = True) -> pd.Series:
    """
    把某一列转成 0-1 分位得分。

    higher_is_better=True:
        数值越大得分越高。

    higher_is_better=False:
        数值越小得分越高，比如 CV。
    """

    s = pd.to_numeric(series, errors="coerce")

    if s.notna().sum() == 0:
        return pd.Series(pd.NA, index=series.index)

    if s.notna().sum() == 1:
        out = pd.Series(pd.NA, index=series.index)
        out.loc[s.notna()] = 1.0
        return out

    rank = s.rank(
        method="average",
        ascending=not higher_is_better,
        pct=True
    )

    return rank


# =========================================================
# 4. 读取数据
# =========================================================

def read_input_data() -> pd.DataFrame:
    """
    优先读取 SQLite 数据库。
    如果 SQLite 不存在或没有 sequence_records 表，再读取 clean_wide CSV。
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
    把不同版本数据库里的字段统一成后续聚合需要的字段。
    """

    out = df.copy()

    # 基础字段
    for c in ["gvp_type", "start_pos", "end_pos"]:
        if c not in out.columns:
            out[c] = pd.NA

    # SP
    if "cluster_species_percent" not in out.columns:
        c = first_existing_col(out, [
            "candidate_species_percent",
            "species_percent",
        ])
        out["cluster_species_percent"] = out[c] if c else pd.NA

    # CV
    if "cluster_cv_start_position" not in out.columns:
        c = first_existing_col(out, [
            "candidate_cv",
            "cv_start_position",
            "cv",
        ])
        out["cluster_cv_start_position"] = out[c] if c else pd.NA

    # MSA 平均分
    if "msa_avg_score" not in out.columns:
        c = first_existing_col(out, [
            "msa_summary_avg_conservation_score",
            "msa_fragment_avg_score",
        ])
        out["msa_avg_score"] = out[c] if c else pd.NA

    # MSA 标准差
    if "msa_std_score" not in out.columns:
        c = first_existing_col(out, [
            "msa_summary_conservation_std",
            "msa_conservation_std",
            "msa_std",
            "msa_std_score",
        ])
        out["msa_std_score"] = out[c] if c else pd.NA

    # MSA 百分位
    if "msa_avg_percentile" not in out.columns:
        c = first_existing_col(out, [
            "msa_summary_avg_percentile",
            "msa_percentile_rank",
        ])
        out["msa_avg_percentile"] = out[c] if c else pd.NA

    # Attention 平均分
    if "attn_avg_attn" not in out.columns:
        c = first_existing_col(out, [
            "attn_summary_avg_attn",
            "attn_attention_score",
        ])
        out["attn_avg_attn"] = out[c] if c else pd.NA

    # Attention 标准差
    if "attn_std_attn" not in out.columns:
        c = first_existing_col(out, [
            "attn_summary_std_attn",
            "attn_std",
            "attn_std_attn",
        ])
        out["attn_std_attn"] = out[c] if c else pd.NA

    # Attention 百分位
    if "attn_avg_percentile" not in out.columns:
        c = first_existing_col(out, [
            "attn_summary_avg_percentile",
            "attn_percentile",
        ])
        out["attn_avg_percentile"] = out[c] if c else pd.NA

    # Alpha 比例
    if "ss_alpha_ratio" not in out.columns:
        c = first_existing_col(out, [
            "alpha_helix_ratio",
            "pymol_alpha_helix_ratio",
            "pymol_detail_helix_ratio",
        ])
        out["ss_alpha_ratio"] = out[c] if c else pd.NA

    # Beta 比例
    if "ss_beta_ratio" not in out.columns:
        c = first_existing_col(out, [
            "beta_sheet_ratio",
            "pymol_beta_sheet_ratio",
            "pymol_detail_sheet_ratio",
        ])
        out["ss_beta_ratio"] = out[c] if c else pd.NA

    # 柔性区域比例
    if "ss_flexible_ratio" not in out.columns:
        c = first_existing_col(out, [
            "flexible_region_ratio",
            "pymol_coil_ratio",
            "pymol_detail_coil_ratio",
        ])
        out["ss_flexible_ratio"] = out[c] if c else pd.NA

    # ss_ratio = alpha + beta
    if "ss_ratio" not in out.columns:
        out["ss_ratio"] = pd.NA

    # 含二级结构序列占比，用于排序，不输出
    if "ss_sequence_ratio" not in out.columns:
        c = first_existing_col(out, [
            "pymol_secondary_structure_fragment_ratio",
            "secondary_structure_fragment_ratio",
        ])
        out["ss_sequence_ratio"] = out[c] if c else pd.NA

    if "ss_has_01" not in out.columns:
        c = first_existing_col(out, [
            "pymol_has_secondary_structure",
            "has_secondary_structure",
        ])
        out["ss_has_01"] = out[c] if c else pd.NA

    numeric_cols = [
        "start_pos",
        "end_pos",

        "cluster_species_percent",
        "cluster_cv_start_position",

        "msa_avg_score",
        "msa_std_score",
        "msa_avg_percentile",

        "attn_avg_attn",
        "attn_std_attn",
        "attn_avg_percentile",

        "ss_has_01",
        "ss_sequence_ratio",
        "ss_alpha_ratio",
        "ss_beta_ratio",
        "ss_flexible_ratio",
        "ss_ratio",
    ]

    out = to_numeric(out, numeric_cols)

    # ss_ratio 缺失时，用 alpha + beta 补
    mask_ss_missing = out["ss_ratio"].isna()
    out.loc[mask_ss_missing, "ss_ratio"] = (
        out.loc[mask_ss_missing, "ss_alpha_ratio"].fillna(0)
        + out.loc[mask_ss_missing, "ss_beta_ratio"].fillna(0)
    )

    # ss_sequence_ratio 缺失时，用同片段 ss_has_01 后续聚合平均补
    return out


# =========================================================
# 6. 按候选片段聚合
# =========================================================

def build_fragment_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    聚合到候选片段级别：
        gvp_type + start_pos + end_pos

    MSA / Attention：
        优先取片段汇总字段。
        如果没有 std 字段，则从同片段序列级分数补算 std。

    二级结构：
        ss_alpha_ratio / ss_beta_ratio / ss_flexible_ratio / ss_ratio 取同片段平均。
        ss_sequence_ratio 优先取现成字段；如果没有，则用 ss_has_01 平均值补。
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
        row = {
            "gvp_type": gvp_type,
            "start_pos": int(start_pos),
            "end_pos": int(end_pos),
            "region": f"{int(start_pos)}-{int(end_pos)}",

            "cluster_species_percent": first_non_null(g["cluster_species_percent"]),
            "cluster_cv_start_position": first_non_null(g["cluster_cv_start_position"]),

            "msa_avg_score": first_non_null(g["msa_avg_score"]),
            "msa_std_score": first_non_null(g["msa_std_score"]),
            "msa_avg_percentile": first_non_null(g["msa_avg_percentile"]),

            "attn_avg_attn": first_non_null(g["attn_avg_attn"]),
            "attn_std_attn": first_non_null(g["attn_std_attn"]),
            "attn_avg_percentile": first_non_null(g["attn_avg_percentile"]),

            "ss_alpha_ratio": mean_non_null(g["ss_alpha_ratio"]),
            "ss_beta_ratio": mean_non_null(g["ss_beta_ratio"]),
            "ss_flexible_ratio": mean_non_null(g["ss_flexible_ratio"]),
            "ss_ratio": mean_non_null(g["ss_ratio"]),

            # 用于排序，不输出
            "ss_sequence_ratio": first_non_null(g["ss_sequence_ratio"]),
        }

        # 如果 MSA std 没有，就用同片段序列分数补算
        if pd.isna(row["msa_std_score"]):
            row["msa_std_score"] = std_non_null(g["msa_avg_score"])

        # 如果 Attention std 没有，就用同片段序列分数补算
        if pd.isna(row["attn_std_attn"]):
            row["attn_std_attn"] = std_non_null(g["attn_avg_attn"])

        # ss_ratio 缺失时，用 alpha + beta 补
        if pd.isna(row["ss_ratio"]):
            alpha = 0.0 if pd.isna(row["ss_alpha_ratio"]) else float(row["ss_alpha_ratio"])
            beta = 0.0 if pd.isna(row["ss_beta_ratio"]) else float(row["ss_beta_ratio"])
            row["ss_ratio"] = alpha + beta

        # ss_sequence_ratio 缺失时，用 ss_has_01 平均值补
        if pd.isna(row["ss_sequence_ratio"]):
            row["ss_sequence_ratio"] = mean_non_null(g["ss_has_01"])

        rows.append(row)

    out = pd.DataFrame(rows)

    if out.empty:
        return out

    return out


# =========================================================
# 7. 综合排序：每种 GVP 取前4
# =========================================================

def add_ranking_scores(fragment_df: pd.DataFrame) -> pd.DataFrame:
    """
    每个 GVP 内部计算综合分。

    四大类：
        1. CV + SP
        2. Attention percentile
        3. MSA percentile
        4. SS ratio + SS sequence ratio
    """

    out_parts = []

    for gvp_type, g in fragment_df.groupby("gvp_type", dropna=False):
        g = g.copy()

        sp_score = percentile_score(
            g["cluster_species_percent"],
            higher_is_better=True
        )

        cv_score = percentile_score(
            g["cluster_cv_start_position"],
            higher_is_better=False
        )

        g["score_cv_sp"] = pd.concat(
            [sp_score, cv_score],
            axis=1
        ).mean(axis=1, skipna=True)

        g["score_attn"] = percentile_score(
            g["attn_avg_percentile"],
            higher_is_better=True
        )

        g["score_msa"] = percentile_score(
            g["msa_avg_percentile"],
            higher_is_better=True
        )

        ss_ratio_score = percentile_score(
            g["ss_ratio"],
            higher_is_better=True
        )

        ss_sequence_score = percentile_score(
            g["ss_sequence_ratio"],
            higher_is_better=True
        )

        g["score_ss"] = pd.concat(
            [ss_ratio_score, ss_sequence_score],
            axis=1
        ).mean(axis=1, skipna=True)

        g["score_total"] = g[
            [
                "score_cv_sp",
                "score_attn",
                "score_msa",
                "score_ss",
            ]
        ].mean(axis=1, skipna=True)

        out_parts.append(g)

    out = pd.concat(out_parts, ignore_index=True)

    return out


def select_top4_per_gvp(fragment_df: pd.DataFrame) -> pd.DataFrame:
    scored = add_ranking_scores(fragment_df)

    scored["__gvp_order"] = scored["gvp_type"].map(GVP_SORT_ORDER).fillna(9999)

    scored = scored.sort_values(
        by=[
            "__gvp_order",
            "score_total",
            "score_cv_sp",
            "score_attn",
            "score_msa",
            "score_ss",
            "start_pos",
            "end_pos",
        ],
        ascending=[
            True,
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

    top4 = (
        scored
        .groupby("gvp_type", group_keys=False, dropna=False)
        .head(4)
        .copy()
    )

    top4 = top4.sort_values(
        by=[
            "__gvp_order",
            "score_total",
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
    )

    return top4


# =========================================================
# 8. 输出整理
# =========================================================

def format_output(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    round_4_cols = [
        "cluster_species_percent",
        "cluster_cv_start_position",
        "msa_avg_score",
        "msa_std_score",
        "attn_avg_attn",
        "attn_std_attn",
        "ss_alpha_ratio",
        "ss_beta_ratio",
        "ss_flexible_ratio",
        "ss_ratio",
    ]

    round_2_cols = [
        "msa_avg_percentile",
        "attn_avg_percentile",
    ]

    for c in round_4_cols:
        if c in out.columns:
            out[c] = out[c].apply(lambda x: round_value(x, 4))

    for c in round_2_cols:
        if c in out.columns:
            out[c] = out[c].apply(lambda x: round_value(x, 2))

    out = out[OUTPUT_FIELDS]

    return out


def print_terminal_table(df: pd.DataFrame):
    pd.set_option("display.max_rows", 1000)
    pd.set_option("display.max_columns", 100)
    pd.set_option("display.width", 260)
    pd.set_option("display.max_colwidth", 80)

    print("\n" + "=" * 160)
    print("表4-11 每种 GVP 综合排序前4候选片段多维度验证结果汇总")
    print("=" * 160)
    print(df.to_string(index=False))
    print("=" * 160)
    print(f"输出候选片段数: {len(df)}")
    print("=" * 160 + "\n")


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

    if fragment_df.empty:
        print("[WARN] 没有可用候选片段记录。")
        return

    top4_df = select_top4_per_gvp(fragment_df)

    final_df = format_output(top4_df)

    print_terminal_table(final_df)
    save_outputs(final_df)


if __name__ == "__main__":
    main()
