#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import copy
import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import esm
import umap
import hdbscan
import requests

from tqdm import tqdm
from scipy import stats
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score
)

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


# =========================================================
# 全局路径配置
# =========================================================
DATA_ROOT = r"C:\Users\r9000\Desktop\毕设（无监督聚类）\新gvp"

OUTPUT_ROOT = r"C:\Users\r9000\Desktop\毕设（无监督聚类）\GVP_Parameter_Sensitivity_Analysis"

os.makedirs(OUTPUT_ROOT, exist_ok=True)


# =========================================================
# 需要分析的GVP类型
# =========================================================
# 只跑GvpA时使用这一行
TARGET_GVPS = ["gvpa"]

# 跑全部8类时使用这一行
# TARGET_GVPS = ["gvpa", "gvpc", "gvpn", "gvpo", "gvpg", "gvpj", "gvpk", "gvpp"]


# =========================================================
# 原始基准参数配置
# =========================================================
GVP_FULL_CONFIG = {
    "gvpa": {
        "window": 30,
        "step": 5,
        "model": "esm2_t12_35M_UR50D",
        "min_cluster": 40,
        "min_cluster_plot": 200,
        "umap_params": {
            "n_neighbors": 20,
            "min_dist": 0.15,
            "metric": "euclidean",
            "random_state": 42
        },
        "hdbscan_params": {
            "min_cluster_size": 40,
            "min_samples": 12,
            "metric": "euclidean"
        }
    },
    "gvpc": {
        "window": 35,
        "step": 8,
        "model": "esm2_t12_35M_UR50D",
        "min_cluster": 20,
        "min_cluster_plot": 50,
        "umap_params": {
            "n_neighbors": 30,
            "min_dist": 0.10,
            "metric": "euclidean",
            "random_state": 42
        },
        "hdbscan_params": {
            "min_cluster_size": 50,
            "min_samples": 20,
            "metric": "euclidean"
        }
    },
    "gvpn": {
        "window": 40,
        "step": 8,
        "model": "esm2_t12_35M_UR50D",
        "min_cluster": 30,
        "min_cluster_plot": 200,
        "umap_params": {
            "n_neighbors": 23,
            "min_dist": 0.08,
            "metric": "euclidean",
            "random_state": 42
        },
        "hdbscan_params": {
            "min_cluster_size": 70,
            "min_samples": 25,
            "metric": "euclidean",
            "cluster_selection_epsilon": 0.15
        }
    },
    "gvpo": {
        "window": 50,
        "step": 5,
        "model": "esm2_t12_35M_UR50D",
        "min_cluster": 80,
        "min_cluster_plot": 200,
        "umap_params": {
            "n_neighbors": 15,
            "min_dist": 0.20,
            "metric": "cosine",
            "random_state": 42
        },
        "hdbscan_params": {
            "min_cluster_size": 20,
            "min_samples": 1,
            "metric": "euclidean",
            "cluster_selection_epsilon": 0.08
        }
    },
    "gvpg": {
        "window": 25,
        "step": 5,
        "model": "esm2_t12_35M_UR50D",
        "min_cluster": 30,
        "min_cluster_plot": 150,
        "umap_params": {
            "n_neighbors": 18,
            "min_dist": 0.10,
            "metric": "euclidean",
            "random_state": 42
        },
        "hdbscan_params": {
            "min_cluster_size": 35,
            "min_samples": 10,
            "metric": "euclidean"
        }
    },
    "gvpj": {
        "window": 35,
        "step": 10,
        "model": "esm2_t12_35M_UR50D",
        "min_cluster": 35,
        "min_cluster_plot": 150,
        "umap_params": {
            "n_neighbors": 22,
            "min_dist": 0.15,
            "metric": "euclidean",
            "random_state": 42
        },
        "hdbscan_params": {
            "min_cluster_size": 40,
            "min_samples": 12,
            "metric": "euclidean",
            "cluster_selection_epsilon": 0.10
        }
    },
    "gvpk": {
        "window": 25,
        "step": 4,
        "model": "esm2_t12_35M_UR50D",
        "min_cluster": 25,
        "min_cluster_plot": 50,
        "umap_params": {
            "n_neighbors": 15,
            "min_dist": 0.10,
            "metric": "euclidean",
            "random_state": 42
        },
        "hdbscan_params": {
            "min_cluster_size": 30,
            "min_samples": 8,
            "metric": "euclidean"
        }
    },
    "gvpp": {
        "window": 40,
        "step": 10,
        "model": "esm2_t12_35M_UR50D",
        "min_cluster": 25,
        "min_cluster_plot": 50,
        "umap_params": {
            "n_neighbors": 15,
            "min_dist": 0.13,
            "metric": "euclidean",
            "random_state": 42
        },
        "hdbscan_params": {
            "min_cluster_size": 20,
            "min_samples": 8,
            "metric": "euclidean",
            "cluster_selection_epsilon": 0.15
        }
    }
}


# =========================================================
# 参数敏感性分析配置
# =========================================================
SENSITIVITY_OUTPUT_DIR = os.path.join(
    OUTPUT_ROOT,
    "Parameter_Sensitivity_Analysis"
)
os.makedirs(SENSITIVITY_OUTPUT_DIR, exist_ok=True)

# 起始位置距离小于等于10aa视为同一候选区域
MATCH_DISTANCE_THRESHOLD = 10

# 5组参数：第1组为原始参数，后4组为小范围扰动
PARAMETER_GROUPS = [
    {
        "group_id": 1,
        "group_name": "Baseline",
        "description": "原始参数组",
        "window_scale": 1.00,
        "step_scale": 1.00,
        "umap_n_neighbors_scale": 1.00,
        "umap_min_dist_scale": 1.00,
        "hdbscan_min_cluster_size_scale": 1.00,
        "hdbscan_min_samples_scale": 1.00
    },
    {
        "group_id": 2,
        "group_name": "Slightly_Smaller",
        "description": "窗口、步长和聚类参数整体小幅减小",
        "window_scale": 0.90,
        "step_scale": 0.90,
        "umap_n_neighbors_scale": 0.90,
        "umap_min_dist_scale": 0.90,
        "hdbscan_min_cluster_size_scale": 0.90,
        "hdbscan_min_samples_scale": 0.90
    },
    {
        "group_id": 3,
        "group_name": "Slightly_Larger",
        "description": "窗口、步长和聚类参数整体小幅增大",
        "window_scale": 1.10,
        "step_scale": 1.10,
        "umap_n_neighbors_scale": 1.10,
        "umap_min_dist_scale": 1.10,
        "hdbscan_min_cluster_size_scale": 1.10,
        "hdbscan_min_samples_scale": 1.10
    },
    {
        "group_id": 4,
        "group_name": "Local_Structure",
        "description": "增强局部结构分辨能力",
        "window_scale": 1.00,
        "step_scale": 1.00,
        "umap_n_neighbors_scale": 0.80,
        "umap_min_dist_scale": 0.80,
        "hdbscan_min_cluster_size_scale": 0.90,
        "hdbscan_min_samples_scale": 0.90
    },
    {
        "group_id": 5,
        "group_name": "Conservative_Clustering",
        "description": "提高聚类保守性",
        "window_scale": 1.00,
        "step_scale": 1.00,
        "umap_n_neighbors_scale": 1.20,
        "umap_min_dist_scale": 1.20,
        "hdbscan_min_cluster_size_scale": 1.20,
        "hdbscan_min_samples_scale": 1.20
    }
]


# =========================================================
# 统计阈值配置
# =========================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CONFIDENCE_LEVEL = 0.95
MIN_SAMPLES_FOR_STATS = 5


# =========================================================
# Nature风格配色
# =========================================================
NATURE_COLORS = [
    "#3B5B92", "#4E8F5A", "#B44745", "#6A5A8C",
    "#B38B4D", "#5A9FB5", "#7A7A7A", "#8C6D62",
    "#B76BA3", "#5E7F70", "#9A7D35", "#4F6F8F",
    "#A55C55", "#6B8E77", "#7C6A9C", "#9C8A5A"
]


def get_nature_colors(n):
    if n <= len(NATURE_COLORS):
        return NATURE_COLORS[:n]
    repeat = n // len(NATURE_COLORS) + 1
    return (NATURE_COLORS * repeat)[:n]


def format_gvp_name(gvp_type):
    return f"Gvp{gvp_type[-1].upper()}"


def apply_nature_axis_style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#4D4D4D")
    ax.spines["bottom"].set_color("#4D4D4D")
    ax.tick_params(
        axis="both",
        labelsize=10,
        colors="#333333",
        direction="out",
        length=4,
        width=0.8
    )


def serialize(obj):
    if isinstance(obj, np.floating):
        value = float(obj)
        if np.isnan(value):
            return None
        return value
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, float) and np.isnan(obj):
        return None
    raise TypeError(f"Type {type(obj)} not serializable")


# =========================================================
# ESM-2嵌入提取
# =========================================================
class ESM2Extractor:
    _instances = {}

    def __new__(cls, model_name="esm2_t12_35M_UR50D"):
        if model_name not in cls._instances:
            instance = super().__new__(cls)
            instance.model_name = model_name
            instance.init_model()
            cls._instances[model_name] = instance
        return cls._instances[model_name]

    def init_model(self):
        self.cache_dir = Path.home() / ".cache" / "torch" / "hub" / "checkpoints"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.model_path = self.cache_dir / f"{self.model_name}.pt"

        if not self.model_path.exists():
            print(f"Downloading {self.model_name}")
            url = f"https://dl.fbaipublicfiles.com/fair-esm/models/{self.model_name}.pt"
            response = requests.get(url, stream=True, timeout=60)
            response.raise_for_status()
            total_size = int(response.headers.get("content-length", 0))

            with open(self.model_path, "wb") as f, tqdm(total=total_size, unit="iB", unit_scale=True) as bar:
                for data in response.iter_content(chunk_size=8192):
                    f.write(data)
                    bar.update(len(data))

        self.model, self.alphabet = getattr(esm.pretrained, self.model_name)()
        self.model.to(DEVICE)
        self.batch_converter = self.alphabet.get_batch_converter()
        self.model.eval()

    def get_embeddings(self, fragments, batch_size=16):
        embeddings = []

        with torch.no_grad():
            for i in range(0, len(fragments), batch_size):
                batch = fragments[i:i + batch_size]

                try:
                    _, _, tokens = self.batch_converter(batch)
                    tokens = tokens.to(DEVICE)

                    res = self.model(tokens, repr_layers=[self.model.num_layers])
                    reprs = res["representations"][self.model.num_layers]

                    for j, (_, seq) in enumerate(batch):
                        emb = reprs[j, 1:len(seq) + 1].mean(dim=0)
                        norm = torch.norm(emb)

                        if norm > 0:
                            emb = emb / norm

                        embeddings.append(emb.cpu().numpy())

                except Exception as e:
                    print(f"Embedding extraction error: {e}")
                    continue

        return np.array(embeddings)


# =========================================================
# 数据读取与滑动窗口
# =========================================================
def load_single_gvp(json_path, gvp_type):
    if not os.path.exists(json_path):
        print(f"File not found: {json_path}")
        return []

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    records = []

    for item in data:
        seq = item.get("sequence", "").strip().upper()
        seq = "".join([c for c in seq if c in "ACDEFGHIKLMNPQRSTVWY"])

        if len(seq) < 10:
            continue

        records.append({
            "seq": seq,
            "acc": item.get("unique_sequence_id"),
            "species": item.get("representative_annotation", {}).get("organism"),
            "gvp_type": gvp_type,
            "full_seq_length": len(seq)
        })

    return records


def slide_window(records, window, step):
    fragments = []
    meta = []

    for rec in records:
        seq = rec["seq"]
        full_len = rec["full_seq_length"]

        if len(seq) < window:
            continue

        for s in range(0, len(seq) - window + 1, step):
            e = s + window

            fragments.append((f"{rec['acc']}_{s}-{e}", seq[s:e]))

            meta.append({
                "acc": rec["acc"],
                "start": s,
                "end": e,
                "center": (s + e) / 2,
                "window_size": window,
                "full_seq_length": full_len,
                "relative_start": s / full_len if full_len > 0 else 0,
                "species": rec["species"],
                "gvp_type": rec["gvp_type"]
            })

    return fragments, meta


# =========================================================
# 聚类评价
# =========================================================
def evaluate_clustering(X, labels):
    mask = labels != -1
    X_valid = X[mask]
    y_valid = labels[mask]

    n_valid = len(X_valid)
    n_clust = len(set(y_valid))
    noise_ratio = 1 - n_valid / len(X) if len(X) > 0 else 0

    ch = np.nan
    db = np.nan
    sil = np.nan

    if n_clust >= 2 and n_valid >= 2:
        try:
            ch = calinski_harabasz_score(X_valid, y_valid)
        except Exception:
            ch = np.nan

        try:
            db = davies_bouldin_score(X_valid, y_valid)
        except Exception:
            db = np.nan

        try:
            sil = silhouette_score(X_valid, y_valid, metric="euclidean")
        except Exception:
            sil = np.nan

    return {
        "CH_score": float(ch) if not np.isnan(ch) else np.nan,
        "DB_score": float(db) if not np.isnan(db) else np.nan,
        "Silhouette": float(sil) if not np.isnan(sil) else np.nan,
        "valid_cluster_num": int(n_clust),
        "noise_ratio": float(noise_ratio),
        "total_samples": int(len(X)),
        "valid_samples": int(n_valid)
    }


def get_clustering_effect(eval_result):
    sil = eval_result["Silhouette"]
    db = eval_result["DB_score"]

    if np.isnan(sil):
        return "有效簇不足"

    if sil > 0.4 and not np.isnan(db) and db < 1.0:
        return "较好"
    elif sil > 0.2:
        return "一般"
    else:
        return "较弱"


# =========================================================
# 候选簇位置稳定性分析
# =========================================================
def analyze_cluster_position_features(meta, labels):
    valid_mask = labels != -1
    valid_meta = [meta[i] for i in range(len(meta)) if valid_mask[i]]
    valid_labels = labels[valid_mask]

    cluster_pos_stats = {}

    CV_HIGH_THRESH = 0.3
    CV_MID_THRESH = 0.6
    CONSERVATION_PERCENT = 0.2

    all_species = set(
        m["species"] for m in meta
        if m["species"] is not None and str(m["species"]).strip() != ""
    )

    total_species = len(all_species)
    species_threshold = max(1, int(total_species * CONSERVATION_PERCENT))

    for cluster_id in sorted(set(valid_labels)):
        cluster_id_py = int(cluster_id)
        cluster_mask = valid_labels == cluster_id
        cluster_meta = [
            valid_meta[i]
            for i in range(len(valid_meta))
            if cluster_mask[i]
        ]

        if len(cluster_meta) < MIN_SAMPLES_FOR_STATS:
            continue

        starts = np.array([m["start"] for m in cluster_meta], dtype=float)
        relative_starts = np.array([m["relative_start"] for m in cluster_meta], dtype=float)

        mean_start = np.mean(starts)
        std_start = np.std(starts, ddof=1)

        cv_start = std_start / mean_start if mean_start > 0 else np.inf

        if np.mean(relative_starts) > 0:
            relative_cv = np.std(relative_starts, ddof=1) / np.mean(relative_starts)
        else:
            relative_cv = np.inf

        se_start = stats.sem(starts)

        try:
            ci_low, ci_high = stats.t.interval(
                CONFIDENCE_LEVEL,
                len(starts) - 1,
                loc=mean_start,
                scale=se_start
            )
        except Exception:
            continue

        if np.isnan(ci_low) or np.isnan(ci_high):
            continue

        species_in_cluster = set(
            m["species"] for m in cluster_meta
            if m["species"] is not None and str(m["species"]).strip() != ""
        )

        species_count = len(species_in_cluster)
        is_species_conserved = species_count >= species_threshold

        if cv_start < CV_HIGH_THRESH and is_species_conserved:
            conservation_level = "高度保守"
            candidate_result = "通过"
        elif cv_start < CV_MID_THRESH and is_species_conserved:
            conservation_level = "中度保守"
            candidate_result = "通过"
        elif cv_start < CV_MID_THRESH:
            conservation_level = "位置保守但物种覆盖不足"
            candidate_result = "保留观察"
        else:
            conservation_level = "高度变异"
            candidate_result = "未通过"

        cluster_pos_stats[cluster_id_py] = {
            "cluster_id": cluster_id_py,
            "sample_count": int(len(cluster_meta)),
            "species_count": int(species_count),
            "total_species_in_dataset": int(total_species),
            "species_percent": round(species_count / total_species * 100, 2) if total_species > 0 else 0,
            "mean_start_position": float(mean_start),
            "std_start_position": float(std_start),
            "cv_start_position": float(cv_start),
            "relative_cv": float(relative_cv),
            "95%_CI_low": float(ci_low),
            "95%_CI_high": float(ci_high),
            "conservation_level": conservation_level,
            "candidate_result": candidate_result
        }

    return cluster_pos_stats


# =========================================================
# 从候选簇统计中提取候选区域
# =========================================================
def candidate_region_from_cluster_stats(cluster_pos_stats, window):
    candidates = []

    for cid, row in cluster_pos_stats.items():
        if row.get("candidate_result") != "通过":
            continue

        mean_start = row.get("mean_start_position", None)

        if mean_start is None:
            continue

        if isinstance(mean_start, float) and np.isnan(mean_start):
            continue

        start = int(math.floor(float(mean_start)))
        end = start + int(window)

        candidates.append({
            "cluster_id": int(row.get("cluster_id", cid)),
            "start": start,
            "end": end,
            "region": f"{start}-{end}",
            "mean_start_position": float(mean_start),
            "sample_count": int(row.get("sample_count", 0)),
            "species_count": int(row.get("species_count", 0)),
            "species_percent": float(row.get("species_percent", 0)),
            "cv_start_position": float(row.get("cv_start_position", np.nan)),
            "conservation_level": row.get("conservation_level", "")
        })

    candidates = sorted(candidates, key=lambda x: x["start"])
    return candidates


# =========================================================
# 参数扰动生成
# =========================================================
def safe_round_int(value, min_value=1):
    value = int(round(value))
    return max(value, min_value)


def make_sensitivity_config(base_config, group):
    cfg = copy.deepcopy(base_config)

    cfg["window"] = safe_round_int(
        base_config["window"] * group["window_scale"],
        min_value=10
    )

    cfg["step"] = safe_round_int(
        base_config["step"] * group["step_scale"],
        min_value=1
    )

    if cfg["step"] >= cfg["window"]:
        cfg["step"] = max(1, cfg["window"] // 3)

    if "n_neighbors" in cfg["umap_params"]:
        cfg["umap_params"]["n_neighbors"] = safe_round_int(
            base_config["umap_params"]["n_neighbors"] * group["umap_n_neighbors_scale"],
            min_value=5
        )

    if "min_dist" in cfg["umap_params"]:
        cfg["umap_params"]["min_dist"] = float(
            base_config["umap_params"]["min_dist"] * group["umap_min_dist_scale"]
        )
        cfg["umap_params"]["min_dist"] = min(
            max(cfg["umap_params"]["min_dist"], 0.01),
            0.80
        )

    if "min_cluster_size" in cfg["hdbscan_params"]:
        cfg["hdbscan_params"]["min_cluster_size"] = safe_round_int(
            base_config["hdbscan_params"]["min_cluster_size"] * group["hdbscan_min_cluster_size_scale"],
            min_value=5
        )

    if "min_samples" in cfg["hdbscan_params"]:
        cfg["hdbscan_params"]["min_samples"] = safe_round_int(
            base_config["hdbscan_params"]["min_samples"] * group["hdbscan_min_samples_scale"],
            min_value=1
        )

    return cfg


def adjust_config_for_sample_size(config, sample_count):
    cfg = copy.deepcopy(config)

    if sample_count <= 2:
        return cfg

    if "n_neighbors" in cfg["umap_params"]:
        cfg["umap_params"]["n_neighbors"] = min(
            cfg["umap_params"]["n_neighbors"],
            max(2, sample_count - 1)
        )

    if "min_cluster_size" in cfg["hdbscan_params"]:
        cfg["hdbscan_params"]["min_cluster_size"] = min(
            cfg["hdbscan_params"]["min_cluster_size"],
            max(2, sample_count)
        )

    if "min_samples" in cfg["hdbscan_params"]:
        cfg["hdbscan_params"]["min_samples"] = min(
            cfg["hdbscan_params"]["min_samples"],
            max(1, cfg["hdbscan_params"]["min_cluster_size"])
        )

    return cfg


# =========================================================
# 单组参数运行
# =========================================================
def run_single_gvp_parameter_group(gvp_type, config, group):
    gvp_suffix = gvp_type[-1].upper()

    json_path = os.path.join(
        DATA_ROOT,
        gvp_type,
        f"Gvp{gvp_suffix}_sequences.json"
    )

    records = load_single_gvp(json_path, gvp_type)

    if not records:
        print(f"No data for {format_gvp_name(gvp_type)}")
        return None

    fragments, meta = slide_window(records, config["window"], config["step"])

    if not fragments:
        print(f"No fragments for {format_gvp_name(gvp_type)} group {group['group_id']}")
        return None

    adjusted_config = adjust_config_for_sample_size(config, len(fragments))

    print(
        f"{format_gvp_name(gvp_type)} group {group['group_id']} | "
        f"window={adjusted_config['window']}, "
        f"step={adjusted_config['step']}, "
        f"fragments={len(fragments)}, "
        f"UMAP_n_neighbors={adjusted_config['umap_params'].get('n_neighbors')}, "
        f"HDBSCAN_min_cluster_size={adjusted_config['hdbscan_params'].get('min_cluster_size')}, "
        f"HDBSCAN_min_samples={adjusted_config['hdbscan_params'].get('min_samples')}"
    )

    extractor = ESM2Extractor(adjusted_config["model"])
    embeddings = extractor.get_embeddings(fragments)

    if len(embeddings) == 0:
        print(f"No embeddings for {format_gvp_name(gvp_type)} group {group['group_id']}")
        return None

    reducer = umap.UMAP(**adjusted_config["umap_params"])
    umap_embeds = reducer.fit_transform(embeddings)

    clusterer = hdbscan.HDBSCAN(**adjusted_config["hdbscan_params"])
    labels = clusterer.fit_predict(umap_embeds)

    for i, m in enumerate(meta):
        m["cluster"] = int(labels[i])

    eval_result = evaluate_clustering(umap_embeds, labels)
    cluster_pos_stats = analyze_cluster_position_features(meta, labels)
    candidates = candidate_region_from_cluster_stats(
        cluster_pos_stats,
        adjusted_config["window"]
    )

    group_out_dir = os.path.join(
        SENSITIVITY_OUTPUT_DIR,
        gvp_type,
        f"group_{group['group_id']}_{group['group_name']}"
    )
    os.makedirs(group_out_dir, exist_ok=True)

    result_json = {
        "gvp_type": gvp_type,
        "GVP_type": format_gvp_name(gvp_type),
        "group_id": group["group_id"],
        "group_name": group["group_name"],
        "description": group["description"],
        "config": adjusted_config,
        "clustering_metrics": eval_result,
        "candidate_count": len(candidates),
        "candidates": candidates,
        "cluster_position_stats": cluster_pos_stats
    }

    with open(
        os.path.join(group_out_dir, f"{gvp_type}_group_{group['group_id']}_sensitivity_result.json"),
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(result_json, f, indent=2, ensure_ascii=False, default=serialize)

    pd.DataFrame(candidates).to_csv(
        os.path.join(group_out_dir, f"{gvp_type}_group_{group['group_id']}_passed_candidates.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    pd.DataFrame.from_dict(cluster_pos_stats, orient="index").to_csv(
        os.path.join(group_out_dir, f"{gvp_type}_group_{group['group_id']}_cluster_position_stats.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    pd.DataFrame({
        "UMAP1": umap_embeds[:, 0],
        "UMAP2": umap_embeds[:, 1],
        "cluster": labels
    }).to_csv(
        os.path.join(group_out_dir, f"{gvp_type}_group_{group['group_id']}_umap_coordinates.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    print(
        f"{format_gvp_name(gvp_type)} group {group['group_id']} completed | "
        f"candidate_count={len(candidates)}, "
        f"valid_cluster_num={eval_result['valid_cluster_num']}, "
        f"noise_ratio={eval_result['noise_ratio']:.4f}"
    )

    return {
        "gvp_type": gvp_type,
        "group_id": group["group_id"],
        "group_name": group["group_name"],
        "description": group["description"],
        "config": adjusted_config,
        "eval_result": eval_result,
        "candidate_count": len(candidates),
        "candidates": candidates
    }


# =========================================================
# 候选区域匹配
# =========================================================
def is_same_candidate_region(base, other, threshold=MATCH_DISTANCE_THRESHOLD):
    distance = abs(int(base["start"]) - int(other["start"]))

    if distance <= threshold:
        return True, distance

    return False, distance


def match_candidates_to_baseline(base_candidates, other_candidates):
    if len(base_candidates) == 0:
        return {
            "matched_count": 0,
            "matched_ratio": np.nan,
            "matched_pairs": []
        }

    possible_pairs = []

    for i, base in enumerate(base_candidates):
        for j, other in enumerate(other_candidates):
            matched, distance = is_same_candidate_region(base, other)

            if matched:
                possible_pairs.append({
                    "base_index": i,
                    "other_index": j,
                    "distance": distance,
                    "base": base,
                    "other": other
                })

    possible_pairs = sorted(possible_pairs, key=lambda x: x["distance"])

    used_base = set()
    used_other = set()
    matched_pairs = []

    for pair in possible_pairs:
        bi = pair["base_index"]
        oi = pair["other_index"]

        if bi in used_base or oi in used_other:
            continue

        used_base.add(bi)
        used_other.add(oi)

        matched_pairs.append({
            "base_cluster_id": pair["base"]["cluster_id"],
            "base_region": pair["base"]["region"],
            "base_start": pair["base"]["start"],
            "base_end": pair["base"]["end"],
            "other_cluster_id": pair["other"]["cluster_id"],
            "other_region": pair["other"]["region"],
            "other_start": pair["other"]["start"],
            "other_end": pair["other"]["end"],
            "start_distance": pair["distance"]
        })

    matched_count = len(matched_pairs)
    matched_ratio = matched_count / len(base_candidates) * 100

    return {
        "matched_count": matched_count,
        "matched_ratio": matched_ratio,
        "matched_pairs": matched_pairs
    }


# =========================================================
# 保存5组参数表
# =========================================================
def save_parameter_group_table():
    rows = []

    for gvp_type in TARGET_GVPS:
        if gvp_type not in GVP_FULL_CONFIG:
            continue

        base_config = GVP_FULL_CONFIG[gvp_type]

        for group in PARAMETER_GROUPS:
            cfg = make_sensitivity_config(base_config, group)

            rows.append({
                "GVP类型": format_gvp_name(gvp_type),
                "原始GVP类型": gvp_type,
                "参数组号": group["group_id"],
                "参数组名称": group["group_name"],
                "参数组说明": group["description"],
                "滑动窗口长度": cfg["window"],
                "滑动步长": cfg["step"],
                "UMAP邻居数": cfg["umap_params"].get("n_neighbors", ""),
                "UMAP最小距离": cfg["umap_params"].get("min_dist", ""),
                "UMAP距离度量": cfg["umap_params"].get("metric", ""),
                "HDBSCAN最小簇大小": cfg["hdbscan_params"].get("min_cluster_size", ""),
                "HDBSCAN最小样本数": cfg["hdbscan_params"].get("min_samples", ""),
                "HDBSCAN距离度量": cfg["hdbscan_params"].get("metric", ""),
                "HDBSCAN_epsilon": cfg["hdbscan_params"].get("cluster_selection_epsilon", "")
            })

    df = pd.DataFrame(rows)

    path = os.path.join(
        SENSITIVITY_OUTPUT_DIR,
        "Parameter_Group_Settings.csv"
    )

    df.to_csv(path, index=False, encoding="utf-8-sig")

    print(f"Saved parameter group table: {path}")

    return df


# =========================================================
# 折线图
# =========================================================
def plot_parameter_sensitivity_line(summary_df):
    if summary_df.empty:
        print("Summary table is empty, skip plotting.")
        return

    plt.figure(figsize=(10, 6))
    ax = plt.gca()

    gvp_list = summary_df["GVP类型"].dropna().unique().tolist()
    colors = get_nature_colors(len(gvp_list))

    for i, gvp_name in enumerate(gvp_list):
        sub = summary_df[summary_df["GVP类型"] == gvp_name].copy()
        sub = sub.sort_values("参数组号")

        ax.plot(
            sub["参数组号"],
            sub["相对原参数匹配比例(%)"],
            marker="o",
            linewidth=2.0,
            markersize=6,
            color=colors[i],
            label=gvp_name
        )

        for _, row in sub.iterrows():
            y = row["相对原参数匹配比例(%)"]

            if pd.isna(y):
                continue

            ax.text(
                row["参数组号"],
                y + 1.5,
                f"{y:.1f}",
                ha="center",
                va="bottom",
                fontsize=8,
                color=colors[i]
            )

    ax.set_xlabel("Parameter group", fontsize=12)
    ax.set_ylabel("Matched candidate regions relative to baseline (%)", fontsize=12)
    ax.set_title(
        "Parameter Sensitivity of Candidate Region Identification",
        fontsize=15,
        fontweight="bold"
    )

    ax.set_xticks([g["group_id"] for g in PARAMETER_GROUPS])
    ax.set_ylim(0, 110)

    ax.grid(
        axis="y",
        linestyle="--",
        linewidth=0.6,
        alpha=0.35
    )

    ax.legend(
        frameon=False,
        fontsize=9,
        loc="best"
    )

    apply_nature_axis_style(ax)

    plt.tight_layout()

    png_path = os.path.join(
        SENSITIVITY_OUTPUT_DIR,
        "Parameter_Sensitivity_Candidate_Match_Line.png"
    )

    pdf_path = os.path.join(
        SENSITIVITY_OUTPUT_DIR,
        "Parameter_Sensitivity_Candidate_Match_Line.pdf"
    )

    plt.savefig(png_path, dpi=600, bbox_inches="tight")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close()

    print(f"Saved figure: {png_path}")
    print(f"Saved PDF: {pdf_path}")


# =========================================================
# 敏感性分析主流程
# =========================================================
def run_parameter_sensitivity_analysis():
    save_parameter_group_table()

    summary_rows = []
    detail_rows = []

    for gvp_type in TARGET_GVPS:
        if gvp_type not in GVP_FULL_CONFIG:
            print(f"Skip unknown GVP type: {gvp_type}")
            continue

        print("\n" + "=" * 100)
        print(f"Parameter sensitivity analysis: {format_gvp_name(gvp_type)}")
        print("=" * 100)

        group_results = {}

        for group in PARAMETER_GROUPS:
            print(
                f"\nRunning {format_gvp_name(gvp_type)} "
                f"group {group['group_id']} ({group['group_name']})"
            )

            cfg = make_sensitivity_config(
                GVP_FULL_CONFIG[gvp_type],
                group
            )

            result = run_single_gvp_parameter_group(
                gvp_type=gvp_type,
                config=cfg,
                group=group
            )

            if result is None:
                continue

            group_results[group["group_id"]] = result

        if 1 not in group_results:
            print(f"Baseline group missing for {format_gvp_name(gvp_type)}, skip comparison.")
            continue

        baseline_candidates = group_results[1]["candidates"]
        baseline_count = len(baseline_candidates)

        for group in PARAMETER_GROUPS:
            gid = group["group_id"]

            if gid not in group_results:
                continue

            current_candidates = group_results[gid]["candidates"]

            if gid == 1:
                matched_count = baseline_count
                matched_ratio = 100.0 if baseline_count > 0 else np.nan
                matched_pairs = []

                for cand in baseline_candidates:
                    matched_pairs.append({
                        "GVP类型": format_gvp_name(gvp_type),
                        "原始GVP类型": gvp_type,
                        "参数组号": gid,
                        "参数组名称": group["group_name"],
                        "基准cluster_id": cand["cluster_id"],
                        "基准区域": cand["region"],
                        "基准起始位置": cand["start"],
                        "基准终止位置": cand["end"],
                        "比较cluster_id": cand["cluster_id"],
                        "比较区域": cand["region"],
                        "比较起始位置": cand["start"],
                        "比较终止位置": cand["end"],
                        "起始位置距离": 0
                    })
            else:
                match_result = match_candidates_to_baseline(
                    baseline_candidates,
                    current_candidates
                )
                matched_count = match_result["matched_count"]
                matched_ratio = match_result["matched_ratio"]
                matched_pairs = match_result["matched_pairs"]

                for pair in matched_pairs:
                    pair["GVP类型"] = format_gvp_name(gvp_type)
                    pair["原始GVP类型"] = gvp_type
                    pair["参数组号"] = gid
                    pair["参数组名称"] = group["group_name"]

                    pair["基准cluster_id"] = pair.pop("base_cluster_id")
                    pair["基准区域"] = pair.pop("base_region")
                    pair["基准起始位置"] = pair.pop("base_start")
                    pair["基准终止位置"] = pair.pop("base_end")

                    pair["比较cluster_id"] = pair.pop("other_cluster_id")
                    pair["比较区域"] = pair.pop("other_region")
                    pair["比较起始位置"] = pair.pop("other_start")
                    pair["比较终止位置"] = pair.pop("other_end")

                    pair["起始位置距离"] = pair.pop("start_distance")

            eval_result = group_results[gid]["eval_result"]

            summary_rows.append({
                "GVP类型": format_gvp_name(gvp_type),
                "原始GVP类型": gvp_type,
                "参数组号": gid,
                "参数组名称": group["group_name"],
                "参数组说明": group["description"],

                "原参数候选区域数": baseline_count,
                "当前参数候选区域数": len(current_candidates),
                "匹配原参数候选区域数": matched_count,
                "相对原参数匹配比例(%)": (
                    round(matched_ratio, 2)
                    if not pd.isna(matched_ratio)
                    else np.nan
                ),

                "有效簇数量": eval_result["valid_cluster_num"],
                "噪声比例": round(eval_result["noise_ratio"], 4),
                "轮廓系数": (
                    round(eval_result["Silhouette"], 4)
                    if not np.isnan(eval_result["Silhouette"])
                    else ""
                ),
                "Calinski-Harabasz指数": (
                    round(eval_result["CH_score"], 4)
                    if not np.isnan(eval_result["CH_score"])
                    else ""
                ),
                "Davies-Bouldin指数": (
                    round(eval_result["DB_score"], 4)
                    if not np.isnan(eval_result["DB_score"])
                    else ""
                ),
                "聚类效果说明": get_clustering_effect(eval_result)
            })

            detail_rows.extend(matched_pairs)

    summary_df = pd.DataFrame(summary_rows)
    detail_df = pd.DataFrame(detail_rows)

    summary_path = os.path.join(
        SENSITIVITY_OUTPUT_DIR,
        "Parameter_Sensitivity_Candidate_Match_Summary.csv"
    )

    detail_path = os.path.join(
        SENSITIVITY_OUTPUT_DIR,
        "Parameter_Sensitivity_Candidate_Match_Details.csv"
    )

    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    detail_df.to_csv(detail_path, index=False, encoding="utf-8-sig")

    print(f"\nSaved summary: {summary_path}")
    print(f"Saved detail: {detail_path}")

    plot_parameter_sensitivity_line(summary_df)

    return summary_df, detail_df


# =========================================================
# 主函数
# =========================================================
def main():
    print("Start parameter sensitivity analysis.")
    print(f"Device: {DEVICE}")
    print(f"Target GVPs: {TARGET_GVPS}")
    print(f"Data root: {DATA_ROOT}")
    print(f"Output directory: {SENSITIVITY_OUTPUT_DIR}")
    print(f"Match distance threshold: {MATCH_DISTANCE_THRESHOLD} aa")

    summary_df, detail_df = run_parameter_sensitivity_analysis()

    print("\nParameter sensitivity summary:")
    if not summary_df.empty:
        pd.set_option("display.max_rows", None)
        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", 200)
        print(summary_df.to_string(index=False))
    else:
        print("Summary table is empty.")

    print("\nAll done.")


if __name__ == "__main__":
    main()
