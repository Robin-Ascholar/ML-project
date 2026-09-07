#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import numpy as np
import umap
import hdbscan
import matplotlib.pyplot as plt
import torch
import esm
from pathlib import Path
import requests
from tqdm import tqdm
import pandas as pd
from scipy import stats
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ================= Global Configuration =================
DATA_ROOT = r"C:\Users\r9000\Desktop\毕设（无监督聚类）\新gvp"
OUTPUT_ROOT = r"C:\Users\r9000\Desktop\毕设（无监督聚类）\GVP_Four_Type_Clustering_Resultsplus"
os.makedirs(OUTPUT_ROOT, exist_ok=True)

# TARGET_GVPS = ["gvpa", "gvpc", "gvpn", "gvpo", "gvpg", "gvpj", "gvpk", "gvpp"]

TARGET_GVPS = ["gvpa"]

GVP_FULL_CONFIG = {
    "gvpa": {
        "window": 30, "step": 5,
        "model": "esm2_t12_35M_UR50D",
        "min_cluster": 40,
        "min_cluster_plot": 200,
        "umap_params": {"n_neighbors": 20, "min_dist": 0.15, "metric": "euclidean", "random_state": 42},
        "hdbscan_params": {"min_cluster_size": 40, "min_samples": 12, "metric": "euclidean"}
    },
    "gvpc": {
        "window": 35, "step": 8,
        "model": "esm2_t12_35M_UR50D",
        "min_cluster": 20,
        "min_cluster_plot": 50,
        "umap_params": {"n_neighbors": 30, "min_dist": 0.1, "metric": "euclidean", "random_state": 42},
        "hdbscan_params": {"min_cluster_size": 50, "min_samples": 20, "metric": "euclidean"}
    },
    "gvpn": {
        "window": 40, "step": 8,
        "model": "esm2_t12_35M_UR50D",
        "min_cluster": 30,
        "min_cluster_plot": 200,
        "umap_params": {"n_neighbors": 23, "min_dist": 0.08, "metric": "euclidean", "random_state": 42},
        "hdbscan_params": {"min_cluster_size": 70, "min_samples": 25, "metric": "euclidean",
                           "cluster_selection_epsilon": 0.15}
    },
    "gvpo": {
        "window": 50, "step": 5,
        "model": "esm2_t12_35M_UR50D",
        "min_cluster": 80,
        "min_cluster_plot": 200,
        "umap_params": {"n_neighbors": 15, "min_dist": 0.2, "metric": "cosine", "random_state": 42},
        "hdbscan_params": {"min_cluster_size": 20, "min_samples": 1, "metric": "euclidean",
                           "cluster_selection_epsilon": 0.08}
    },
    "gvpg": {
        "window": 25,
        "step": 5,
        "model": "esm2_t12_35M_UR50D",
        "min_cluster": 30,
        "min_cluster_plot": 150,
        "umap_params": {"n_neighbors": 18, "min_dist": 0.1, "metric": "euclidean", "random_state": 42},
        "hdbscan_params": {"min_cluster_size": 35, "min_samples": 10, "metric": "euclidean"}
    },
    "gvpj": {
        "window": 35,
        "step": 10,
        "model": "esm2_t12_35M_UR50D",
        "min_cluster": 35,
        "min_cluster_plot": 150,
        "umap_params": {"n_neighbors": 22, "min_dist": 0.15, "metric": "euclidean", "random_state": 42},
        "hdbscan_params": {"min_cluster_size": 40, "min_samples": 12, "metric": "euclidean",
                           "cluster_selection_epsilon": 0.1}
    },
    "gvpk": {
        "window": 25,
        "step": 4,
        "model": "esm2_t12_35M_UR50D",
        "min_cluster": 25,
        "min_cluster_plot": 50,
        "umap_params": {"n_neighbors": 15, "min_dist": 0.1, "metric": "euclidean", "random_state": 42},
        "hdbscan_params": {"min_cluster_size": 30, "min_samples": 8, "metric": "euclidean"}
    },
    "gvpp": {
        "window": 40,
        "step": 10,
        "model": "esm2_t12_35M_UR50D",
        "min_cluster": 25,
        "min_cluster_plot": 50,
        "umap_params": {"n_neighbors": 15, "min_dist": 0.13, "metric": "euclidean", "random_state": 42},
        "hdbscan_params": {"min_cluster_size": 20, "min_samples": 8, "metric": "euclidean",
                           "cluster_selection_epsilon": 0.15}
    },
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CONFIDENCE_LEVEL = 0.95
MIN_SAMPLES_FOR_STATS = 5


# ================= Nature风格配色 =================
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
        labelsize=9,
        colors="#333333",
        direction="out",
        length=3,
        width=0.8
    )


class ESM2Extractor:
    _instance = None

    def __new__(cls, model_name="esm2_t12_35M_UR50D"):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.model_name = model_name
            cls._instance.init_model()
        return cls._instance

    def init_model(self):
        self.cache_dir = Path.home() / ".cache" / "torch" / "hub" / "checkpoints"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.model_path = self.cache_dir / f"{self.model_name}.pt"

        if not self.model_path.exists():
            print(f"Downloading {self.model_name}")
            url = f"https://dl.fbaipublicfiles.com/fair-esm/models/{self.model_name}.pt"
            response = requests.get(url, stream=True, timeout=30)
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
                        emb = emb / torch.norm(emb)
                        embeddings.append(emb.cpu().numpy())

                except Exception as e:
                    print(f"Embedding extraction error: {e}")
                    continue

        return np.array(embeddings)


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
    fragments, meta = [], []

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


def evaluate_clustering(X, labels):
    from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score

    mask = labels != -1
    X_valid = X[mask]
    y_valid = labels[mask]

    n_valid = len(X_valid)
    n_clust = len(set(y_valid))
    noise_ratio = 1 - n_valid / len(X) if len(X) > 0 else 0

    ch = db = sil = np.nan

    if n_clust >= 2 and n_valid >= 2:
        ch = calinski_harabasz_score(X_valid, y_valid)
        db = davies_bouldin_score(X_valid, y_valid)
        sil = silhouette_score(X_valid, y_valid, metric="euclidean")

    return {
        "CH_score": float(ch),
        "DB_score": float(db),
        "Silhouette": float(sil),
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

    if sil > 0.4 and db < 1.0:
        return "较好"
    elif sil > 0.2:
        return "一般"
    else:
        return "较弱"


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
        cluster_meta = [valid_meta[i] for i in range(len(valid_meta)) if cluster_mask[i]]

        if len(cluster_meta) < MIN_SAMPLES_FOR_STATS:
            continue

        starts = np.array([m["start"] for m in cluster_meta])
        relative_starts = np.array([m["relative_start"] for m in cluster_meta])

        mean_start = np.mean(starts)
        std_start = np.std(starts, ddof=1)
        cv_start = std_start / mean_start if mean_start > 0 else np.inf
        relative_cv = (
            np.std(relative_starts, ddof=1) / np.mean(relative_starts)
            if np.mean(relative_starts) > 0 else np.inf
        )

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


def serialize(obj):
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, float) and np.isnan(obj):
        return None
    raise TypeError(f"Type {type(obj)} not serializable")


def plot_single_nature_cluster(
    umap_embeds,
    labels,
    cluster_stats,
    valid_lbls,
    major,
    gvp_type,
    out_dir,
    min_cluster_plot
):
    plt.figure(figsize=(10, 8))
    ax = plt.gca()

    noise_mask = labels == -1

    if noise_mask.any():
        ax.scatter(
            umap_embeds[noise_mask, 0],
            umap_embeds[noise_mask, 1],
            c="#D9D9D9",
            s=5,
            alpha=0.25,
            linewidths=0,
            label="Noise"
        )

    colors = get_nature_colors(len(major))

    for i, lbl in enumerate(major):
        mask = labels == lbl
        ax.scatter(
            umap_embeds[mask, 0],
            umap_embeds[mask, 1],
            c=colors[i],
            s=9,
            alpha=0.78,
            linewidths=0,
            label=f"Cluster {lbl} ({cluster_stats[lbl]['fragment_num']})"
        )

    small_clusters = len(valid_lbls) - len(major)
    if small_clusters > 0:
        ax.text(
            0.02,
            0.98,
            f"{small_clusters} smaller clusters not shown",
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment="top",
            bbox=dict(
                boxstyle="round,pad=0.35",
                facecolor="white",
                edgecolor="#BFBFBF",
                alpha=0.85
            )
        )

    ax.set_title(
        f"{format_gvp_name(gvp_type)} Fragment Clustering",
        fontsize=15,
        fontweight="bold"
    )
    ax.set_xlabel("UMAP 1", fontsize=12)
    ax.set_ylabel("UMAP 2", fontsize=12)

    apply_nature_axis_style(ax)

    ax.legend(
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        fontsize=8,
        frameon=False,
        markerscale=1.5
    )

    plt.tight_layout()

    save_path = os.path.join(out_dir, f"{gvp_type}_clustering_nature_style.png")
    plt.savefig(save_path, dpi=600, bbox_inches="tight")
    plt.close()

    print(f"Saved figure: {save_path}")


def plot_combined_other_gvps(all_plot_data, output_root):
    other_gvps = ["gvpg", "gvpj", "gvpk", "gvpn", "gvpo", "gvpp"]

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    for idx, gvp_type in enumerate(other_gvps):
        ax = axes[idx]

        if gvp_type not in all_plot_data:
            ax.axis("off")
            continue

        data = all_plot_data[gvp_type]

        umap_embeds = data["umap_embeds"]
        labels = data["labels"]
        cluster_stats = data["cluster_stats"]
        valid_lbls = data["valid_lbls"]
        major = data["major"]

        noise_mask = labels == -1

        if noise_mask.any():
            ax.scatter(
                umap_embeds[noise_mask, 0],
                umap_embeds[noise_mask, 1],
                c="#D9D9D9",
                s=4,
                alpha=0.22,
                linewidths=0
            )

        colors = get_nature_colors(len(major))

        for i, lbl in enumerate(major):
            mask = labels == lbl
            ax.scatter(
                umap_embeds[mask, 0],
                umap_embeds[mask, 1],
                c=colors[i],
                s=5,
                alpha=0.75,
                linewidths=0
            )

        small_clusters = len(valid_lbls) - len(major)
        if small_clusters > 0:
            ax.text(
                0.02,
                0.96,
                f"{small_clusters} smaller clusters not shown",
                transform=ax.transAxes,
                fontsize=8,
                verticalalignment="top",
                bbox=dict(
                    boxstyle="round,pad=0.25",
                    facecolor="white",
                    edgecolor="#BFBFBF",
                    alpha=0.8
                )
            )

        ax.set_title(format_gvp_name(gvp_type), fontsize=13, fontweight="bold")
        ax.set_xlabel("UMAP 1", fontsize=10)
        ax.set_ylabel("UMAP 2", fontsize=10)

        apply_nature_axis_style(ax)

    plt.suptitle(
        "Fragment Clustering of Other GVP Types",
        fontsize=17,
        fontweight="bold",
        y=0.98
    )

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    save_path = os.path.join(output_root, "Other_GVP_Clustering_Subplots_Nature_Style.png")
    plt.savefig(save_path, dpi=600, bbox_inches="tight")
    plt.close()

    print(f"Saved combined subplot: {save_path}")


def cluster_and_plot(embeddings, meta, gvp_type, out_dir, config):
    if len(embeddings) == 0:
        print(f"No embeddings for {gvp_type}")
        return None

    min_cluster_plot = config.get("min_cluster_plot", 200)

    reducer = umap.UMAP(**config["umap_params"])
    umap_embeds = reducer.fit_transform(embeddings)

    clusterer = hdbscan.HDBSCAN(**config["hdbscan_params"])
    labels = clusterer.fit_predict(umap_embeds)

    for i, m in enumerate(meta):
        m["cluster"] = int(labels[i])

    eval_result = evaluate_clustering(umap_embeds, labels)
    cluster_pos_stats = analyze_cluster_position_features(meta, labels)

    valid_lbls = [int(l) for l in np.unique(labels) if l != -1]
    noise_num = int(np.sum(labels == -1))
    non_noise_num = int(np.sum(labels != -1))

    cluster_stats = {}

    for lbl in valid_lbls:
        mask = labels == lbl
        sub = [meta[i] for i in range(len(meta)) if mask[i]]

        cluster_stats[lbl] = {
            "cluster_id": lbl,
            "fragment_num": int(len(sub)),
            "species_num": int(len(set(x["species"] for x in sub if x["species"]))),
            "acc_num": int(len(set(x["acc"] for x in sub if x["acc"])))
        }

    major = [
        l for l in valid_lbls
        if cluster_stats[l]["fragment_num"] >= min_cluster_plot
    ]

    if len(major) == 0 and len(valid_lbls) > 0:
        print(f"No cluster meets min_cluster_plot={min_cluster_plot}, showing all clusters for {gvp_type}")
        major = valid_lbls

    max_cluster_size = max(
        [cluster_stats[l]["fragment_num"] for l in valid_lbls],
        default=0
    )

    passed_candidate_clusters = [
        c for c in cluster_pos_stats.values()
        if c["candidate_result"] == "通过"
    ]

    observed_candidate_clusters = [
        c for c in cluster_pos_stats.values()
        if c["candidate_result"] == "保留观察"
    ]

    high_conserved_clusters = [
        c for c in cluster_pos_stats.values()
        if c["conservation_level"] == "高度保守"
    ]

    mid_conserved_clusters = [
        c for c in cluster_pos_stats.values()
        if c["conservation_level"] == "中度保守"
    ]

    # ================= 表4-4 行 =================
    table_4_4_row = {
        "GVP类型": format_gvp_name(gvp_type),
        "片段总数": int(len(labels)),
        "有效簇数量": int(len(valid_lbls)),
        "非噪声片段数": non_noise_num,
        "噪声片段数": noise_num,
        "噪声比例": round(noise_num / len(labels), 4) if len(labels) > 0 else 0,
        "最大簇片段数": int(max_cluster_size),
        "通过候选簇数": int(len(passed_candidate_clusters)),
        "保留观察簇数": int(len(observed_candidate_clusters)),
        "高度保守簇数": int(len(high_conserved_clusters)),
        "中度保守簇数": int(len(mid_conserved_clusters))
    }

    # ================= 表4-5 行 =================
    table_4_5_row = {
        "GVP类型": format_gvp_name(gvp_type),
        "有效簇数量": eval_result["valid_cluster_num"],
        "轮廓系数": round(eval_result["Silhouette"], 4) if not np.isnan(eval_result["Silhouette"]) else "",
        "Calinski-Harabasz指数": round(eval_result["CH_score"], 4) if not np.isnan(eval_result["CH_score"]) else "",
        "Davies-Bouldin指数": round(eval_result["DB_score"], 4) if not np.isnan(eval_result["DB_score"]) else "",
        "噪声比例": round(eval_result["noise_ratio"], 4),
        "聚类效果说明": get_clustering_effect(eval_result)
    }

    # 保存单类结果
    output_data = {
        "clustering_metrics": eval_result,
        "cluster_position_stats": cluster_pos_stats
    }

    with open(os.path.join(out_dir, f"{gvp_type}_complete_analysis.json"), "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False, default=serialize)

    pd.DataFrame.from_dict(cluster_stats, orient="index").to_csv(
        os.path.join(out_dir, f"{gvp_type}_cluster_report.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    pd.DataFrame.from_dict(cluster_pos_stats, orient="index").to_csv(
        os.path.join(out_dir, f"{gvp_type}_candidate_cluster_position_stats.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    with open(os.path.join(out_dir, f"{gvp_type}_clustering_results.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False, default=serialize)

    pd.DataFrame({
        "UMAP1": umap_embeds[:, 0],
        "UMAP2": umap_embeds[:, 1],
        "cluster": labels
    }).to_csv(
        os.path.join(out_dir, f"{gvp_type}_umap_cluster_coordinates.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    # GvpA/GvpC单独出图
    if gvp_type in ["gvpa", "gvpc"]:
        plot_single_nature_cluster(
            umap_embeds,
            labels,
            cluster_stats,
            valid_lbls,
            major,
            gvp_type,
            out_dir,
            min_cluster_plot
        )

    print(f"\n{format_gvp_name(gvp_type)} 表4-4统计：")
    print(table_4_4_row)

    print(f"\n{format_gvp_name(gvp_type)} 表4-5统计：")
    print(table_4_5_row)

    return {
        "gvp_type": gvp_type,
        "table_4_4_row": table_4_4_row,
        "table_4_5_row": table_4_5_row,
        "plot_data": {
            "umap_embeds": umap_embeds,
            "labels": labels,
            "cluster_stats": cluster_stats,
            "valid_lbls": valid_lbls,
            "major": major
        }
    }


def run_single_gvp(gvp_type, config):
    gvp_suffix = gvp_type[-1].upper()

    json_path = os.path.join(
        DATA_ROOT,
        gvp_type,
        f"Gvp{gvp_suffix}_sequences.json"
    )

    out_dir = os.path.join(OUTPUT_ROOT, gvp_type)
    os.makedirs(out_dir, exist_ok=True)

    records = load_single_gvp(json_path, gvp_type)

    if not records:
        print(f"No data for {gvp_type}")
        return None

    fragments, meta = slide_window(records, config["window"], config["step"])

    if not fragments:
        print(f"No valid fragments for {gvp_type}")
        return None

    extractor = ESM2Extractor(config["model"])
    embeds = extractor.get_embeddings(fragments)

    if len(embeds) == 0:
        print(f"No embeddings generated for {gvp_type}")
        return None

    return cluster_and_plot(embeds, meta, gvp_type, out_dir, config)


def main():
    target_gvps = [
        g for g in TARGET_GVPS
        if g in GVP_FULL_CONFIG
    ]

    print("Target GVPs to process:", target_gvps)

    table_4_4_rows = []
    table_4_5_rows = []
    all_plot_data = {}

    for gvp in target_gvps:
        print("\n" + "=" * 60)
        print(f"Processing {gvp.upper()}...")
        print("=" * 60)

        result = run_single_gvp(gvp, GVP_FULL_CONFIG[gvp])

        if result is None:
            continue

        table_4_4_rows.append(result["table_4_4_row"])
        table_4_5_rows.append(result["table_4_5_row"])
        all_plot_data[gvp] = result["plot_data"]

    # ================= 导出表4-4 =================
    df_4_4 = pd.DataFrame(table_4_4_rows)

    table_4_4_path = os.path.join(
        OUTPUT_ROOT,
        "Table_4_4_HDBSCAN_clustering_statistics.csv"
    )

    df_4_4.to_csv(table_4_4_path, index=False, encoding="utf-8-sig")

    print("\n表4-4 不同GVP类型HDBSCAN聚类结果统计")
    print(df_4_4)
    print(f"Saved: {table_4_4_path}")

    # ================= 导出表4-5 =================
    df_4_5 = pd.DataFrame(table_4_5_rows)

    table_4_5_path = os.path.join(
        OUTPUT_ROOT,
        "Table_4_5_clustering_quality_evaluation.csv"
    )

    df_4_5.to_csv(table_4_5_path, index=False, encoding="utf-8-sig")

    print("\n表4-5 不同GVP类型聚类质量评价结果")
    print(df_4_5)
    print(f"Saved: {table_4_5_path}")

    # ================= 其他GVP合并子图 =================
    plot_combined_other_gvps(all_plot_data, OUTPUT_ROOT)

    print(f"\nAll processing completed. Results saved to: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
