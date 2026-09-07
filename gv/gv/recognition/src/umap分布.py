#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
GVP Fragment UMAP Visualization Only
GvpA、GvpC单独出图
GvpG、GvpJ、GvpK、GvpN、GvpO、GvpP合并为2×3子图
不进行HDBSCAN聚类
"""

import os
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import esm
import umap
import requests

from tqdm import tqdm

warnings.filterwarnings("ignore")

# ================= 路径配置 =================
DATA_ROOT = r"C:\Users\r9000\Desktop\毕设（无监督聚类）\新gvp"
OUTPUT_ROOT = r"C:\Users\r9000\Desktop\毕设（无监督聚类）\UMAP_Only_Results"

os.makedirs(OUTPUT_ROOT, exist_ok=True)

# ================= GVP类型 =================
TARGET_GVPS = [
    "gvpa",
    "gvpc",
    "gvpg",
    "gvpj",
    "gvpk",
    "gvpn",
    "gvpo",
    "gvpp"
]

# ================= 参数配置 =================
GVP_CONFIG = {
    "gvpa": {
        "window": 30,
        "step": 5,
        "model": "esm2_t12_35M_UR50D",
        "umap_params": {
            "n_neighbors": 20,
            "min_dist": 0.15,
            "metric": "euclidean",
            "random_state": 42
        }
    },
    "gvpc": {
        "window": 35,
        "step": 8,
        "model": "esm2_t12_35M_UR50D",
        "umap_params": {
            "n_neighbors": 30,
            "min_dist": 0.10,
            "metric": "euclidean",
            "random_state": 42
        }
    },
    "gvpg": {
        "window": 25,
        "step": 5,
        "model": "esm2_t12_35M_UR50D",
        "umap_params": {
            "n_neighbors": 18,
            "min_dist": 0.10,
            "metric": "euclidean",
            "random_state": 42
        }
    },
    "gvpj": {
        "window": 35,
        "step": 10,
        "model": "esm2_t12_35M_UR50D",
        "umap_params": {
            "n_neighbors": 22,
            "min_dist": 0.15,
            "metric": "euclidean",
            "random_state": 42
        }
    },
    "gvpk": {
        "window": 25,
        "step": 4,
        "model": "esm2_t12_35M_UR50D",
        "umap_params": {
            "n_neighbors": 15,
            "min_dist": 0.10,
            "metric": "euclidean",
            "random_state": 42
        }
    },
    "gvpn": {
        "window": 40,
        "step": 8,
        "model": "esm2_t12_35M_UR50D",
        "umap_params": {
            "n_neighbors": 23,
            "min_dist": 0.08,
            "metric": "euclidean",
            "random_state": 42
        }
    },
    "gvpo": {
        "window": 50,
        "step": 5,
        "model": "esm2_t12_35M_UR50D",
        "umap_params": {
            "n_neighbors": 15,
            "min_dist": 0.20,
            "metric": "cosine",
            "random_state": 42
        }
    },
    "gvpp": {
        "window": 40,
        "step": 10,
        "model": "esm2_t12_35M_UR50D",
        "umap_params": {
            "n_neighbors": 15,
            "min_dist": 0.13,
            "metric": "euclidean",
            "random_state": 42
        }
    }
}

# ================= Nature风格配色 =================
COLOR_MAP = {
    "gvpa": "#3B5B92",
    "gvpc": "#4E8F5A",
    "gvpg": "#B44745",
    "gvpj": "#6A5A8C",
    "gvpk": "#B38B4D",
    "gvpn": "#5A9FB5",
    "gvpo": "#7A7A7A",
    "gvpp": "#8C6D62"
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


# ========================================================
# ESM-2 Embedding提取器
# ========================================================
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
            print(f"Downloading {self.model_name}...")

            url = f"https://dl.fbaipublicfiles.com/fair-esm/models/{self.model_name}.pt"
            response = requests.get(url, stream=True, timeout=60)
            response.raise_for_status()

            total_size = int(response.headers.get("content-length", 0))

            with open(self.model_path, "wb") as f, tqdm(
                total=total_size,
                unit="iB",
                unit_scale=True
            ) as bar:
                for data in response.iter_content(chunk_size=8192):
                    f.write(data)
                    bar.update(len(data))

        self.model, self.alphabet = getattr(esm.pretrained, self.model_name)()
        self.model.to(DEVICE)
        self.model.eval()

        self.batch_converter = self.alphabet.get_batch_converter()

    def get_embeddings(self, fragments, batch_size=16):
        embeddings = []

        with torch.no_grad():
            for i in range(0, len(fragments), batch_size):
                batch = fragments[i:i + batch_size]

                try:
                    _, _, tokens = self.batch_converter(batch)
                    tokens = tokens.to(DEVICE)

                    results = self.model(
                        tokens,
                        repr_layers=[self.model.num_layers]
                    )

                    reps = results["representations"][self.model.num_layers]

                    for j, (_, seq) in enumerate(batch):
                        emb = reps[j, 1:len(seq) + 1].mean(dim=0)
                        emb = emb / torch.norm(emb)
                        embeddings.append(emb.cpu().numpy())

                except Exception as e:
                    print(f"Embedding extraction error: {e}")

        return np.array(embeddings)


# ========================================================
# 读取单类GVP序列
# ========================================================
def load_single_gvp(json_path, gvp_type):
    if not os.path.exists(json_path):
        print(f"File not found: {json_path}")
        return []

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    records = []

    for item in data:
        seq = item.get("sequence", "").strip().upper()

        seq = "".join(
            [c for c in seq if c in "ACDEFGHIKLMNPQRSTVWY"]
        )

        if len(seq) < 10:
            continue

        records.append({
            "seq": seq,
            "acc": item.get("unique_sequence_id"),
            "species": item.get(
                "representative_annotation",
                {}
            ).get("organism"),
            "gvp_type": gvp_type,
            "full_seq_length": len(seq)
        })

    return records


# ========================================================
# 滑动窗口切分
# ========================================================
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

            fragments.append(
                (f"{rec['acc']}_{s}-{e}", seq[s:e])
            )

            meta.append({
                "acc": rec["acc"],
                "start": s,
                "end": e,
                "center": (s + e) / 2,
                "window_size": window,
                "full_seq_length": full_len,
                "species": rec["species"],
                "gvp_type": rec["gvp_type"]
            })

    return fragments, meta


# ========================================================
# 保存UMAP坐标
# ========================================================
def save_umap_coordinates(umap_embeds, meta, out_dir, gvp_type):
    rows = []

    for i, m in enumerate(meta):
        rows.append({
            "acc": m["acc"],
            "species": m["species"],
            "gvp_type": m["gvp_type"],
            "start": m["start"],
            "end": m["end"],
            "center": m["center"],
            "window_size": m["window_size"],
            "full_seq_length": m["full_seq_length"],
            "UMAP1": umap_embeds[i, 0],
            "UMAP2": umap_embeds[i, 1]
        })

    df = pd.DataFrame(rows)

    save_path = os.path.join(
        out_dir,
        f"{gvp_type}_umap_coordinates.csv"
    )

    df.to_csv(save_path, index=False, encoding="utf-8-sig")

    print(f"Saved coordinates: {save_path}")


# ========================================================
# Nature风格坐标轴
# ========================================================
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


# ========================================================
# GvpA、GvpC单独绘图
# ========================================================
def save_single_umap_plot(umap_embeds, gvp_type, out_dir):
    plt.figure(figsize=(10, 8))

    plt.scatter(
        umap_embeds[:, 0],
        umap_embeds[:, 1],
        c=COLOR_MAP[gvp_type],
        s=7,
        alpha=0.65,
        linewidths=0
    )

    plt.title(
        f"{gvp_type.upper()} Fragment UMAP Projection",
        fontsize=16,
        fontweight="bold"
    )

    plt.xlabel("UMAP 1", fontsize=13)
    plt.ylabel("UMAP 2", fontsize=13)

    ax = plt.gca()
    apply_nature_axis_style(ax)

    plt.tight_layout()

    save_path = os.path.join(
        out_dir,
        f"{gvp_type}_umap_nature_style.png"
    )

    plt.savefig(
        save_path,
        dpi=600,
        bbox_inches="tight"
    )

    plt.close()

    print(f"Saved plot: {save_path}")


# ========================================================
# 其他GVP类型合并为2×3子图
# ========================================================
def save_other_gvp_subplot(all_umap_results):
    subplot_gvps = [
        "gvpg",
        "gvpj",
        "gvpk",
        "gvpn",
        "gvpo",
        "gvpp"
    ]

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(16, 10)
    )

    axes = axes.flatten()

    for idx, gvp_type in enumerate(subplot_gvps):
        ax = axes[idx]

        if gvp_type not in all_umap_results:
            ax.axis("off")
            continue

        umap_embeds = all_umap_results[gvp_type]

        ax.scatter(
            umap_embeds[:, 0],
            umap_embeds[:, 1],
            c=COLOR_MAP[gvp_type],
            s=5,
            alpha=0.65,
            linewidths=0
        )

        ax.set_title(
            f"{gvp_type.upper()}",
            fontsize=14,
            fontweight="bold"
        )

        ax.set_xlabel("UMAP 1", fontsize=10)
        ax.set_ylabel("UMAP 2", fontsize=10)

        apply_nature_axis_style(ax)

    plt.suptitle(
        "UMAP Projection of Other GVP Types",
        fontsize=18,
        fontweight="bold",
        y=0.98
    )

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    save_path = os.path.join(
        OUTPUT_ROOT,
        "Other_GVP_UMAP_Subplots_Nature_Style.png"
    )

    plt.savefig(
        save_path,
        dpi=600,
        bbox_inches="tight"
    )

    plt.close()

    print(f"Saved subplot: {save_path}")


# ========================================================
# 单个GVP完整流程
# ========================================================
def process_single_gvp(gvp_type, all_umap_results):
    print("\n" + "=" * 60)
    print(f"Processing {gvp_type.upper()}")
    print("=" * 60)

    config = GVP_CONFIG[gvp_type]
    gvp_suffix = gvp_type[-1].upper()

    json_path = os.path.join(
        DATA_ROOT,
        gvp_type,
        f"Gvp{gvp_suffix}_sequences.json"
    )

    out_dir = os.path.join(
        OUTPUT_ROOT,
        gvp_type
    )

    os.makedirs(out_dir, exist_ok=True)

    records = load_single_gvp(json_path, gvp_type)

    if not records:
        print(f"No valid records for {gvp_type}")
        return

    print(f"Loaded sequences: {len(records)}")

    fragments, meta = slide_window(
        records,
        config["window"],
        config["step"]
    )

    if not fragments:
        print(f"No fragments for {gvp_type}")
        return

    print(f"Generated fragments: {len(fragments)}")

    extractor = ESM2Extractor(config["model"])

    embeddings = extractor.get_embeddings(fragments)

    if len(embeddings) == 0:
        print(f"No embeddings for {gvp_type}")
        return

    print(f"Embedding shape: {embeddings.shape}")

    reducer = umap.UMAP(**config["umap_params"])
    umap_embeds = reducer.fit_transform(embeddings)

    print(f"UMAP shape: {umap_embeds.shape}")

    save_umap_coordinates(
        umap_embeds,
        meta,
        out_dir,
        gvp_type
    )

    all_umap_results[gvp_type] = umap_embeds

    if gvp_type in ["gvpa", "gvpc"]:
        save_single_umap_plot(
            umap_embeds,
            gvp_type,
            out_dir
        )


# ========================================================
# 主函数
# ========================================================
def main():
    print("\nStarting GVP UMAP-only visualization pipeline...")
    print("No HDBSCAN clustering will be performed.\n")

    all_umap_results = {}

    for gvp_type in TARGET_GVPS:
        process_single_gvp(
            gvp_type,
            all_umap_results
        )

    save_other_gvp_subplot(all_umap_results)

    print("\nAll processing completed!")
    print(f"Results saved to: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
