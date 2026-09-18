#!/usr/bin/env python3
"""RQ3: compare generated GvpA sequences with real sequences in ESM-2 space.

This is deliberately a distribution analysis, not a functional prediction.  It
uses the frozen cyanobacterial GvpA set as the target distribution, a real-vs-
real split as the natural-variation baseline, and reports MMD plus an outlier
rate calibrated on the held-out real sequences.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from transformers import AutoModel, AutoTokenizer

try:
    import umap.umap_ as umap
except ImportError:  # pragma: no cover - PCA fallback keeps the script useful.
    umap = None

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.fasta import read_fasta

MODEL_ID = "facebook/esm2_t6_8M_UR50D"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RQ3 ESM-2 embedding distribution analysis.")
    parser.add_argument("--manifest", required=True, help="The final_model_manifest.json file.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--bootstrap-repeats", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size < 1 or args.bootstrap_repeats < 20:
        raise SystemExit("batch-size must be positive and bootstrap-repeats must be at least 20")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(args.manifest)
    manifest = json.loads(manifest_path.read_text())
    base_dir = manifest_path.parent
    reference_path = resolve_path(base_dir, manifest["reference"])
    train_path = reference_path.parent / "train.fasta"
    real_records = read_fasta(reference_path)
    train_records = read_fasta(train_path)
    train_ids = {record.record_id for record in train_records}

    groups: list[dict[str, Any]] = [
        {
            "key": "real",
            "label": "Real cyanobacterial GvpA",
            "model_type": "real",
            "records": real_records,
        }
    ]
    for item in manifest["runs"]:
        groups.append(
            {
                "key": slug(item["model"]),
                "label": item["model"],
                "model_type": item["model_type"],
                "records": read_fasta(resolve_path(base_dir, item["generated"])),
            }
        )

    ids, sequences, labels, kinds = flatten_groups(groups)
    device = select_device(args.device)
    embeddings = embed_sequences(sequences, args.model_id, args.batch_size, device)
    embeddings = l2_normalize(embeddings)
    np.savez_compressed(
        output_dir / "esm2_embeddings.npz",
        ids=np.asarray(ids),
        labels=np.asarray(labels),
        kinds=np.asarray(kinds),
        embeddings=embeddings,
    )

    coordinates, projection_name = project_2d(embeddings, args.seed)
    real_mask = np.asarray(labels) == "Real cyanobacterial GvpA"
    train_mask = np.asarray([record_id in train_ids for record_id in ids]) & real_mask
    heldout_mask = real_mask & ~train_mask
    if not train_mask.any() or not heldout_mask.any():
        raise RuntimeError("Could not recover the frozen train/held-out split for outlier calibration.")
    train_embeddings = embeddings[train_mask]
    heldout_embeddings = embeddings[heldout_mask]
    heldout_distances = nearest_distances(heldout_embeddings, train_embeddings)
    outlier_threshold = float(np.quantile(heldout_distances, 0.95))

    rng = np.random.default_rng(args.seed)
    real_embeddings = embeddings[real_mask]
    bandwidth2 = median_squared_distance(real_embeddings)
    real_split_mmd = real_split_baseline(
        real_embeddings, args.bootstrap_repeats, rng, bandwidth2
    )
    summary_rows: list[dict[str, Any]] = []
    per_sequence_rows: list[dict[str, Any]] = []
    start = 0
    for group in groups:
        count = len(group["records"])
        indexes = np.arange(start, start + count)
        group_embeddings = embeddings[indexes]
        group_distances = nearest_distances(group_embeddings, train_embeddings)
        group_outliers = group_distances > outlier_threshold
        group_mmd = generated_vs_real_mmd(
            group_embeddings,
            real_embeddings,
            args.bootstrap_repeats,
            rng,
            bandwidth2,
        ) if group["key"] != "real" else real_split_mmd
        if group["key"] == "real":
            group_outliers = np.zeros(count, dtype=bool)
        row = summarize_group(
            group,
            group_mmd,
            real_split_mmd,
            group_distances,
            group_outliers,
            outlier_threshold,
        )
        summary_rows.append(row)
        for local_index, record in enumerate(group["records"]):
            per_sequence_rows.append(
                {
                    "sequence_id": record.record_id,
                    "source": group["label"],
                    "source_type": group["model_type"],
                    "length": len(record.sequence),
                    "projection_1": coordinates[start + local_index, 0],
                    "projection_2": coordinates[start + local_index, 1],
                    "nearest_train_embedding_distance": group_distances[local_index],
                    "embedding_outlier": bool(group_outliers[local_index]),
                }
            )
        start += count

    write_csv(output_dir / "per_sequence_embeddings.csv", per_sequence_rows)
    write_csv(output_dir / "distribution_summary.csv", summary_rows)
    plot_projection(coordinates, labels, output_dir / "esm2_projection.png", projection_name)
    metadata = {
        "rq": "RQ3",
        "model_id": args.model_id,
        "pooling": "last-layer mean pooling over residue tokens; special and padding tokens excluded",
        "normalization": "L2 normalization before distances, MMD and projection",
        "reference_fasta": str(reference_path),
        "n_real": len(real_records),
        "n_train_reference": len(train_records),
        "n_heldout_reference": int(heldout_mask.sum()),
        "projection": projection_name,
        "mmd": "biased RBF MMD^2 with a single bandwidth calibrated on all real embeddings; repeated matched-size subsamples",
        "rbf_bandwidth_squared": bandwidth2,
        "real_split_baseline": summarize_distribution(real_split_mmd),
        "outlier_threshold": {
            "definition": "95th percentile of held-out real sequence distance to frozen training embeddings",
            "value": outlier_threshold,
        },
        "random_seed": args.seed,
    }
    (output_dir / "analysis_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (output_dir / "rq3_report.md").write_text(
        build_report(
            summary_rows,
            metadata,
            manifest.get("reference_label", str(reference_path)),
            sampling_seed_retest=bool(manifest.get("sampling_seed_retest", False)),
        )
    )


def resolve_path(base_dir: Path, value: str) -> Path:
    candidate = Path(value)
    return candidate if candidate.is_absolute() else base_dir / candidate


def flatten_groups(groups: list[dict[str, Any]]) -> tuple[list[str], list[str], list[str], list[str]]:
    ids: list[str] = []
    sequences: list[str] = []
    labels: list[str] = []
    kinds: list[str] = []
    for group in groups:
        for record in group["records"]:
            ids.append(record.record_id)
            sequences.append(record.sequence)
            labels.append(group["label"])
            kinds.append(group["model_type"])
    return ids, sequences, labels, kinds


def select_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is unavailable.")
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def embed_sequences(
    sequences: list[str], model_id: str, batch_size: int, device: torch.device
) -> np.ndarray:
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id).to(device)
    model.eval()
    vectors: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(sequences), batch_size):
            batch = sequences[start : start + batch_size]
            encoded = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                return_special_tokens_mask=True,
            )
            special_mask = encoded.pop("special_tokens_mask")
            encoded = {key: value.to(device) for key, value in encoded.items()}
            hidden = model(**encoded).last_hidden_state
            residue_mask = (
                encoded["attention_mask"].bool()
                & ~special_mask.to(device).bool()
            )
            pooled = (hidden * residue_mask.unsqueeze(-1)).sum(dim=1)
            pooled = pooled / residue_mask.sum(dim=1, keepdim=True).clamp(min=1)
            vectors.append(pooled.cpu().numpy())
    return np.concatenate(vectors, axis=0)


def l2_normalize(values: np.ndarray) -> np.ndarray:
    return values / np.linalg.norm(values, axis=1, keepdims=True).clip(min=1e-12)


def project_2d(values: np.ndarray, seed: int) -> tuple[np.ndarray, str]:
    if umap is not None:
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.15, metric="cosine", random_state=seed)
        return reducer.fit_transform(values), "UMAP (cosine, n_neighbors=15, min_dist=0.15)"
    return PCA(n_components=2, random_state=seed).fit_transform(values), "PCA (UMAP unavailable)"


def nearest_distances(query: np.ndarray, reference: np.ndarray) -> np.ndarray:
    squared = (
        np.sum(query * query, axis=1, keepdims=True)
        + np.sum(reference * reference, axis=1)
        - 2 * query @ reference.T
    )
    return np.sqrt(np.maximum(squared, 0.0)).min(axis=1)


def rbf_mmd2(left: np.ndarray, right: np.ndarray, bandwidth2: float) -> float:
    kernel_xx = np.exp(-pairwise_squared_distances(left, left) / (2 * bandwidth2))
    kernel_yy = np.exp(-pairwise_squared_distances(right, right) / (2 * bandwidth2))
    kernel_xy = np.exp(-pairwise_squared_distances(left, right) / (2 * bandwidth2))
    return float(kernel_xx.mean() + kernel_yy.mean() - 2 * kernel_xy.mean())


def pairwise_squared_distances(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.maximum(
        np.sum(left * left, axis=1, keepdims=True)
        + np.sum(right * right, axis=1)
        - 2 * left @ right.T,
        0.0,
    )


def median_squared_distance(values: np.ndarray) -> float:
    distances = pairwise_squared_distances(values, values)
    upper = distances[np.triu_indices_from(distances, k=1)]
    positive = upper[upper > 0]
    return float(np.median(positive)) if len(positive) else 1.0


def real_split_baseline(
    values: np.ndarray, repeats: int, rng: np.random.Generator, bandwidth2: float
) -> np.ndarray:
    size = len(values) // 2
    results = []
    for _ in range(repeats):
        indexes = rng.permutation(len(values))
        results.append(
            rbf_mmd2(values[indexes[:size]], values[indexes[size : 2 * size]], bandwidth2)
        )
    return np.asarray(results)


def generated_vs_real_mmd(
    generated: np.ndarray,
    real: np.ndarray,
    repeats: int,
    rng: np.random.Generator,
    bandwidth2: float,
) -> np.ndarray:
    size = min(len(generated), len(real) // 2)
    results = []
    for _ in range(repeats):
        generated_indexes = rng.choice(len(generated), size=size, replace=False)
        real_indexes = rng.choice(len(real), size=size, replace=False)
        results.append(rbf_mmd2(generated[generated_indexes], real[real_indexes], bandwidth2))
    return np.asarray(results)


def summarize_group(
    group: dict[str, Any],
    mmd_values: np.ndarray,
    real_split_mmd: np.ndarray,
    distances: np.ndarray,
    outliers: np.ndarray,
    outlier_threshold: float,
) -> dict[str, Any]:
    mmd_summary = summarize_distribution(mmd_values)
    baseline_q95 = float(np.quantile(real_split_mmd, 0.95))
    return {
        "source": group["label"],
        "source_type": group["model_type"],
        "n_sequences": len(group["records"]),
        "mmd2_median": mmd_summary["median"],
        "mmd2_q025": mmd_summary["q025"],
        "mmd2_q975": mmd_summary["q975"],
        "mmd2_exceeds_real_split_q95_fraction": float(np.mean(mmd_values > baseline_q95)),
        "mean_nearest_train_embedding_distance": float(np.mean(distances)),
        "outlier_threshold": outlier_threshold,
        "embedding_outlier_rate": float(np.mean(outliers)),
        "distribution_assessment": assess_distribution(
            group["model_type"], mmd_values, baseline_q95, float(np.mean(outliers))
        ),
    }


def summarize_distribution(values: np.ndarray) -> dict[str, float]:
    return {
        "median": float(np.median(values)),
        "q025": float(np.quantile(values, 0.025)),
        "q975": float(np.quantile(values, 0.975)),
    }


def assess_distribution(kind: str, values: np.ndarray, baseline_q95: float, outlier_rate: float) -> str:
    if kind == "real":
        return "Natural real-vs-real baseline; not a generated-model judgment."
    if np.median(values) <= baseline_q95 and outlier_rate <= 0.05:
        return "Compatible with the real embedding distribution at this resolution."
    if outlier_rate > 0.20:
        return "Substantial embedding outlier rate; evidence of distributional drift."
    return "Partly overlaps real GvpA, but distributional mismatch exceeds the real-vs-real baseline."


def plot_projection(
    coordinates: np.ndarray, labels: list[str], output_path: Path, projection_name: str
) -> None:
    order = ["Real cyanobacterial GvpA"] + [label for label in dict.fromkeys(labels) if label != "Real cyanobacterial GvpA"]
    colors = ["#222222", "#7f7f7f", "#bcbd22", "#1f77b4", "#ff7f0e", "#d62728"]
    figure, axis = plt.subplots(figsize=(9.5, 6.5), dpi=180)
    labels_array = np.asarray(labels)
    for index, label in enumerate(order):
        mask = labels_array == label
        axis.scatter(
            coordinates[mask, 0],
            coordinates[mask, 1],
            label=label,
            s=20 if label == "Real cyanobacterial GvpA" else 28,
            alpha=0.62 if label == "Real cyanobacterial GvpA" else 0.78,
            color=colors[index % len(colors)],
            edgecolors="none",
        )
    axis.set_title("RQ3: ESM-2 embedding projection")
    axis.set_xlabel(f"{projection_name} component 1")
    axis.set_ylabel(f"{projection_name} component 2")
    axis.legend(loc="best", fontsize=7, frameon=True)
    figure.tight_layout()
    figure.savefig(output_path)
    plt.close(figure)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def build_report(
    rows: list[dict[str, Any]],
    metadata: dict[str, Any],
    reference_label: str,
    sampling_seed_retest: bool = False,
) -> str:
    baseline = metadata["real_split_baseline"]
    lines = [
        "# RQ3: PLM Embedding Distribution Analysis",
        "",
        "## Question and scope",
        "",
        "This analysis asks whether the generated sets occupy a representation-space distribution compatible with frozen cyanobacterial GvpA, rather than merely matching simple sequence statistics. It is computational evidence only: it does not prove protein function.",
        "",
        "## Method",
        "",
        f"- PLM: `{metadata['model_id']}`; {metadata['pooling']}",
        f"- Target distribution: {metadata['n_real']} frozen cyanobacterial GvpA sequences from `{reference_label}`.",
        f"- Natural-variation baseline: repeated random real-vs-real half splits; MMD^2 median {baseline['median']:.4f}, 95% interval [{baseline['q025']:.4f}, {baseline['q975']:.4f}].",
        f"- Outlier threshold: {metadata['outlier_threshold']['value']:.4f}, the 95th percentile of held-out real-to-train embedding distances.",
        "- The two baselines are negative controls: they remain in the plot/table even when their family-profile evidence is incomplete.",
        "",
        "## Results",
        "",
        "| Source | MMD^2 median | MMD beyond real q95 | Outlier rate | Assessment |",
        "|---|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['source']} | {row['mmd2_median']:.4f} | "
            f"{100 * row['mmd2_exceeds_real_split_q95_fraction']:.1f}% | "
            f"{100 * row['embedding_outlier_rate']:.1f}% | {row['distribution_assessment']} |"
        )
    limitation = (
        "Each row is a sampling-seed replicate from a selected checkpoint. "
        "Compare its mean/range across seeds for sampling stability; this does not replace independent model-retraining repeats. "
        "ESM-2 embeddings are not functional assays and should be interpreted alongside profile, conservation and composition evidence."
        if sampling_seed_retest
        else "The current generated sets are single selected samples per method. The next robustness step is to repeat the same embedding analysis for the available generation seeds, and report a mean/range rather than a single row. "
        "ESM-2 embeddings are not functional assays and should be interpreted alongside profile, conservation and composition evidence."
    )
    lines.extend([
        "",
        "## Interpretation rule",
        "",
        "A small MMD relative to real-vs-real variation and a low calibrated outlier rate support distributional compatibility. A high profile hit rate without this support indicates that local family motifs can be present even when the global representation distribution drifts. UMAP is a visualization only; conclusions use the high-dimensional MMD and outlier statistics.",
        "",
        "## Limitations",
        "",
        limitation,
        "",
    ])
    return "\n".join(lines)


def slug(value: str) -> str:
    return "".join(character.lower() if character.isalnum() else "_" for character in value).strip("_")


if __name__ == "__main__":
    main()
