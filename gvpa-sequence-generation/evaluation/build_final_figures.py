#!/usr/bin/env python3
"""Build the M4 ablation summaries and final-report figures from saved runs.

This script intentionally separates two kinds of evidence:
1. optimisation-package comparisons (several changes were made together, so these
   are not claimed as one-factor causal ablations); and
2. controlled sampling-parameter comparisons from a fixed selected checkpoint.

It only reads existing FASTA files and evaluator metrics; it never retrains a
model or rewrites a prior experimental result.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHINESE_FONT = Path("/home/robin/.local/share/fonts/ke-ppt-cjk/NotoSansCJK-Regular.ttc")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build final M4 ablation tables and figures.")
    parser.add_argument("--output-dir", default="runs/final_figures")
    return parser.parse_args()


def read_metrics(relative_path: str) -> dict[str, Any]:
    with (PROJECT_ROOT / relative_path).open() as handle:
        return json.load(handle)


def fasta_lengths(relative_path: str) -> list[int]:
    lengths: list[int] = []
    current: list[str] = []
    with (PROJECT_ROOT / relative_path).open() as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current:
                    lengths.append(len("".join(current)))
                    current = []
            else:
                current.append(line)
    if current:
        lengths.append(len("".join(current)))
    return lengths


def metric_row(
    *,
    model: str,
    condition: str,
    metrics_path: str,
    fasta_path: str,
    comparison_type: str,
    note: str,
) -> dict[str, Any]:
    metrics = read_metrics(metrics_path)
    lengths = fasta_lengths(fasta_path)
    return {
        "model": model,
        "condition": condition,
        "comparison_type": comparison_type,
        "n_sequences": metrics["n_generated"],
        "profile_hit_rate": metrics.get("family_profile_hit_rate"),
        "novel_rate_train_identity_lt_0_95": metrics["novel_sequence_rate"],
        "pairwise_diversity": metrics["mean_pairwise_diversity"],
        "aa_composition_l1": metrics["aa_composition_l1_distance"],
        "exact_copy_train_rate": metrics["exact_copy_train_rate"],
        "length_in_train_range_rate": metrics["length_in_train_range_rate"],
        "mean_length": metrics["mean_length"],
        "length_at_91_rate": sum(length == 91 for length in lengths) / len(lengths),
        "metrics_source": metrics_path,
        "fasta_source": fasta_path,
        "note": note,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def setup_style() -> None:
    if CHINESE_FONT.exists():
        font_manager.fontManager.addfont(str(CHINESE_FONT))
        font_family = font_manager.FontProperties(fname=str(CHINESE_FONT)).get_name()
    else:  # Keep plots usable on a clean machine, even if Chinese glyphs are unavailable.
        font_family = "DejaVu Sans"
    plt.rcParams.update(
        {
            "font.family": font_family,
            "font.size": 10,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "figure.dpi": 160,
            "savefig.dpi": 240,
        }
    )


def plot_package_comparisons(rows: list[dict[str, Any]], output_path: Path) -> None:
    pairs = {
        "LSTM": ("原始 LSTM", "优化 LSTM"),
        "VAE": ("自由采样 VAE", "长度控制 VAE"),
        "GAN": ("原始 GAN", "约束 GAN"),
    }
    metric_specs = {
        "LSTM": [
            ("profile_hit_rate", "PF00741 通过率 ↑"),
            ("novel_rate_train_identity_lt_0_95", "新颖率 ↑"),
            ("pairwise_diversity", "批内多样性 ↑"),
            ("exact_copy_train_rate", "训练集精确复制率 ↓"),
        ],
        "VAE": [
            ("profile_hit_rate", "PF00741 通过率 ↑"),
            ("novel_rate_train_identity_lt_0_95", "新颖率 ↑"),
            ("pairwise_diversity", "批内多样性 ↑"),
            ("length_at_91_rate", "长度卡在 91 aa 的比例 ↓"),
        ],
        "GAN": [
            ("profile_hit_rate", "PF00741 通过率 ↑"),
            ("novel_rate_train_identity_lt_0_95", "新颖率 ↑"),
            ("pairwise_diversity", "批内多样性 ↑"),
            ("aa_composition_l1", "AA 组成 L1 距离 ↓"),
        ],
    }
    by_key = {(row["model"], row["condition"]): row for row in rows}
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.3), sharex=True)
    for axis, model in zip(axes, pairs):
        before, after = (by_key[(model, name)] for name in pairs[model])
        specs = metric_specs[model]
        positions = np.arange(len(specs))[::-1]
        for y, (key, _) in zip(positions, specs):
            start, end = before[key], after[key]
            axis.plot([start, end], [y, y], color="#97a1ad", linewidth=2, zorder=1)
            axis.scatter(start, y, color="#6d7680", s=58, zorder=2, label="优化前" if y == positions[0] else None)
            axis.scatter(end, y, color="#1677b8", s=58, zorder=3, label="优化后" if y == positions[0] else None)
            axis.text(start, y + 0.14, f"{start:.2f}", ha="center", va="bottom", color="#4f5861", fontsize=9)
            axis.text(end, y - 0.16, f"{end:.2f}", ha="center", va="top", color="#075985", fontsize=9)
        axis.set_yticks(positions, [label for _, label in specs])
        axis.set_xlim(-0.04, 1.06)
        axis.grid(axis="x", alpha=0.25)
        axis.set_title(model)
        axis.spines[["top", "right", "left"]].set_visible(False)
        axis.tick_params(axis="y", length=0)
    axes[0].legend(loc="lower right", frameon=False)
    fig.suptitle("优化/约束比较：保真、新颖性与多样性的取舍", y=0.98, fontsize=16, fontweight="bold")
    fig.text(
        0.5,
        0.01,
        "注：每列比较的是保存的完整优化方案，不是严格的单因素因果消融；VAE 的长度卡点比例由对应 FASTA 直接统计。",
        ha="center",
        color="#4b5563",
    )
    fig.tight_layout(rect=(0, 0.07, 1, 0.93))
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_sampling_controls(rows: list[dict[str, Any]], output_path: Path) -> None:
    models = ["LSTM", "VAE", "GAN"]
    colors = {"profile_hit_rate": "#1677b8", "novel_rate_train_identity_lt_0_95": "#d97706", "pairwise_diversity": "#2f855a"}
    labels = {
        "profile_hit_rate": "PF00741 通过率",
        "novel_rate_train_identity_lt_0_95": "新颖率",
        "pairwise_diversity": "批内多样性",
    }
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharey=True)
    for axis, model in zip(axes, models):
        subset = [row for row in rows if row["model"] == model]
        x = np.arange(len(subset))
        for key in colors:
            axis.plot(x, [row[key] for row in subset], marker="o", linewidth=2, color=colors[key], label=labels[key])
        axis.set_xticks(x, [row["condition"] for row in subset], rotation=18, ha="right")
        axis.set_ylim(-0.03, 1.06)
        axis.grid(axis="y", alpha=0.25)
        axis.set_title(f"{model}：固定 checkpoint 的生成参数")
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("指标值")
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("采样参数对结果的影响（可重复的生成阶段对照）", y=1.02, fontsize=16, fontweight="bold")
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_final_tradeoff(rows: list[dict[str, Any]], output_path: Path) -> None:
    fig, axis = plt.subplots(figsize=(8.6, 6.4))
    cmap = plt.colormaps["YlOrRd"]
    colors = [row["aa_composition_l1"] for row in rows]
    dots = axis.scatter(
        [row["strict_novelty_all_165"] for row in rows],
        [row["profile_hit_rate"] for row in rows],
        s=[300 + 1800 * row["pairwise_diversity"] for row in rows],
        c=colors,
        cmap=cmap,
        edgecolor="#1f2937",
        linewidth=0.8,
        zorder=3,
    )
    for row in rows:
        axis.annotate(
            row["model"],
            (row["strict_novelty_all_165"], row["profile_hit_rate"]),
            xytext=(8, 8),
            textcoords="offset points",
            fontweight="bold",
        )
    axis.axhline(0.95, color="#64748b", linestyle="--", linewidth=1.2, label="PF00741 筛选线 (0.95)")
    axis.set(xlim=(0.35, 1.05), ylim=(0.94, 1.01), xlabel="严格新颖率（对全部 165 条冻结蓝藻 GvpA，↑）", ylabel="PF00741 通过率（↑）")
    axis.grid(alpha=0.25)
    axis.legend(loc="lower left", frameon=False)
    colorbar = fig.colorbar(dots, ax=axis, pad=0.02)
    colorbar.set_label("AA 组成 L1（↓，越浅越接近）")
    axis.text(0.99, 0.943, "圆面积：批内多样性（↑）", ha="right", va="bottom", color="#4b5563")
    axis.set_title("最终候选的三目标权衡", fontsize=16, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_rq3(rows: list[dict[str, Any]], output_path: Path) -> None:
    labels = [row["model"] for row in rows]
    mmd = np.asarray([row["mmd2_mean"] for row in rows])
    mmd_min = np.asarray([row["mmd2_min"] for row in rows])
    mmd_max = np.asarray([row["mmd2_max"] for row in rows])
    outlier = np.asarray([row["embedding_outlier_rate_mean"] for row in rows])
    outlier_min = np.asarray([row["embedding_outlier_rate_min"] for row in rows])
    outlier_max = np.asarray([row["embedding_outlier_rate_max"] for row in rows])
    x = np.arange(len(rows))
    colors = ["#1677b8", "#d97706", "#c2410c"]
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.2))
    axes[0].bar(x, mmd, color=colors, width=0.62)
    axes[0].errorbar(x, mmd, yerr=np.vstack((mmd - mmd_min, mmd_max - mmd)), fmt="none", color="#334155", capsize=5)
    axes[0].axhspan(0.0044, 0.0206, color="#cbd5e1", alpha=0.65, label="真实序列随机二分的 95% 区间")
    axes[0].axhline(0.0089, color="#64748b", linestyle="--", label="真实-真实 MMD² 中位数")
    axes[0].set_yscale("log")
    axes[0].set_xticks(x, labels)
    axes[0].set_ylabel("ESM-2 MMD²（对数轴，↓）")
    axes[0].set_title("全局表示分布差异")
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].bar(x, outlier * 100, color=colors, width=0.62)
    axes[1].errorbar(x, outlier * 100, yerr=np.vstack(((outlier - outlier_min) * 100, (outlier_max - outlier) * 100)), fmt="none", color="#334155", capsize=5)
    axes[1].axhline(5, color="#64748b", linestyle="--", label="真实留出集校准阈值（5%）")
    axes[1].set_xticks(x, labels)
    axes[1].set_ylim(0, 105)
    axes[1].set_ylabel("嵌入离群率（%，↓）")
    axes[1].set_title("落在真实训练分布之外的比例")
    axes[1].legend(frameon=False, fontsize=8)
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", alpha=0.2)
    fig.suptitle("RQ3：三个采样种子复测后的 ESM-2 分布结果", fontsize=16, fontweight="bold")
    fig.text(0.5, 0.01, "柱为三个采样种子的均值，误差线为最小值至最大值；这检验采样稳定性，并非独立重新训练。", ha="center", color="#4b5563")
    fig.tight_layout(rect=(0, 0.06, 1, 0.93))
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_lengths(output_path: Path) -> None:
    series = [
        ("真实蓝藻 GvpA", "data/processed/dataset_v1/all.fasta", "#475569"),
        ("VAE 自由采样", "runs/formal_vae_50ep_seed42/generated.fasta", "#d97706"),
        ("VAE 长度控制", "runs/formal_vae_50ep_seed42/generated_empirical_s0.75_t0.6_masked.fasta", "#1677b8"),
    ]
    fig, axis = plt.subplots(figsize=(10, 5.2))
    bins = np.arange(58.5, 92.6, 1)
    for label, path, color in series:
        axis.hist(fasta_lengths(path), bins=bins, histtype="step", linewidth=2.2, density=True, label=label, color=color)
    axis.axvline(91, linestyle="--", color="#991b1b", linewidth=1.2, label="训练集最大长度 91")
    axis.set(xlim=(58, 92), xlabel="序列长度（aa）", ylabel="密度", title="VAE 长度控制修复了自由采样的上限堆积")
    axis.grid(axis="y", alpha=0.2)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_style()

    package_rows = [
        metric_row(model="LSTM", condition="原始 LSTM", comparison_type="optimisation_package", metrics_path="runs/formal_lstm_50ep_seed42/eval_profile_v2/metrics.json", fasta_path="runs/formal_lstm_50ep_seed42/generated.fasta", note="原始自回归模型。"),
        metric_row(model="LSTM", condition="优化 LSTM", comparison_type="optimisation_package", metrics_path="runs/opt_lstm_pretrain_finetune_seed42/eval_pf_t1.0_masked/metrics.json", fasta_path="runs/opt_lstm_pretrain_finetune_seed42/generated_t1.0_masked.fasta", note="辅助预训练、蓝藻微调、正则化和温度 1.0 的组合。"),
        metric_row(model="VAE", condition="自由采样 VAE", comparison_type="optimisation_package", metrics_path="runs/formal_vae_50ep_seed42/eval_profile_v2/metrics.json", fasta_path="runs/formal_vae_50ep_seed42/generated.fasta", note="原始 VAE，自由 EOS 结束。"),
        metric_row(model="VAE", condition="长度控制 VAE", comparison_type="optimisation_package", metrics_path="runs/formal_vae_50ep_seed42/eval_pf_empirical_s0.75_t0.6_masked/metrics.json", fasta_path="runs/formal_vae_50ep_seed42/generated_empirical_s0.75_t0.6_masked.fasta", note="经验长度目标、latent scale 0.75、temperature 0.6。"),
        metric_row(model="GAN", condition="原始 GAN", comparison_type="optimisation_package", metrics_path="runs/formal_gan_100ep_seed42/eval_pf_t0.1/metrics.json", fasta_path="runs/formal_gan_100ep_seed42/generated_t0.1.fasta", note="未加组成匹配和疏水 run 约束。"),
        metric_row(model="GAN", condition="约束 GAN", comparison_type="optimisation_package", metrics_path="runs/opt_gan_composition_hydro_seed42/eval_pf_t0.1_filtered/metrics.json", fasta_path="runs/opt_gan_composition_hydro_seed42/generated_t0.1_filtered.fasta", note="组成匹配、疏水 run 惩罚及最终过滤的组合。"),
    ]
    sampling_rows = [
        metric_row(model="LSTM", condition="T=0.8", comparison_type="sampling_control", metrics_path="runs/opt_lstm_pretrain_finetune_seed42/eval_pf_t0.8/metrics.json", fasta_path="runs/opt_lstm_pretrain_finetune_seed42/generated_t0.8.fasta", note="同一优化 checkpoint。"),
        metric_row(model="LSTM", condition="T=1.0", comparison_type="sampling_control", metrics_path="runs/opt_lstm_pretrain_finetune_seed42/eval_pf_t1.0_masked/metrics.json", fasta_path="runs/opt_lstm_pretrain_finetune_seed42/generated_t1.0_masked.fasta", note="同一优化 checkpoint；选定设置。"),
        metric_row(model="LSTM", condition="T=1.2", comparison_type="sampling_control", metrics_path="runs/opt_lstm_pretrain_finetune_seed42/eval_pf_t1.2/metrics.json", fasta_path="runs/opt_lstm_pretrain_finetune_seed42/generated_t1.2.fasta", note="同一优化 checkpoint。"),
        metric_row(model="VAE", condition="s=0.5, T=0.8", comparison_type="sampling_control", metrics_path="runs/formal_vae_50ep_seed42/eval_pf_empirical_s0.5_t0.8/metrics.json", fasta_path="runs/formal_vae_50ep_seed42/generated_empirical_s0.5_t0.8.fasta", note="同一 VAE checkpoint；经验长度控制。"),
        metric_row(model="VAE", condition="s=0.75, T=0.8", comparison_type="sampling_control", metrics_path="runs/formal_vae_50ep_seed42/eval_pf_empirical_s0.75_t0.8/metrics.json", fasta_path="runs/formal_vae_50ep_seed42/generated_empirical_s0.75_t0.8.fasta", note="同一 VAE checkpoint；经验长度控制。"),
        metric_row(model="VAE", condition="s=0.75, T=0.6", comparison_type="sampling_control", metrics_path="runs/formal_vae_50ep_seed42/eval_pf_empirical_s0.75_t0.6_masked/metrics.json", fasta_path="runs/formal_vae_50ep_seed42/generated_empirical_s0.75_t0.6_masked.fasta", note="同一 VAE checkpoint；选定设置。"),
        metric_row(model="VAE", condition="s=1.0, T=0.8", comparison_type="sampling_control", metrics_path="runs/formal_vae_50ep_seed42/eval_pf_empirical_s1.0_t0.8/metrics.json", fasta_path="runs/formal_vae_50ep_seed42/generated_empirical_s1.0_t0.8.fasta", note="同一 VAE checkpoint；经验长度控制。"),
        metric_row(model="GAN", condition="T=0.1", comparison_type="sampling_control", metrics_path="runs/opt_gan_composition_hydro_seed42/eval_pf_t0.1_filtered/metrics.json", fasta_path="runs/opt_gan_composition_hydro_seed42/generated_t0.1_filtered.fasta", note="同一约束 GAN checkpoint；选定设置及过滤。"),
        metric_row(model="GAN", condition="T=0.2", comparison_type="sampling_control", metrics_path="runs/opt_gan_composition_hydro_seed42/eval_pf_t0.2/metrics.json", fasta_path="runs/opt_gan_composition_hydro_seed42/generated_t0.2.fasta", note="同一约束 GAN checkpoint。"),
    ]
    write_csv(output_dir / "optimisation_package_comparison.csv", package_rows)
    write_csv(output_dir / "sampling_parameter_comparison.csv", sampling_rows)

    final_rows = [
        {"model": "LSTM", "profile_hit_rate": 0.98, "strict_novelty_all_165": 0.42, "pairwise_diversity": 0.131, "aa_composition_l1": 0.040},
        {"model": "VAE", "profile_hit_rate": 0.97, "strict_novelty_all_165": 0.62, "pairwise_diversity": 0.176, "aa_composition_l1": 0.098},
        {"model": "GAN", "profile_hit_rate": 1.00, "strict_novelty_all_165": 1.00, "pairwise_diversity": 0.157, "aa_composition_l1": 0.170},
    ]
    rq3_rows = [
        {"model": "LSTM", "mmd2_mean": 0.0264, "mmd2_min": 0.0221, "mmd2_max": 0.0330, "embedding_outlier_rate_mean": 0.0767, "embedding_outlier_rate_min": 0.06, "embedding_outlier_rate_max": 0.09},
        {"model": "VAE", "mmd2_mean": 0.2858, "mmd2_min": 0.2710, "mmd2_max": 0.3075, "embedding_outlier_rate_mean": 0.6667, "embedding_outlier_rate_min": 0.61, "embedding_outlier_rate_max": 0.70},
        {"model": "GAN", "mmd2_mean": 0.7395, "mmd2_min": 0.7216, "mmd2_max": 0.7587, "embedding_outlier_rate_mean": 0.9633, "embedding_outlier_rate_min": 0.95, "embedding_outlier_rate_max": 0.97},
    ]
    write_csv(output_dir / "final_model_tradeoff.csv", final_rows)
    write_csv(output_dir / "rq3_seed_summary.csv", rq3_rows)

    plot_package_comparisons(package_rows, output_dir / "fig_01_optimisation_package_comparison.png")
    plot_sampling_controls(sampling_rows, output_dir / "fig_02_sampling_parameter_controls.png")
    plot_final_tradeoff(final_rows, output_dir / "fig_03_final_three_objective_tradeoff.png")
    plot_rq3(rq3_rows, output_dir / "fig_04_rq3_distribution_summary.png")
    plot_lengths(output_dir / "fig_05_vae_length_control.png")
    print(f"Wrote tables and figures to {output_dir}")


if __name__ == "__main__":
    main()
