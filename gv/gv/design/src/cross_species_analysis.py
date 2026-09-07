#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
cross_species_analysis.py
任务4：分析生成序列的跨物种保守性与潜在新功能区域

功能：
1. 多物种生成序列的MSA构建与位置特异性保守性分析
2. 跨物种保守区域（功能核心）与物种特异性变异（适应性进化）检测
3. 潜在新功能区域预测（基于理化性质突变、变异热点、结构无序区预测）
4. 输出可视化图表与结构化报告
"""

import os
import re
import warnings
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import entropy
from Bio import SeqIO, AlignIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

# 兼容你的数据体系
from data_loader import GvpDataset, AA_VOCAB

# ==================== 配置 ====================
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12

# 氨基酸理化性质表（用于功能推断）
AA_PROPERTIES = {
    'hydrophobic': set('AVILMFWY'),
    'polar': set('STNQ'),
    'basic': set('KRH'),
    'acidic': set('DE'),
    'special': set('CGP'),
    'aromatic': set('FWY')
}

AA_HYDROPATHY = {
    'A': 1.8, 'C': 2.5, 'D': -3.5, 'E': -3.5, 'F': 2.8,
    'G': -0.4, 'H': -3.2, 'I': 4.5, 'K': -3.9, 'L': 3.8,
    'M': 1.9, 'N': -3.5, 'P': -1.6, 'Q': -3.5, 'R': -4.5,
    'S': -0.8, 'T': -0.7, 'V': 4.2, 'W': -0.9, 'Y': -1.3
}

# Gvp家族已知功能基序（文献参考）
KNOWN_MOTIFS = {
    'GvpA_core': r'[AILV]S[AILV]E[AILV]',      # 疏水核心重复单元
    'Alpha_helix_capping': r'[STNQ][AILV]{3,5}[KRH]',  # 螺旋帽
    'Lipid_binding': r'[AILV]{4,6}[FWY]',           # 脂质结合倾向
    'Self_assembly': r'[AILV]{3}[STNQ][AILV]{3}',   # 自组装界面
}


@dataclass
class PositionStats:
    """单个位点的跨物种统计"""
    position: int
    consensus: str
    entropy: float
    occupancy: Dict[str, int]  # aa -> count
    species_variants: Dict[str, str]  # species -> aa
    property_shift: str  # 理化性质变化描述
    is_conserved: bool
    is_hypervariable: bool
    is_novel_pattern: bool


class CrossSpeciesAnalyzer:
    """跨物种保守性与新功能区域分析器"""

    def __init__(self, generated_csv: str, natural_fasta: str, 
                 msa_file: Optional[str] = None, output_dir: str = 'analysis_output'):
        """
        Args:
            generated_csv: generate_and_evaluate.py 输出的CSV路径（如 generated_candidates_all_models.csv）
            natural_fasta: 天然GVP序列FASTA路径
            msa_file: 天然MSA路径（可选，用于对比）
            output_dir: 输出目录
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # 加载生成序列
        self.gen_df = pd.read_csv(generated_csv)
        # 过滤有效序列
        self.gen_df = self.gen_df[self.gen_df['seq'].apply(lambda x: isinstance(x, str) and len(x) > 10)]

        # 加载天然序列
        self.natural_seqs = []
        self.natural_species = []
        for rec in SeqIO.parse(natural_fasta, 'fasta'):
            self.natural_seqs.append(str(rec.seq))
            sp = self._parse_species_from_header(rec.id)
            self.natural_species.append(sp)

        # 加载天然MSA（如有）
        self.natural_msa = None
        if msa_file and os.path.exists(msa_file):
            self.natural_msa = AlignIO.read(msa_file, 'fasta')

        # 按物种分组
        self.species_groups = self._group_by_species()

        print(f"[Analyzer] 加载完成：生成序列 {len(self.gen_df)} 条，"
              f"天然序列 {len(self.natural_seqs)} 条，"
              f"涉及 {len(self.species_groups)} 个物种/条件")

    def _parse_species_from_header(self, header: str) -> str:
        """从header解析物种，与data_loader.py保持一致"""
        if header.startswith('sp|') or header.startswith('tr|'):
            parts = header.split('|')[-1].split('_')
            return parts[-1] if len(parts) > 1 else 'unknown'
        parts = header.split('_')
        if len(parts) >= 2:
            return '_'.join(parts[1:])
        return 'unknown'

    def _group_by_species(self) -> Dict[str, List[str]]:
        """将生成序列按模型/条件分组（模拟跨物种）"""
        groups = defaultdict(list)

        # 优先使用CSV中的model列作为"物种/谱系"标识
        if 'model' in self.gen_df.columns:
            for _, row in self.gen_df.iterrows():
                groups[row['model']].append(row['seq'])
        else:
            groups['generated'] = self.gen_df['seq'].tolist()

        # 同时加入天然序列作为对照组
        for sp, seq in zip(self.natural_species, self.natural_seqs):
            groups[f"natural_{sp}"].append(seq)

        return dict(groups)

    def build_msa(self, sequences: List[str], ids: List[str], 
                  method: str = 'simple') -> List[SeqRecord]:
        """
        构建简化MSA（截断到最短长度）
        实际项目中建议替换为MUSCLE/MAFFT调用
        """
        if not sequences:
            return []

        if method == 'simple':
            min_len = min(len(s) for s in sequences)
            aligned = []
            for seq, sid in zip(sequences, ids):
                aligned.append(SeqRecord(Seq(seq[:min_len]), id=sid, description=""))
            return aligned
        else:
            raise NotImplementedError("请安装MUSCLE或MAFFT进行真实比对")

    def calculate_position_entropy(self, aligned_seqs: List[str]) -> List[PositionStats]:
        """
        计算每个位置的Shannon熵和保守性统计
        """
        if not aligned_seqs:
            return []

        length = len(aligned_seqs[0])
        n_seqs = len(aligned_seqs)
        stats = []

        for i in range(length):
            col = [seq[i] for seq in aligned_seqs if i < len(seq)]
            col = [aa for aa in col if aa != '-']
            if not col:
                continue

            counts = Counter(col)
            total = sum(counts.values())
            freqs = np.array([counts.get(aa, 0) / total for aa in 'ACDEFGHIKLMNPQRSTVWY'])
            freqs = freqs[freqs > 0]

            # Shannon熵（0 = 完全保守，~4.32 = 完全随机）
            H = entropy(freqs, base=2) if len(freqs) > 0 else 0

            # 共识氨基酸
            consensus = counts.most_common(1)[0][0]

            # 理化性质变化分析
            prop_shift = self._analyze_property_shift(counts)

            # 判定类别
            is_conserved = H < 0.5 and counts.most_common(1)[0][1] / total > 0.9
            is_hypervariable = H > 2.5

            # 新功能区域：高变异但具有特定理化性质约束（非随机）
            is_novel = (1.0 < H < 2.5) and self._is_functionally_constrained(counts)

            stats.append(PositionStats(
                position=i,
                consensus=consensus,
                entropy=float(H),
                occupancy=dict(counts),
                species_variants={},  # 后续填充
                property_shift=prop_shift,
                is_conserved=is_conserved,
                is_hypervariable=is_hypervariable,
                is_novel_pattern=is_novel
            ))

        return stats

    def _analyze_property_shift(self, counts: Counter) -> str:
        """分析该位点的理化性质主导类型"""
        total = sum(counts.values())
        prop_scores = {}
        for prop_name, aas in AA_PROPERTIES.items():
            score = sum(counts.get(aa, 0) for aa in aas) / total
            prop_scores[prop_name] = score

        dominant = max(prop_scores, key=prop_scores.get)
        score = prop_scores[dominant]

        if score > 0.7:
            return f"strong_{dominant}"
        elif score > 0.4:
            return f"mixed_{dominant}"
        else:
            return "balanced"

    def _is_functionally_constrained(self, counts: Counter) -> bool:
        """判断变异是否受功能约束（非完全随机）"""
        total = sum(counts.values())
        # 检查是否所有变异氨基酸具有相似理化性质（如都是疏水）
        hydropathy_values = [AA_HYDROPATHY.get(aa, 0) for aa in counts.keys()]
        if len(hydropathy_values) > 1:
            hydropathy_std = np.std(hydropathy_values)
            # 低标准差 = 理化性质保守，即使序列不保守
            if hydropathy_std < 1.5:
                return True

        # 检查电荷守恒
        charged = set('KRHDE')
        charged_ratio = sum(counts.get(aa, 0) for aa in charged) / total
        if charged_ratio > 0.8 or charged_ratio < 0.2:
            return True

        return False

    def find_conserved_blocks(self, stats: List[PositionStats], 
                              min_len: int = 5) -> List[Tuple[int, int, str]]:
        """查找连续保守区块 [start, end, type]"""
        blocks = []
        current_start = None
        current_type = None

        for st in stats:
            pos = st.position
            if st.is_conserved:
                btype = "strict_conserved"
            elif st.is_novel_pattern:
                btype = "functional_constrained"
            else:
                btype = None

            if btype:
                if current_start is None:
                    current_start = pos
                    current_type = btype
                elif current_type != btype:
                    if pos - current_start >= min_len:
                        blocks.append((current_start, pos - 1, current_type))
                    current_start = pos
                    current_type = btype
            else:
                if current_start is not None and pos - current_start >= min_len:
                    blocks.append((current_start, pos - 1, current_type))
                current_start = None
                current_type = None

        # 处理末尾
        if current_start is not None and len(stats) - current_start >= min_len:
            blocks.append((current_start, len(stats) - 1, current_type))

        return blocks

    def predict_functional_regions(self, stats: List[PositionStats], 
                                   seq_example: str) -> Dict[str, List[dict]]:
        """
        预测潜在新功能区域
        基于：变异模式 + 已知motif匹配 + 理化性质异常
        """
        regions = {
            'cross_species_conserved': [],    # 跨物种保守 = 功能核心
            'species_specific_adaptive': [],   # 物种特异 = 适应性
            'novel_functional_candidate': [],  # 潜在新功能
            'disordered_prone': []             # 无序区（柔性，可能参与互作）
        }

        # 1. 保守区块
        conserved_blocks = self.find_conserved_blocks(stats, min_len=4)
        for start, end, btype in conserved_blocks:
            seq_slice = seq_example[start:end+1]
            regions['cross_species_conserved'].append({
                'start': start, 'end': end,
                'type': btype,
                'sequence': seq_slice,
                'note': '跨物种高度保守，可能为结构/功能核心'
            })

        # 2. 物种特异性位点（仅在某一生成模型中出现）
        for st in stats:
            if st.entropy > 1.5 and st.entropy < 3.0:
                # 检查是否某类氨基酸占主导但非绝对保守
                top_aa, top_count = Counter(st.occupancy).most_common(1)[0]
                if 0.5 < top_count / sum(st.occupancy.values()) < 0.9:
                    regions['species_specific_adaptive'].append({
                        'position': st.position,
                        'consensus': st.consensus,
                        'entropy': st.entropy,
                        'note': '部分保守，可能存在物种特异性适应'
                    })

        # 3. 潜在新功能：理化性质剧变区
        for i in range(1, len(stats)):
            prev_prop = stats[i-1].property_shift
            curr_prop = stats[i].property_shift
            if prev_prop != curr_prop and 'strong' in curr_prop and stats[i].entropy > 1.0:
                regions['novel_functional_candidate'].append({
                    'position': stats[i].position,
                    'property_shift': f"{prev_prop} -> {curr_prop}",
                    'entropy': stats[i].entropy,
                    'note': '理化性质突变+中度变异，可能产生新功能界面'
                })

        # 4. 无序区预测：基于疏水性（低疏水性 = 高无序倾向）
        window = 15
        for i in range(len(seq_example) - window):
            slice_seq = seq_example[i:i+window]
            avg_hydro = np.mean([AA_HYDROPATHY.get(aa, 0) for aa in slice_seq])
            if avg_hydro < -1.0:  # 高亲水性 = 倾向无序
                # 检查是否已有记录
                if not any(r['start'] <= i <= r['end'] for r in regions['disordered_prone']):
                    regions['disordered_prone'].append({
                        'start': i, 'end': i + window,
                        'avg_hydropathy': avg_hydro,
                        'note': '低疏水性区域，可能为柔性环/互作界面'
                    })

        return regions

    def run_full_analysis(self, top_n: int = 100):
        """执行完整分析流程"""
        print("\n" + "="*60)
        print("【任务4】跨物种保守性与潜在新功能区域分析")
        print("="*60)

        # 1. 选取Top序列（按overall排序）
        top_df = self.gen_df.nlargest(top_n, 'overall')
        top_seqs = top_df['seq'].tolist()
        top_ids = [f"gen_{i}" for i in range(len(top_seqs))]

        # 2. 构建MSA
        print(f"\n[1/5] 构建MSA（{len(top_seqs)}条序列）...")
        msa_records = self.build_msa(top_seqs, top_ids)
        aligned = [str(rec.seq) for rec in msa_records]

        # 3. 位置特异性分析
        print("[2/5] 计算位置保守性统计...")
        stats = self.calculate_position_entropy(aligned)

        # 4. 功能区域预测
        print("[3/5] 预测功能区域...")
        example_seq = top_seqs[0] if top_seqs else ""
        regions = self.predict_functional_regions(stats, example_seq)

        # 5. 与天然序列对比（如果有）
        natural_stats = None
        if self.natural_seqs:
            print("[4/5] 天然序列对照分析...")
            nat_msa = self.build_msa(self.natural_seqs[:50], 
                                     [f"nat_{i}" for i in range(min(50, len(self.natural_seqs)))])
            natural_stats = self.calculate_position_entropy([str(rec.seq) for rec in nat_msa])

        # 6. 可视化与报告
        print("[5/5] 生成可视化与报告...")
        self._plot_conservation_landscape(stats, natural_stats, regions)
        report = self._generate_text_report(stats, regions, top_df)

        # 保存
        report_path = os.path.join(self.output_dir, 'cross_species_report.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)

        # 保存区域表格
        all_regions = []
        for rtype, items in regions.items():
            for item in items:
                item['region_type'] = rtype
                all_regions.append(item)

        if all_regions:
            reg_df = pd.DataFrame(all_regions)
            reg_df.to_csv(os.path.join(self.output_dir, 'functional_regions.csv'), index=False)

        print(f"\n✅ 分析完成！结果保存在: {self.output_dir}/")
        print(f"   - cross_species_report.txt (详细文本报告)")
        print(f"   - functional_regions.csv (功能区域表格)")
        print(f"   - conservation_landscape.png (保守性景观图)")

        return stats, regions

    def _plot_conservation_landscape(self, stats, natural_stats, regions):
        """绘制保守性景观图"""
        fig, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=True,
                                gridspec_kw={'height_ratios': [1, 1, 0.8]})

        positions = [s.position for s in stats]
        entropies = [s.entropy for s in stats]

        # 图1: 熵值曲线
        ax1 = axes[0]
        ax1.fill_between(positions, entropies, alpha=0.3, color='steelblue')
        ax1.plot(positions, entropies, color='steelblue', linewidth=1.5, label='Shannon Entropy')

        # 标记保守区
        for r in regions.get('cross_species_conserved', []):
            ax1.axvspan(r['start'], r['end'], alpha=0.2, color='green', 
                       label='Conserved Block' if r == regions['cross_species_conserved'][0] else "")

        # 标记新功能候选区
        for r in regions.get('novel_functional_candidate', []):
            ax1.axvline(r['position'], color='red', alpha=0.3, linestyle='--')

        ax1.axhline(0.5, color='green', linestyle='--', alpha=0.5, label='Strict threshold')
        ax1.axhline(2.5, color='orange', linestyle='--', alpha=0.5, label='Hypervariable threshold')
        ax1.set_ylabel('Entropy (bits)')
        ax1.set_title('A. Cross-Species Conservation Landscape', fontweight='bold')
        ax1.legend(loc='upper right', fontsize=8)
        ax1.set_ylim(0, max(entropies + [1]) * 1.1)

        # 图2: 与天然序列对比
        ax2 = axes[1]
        if natural_stats:
            nat_pos = [s.position for s in natural_stats[:len(stats)]]
            nat_ent = [s.entropy for s in natural_stats[:len(stats)]]
            ax2.plot(positions, entropies, color='steelblue', alpha=0.7, label='Generated')
            ax2.plot(nat_pos, nat_ent, color='gray', alpha=0.7, linestyle='--', label='Natural')
            ax2.fill_between(positions, 
                           [min(g, n) for g, n in zip(entropies, nat_ent)],
                           [max(g, n) for g, n in zip(entropies, nat_ent)],
                           alpha=0.2, color='purple', label='Divergence')
        else:
            ax2.plot(positions, entropies, color='steelblue', label='Generated')

        ax2.set_ylabel('Entropy (bits)')
        ax2.set_title('B. Generated vs Natural Conservation', fontweight='bold')
        ax2.legend()

        # 图3: 理化性质映射
        ax3 = axes[2]
        prop_colors = {
            'strong_hydrophobic': '#FFD93D',
            'strong_polar': '#95E1D3',
            'strong_basic': '#FF6B6B',
            'strong_acidic': '#4ECDC4',
            'mixed_hydrophobic': '#FFE66D',
            'balanced': '#E8E8E8'
        }

        for s in stats:
            color = prop_colors.get(s.property_shift, '#CCCCCC')
            ax3.barh(0, 1, left=s.position, height=0.8, color=color, alpha=0.8)

        ax3.set_xlim(0, len(stats))
        ax3.set_ylim(-0.5, 0.5)
        ax3.set_yticks([])
        ax3.set_xlabel('Position')
        ax3.set_title('C. Physicochemical Property Dominance', fontweight='bold')

        # 添加图例
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=c, label=k.replace('_', ' ').title()) 
                          for k, c in prop_colors.items() if any(s.property_shift == k for s in stats)]
        ax3.legend(handles=legend_elements, loc='upper right', ncol=3, fontsize=8)

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'conservation_landscape.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()

    def _generate_text_report(self, stats, regions, top_df) -> str:
        """生成详细文本报告"""
        lines = []
        lines.append("="*70)
        lines.append("跨物种保守性与潜在新功能区域分析报告")
        lines.append("="*70)
        lines.append("")

        # 统计摘要
        n_conserved = sum(1 for s in stats if s.is_conserved)
        n_hyper = sum(1 for s in stats if s.is_hypervariable)
        n_novel = sum(1 for s in stats if s.is_novel_pattern)

        lines.append(f"【统计摘要】")
        lines.append(f"  分析序列数: {len(top_df)}")
        lines.append(f"  比对长度: {len(stats)} aa")
        lines.append(f"  严格保守位点: {n_conserved} ({n_conserved/len(stats)*100:.1f}%)")
        lines.append(f"  高变位点: {n_hyper} ({n_hyper/len(stats)*100:.1f}%)")
        lines.append(f"  潜在新功能位点: {n_novel} ({n_novel/len(stats)*100:.1f}%)")
        lines.append("")

        # 保守区域
        lines.append("【跨物种保守区域（功能核心）】")
        if regions['cross_species_conserved']:
            for r in regions['cross_species_conserved'][:10]:
                lines.append(f"  位置 {r['start']}-{r['end']}: {r['type']}")
                lines.append(f"    序列: {r['sequence']}")
                lines.append(f"    注释: {r['note']}")
        else:
            lines.append("  未检测到显著保守区块")
        lines.append("")

        # 新功能候选
        lines.append("【潜在新功能区域】")
        if regions['novel_functional_candidate']:
            for r in regions['novel_functional_candidate'][:10]:
                lines.append(f"  位置 {r['position']}: {r['property_shift']}")
                lines.append(f"    熵值: {r['entropy']:.2f} bits")
                lines.append(f"    注释: {r['note']}")
        else:
            lines.append("  未检测到显著新功能候选区")
        lines.append("")

        # 无序区
        lines.append("【预测柔性/无序区域】")
        if regions['disordered_prone']:
            for r in regions['disordered_prone'][:5]:
                lines.append(f"  位置 {r['start']}-{r['end']}, 平均疏水性: {r['avg_hydropathy']:.2f}")
        lines.append("")

        # 位点明细
        lines.append("【Top 20 保守位点明细】")
        sorted_stats = sorted(stats, key=lambda x: x.entropy)[:20]
        for s in sorted_stats:
            occ = sorted(s.occupancy.items(), key=lambda x: x[1], reverse=True)[:3]
            occ_str = ", ".join([f"{aa}:{cnt}" for aa, cnt in occ])
            lines.append(f"  Pos {s.position:3d}: {s.consensus} | 熵={s.entropy:.2f} | {occ_str}")

        lines.append("")
        lines.append("="*70)
        lines.append("报告生成完成")

        return "\n".join(lines)


# ==================== 命令行接口 ====================
def main():
    import argparse
    parser = argparse.ArgumentParser(description='跨物种保守性分析')
    parser.add_argument('--generated-csv', default='generated_candidates_all_models.csv',
                       help='生成序列CSV路径')
    parser.add_argument('--natural-fasta', default='data/natural_gvp.fasta',
                       help='天然序列FASTA路径')
    parser.add_argument('--msa', default=None, help='天然MSA路径（可选）')
    parser.add_argument('--output-dir', default='analysis_output', help='输出目录')
    parser.add_argument('--top-n', type=int, default=100, help='分析Top N序列')
    args = parser.parse_args()

    analyzer = CrossSpeciesAnalyzer(
        args.generated_csv, args.natural_fasta, args.msa, args.output_dir
    )
    analyzer.run_full_analysis(top_n=args.top_n)


if __name__ == '__main__':
    main()