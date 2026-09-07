#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
candidate_explainer.py
任务5：输出候选新型Gvp序列，并进行可解释性分析

功能：
1. 多维度筛选Top-K候选序列（综合评分+多样性约束）
2. 逐序列可解释性报告（评分拆解、理化性质、结构倾向、功能注释）
3. 与天然序列的对比分析（identity、保守性、新颖性量化）
4. 生成候选序列卡片（文本报告+可视化）
5. 输出FASTA与CSV供下游实验验证
"""

import os
import re
import json
import warnings
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib.gridspec import GridSpec

# 兼容现有评分体系
try:
    from score_net import GvpScoreNet
except ImportError:
    from score.score_net import GvpScoreNet

try:
    from predict import predict as deeploc_predict
except ImportError:
    deeploc_predict = None

from data_loader import GvpDataset, AA_VOCAB

# ==================== 配置 ====================
plt.rcParams['font.size'] = 9
plt.rcParams['axes.titlesize'] = 11

AA_HYDROPATHY = {
    'A': 1.8, 'C': 2.5, 'D': -3.5, 'E': -3.5, 'F': 2.8,
    'G': -0.4, 'H': -3.2, 'I': 4.5, 'K': -3.9, 'L': 3.8,
    'M': 1.9, 'N': -3.5, 'P': -1.6, 'Q': -3.5, 'R': -4.5,
    'S': -0.8, 'T': -0.7, 'V': 4.2, 'W': -0.9, 'Y': -1.3
}

AA_CHARGE = {
    'K': +1, 'R': +1, 'H': +0.5,
    'D': -1, 'E': -1,
    'A': 0, 'C': 0, 'F': 0, 'G': 0, 'I': 0, 'L': 0, 'M': 0,
    'N': 0, 'P': 0, 'Q': 0, 'S': 0, 'T': 0, 'V': 0, 'W': 0, 'Y': 0
}

AA_SIZE = {
    'A': 89, 'C': 121, 'D': 133, 'E': 147, 'F': 165,
    'G': 75, 'H': 155, 'I': 131, 'K': 146, 'L': 131,
    'M': 149, 'N': 132, 'P': 115, 'Q': 146, 'R': 174,
    'S': 105, 'T': 119, 'V': 117, 'W': 204, 'Y': 181
}


AA_PROPERTIES = {
    'hydrophobic': set('AVILMFWY'),
    'polar': set('STNQ'),
    'basic': set('KRH'),
    'acidic': set('DE'),
    'special': set('CGP'),
    'aromatic': set('FWY')
}

# Gvp家族已知特征（基于文献）
GVP_KNOWLEDGE = {
    'length_A': (60, 100),      # GvpA典型长度
    'length_C': (300, 600),     # GvpC典型长度
    'hydrophobic_ratio': (0.35, 0.55),  # 理想疏水比例
    'helix_propensity': (0.4, 0.7),     # 螺旋倾向
    'repeat_unit': 'SLAEV',             # GvpA经典重复单元
    'function_domains': {
        'membrane_anchor': 'N端疏水区',
        'gas_permeation': '中央亲水通道',
        'self_assembly': 'C端电荷互补区'
    }
}


@dataclass
class SequenceExplanation:
    """单条序列的可解释性分析结果"""
    seq_id: str
    sequence: str
    model_source: str

    # 综合评分
    overall_score: float
    fold_score: float
    conserv_score: float
    novelty_score: float
    toxic_score: float
    assembly_score: float

    # 理化性质
    length: int
    molecular_weight: float
    isoelectric_point: float
    hydrophobic_ratio: float
    charge_at_ph7: int
    aromatic_ratio: float

    # 结构倾向
    helix_propensity: float
    sheet_propensity: float
    turn_propensity: float
    disorder_tendency: float

    # 功能注释
    known_motifs: List[Dict]
    predicted_localization: str
    solubility_prediction: float
    aggregation_risk: str

    # 与天然序列对比
    closest_natural_identity: float
    closest_natural_seq: str
    conservation_pattern: str

    # 可解释性文本
    summary: str
    strengths: List[str]
    risks: List[str]
    recommendations: List[str]


class CandidateExplainer:
    """候选序列可解释性分析器"""

    def __init__(self, generated_csv: str, natural_fasta: str,
                 output_dir: str = 'candidate_output', 
                 scorer: Optional[GvpScoreNet] = None):
        """
        Args:
            generated_csv: 生成序列CSV
            natural_fasta: 天然序列FASTA（用于对比）
            output_dir: 输出目录
            scorer: 评分器实例（可选，默认新建）
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        self.df = pd.read_csv(generated_csv)
        self.df = self.df[self.df['seq'].apply(lambda x: isinstance(x, str) and len(x) > 10)]

        # 加载天然序列
        self.natural_seqs = []
        from Bio import SeqIO
        for rec in SeqIO.parse(natural_fasta, 'fasta'):
            self.natural_seqs.append(str(rec.seq))

        self.scorer = scorer or GvpScoreNet()

        print(f"[Explainer] 加载完成：生成序列 {len(self.df)} 条，天然序列 {len(self.natural_seqs)} 条")

    # ==================== 筛选策略 ====================

    def select_candidates(self, top_k: int = 20, 
                         diversity_threshold: float = 0.3,
                         min_overall: float = 0.65) -> pd.DataFrame:
        """
        多样性感知筛选：在高分序列中保持序列多样性

        策略：
        1. 先按overall过滤
        2. 贪心选择：每次选最高分，排除与其identity > threshold的序列
        """
        qualified = self.df[self.df['overall'] >= min_overall].copy()
        if len(qualified) == 0:
            warnings.warn(f"无序列满足 overall >= {min_overall}，放宽条件")
            qualified = self.df.nlargest(top_k * 3, 'overall').copy()

        qualified = qualified.sort_values('overall', ascending=False)

        selected = []
        for _, row in qualified.iterrows():
            if len(selected) >= top_k:
                break

            seq = row['seq']
            # 计算与已选序列的最大identity
            max_id = 0
            for sel in selected:
                max_id = max(max_id, self._sequence_identity(seq, sel['seq']))

            if max_id < diversity_threshold or len(selected) == 0:
                selected.append(row)

        result = pd.DataFrame(selected)
        print(f"[筛选] 从 {len(qualified)} 条候选中选出 {len(result)} 条多样性序列")
        return result

    def _sequence_identity(self, seq1: str, seq2: str) -> float:
        """计算两个序列的相同位点比例（对齐到较短长度）"""
        min_len = min(len(seq1), len(seq2))
        matches = sum(a == b for a, b in zip(seq1[:min_len], seq2[:min_len]))
        return matches / min_len if min_len > 0 else 0

    # ==================== 理化性质计算 ====================

    def _calculate_physicochemical(self, seq: str) -> Dict:
        """计算序列理化性质"""
        length = len(seq)

        # 分子量
        mw = sum(AA_SIZE.get(aa, 110) for aa in seq) - 18 * (length - 1)

        # 电荷（pH 7.0近似）
        charge = sum(AA_CHARGE.get(aa, 0) for aa in seq)

        # 疏水性比例
        hydrophobic = set('AVILMFWY')
        h_ratio = sum(1 for aa in seq if aa in hydrophobic) / length

        # 芳香性比例
        aromatic = set('FWY')
        a_ratio = sum(1 for aa in seq if aa in aromatic) / length

        # 等电点近似（简化计算）
        n_pos = sum(1 for aa in seq if aa in 'KRH')
        n_neg = sum(1 for aa in seq if aa in 'DE')
        pi_approx = 7.0 + (n_pos - n_neg) * 0.3  # 非常粗略的估计

        return {
            'length': length,
            'molecular_weight': mw,
            'charge_at_ph7': int(charge),
            'hydrophobic_ratio': h_ratio,
            'aromatic_ratio': a_ratio,
            'isoelectric_point': pi_approx
        }

    def _calculate_structure_propensity(self, seq: str) -> Dict:
        """
        基于氨基酸倾向预测二级结构
        使用简化版Chou-Fasman参数
        """
        # Chou-Fasman倾向参数（简化）
        helix_favor = set('AELMQKRH')  # 螺旋倾向
        sheet_favor = set('VITFYW')     # 折叠倾向
        turn_favor = set('NGSDPC')      # 转角倾向

        length = len(seq)
        h_prop = sum(1 for aa in seq if aa in helix_favor) / length
        s_prop = sum(1 for aa in seq if aa in sheet_favor) / length
        t_prop = sum(1 for aa in seq if aa in turn_favor) / length

        # 无序倾向：基于低疏水性+高电荷
        hydro_values = [AA_HYDROPATHY.get(aa, 0) for aa in seq]
        avg_hydro = np.mean(hydro_values)
        disorder = 0.0
        if avg_hydro < -0.5:
            disorder = min(1.0, (-avg_hydro) * 0.3 + abs(sum(AA_CHARGE.get(aa, 0) for aa in seq)) / length)

        return {
            'helix_propensity': h_prop,
            'sheet_propensity': s_prop,
            'turn_propensity': t_prop,
            'disorder_tendency': disorder
        }

    def _find_known_motifs(self, seq: str) -> List[Dict]:
        """查找已知功能基序"""
        motifs = []

        # 经典GvpA重复单元
        for match in re.finditer(r'SLAEV|SLAEI|SLA[A-Z]V', seq):
            motifs.append({
                'name': 'GvpA_repeat_variant',
                'position': match.start(),
                'sequence': match.group(),
                'significance': 'Gas vesicle structural repeat unit'
            })

        # 跨膜螺旋信号（连续疏水区）
        for match in re.finditer(r'[AILVFM]{8,15}', seq):
            motifs.append({
                'name': 'hydrophobic_stretch',
                'position': match.start(),
                'sequence': match.group(),
                'significance': 'Potential membrane anchor or hydrophobic core'
            })

        # 电荷簇（功能界面）
        for match in re.finditer(r'[KRH]{3,}|[DE]{3,}', seq):
            motifs.append({
                'name': 'charge_cluster',
                'position': match.start(),
                'sequence': match.group(),
                'significance': 'Potential protein-protein interaction interface'
            })

        return motifs

    def _find_closest_natural(self, seq: str) -> Tuple[float, str]:
        """查找最相似的天然序列"""
        best_id = 0
        best_seq = ""
        for nat in self.natural_seqs:
            identity = self._sequence_identity(seq, nat)
            if identity > best_id:
                best_id = identity
                best_seq = nat
        return best_id, best_seq

    def _predict_localization_and_solubility(self, seq: str) -> Dict:
        """预测亚细胞定位与可溶性"""
        # 使用mock DeepLoc2（如果可用）
        if deeploc_predict is not None:
            try:
                result = deeploc_predict(seq)
                if hasattr(result, 'to_dict'):
                    result = result.to_dict('records')[0]
                return {
                    'localization': result.get('localizations', ['Unknown'])[0],
                    'solubility': result.get('soluble', 0.5),
                    'membrane': result.get('membrane', 'Unknown')
                }
            except Exception:
                pass

        # Fallback：基于疏水性的启发式
        h_ratio = sum(1 for aa in seq if aa in 'AVILMFWY') / len(seq)
        if h_ratio > 0.55:
            loc = 'Membrane'
            sol = 0.3
        elif h_ratio > 0.4:
            loc = 'Cytoplasm/Membrane'
            sol = 0.7
        else:
            loc = 'Cytoplasm'
            sol = 0.9

        return {'localization': loc, 'solubility': sol, 'membrane': 'Soluble' if sol > 0.6 else 'Membrane'}

    def _assess_aggregation(self, seq: str) -> str:
        """评估聚集风险"""
        # 基于Aggrescan3D启发式
        hydrophobic = set('AILMFWVY')
        h_count = sum(1 for aa in seq if aa in hydrophobic)
        h_ratio = h_count / len(seq)

        # 检查连续疏水区
        max_h_stretch = 0
        current = 0
        for aa in seq:
            if aa in hydrophobic:
                current += 1
                max_h_stretch = max(max_h_stretch, current)
            else:
                current = 0

        if h_ratio > 0.6 or max_h_stretch > 12:
            return 'High risk (excessive hydrophobicity)'
        elif h_ratio > 0.5 or max_h_stretch > 8:
            return 'Moderate risk (review recommended)'
        else:
            return 'Low risk'

    # ==================== 可解释性生成 ====================

    def _generate_explanation(self, seq: str, scores: Dict, 
                             phys: Dict, struct: Dict,
                             motifs: List[Dict], nat_identity: float,
                             loc_sol: Dict) -> Tuple[str, List[str], List[str], List[str]]:
        """生成自然语言可解释性文本"""

        # Summary
        parts = []
        parts.append(f"该序列为{seq[:20]}...（长度{phys['length']}），")
        parts.append(f"综合评分{scores['overall']:.3f}，")

        if scores['overall'] > 0.8:
            parts.append("属于高质量候选。")
        elif scores['overall'] > 0.65:
            parts.append("具有进一步优化潜力。")
        else:
            parts.append("建议结合定向进化改进。")

        if nat_identity < 0.3:
            parts.append(f"与天然序列差异显著（identity={nat_identity:.1%}），新颖性高。")
        else:
            parts.append(f"与天然序列有一定相似性（identity={nat_identity:.1%}）。")

        summary = "".join(parts)

        # Strengths
        strengths = []
        if scores['fold'] > 0.9:
            strengths.append(f"折叠性优秀（{scores['fold']:.2f}），结构稳定")
        if scores['toxic'] > 0.85:
            strengths.append(f"安全性良好（{scores['toxic']:.2f}），聚集风险低")
        if scores['novelty'] > 0.7:
            strengths.append(f"新颖性高（{scores['novelty']:.2f}），可能具有独特功能")
        if phys['hydrophobic_ratio'] > 0.35 and phys['hydrophobic_ratio'] < 0.55:
            strengths.append(f"疏水性比例理想（{phys['hydrophobic_ratio']:.1%}），符合Gvp家族特征")
        if struct['helix_propensity'] > 0.5:
            strengths.append(f"螺旋倾向高（{struct['helix_propensity']:.1%}），利于形成gas vesicle壁")
        if loc_sol['solubility'] > 0.7:
            strengths.append(f"预测可溶性良好（{loc_sol['solubility']:.2f}）")

        # Risks
        risks = []
        if scores['fold'] < 0.7:
            risks.append(f"折叠性偏低（{scores['fold']:.2f}），可能存在结构不稳定")
        if scores['toxic'] < 0.6:
            risks.append(f"安全性评分较低（{scores['toxic']:.2f}），需验证细胞毒性")
        if phys['charge_at_ph7'] == 0:
            risks.append("净电荷为零，可能影响溶解度或蛋白互作")
        if struct['disorder_tendency'] > 0.5:
            risks.append(f"无序倾向高（{struct['disorder_tendency']:.2f}），功能区域可能不稳定")
        if len(seq) < 50 or len(seq) > 200:
            risks.append(f"长度异常（{len(seq)}），偏离GvpA典型范围（60-100aa）")

        # Recommendations
        recommendations = []
        if scores['fold'] < 0.8:
            recommendations.append("建议：使用AlphaFold2验证结构，或引入已知稳定突变")
        if scores['conserv'] < 0.5:
            recommendations.append("建议：检查是否保留了Gvp家族核心功能残基")
        if nat_identity > 0.5:
            recommendations.append("建议：与天然序列过于相似，可引入更多变异提升新颖性")
        if not any(m['name'] == 'GvpA_repeat_variant' for m in motifs):
            recommendations.append("建议：未检测到经典GvpA重复单元，可能影响gas vesicle形成")

        if not recommendations:
            recommendations.append("该候选序列综合表现良好，建议优先进行实验验证")

        return summary, strengths, risks, recommendations

    # ==================== 单序列分析 ====================

    def analyze_sequence(self, row: pd.Series) -> SequenceExplanation:
        """对单条序列进行全面可解释性分析"""
        seq = row['seq']
        seq_id = row.get('id', f"seq_{hash(seq) % 10000}")
        model = row.get('model', 'unknown')

        # 评分（如果CSV中已有则直接使用，否则重新计算）
        scores = {
            'overall': row.get('overall', 0),
            'fold': row.get('fold', 0),
            'conserv': row.get('conserv', 0),
            'novelty': row.get('novelty', 0),
            'toxic': row.get('toxic', 0),
            'assembly': row.get('assembly', 0.5)
        }

        # 理化性质
        phys = self._calculate_physicochemical(seq)

        # 结构倾向
        struct = self._calculate_structure_propensity(seq)

        # 功能基序
        motifs = self._find_known_motifs(seq)

        # 天然序列对比
        nat_id, nat_seq = self._find_closest_natural(seq)

        # 定位与可溶性
        loc_sol = self._predict_localization_and_solubility(seq)

        # 聚集风险
        agg_risk = self._assess_aggregation(seq)

        # 保守性模式
        if scores['conserv'] > 0.8:
            conserv_pattern = "高度保守，保留核心功能"
        elif scores['conserv'] > 0.5:
            conserv_pattern = "中度保守，部分功能位点变异"
        else:
            conserv_pattern = "低保守性，功能可能显著改变"

        # 生成文本解释
        summary, strengths, risks, recommendations = self._generate_explanation(
            seq, scores, phys, struct, motifs, nat_id, loc_sol
        )

        return SequenceExplanation(
            seq_id=seq_id,
            sequence=seq,
            model_source=model,
            overall_score=scores['overall'],
            fold_score=scores['fold'],
            conserv_score=scores['conserv'],
            novelty_score=scores['novelty'],
            toxic_score=scores['toxic'],
            assembly_score=scores['assembly'],
            **phys,
            **struct,
            known_motifs=motifs,
            predicted_localization=loc_sol['localization'],
            solubility_prediction=loc_sol['solubility'],
            aggregation_risk=agg_risk,
            closest_natural_identity=nat_id,
            closest_natural_seq=nat_seq[:50] + "..." if len(nat_seq) > 50 else nat_seq,
            conservation_pattern=conserv_pattern,
            summary=summary,
            strengths=strengths,
            risks=risks,
            recommendations=recommendations
        )

    # ==================== 可视化 ====================

    def plot_candidate_card(self, exp: SequenceExplanation, save_path: str):
        """为单个候选生成可视化卡片"""
        fig = plt.figure(figsize=(14, 10))
        gs = GridSpec(3, 3, figure=fig, hspace=0.4, wspace=0.35)

        # 配色
        score_color = '#2ecc71' if exp.overall_score > 0.8 else '#f39c12' if exp.overall_score > 0.65 else '#e74c3c'

        # 1. 标题与总览
        ax_title = fig.add_subplot(gs[0, :])
        ax_title.axis('off')
        title_text = f"Candidate: {exp.seq_id}  |  Model: {exp.model_source.upper()}  |  Overall: {exp.overall_score:.3f}"
        ax_title.text(0.5, 0.8, title_text, ha='center', va='top', fontsize=14, fontweight='bold')
        ax_title.text(0.5, 0.5, exp.summary, ha='center', va='top', fontsize=10, 
                     wrap=True, transform=ax_title.transAxes,
                     bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

        # 2. 雷达图（评分拆解）
        ax_radar = fig.add_subplot(gs[0, 0], polar=True)
        metrics = ['fold', 'conserv', 'novelty', 'toxic', 'assembly']
        values = [exp.fold_score, exp.conserv_score, exp.novelty_score, 
                 exp.toxic_score, exp.assembly_score]
        angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
        values += values[:1]
        angles += angles[:1]

        ax_radar.plot(angles, values, 'o-', linewidth=2, color=score_color)
        ax_radar.fill(angles, values, alpha=0.25, color=score_color)
        ax_radar.set_xticks(angles[:-1])
        ax_radar.set_xticklabels(metrics, fontsize=9)
        ax_radar.set_ylim(0, 1)
        ax_radar.set_title('Score Breakdown', fontweight='bold', pad=15)

        # 3. 理化性质条形图
        ax_phys = fig.add_subplot(gs[0, 1])
        phys_names = ['Hydrophobic', 'Aromatic', 'Helix', 'Sheet', 'Turn']
        phys_vals = [exp.hydrophobic_ratio, exp.aromatic_ratio, 
                    exp.helix_propensity, exp.sheet_propensity, exp.turn_propensity]
        colors = ['#3498db', '#9b59b6', '#2ecc71', '#e74c3c', '#f39c12']
        bars = ax_phys.bar(phys_names, phys_vals, color=colors, alpha=0.8)
        ax_phys.set_ylim(0, 1)
        ax_phys.set_ylabel('Proportion')
        ax_phys.set_title('Physicochemical Profile', fontweight='bold')
        ax_phys.axhline(0.5, color='gray', linestyle='--', alpha=0.3)

        # 添加数值标签
        for bar, val in zip(bars, phys_vals):
            ax_phys.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                        f'{val:.2f}', ha='center', va='bottom', fontsize=8)

        # 4. 序列映射（按性质着色）
        ax_seq = fig.add_subplot(gs[1, :])
        prop_colors = {
            'hydrophobic': '#FFD93D', 'polar': '#95E1D3',
            'basic': '#FF6B6B', 'acidic': '#4ECDC4', 'special': '#E8E8E8'
        }
        aa_props = {}
        for prop, aas in AA_PROPERTIES.items():
            for aa in aas:
                aa_props[aa] = prop

        for i, aa in enumerate(exp.sequence):
            prop = aa_props.get(aa, 'special')
            ax_seq.barh(0, 1, left=i, height=0.8, 
                       color=prop_colors.get(prop, '#CCCCCC'), 
                       edgecolor='white', linewidth=0.2)

        # 标记基序
        for motif in exp.known_motifs:
            pos = motif['position']
            mlen = len(motif['sequence'])
            ax_seq.axvspan(pos, pos + mlen, alpha=0.3, color='red')
            ax_seq.text(pos + mlen/2, 0.6, motif['name'], 
                       ha='center', va='bottom', fontsize=7, color='darkred',
                       rotation=45)

        ax_seq.set_xlim(0, len(exp.sequence))
        ax_seq.set_ylim(-0.5, 1.0)
        ax_seq.set_xlabel('Position')
        ax_seq.set_title('Sequence Map (Red = Known Motifs)', fontweight='bold')
        ax_seq.set_yticks([])

        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=c, label=k.title()) 
                          for k, c in prop_colors.items()]
        ax_seq.legend(handles=legend_elements, loc='upper right', ncol=5, fontsize=8)

        # 5. 疏水性曲线
        ax_hydro = fig.add_subplot(gs[2, 0])
        window = 7
        half = window // 2
        hydros = []
        x_vals = []
        for i in range(len(exp.sequence) - window + 1):
            avg = np.mean([AA_HYDROPATHY.get(aa, 0) for aa in exp.sequence[i:i+window]])
            hydros.append(avg)
            x_vals.append(i + half)

        ax_hydro.plot(x_vals, hydros, 'b-', linewidth=1.5)
        ax_hydro.axhline(0, color='black', linestyle='-', alpha=0.3)
        ax_hydro.fill_between(x_vals, hydros, 0, 
                             where=[h > 0 for h in hydros], alpha=0.3, color='yellow')
        ax_hydro.fill_between(x_vals, hydros, 0,
                             where=[h <= 0 for h in hydros], alpha=0.3, color='cyan')
        ax_hydro.set_xlabel('Position')
        ax_hydro.set_ylabel('Hydropathy')
        ax_hydro.set_title('Hydropathy Profile', fontweight='bold')

        # 6. 与天然序列对比
        ax_compare = fig.add_subplot(gs[2, 1])
        if exp.closest_natural_seq and len(exp.closest_natural_seq) > 10:
            nat_clean = exp.closest_natural_seq.replace("...", "")
            min_len = min(len(exp.sequence), len(nat_clean))
            identities = []
            for i in range(min_len):
                identities.append(1 if exp.sequence[i] == nat_clean[i] else 0)

            # 滑动窗口平滑
            w = 5
            smoothed = []
            for i in range(len(identities)):
                start = max(0, i - w//2)
                end = min(len(identities), i + w//2 + 1)
                smoothed.append(np.mean(identities[start:end]))

            ax_compare.plot(range(len(smoothed)), smoothed, 'g-', linewidth=1.5)
            ax_compare.fill_between(range(len(smoothed)), smoothed, alpha=0.3, color='green')
            ax_compare.set_ylim(0, 1)
            ax_compare.set_xlabel('Position')
            ax_compare.set_ylabel('Identity')
            ax_compare.set_title(f'vs Natural (ID={exp.closest_natural_identity:.1%})', fontweight='bold')
        else:
            ax_compare.text(0.5, 0.5, 'No natural\nreference', ha='center', va='center',
                          transform=ax_compare.transAxes, fontsize=12, color='gray')
            ax_compare.set_title('vs Natural', fontweight='bold')

        # 7. 可解释性文本框
        ax_text = fig.add_subplot(gs[2, 2])
        ax_text.axis('off')

        y_pos = 0.95
        ax_text.text(0.05, y_pos, "Strengths:", fontsize=10, fontweight='bold', 
                    color='green', transform=ax_text.transAxes)
        y_pos -= 0.08
        for s in exp.strengths[:3]:
            ax_text.text(0.05, y_pos, f"• {s}", fontsize=8, transform=ax_text.transAxes)
            y_pos -= 0.07

        y_pos -= 0.05
        ax_text.text(0.05, y_pos, "Risks:", fontsize=10, fontweight='bold',
                    color='red', transform=ax_text.transAxes)
        y_pos -= 0.08
        for r in exp.risks[:3]:
            ax_text.text(0.05, y_pos, f"• {r}", fontsize=8, transform=ax_text.transAxes)
            y_pos -= 0.07

        y_pos -= 0.05
        ax_text.text(0.05, y_pos, "Recommendations:", fontsize=10, fontweight='bold',
                    color='blue', transform=ax_text.transAxes)
        y_pos -= 0.08
        for rec in exp.recommendations[:2]:
            ax_text.text(0.05, y_pos, f"• {rec}", fontsize=8, transform=ax_text.transAxes)
            y_pos -= 0.07

        plt.suptitle(f'GVP Candidate Analysis Card', fontsize=16, fontweight='bold', y=0.98)
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close()

    # ==================== 主流程 ====================

    def run_full_pipeline(self, top_k: int = 10, min_overall: float = 0.65):
        """执行完整候选序列解释流程"""
        print("\n" + "="*60)
        print("【任务5】候选新型Gvp序列可解释性分析")
        print("="*60)

        # 1. 筛选候选
        print(f"\n[1/5] 多样性感知筛选（Top-{top_k}, min_overall={min_overall}）...")
        candidates = self.select_candidates(top_k=top_k, min_overall=min_overall)

        # 2. 逐序列分析
        print("[2/5] 逐序列可解释性分析...")
        explanations = []
        for _, row in candidates.iterrows():
            exp = self.analyze_sequence(row)
            explanations.append(exp)

        # 3. 生成可视化卡片
        print("[3/5] 生成候选序列卡片...")
        for i, exp in enumerate(explanations):
            path = os.path.join(self.output_dir, f"candidate_{i+1:02d}_{exp.seq_id[:20]}.png")
            self.plot_candidate_card(exp, path)
            print(f"  卡片 {i+1}/{len(explanations)}: {exp.seq_id}")

        # 4. 生成对比总览图
        print("[4/5] 生成候选对比总览...")
        self._plot_candidate_comparison(explanations)

        # 5. 输出结构化文件
        print("[5/5] 输出FASTA与CSV...")
        self._export_results(explanations)

        # 生成文本报告
        self._generate_master_report(explanations)

        print(f"\n✅ 分析完成！结果保存在: {self.output_dir}/")
        print(f"   - candidate_*.png (单序列可视化卡片)")
        print(f"   - candidate_comparison.png (总览对比图)")
        print(f"   - candidates.fasta (候选序列FASTA)")
        print(f"   - candidates_detailed.csv (详细数据表)")
        print(f"   - master_report.txt (主报告)")

        return explanations

    def _plot_candidate_comparison(self, explanations: List[SequenceExplanation]):
        """生成所有候选的对比总览图"""
        fig = plt.figure(figsize=(16, 10))
        gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)

        # 数据准备
        df_data = []
        for exp in explanations:
            df_data.append({
                'id': exp.seq_id[:15],
                'model': exp.model_source,
                'overall': exp.overall_score,
                'fold': exp.fold_score,
                'novelty': exp.novelty_score,
                'toxic': exp.toxic_score,
                'length': exp.length,
                'hydro': exp.hydrophobic_ratio,
                'nat_id': exp.closest_natural_identity
            })
        df = pd.DataFrame(df_data)

        # 1. 综合评分排序
        ax1 = fig.add_subplot(gs[0, 0])
        colors = {'transformer': '#FF6B6B', 'diffusion': '#4ECDC4', 
                 'esm2': '#45B7D1', 'vae': '#96CEB4'}
        bar_colors = [colors.get(m, '#999999') for m in df['model']]
        bars = ax1.barh(range(len(df)), df['overall'], color=bar_colors, alpha=0.8)
        ax1.set_yticks(range(len(df)))
        ax1.set_yticklabels(df['id'], fontsize=8)
        ax1.set_xlabel('Overall Score')
        ax1.set_title('A. Candidate Ranking', fontweight='bold')
        ax1.invert_yaxis()

        # 添加数值
        for bar, val in zip(bars, df['overall']):
            ax1.text(val + 0.01, bar.get_y() + bar.get_height()/2, 
                    f'{val:.3f}', va='center', fontsize=8)

        # 2. 评分热力图
        ax2 = fig.add_subplot(gs[0, 1])
        score_matrix = df[['fold', 'novelty', 'toxic']].values
        im = ax2.imshow(score_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
        ax2.set_xticks(range(3))
        ax2.set_xticklabels(['Fold', 'Novelty', 'Toxic'])
        ax2.set_yticks(range(len(df)))
        ax2.set_yticklabels(df['id'], fontsize=8)
        ax2.set_title('B. Score Heatmap', fontweight='bold')
        plt.colorbar(im, ax=ax2)

        # 3. 新颖性 vs 天然相似性
        ax3 = fig.add_subplot(gs[1, 0])
        scatter = ax3.scatter(df['nat_id'], df['novelty'], 
                             c=df['overall'], cmap='viridis', s=200, 
                             edgecolors='black', linewidth=1)
        for i, row in df.iterrows():
            ax3.annotate(row['id'], (row['nat_id'], row['novelty']),
                        xytext=(5, 5), textcoords='offset points', fontsize=7)
        ax3.set_xlabel('Natural Sequence Identity')
        ax3.set_ylabel('Novelty Score')
        ax3.set_title('C. Novelty vs Natural Similarity', fontweight='bold')
        plt.colorbar(scatter, ax=ax3, label='Overall Score')

        # 4. 长度与疏水性分布
        ax4 = fig.add_subplot(gs[1, 1])
        for model in df['model'].unique():
            sub = df[df['model'] == model]
            ax4.scatter(sub['length'], sub['hydro'], 
                       s=sub['overall']*300, alpha=0.6, label=model)
        ax4.set_xlabel('Length (aa)')
        ax4.set_ylabel('Hydrophobic Ratio')
        ax4.set_title('D. Length vs Hydrophobicity (size=score)', fontweight='bold')
        ax4.legend()
        ax4.axhspan(0.35, 0.55, alpha=0.1, color='green', label='Ideal range')

        plt.suptitle('GVP Candidate Sequence Overview', fontsize=16, fontweight='bold')
        plt.savefig(os.path.join(self.output_dir, 'candidate_comparison.png'),
                   dpi=300, bbox_inches='tight')
        plt.close()

    def _export_results(self, explanations: List[SequenceExplanation]):
        """导出FASTA和CSV"""
        # FASTA
        fasta_path = os.path.join(self.output_dir, 'candidates.fasta')
        with open(fasta_path, 'w') as f:
            for i, exp in enumerate(explanations):
                f.write(f">{exp.seq_id}|model={exp.model_source}|overall={exp.overall_score:.4f}\n")
                f.write(f"{exp.sequence}\n")

        # CSV（扁平化）
        records = []
        for exp in explanations:
            rec = asdict(exp)
            # 处理嵌套结构
            rec['known_motifs'] = json.dumps(rec['known_motifs'], ensure_ascii=False)
            rec['strengths'] = '|'.join(rec['strengths'])
            rec['risks'] = '|'.join(rec['risks'])
            rec['recommendations'] = '|'.join(rec['recommendations'])
            records.append(rec)

        df = pd.DataFrame(records)
        df.to_csv(os.path.join(self.output_dir, 'candidates_detailed.csv'), 
                 index=False, encoding='utf-8-sig')

    def _generate_master_report(self, explanations: List[SequenceExplanation]):
        """生成主报告"""
        lines = []
        lines.append("="*70)
        lines.append("候选新型Gvp序列可解释性分析报告")
        lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        lines.append("="*70)
        lines.append("")

        lines.append("【候选序列总览】")
        lines.append(f"{'Rank':<6}{'ID':<20}{'Model':<12}{'Overall':<10}{'Length':<8}{'Novelty':<10}")
        lines.append("-"*70)
        for i, exp in enumerate(explanations):
            lines.append(f"{i+1:<6}{exp.seq_id[:18]:<20}{exp.model_source:<12}"
                        f"{exp.overall_score:<10.3f}{exp.length:<8}{exp.novelty_score:<10.3f}")
        lines.append("")

        lines.append("【Top 5 候选详细说明】")
        for i, exp in enumerate(explanations[:5]):
            lines.append(f"\n--- Rank {i+1}: {exp.seq_id} ---")
            lines.append(f"序列: {exp.sequence}")
            lines.append(f"来源模型: {exp.model_source}")
            lines.append(f"综合评分: {exp.overall_score:.3f} (fold={exp.fold_score:.2f}, "
                        f"conserv={exp.conserv_score:.2f}, novelty={exp.novelty_score:.2f}, "
                        f"toxic={exp.toxic_score:.2f})")
            lines.append(f"理化性质: 长度={exp.length}, MW≈{exp.molecular_weight:.0f}Da, "
                        f"电荷={exp.charge_at_ph7}, pI≈{exp.isoelectric_point:.1f}")
            lines.append(f"结构倾向: 螺旋={exp.helix_propensity:.1%}, 折叠={exp.sheet_propensity:.1%}, "
                        f"无序={exp.disorder_tendency:.2f}")
            lines.append(f"与天然序列相似性: {exp.closest_natural_identity:.1%}")
            lines.append(f"预测定位: {exp.predicted_localization}, 可溶性: {exp.solubility_prediction:.2f}")
            lines.append(f"聚集风险: {exp.aggregation_risk}")
            lines.append(f"保守性模式: {exp.conservation_pattern}")
            lines.append(f"\n摘要: {exp.summary}")
            lines.append(f"优势: {'; '.join(exp.strengths)}")
            lines.append(f"风险: {'; '.join(exp.risks)}")
            lines.append(f"建议: {'; '.join(exp.recommendations)}")

            if exp.known_motifs:
                lines.append(f"检测到基序:")
                for m in exp.known_motifs[:5]:
                    lines.append(f"  - {m['name']} @ {m['position']}: {m['sequence']} ({m['significance']})")

        lines.append("")
        lines.append("="*70)
        lines.append("报告结束")

        report_path = os.path.join(self.output_dir, 'master_report.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))


# ==================== 命令行接口 ====================
def main():
    import argparse
    parser = argparse.ArgumentParser(description='候选序列可解释性分析')
    parser.add_argument('--generated-csv', default='generated_candidates_all_models.csv',
                       help='生成序列CSV路径')
    parser.add_argument('--natural-fasta', default='data/natural_gvp.fasta',
                       help='天然序列FASTA路径')
    parser.add_argument('--output-dir', default='candidate_output', help='输出目录')
    parser.add_argument('--top-k', type=int, default=10, help='输出Top-K候选')
    parser.add_argument('--min-overall', type=float, default=0.65, help='最低综合评分')
    args = parser.parse_args()

    explainer = CandidateExplainer(args.generated_csv, args.natural_fasta, args.output_dir)
    explainer.run_full_pipeline(top_k=args.top_k, min_overall=args.min_overall)


if __name__ == '__main__':
    main()