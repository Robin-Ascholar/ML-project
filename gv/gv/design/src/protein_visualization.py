"""
蛋白质生成模型结果可视化完整代码（修复版）
包含：统计图表、序列特征分析、交互式筛选工具
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from collections import Counter
import re
import warnings
from scipy import stats
from scipy.stats import gaussian_kde, entropy

warnings.filterwarnings('ignore')

# ==================== 配置 ====================
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 10

# 模型配色方案
MODEL_COLORS = {
    'transformer': '#FF6B6B',
    'diffusion': '#4ECDC4', 
    'esm2': '#45B7D1',
    'vae': '#96CEB4'
}

# 氨基酸物理化学性质分类
AA_PROPERTIES = {
    'hydrophobic': set('AVILMFWY'),
    'polar': set('STNQ'),
    'basic': set('KRH'),
    'acidic': set('DE'),
    'special': set('CGP')
}

# ==================== 数据加载 ====================
def load_data():
    """加载所有CSV文件"""
    files = {
        'all': 'generated_candidates_all_models.csv',
        'diffusion': 'generated_candidates_diffusion.csv',
        'esm2': 'generated_candidates_esm2.csv',
        'transformer': 'generated_candidates_transformer.csv',
        'vae': 'generated_candidates_vae.csv',
        'scored': 'generated_candidates_scored.csv'
    }
    dfs = {}
    for name, path in files.items():
        dfs[name] = pd.read_csv(path)
    return dfs

# ==================== 序列特征计算 ====================
def max_consecutive_repeat(seq):
    """计算最大连续重复长度"""
    max_repeat = current = 1
    for i in range(1, len(seq)):
        current = current + 1 if seq[i] == seq[i-1] else 1
        max_repeat = max(max_repeat, current)
    return max_repeat

def kmer_diversity(seq, k=3):
    """计算k-mer多样性"""
    kmers = [seq[i:i+k] for i in range(len(seq)-k+1)]
    return len(set(kmers)) / len(kmers) if kmers else 0

def get_aa_composition(seq):
    """计算氨基酸组成"""
    counts = Counter(seq)
    total = len(seq)
    return {aa: counts.get(aa, 0)/total for aa in 'ACDEFGHIKLMNPQRSTVWY'}

def get_blockiness(seq):
    """计算序列区块化程度"""
    hydrophobic = set('AVILMFWY')
    blocks = []
    current_type = 'H' if seq[0] in hydrophobic else 'P'
    current_len = 1
    
    for aa in seq[1:]:
        aa_type = 'H' if aa in hydrophobic else 'P'
        if aa_type == current_type:
            current_len += 1
        else:
            blocks.append(current_len)
            current_type = aa_type
            current_len = 1
    blocks.append(current_len)
    return max(blocks), np.mean(blocks), len(blocks)

def find_motif_positions(seq, motif):
    """查找基序所有出现位置"""
    return [m.start() for m in re.finditer(motif, seq)]

def calculate_complexity(seq, window=10):
    """计算局部复杂度（滑动窗口）"""
    complexities = []
    for i in range(len(seq) - window + 1):
        window_seq = seq[i:i+window]
        unique = len(set(window_seq))
        complexities.append(unique / window)
    return complexities

# ==================== 图1: 综合性能概览 ====================
def plot_overview(dfs, save_path='fig1_overview.png'):
    """生成模型综合性能对比图"""
    fig = plt.figure(figsize=(20, 12))
    gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    df_all = dfs['all']
    models = ['transformer', 'diffusion', 'esm2', 'vae']
    colors = [MODEL_COLORS[m] for m in models]
    
    # 1.1 Overall分布箱线图
    ax1 = fig.add_subplot(gs[0, 0])
    box_data = [df_all[df_all['model']==m]['overall'].values for m in models]
    bp = ax1.boxplot(box_data, labels=[m.capitalize() for m in models], 
                     patch_artist=True, notch=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax1.set_ylabel('Overall Score')
    ax1.set_title('A. Overall Score Distribution', fontweight='bold')
    ax1.axhline(y=0.7, color='red', linestyle='--', alpha=0.5)
    
    # 1.2 雷达图
    ax2 = fig.add_subplot(gs[0, 1], polar=True)
    metrics = ['fold', 'conserv', 'novelty', 'toxic']
    angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]
    
    for model, color in zip(models, colors):
        values = df_all[df_all['model']==model][metrics].mean().values.tolist()
        values += values[:1]
        ax2.plot(angles, values, 'o-', linewidth=2, color=color, label=model.capitalize())
        ax2.fill(angles, values, alpha=0.15, color=color)
    
    ax2.set_xticks(angles[:-1])
    ax2.set_xticklabels(metrics)
    ax2.set_ylim(0, 1)
    ax2.set_title('B. Average Metrics Radar', fontweight='bold', pad=20)
    ax2.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    
    # 1.3 Top 10排名
    ax3 = fig.add_subplot(gs[0, 2])
    top10 = df_all.nlargest(10, 'overall')
    bar_colors = [MODEL_COLORS[m] for m in top10['model']]
    ax3.barh(range(10), top10['overall'].values, color=bar_colors, alpha=0.8)
    ax3.set_yticks(range(10))
    ax3.set_yticklabels([f"{id[:15]}..." for id in top10['id']], fontsize=8)
    ax3.set_xlabel('Overall Score')
    ax3.set_title('C. Top 10 Sequences', fontweight='bold')
    ax3.invert_yaxis()
    
    # 1.4 Fold vs Novelty散点
    ax4 = fig.add_subplot(gs[1, 0])
    for model, color in zip(models, colors):
        data = df_all[df_all['model']==model]
        ax4.scatter(data['novelty'], data['fold'], c=color, alpha=0.6, s=50, 
                   label=model.capitalize(), edgecolors='white', linewidth=0.5)
    ax4.set_xlabel('Novelty')
    ax4.set_ylabel('Fold')
    ax4.set_title('D. Fold vs Novelty', fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 1.5 序列长度分布 - 改为箱线图
    ax5 = fig.add_subplot(gs[1, 1])
    box_data = [df_all[df_all['model']==m]['seq'].str.len().values for m in models]
    bp = ax5.boxplot(box_data, labels=[m.capitalize() for m in models], 
                     patch_artist=True, showfliers=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax5.set_ylabel('Sequence Length (aa)')
    ax5.set_title('E. Length Distribution', fontweight='bold')
    ax5.annotate('All models generate\nsimilar lengths', xy=(0.5, 0.95), 
                xycoords='axes fraction', ha='center', va='top',
                fontsize=9, style='italic', color='gray')
    
    # 1.6 指标分组柱状图（替代小提琴图）
    ax6 = fig.add_subplot(gs[1, 2])
    x = np.arange(len(metrics))
    width = 0.2
    
    for i, (model, color) in enumerate(zip(models, colors)):
        means = [df_all[df_all['model']==model][m].mean() for m in metrics]
        stds = [df_all[df_all['model']==model][m].std() for m in metrics]
        ax6.bar(x + i*width, means, width, yerr=stds, label=model.capitalize(),
                color=color, alpha=0.7, capsize=3)
    
    ax6.set_xticks(x + width * 1.5)
    ax6.set_xticklabels(metrics)
    ax6.set_ylabel('Score')
    ax6.set_title('F. Average Metrics by Model', fontweight='bold')
    ax6.legend(fontsize=8)
    ax6.set_ylim(0, 1)
    ax6.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Protein Generation Models: Performance Overview', 
                 fontsize=16, fontweight='bold', y=0.98)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    return fig

# ==================== 图2: 序列特征深度分析 ====================
def plot_sequence_features(dfs, save_path='fig2_sequence_features.png'):
    """生成序列特征分析图"""
    fig = plt.figure(figsize=(20, 14))
    gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.35)
    
    df_all = dfs['all']
    models = ['transformer', 'diffusion', 'esm2', 'vae']
    colors = [MODEL_COLORS[m] for m in models]
    
    # 预计算特征
    features = {}
    for model in models:
        df_model = df_all[df_all['model']==model]
        feats = {
            'lengths': df_model['seq'].str.len().values,
            'max_repeats': [max_consecutive_repeat(s) for s in df_model['seq']],
            'kmer_div': [kmer_diversity(s, 3) for s in df_model['seq']]
        }
        block_data = [get_blockiness(s) for s in df_model['seq']]
        feats['max_blocks'] = [b[0] for b in block_data]
        features[model] = feats
    
    # 2.1 连续重复长度 - 添加均值虚线，移除无意义x=10线
    ax1 = fig.add_subplot(gs[0, 0])
    for model, color in zip(models, colors):
        ax1.hist(features[model]['max_repeats'], bins=15, alpha=0.5, 
                color=color, label=model.capitalize(), density=True)
    
    for model, color in zip(models, colors):
        mean_repeat = np.mean(features[model]['max_repeats'])
        ax1.axvline(x=mean_repeat, color=color, linestyle='--', alpha=0.7, linewidth=1.5)
    
    ax1.set_xlabel('Max Consecutive Repeat')
    ax1.set_ylabel('Density')
    ax1.set_title('A. Consecutive Repeats', fontweight='bold')
    ax1.legend(fontsize=8)
    ax1.text(0.95, 0.95, 'Dashed lines = mean\nAll models: repeats < 5', 
             transform=ax1.transAxes, ha='right', va='top',
             fontsize=8, style='italic', color='gray',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 2.2 区块化程度
    ax2 = fig.add_subplot(gs[0, 1])
    box_data = [features[m]['max_blocks'] for m in models]
    bp = ax2.boxplot(box_data, labels=[m.capitalize() for m in models], patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax2.set_ylabel('Max Block Length')
    ax2.set_title('B. Hydrophobic/Philic Blocks', fontweight='bold')
    
    # 2.3 氨基酸组成热图 - 改为Z-score归一化
    ax3 = fig.add_subplot(gs[0, 2])
    aa_order = list('ACDEFGHIKLMNPQRSTVWY')
    comp_matrix = np.zeros((4, 20))
    for i, model in enumerate(models):
        all_seq = ''.join(df_all[df_all['model']==model]['seq'])
        total = len(all_seq)
        for j, aa in enumerate(aa_order):
            comp_matrix[i, j] = all_seq.count(aa) / total
    
    comp_matrix_z = stats.zscore(comp_matrix, axis=1)
    im = ax3.imshow(comp_matrix_z, cmap='RdBu_r', aspect='auto', vmin=-3, vmax=3)
    ax3.set_xticks(range(20))
    ax3.set_xticklabels(aa_order)
    ax3.set_yticks(range(4))
    ax3.set_yticklabels([m.capitalize() for m in models])
    ax3.set_title('C. AA Composition (Z-score)', fontweight='bold')
    cbar = plt.colorbar(im, ax=ax3, label='Z-score')
    cbar.ax.axhline(y=0, color='black', linewidth=0.5)
    
    # 标注显著偏离
    for i in range(4):
        for j in range(20):
            if abs(comp_matrix_z[i, j]) > 2:
                ax3.text(j, i, f'{comp_matrix_z[i,j]:.1f}', 
                        ha='center', va='center', fontsize=6,
                        color='white' if abs(comp_matrix_z[i,j]) > 2.5 else 'black')
    
    # 2.4 ESM2序列可视化
    ax4 = fig.add_subplot(gs[1, :2])
    esm2_top = dfs['esm2'].nlargest(3, 'overall')
    prop_colors = {'hydrophobic': '#FFD93D', 'basic': '#FF6B6B', 
                   'acidic': '#4ECDC4', 'polar': '#95E1D3', 'special': '#E8E8E8'}
    
    for idx, (_, row) in enumerate(esm2_top.iterrows()):
        y = idx * 3
        seq = row['seq']
        for j, aa in enumerate(seq):
            for prop, aas in AA_PROPERTIES.items():
                if aa in aas:
                    color = prop_colors[prop]
                    break
            ax4.barh(y, 1, left=j, height=0.8, color=color, edgecolor='white', linewidth=0.1)
        ax4.text(len(seq)+2, y, f"{row['overall']:.3f}", va='center', fontsize=9)
    
    ax4.set_xlim(0, 130)
    ax4.set_ylim(-1, 9)
    ax4.set_yticks([0, 3, 6])
    ax4.set_yticklabels(['ESM2 #1', 'ESM2 #2', 'ESM2 #3'])
    ax4.set_xlabel('Position')
    ax4.set_title('D. ESM2 Sequence Blockiness', fontweight='bold')
    
    legend_elements = [mpatches.Patch(color=c, label=k.capitalize()) 
                       for k, c in prop_colors.items()]
    ax4.legend(handles=legend_elements, loc='upper right', ncol=5, fontsize=8)
    
    # 2.5 VAE重复模式 - 添加未找到处理
    ax5 = fig.add_subplot(gs[1, 2])
    vae_seq = dfs['vae'].iloc[0]['seq'][:200]
    motif = "SLAEV"
    positions = find_motif_positions(vae_seq, motif)
    
    ax5.plot(range(len(vae_seq)), [1]*len(vae_seq), 'k-', alpha=0.3)
    
    if positions:
        for pos in positions:
            ax5.axvspan(pos, pos+len(motif), alpha=0.5, color='red')
        for i, pos in enumerate(positions[:3]):
            ax5.text(pos + len(motif)/2, 1.15, f'{motif}\n@{pos}', 
                    ha='center', va='bottom', fontsize=7, color='red')
    else:
        ax5.text(0.5, 0.5, f'Motif "{motif}"\nnot found', 
                transform=ax5.transAxes, ha='center', va='center',
                fontsize=12, color='gray', style='italic')
    
    ax5.set_xlim(0, 200)
    ax5.set_ylim(0.5, 1.5)
    ax5.set_xlabel('Position')
    ax5.set_title(f'E. VAE Motif "{motif}"', fontweight='bold')
    ax5.set_yticks([])
    
    # 2.6 K-mer多样性
    ax6 = fig.add_subplot(gs[2, 0])
    for model, color in zip(models, colors):
        ax6.hist(features[model]['kmer_div'], bins=15, alpha=0.5,
                color=color, label=model.capitalize(), density=True)
    ax6.set_xlabel('3-mer Diversity')
    ax6.set_ylabel('Density')
    ax6.set_title('F. Sequence Complexity', fontweight='bold')
    ax6.legend()
    
    # 2.7 基序频率
    ax7 = fig.add_subplot(gs[2, 1])
    motifs = {'SLAEV': ['transformer', 'vae'], 'IDKG': ['transformer', 'vae'],
              'LKYAEA': ['transformer'], 'GLLRD': ['vae'], 'AAAAA': ['esm2']}
    motif_names = list(motifs.keys())
    x = np.arange(len(motif_names))
    width = 0.2
    
    for i, model in enumerate(models):
        counts = []
        for motif in motif_names:
            df_model = df_all[df_all['model']==model]
            count = sum(1 for s in df_model['seq'] if motif in s)
            counts.append(count)
        ax7.bar(x + i*width, counts, width, label=model.capitalize(), color=colors[i])
    
    ax7.set_xticks(x + width*1.5)
    ax7.set_xticklabels(motif_names, rotation=45)
    ax7.set_ylabel('Frequency')
    ax7.set_title('G. Motif Occurrence', fontweight='bold')
    ax7.legend()
    
    # 2.8 长度vs质量
    ax8 = fig.add_subplot(gs[2, 2])
    for model, color in zip(models, colors):
        data = df_all[df_all['model']==model]
        ax8.scatter(data['seq'].str.len(), data['overall'], c=color, alpha=0.5, s=40,
                   label=model.capitalize(), edgecolors='white', linewidth=0.5)
    ax8.set_xlabel('Length (aa)')
    ax8.set_ylabel('Overall Score')
    ax8.set_title('H. Length vs Quality', fontweight='bold')
    ax8.legend()
    ax8.grid(True, alpha=0.3)
    
    plt.suptitle('Sequence Feature Analysis', fontsize=16, fontweight='bold', y=0.98)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    return fig

# ==================== 图3: 序列Logo风格可视化 ====================
def plot_sequence_logo_style(dfs, save_path='fig3_sequence_logo.png'):
    """生成类Logo的序列保守性可视化"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    models = ['transformer', 'diffusion', 'esm2', 'vae']
    
    for ax, model in zip(axes.flat, models):
        df_model = dfs['all'][dfs['all']['model']==model]
        top20 = df_model.nlargest(20, 'overall')['seq']
        
        max_len = min(50, min(len(s) for s in top20))
        position_counts = {i: Counter() for i in range(max_len)}
        
        for seq in top20:
            for i in range(max_len):
                if i < len(seq):
                    position_counts[i][seq[i]] += 1
        
        # 计算信息含量
        aa_order = list('ACDEFGHIKLMNPQRSTVWY')
        ic_values = []
        for i in range(max_len):
            counts = position_counts[i]
            freqs = np.array([counts.get(aa, 0) / len(top20) for aa in aa_order])
            freqs = freqs[freqs > 0]
            H = entropy(freqs, base=2) if len(freqs) > 0 else 0
            ic = np.log2(20) - H
            ic_values.append(max(0, ic))
        
        # 绘制每个位置的氨基酸频率（高度加权信息含量）
        x_pos = np.arange(max_len)
        bottom = np.zeros(max_len)
        
        for aa in aa_order:
            heights = [position_counts[i].get(aa, 0) / len(top20) * ic_values[i] 
                      for i in range(max_len)]
            color = '#FFD93D' if aa in 'AVILMFWY' else \
                    '#FF6B6B' if aa in 'KRH' else \
                    '#4ECDC4' if aa in 'DE' else \
                    '#95E1D3' if aa in 'STNQ' else '#E8E8E8'
            ax.bar(x_pos, heights, bottom=bottom, color=color, width=1.0, edgecolor='white', linewidth=0.1)
            bottom += heights
        
        # 添加共识序列
        consensus = []
        for i in range(max_len):
            if position_counts[i]:
                consensus.append(max(position_counts[i], key=position_counts[i].get))
            else:
                consensus.append('-')
        
        # 在底部显示共识序列（前20位）
        for i in range(min(20, max_len)):
            ax.text(i + 0.5, -0.05, consensus[i], ha='center', va='top', 
                   fontsize=6, transform=ax.get_xaxis_transform())
        
        ax.set_xlim(0, max_len)
        ax.set_ylim(0, max(ic_values) * 1.2 if ic_values else 1)
        ax.set_xlabel('Position')
        ax.set_ylabel('Information (bits)')
        ax.set_title(f'{model.capitalize()} (Top 20)', fontweight='bold')
        ax.set_xticks(range(0, max_len, 10))
    
    plt.suptitle('Position-Specific Amino Acid Composition (Logo Style)', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    return fig

# ==================== 图4: 交互式质量筛选工具 ====================
def create_quality_filter(dfs, save_path='fig4_quality_filter.png'):
    """生成质量筛选决策图"""
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    df_all = dfs['all']
    
    # 4.1 质量-新颖性权衡
    ax1 = fig.add_subplot(gs[0, 0])
    scatter = ax1.scatter(df_all['novelty'], df_all['overall'], 
                         c=df_all['fold'], cmap='RdYlGn', s=50, alpha=0.7,
                         edgecolors='black', linewidth=0.3)
    ax1.set_xlabel('Novelty')
    ax1.set_ylabel('Overall Score')
    ax1.set_title('A. Quality vs Novelty (colored by Fold)', fontweight='bold')
    plt.colorbar(scatter, ax=ax1, label='Fold Score')
    
    ax1.axhline(y=0.72, color='red', linestyle='--', alpha=0.7, label='High quality')
    ax1.axvline(x=0.6, color='blue', linestyle='--', alpha=0.7, label='High novelty')
    ax1.fill_between([0.6, 1.0], [0.72, 0.72], [1.0, 1.0], alpha=0.1, color='green')
    ax1.text(0.8, 0.85, 'Sweet Spot', ha='center', fontsize=12, 
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    ax1.legend(loc='lower left')
    
    # 4.2 多指标联合分布 - 添加2D密度估计
    ax2 = fig.add_subplot(gs[0, 1])
    df_all['quality_score'] = (df_all['fold'] + df_all['toxic']) / 2
    df_all['diversity_score'] = (df_all['novelty'] + df_all['conserv']) / 2
    
    # 绘制2D密度背景
    all_x = df_all['diversity_score'].values
    all_y = df_all['quality_score'].values
    xy = np.vstack([all_x, all_y])
    kde = gaussian_kde(xy)
    xi, yi = np.mgrid[all_x.min():all_x.max():100j, all_y.min():all_y.max():100j]
    zi = kde(np.vstack([xi.ravel(), yi.ravel()])).reshape(xi.shape)
    ax2.pcolormesh(xi, yi, zi, shading='auto', cmap='Greys', alpha=0.3)
    
    for model, color in MODEL_COLORS.items():
        data = df_all[df_all['model']==model]
        ax2.scatter(data['diversity_score'], data['quality_score'], 
                   c=color, alpha=0.6, s=50, label=model.capitalize(),
                   edgecolors='white', linewidth=0.5)
    
    ax2.set_xlabel('Diversity Score (Novelty+Conserv)/2')
    ax2.set_ylabel('Quality Score (Fold+Toxic)/2')
    ax2.set_title('B. Quality-Diversity Landscape', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 4.3 筛选条件效果 - 添加0值解释
    ax3 = fig.add_subplot(gs[1, 0])
    conditions = [
        ('No filter', df_all),
        ('Overall>0.7', df_all[df_all['overall'] > 0.7]),
        ('Fold>0.95 &\nNovelty>0.6', df_all[(df_all['fold']>0.95) & (df_all['novelty']>0.6)]),
        ('No long repeats', df_all[df_all['seq'].apply(lambda s: max_consecutive_repeat(s) < 15)]),
        ('All combined', df_all[(df_all['overall']>0.7) & (df_all['fold']>0.95) & 
                               (df_all['novelty']>0.6) & 
                               (df_all['seq'].apply(lambda s: max_consecutive_repeat(s) < 15))])
    ]
    
    counts = [len(d) for _, d in conditions]
    colors_bar = ['gray', 'lightblue', 'lightgreen', 'lightyellow', 'darkgreen']
    bars = ax3.bar(range(len(conditions)), counts, color=colors_bar, alpha=0.8)
    ax3.set_xticks(range(len(conditions)))
    ax3.set_xticklabels([c[0] for c in conditions], rotation=45, ha='right')
    ax3.set_ylabel('Remaining Sequences')
    ax3.set_title('C. Filter Effects', fontweight='bold')
    
    for bar, count in zip(bars, counts):
        y_pos = bar.get_height() + 5 if count > 0 else 5
        ax3.text(bar.get_x() + bar.get_width()/2, y_pos, 
                str(count), ha='center', va='bottom', fontsize=10, fontweight='bold')
        if count == 0:
            ax3.text(bar.get_x() + bar.get_width()/2, 15, 
                    'Too strict!', ha='center', va='bottom', 
                    fontsize=8, color='red', style='italic')
    
    ax3.text(0.5, -0.35, 'Note: Strict combined filters yield no candidates.\nConsider relaxed thresholds or weighted scoring.', 
             transform=ax3.transAxes, ha='center', va='top',
             fontsize=8, style='italic', color='darkred',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    # 4.4 推荐序列列表
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis('off')
    
    recommended = df_all[
        (df_all['overall'] > 0.72) & 
        (df_all['fold'] > 0.95) & 
        (df_all['novelty'] > 0.5) &
        (df_all['seq'].str.len() < 250) &
        (df_all['seq'].apply(lambda s: max_consecutive_repeat(s) < 12))
    ].sort_values('overall', ascending=False).head(10)
    
    table_data = []
    for _, row in recommended.iterrows():
        table_data.append([
            row['id'][:20],
            row['model'].capitalize(),
            f"{row['overall']:.3f}",
            f"{row['fold']:.2f}",
            f"{row['novelty']:.2f}",
            str(len(row['seq']))
        ])
    
    table = ax4.table(cellText=table_data,
                     colLabels=['ID', 'Model', 'Overall', 'Fold', 'Novelty', 'Len'],
                     cellLoc='center', loc='center',
                     colColours=['#4472C4']*6)
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.8)
    
    for i, (_, row) in enumerate(recommended.iterrows()):
        for j in range(6):
            table[(i+1, j)].set_facecolor(MODEL_COLORS[row['model']])
            table[(i+1, j)].set_text_props(color='white' if row['model'] in ['transformer', 'esm2'] else 'black')
    
    ax4.set_title('D. Recommended Sequences (Top 10)', fontweight='bold', pad=20)
    
    plt.suptitle('Quality Filter & Sequence Selection Tool', fontsize=16, fontweight='bold', y=0.98)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    return fig

# ==================== 图5: 单序列详细分析 ====================
def plot_single_sequence_analysis(seq, save_path='fig5_single_seq.png'):
    """分析单个序列的详细特征"""
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(3, 2, figure=fig, hspace=0.35, wspace=0.3)
    
    # 5.1 序列条形图（按性质着色）
    ax1 = fig.add_subplot(gs[0, :])
    prop_colors = {'hydrophobic': '#FFD93D', 'polar': '#95E1D3', 
                   'basic': '#FF6B6B', 'acidic': '#4ECDC4', 'special': '#E8E8E8'}
    
    for i, aa in enumerate(seq):
        for prop, aas in AA_PROPERTIES.items():
            if aa in aas:
                color = prop_colors[prop]
                break
        ax1.barh(0, 1, left=i, height=0.8, color=color, edgecolor='white', linewidth=0.2)
    
    ax1.set_xlim(0, len(seq))
    ax1.set_ylim(-0.5, 0.5)
    ax1.set_xlabel('Position')
    ax1.set_title(f'Sequence Map (Length: {len(seq)})', fontweight='bold')
    ax1.set_yticks([])
    
    legend_elements = [mpatches.Patch(color=c, label=k.capitalize()) 
                       for k, c in prop_colors.items()]
    ax1.legend(handles=legend_elements, loc='upper right', ncol=5)
    
    # 5.2 疏水性滑动窗口 - 修复维度匹配和Y轴缩放
    ax2 = fig.add_subplot(gs[1, 0])
    hydrophobicity = {'A': 1.8, 'C': 2.5, 'D': -3.5, 'E': -3.5, 'F': 2.8,
                      'G': -0.4, 'H': -3.2, 'I': 4.5, 'K': -3.9, 'L': 3.8,
                      'M': 1.9, 'N': -3.5, 'P': -1.6, 'Q': -3.5, 'R': -4.5,
                      'S': -0.8, 'T': -0.7, 'V': 4.2, 'W': -0.9, 'Y': -1.3}
    
    window = 7
    half_window = window // 2
    
    hydropathy = []
    for i in range(len(seq) - window + 1):
        avg = np.mean([hydrophobicity.get(aa, 0) for aa in seq[i:i+window]])
        hydropathy.append(avg)
    
    x_vals = list(range(half_window, len(seq) - half_window))
    min_len = min(len(x_vals), len(hydropathy))
    x_vals = x_vals[:min_len]
    hydropathy = hydropathy[:min_len]
    
    # 自动Y轴范围
    y_min, y_max = min(hydropathy), max(hydropathy)
    y_margin = (y_max - y_min) * 0.15
    ax2.set_ylim(y_min - y_margin, y_max + y_margin)
    
    ax2.plot(x_vals, hydropathy, 'b-', linewidth=1.5)
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    ax2.fill_between(x_vals, hydropathy, 0, 
                     where=[h > 0 for h in hydropathy], alpha=0.3, color='yellow', label='Hydrophobic')
    ax2.fill_between(x_vals, hydropathy, 0, 
                     where=[h <= 0 for h in hydropathy], alpha=0.3, color='cyan', label='Hydrophilic')
    
    # 添加极值标注
    max_idx = np.argmax(hydropathy)
    min_idx = np.argmin(hydropathy)
    ax2.annotate(f'Max: {hydropathy[max_idx]:.1f}', 
                xy=(x_vals[max_idx], hydropathy[max_idx]),
                xytext=(10, 10), textcoords='offset points',
                fontsize=8, color='darkgoldenrod',
                arrowprops=dict(arrowstyle='->', color='darkgoldenrod'))
    ax2.annotate(f'Min: {hydropathy[min_idx]:.1f}', 
                xy=(x_vals[min_idx], hydropathy[min_idx]),
                xytext=(10, -15), textcoords='offset points',
                fontsize=8, color='darkcyan',
                arrowprops=dict(arrowstyle='->', color='darkcyan'))
    
    ax2.set_xlabel('Position (Window Center)')
    ax2.set_ylabel('Hydropathy (Kyte-Doolittle)')
    ax2.set_title('A. Hydropathy Profile', fontweight='bold')
    ax2.legend()
    
    # 5.3 局部复杂度 - X轴对齐到窗口中心
    ax3 = fig.add_subplot(gs[1, 1])
    complexity = calculate_complexity(seq, window=10)
    
    comp_window = 10
    comp_half = comp_window // 2
    comp_x = list(range(comp_half, len(seq) - comp_half))
    
    min_len_c = min(len(comp_x), len(complexity))
    comp_x = comp_x[:min_len_c]
    complexity = complexity[:min_len_c]
    
    ax3.plot(comp_x, complexity, 'g-', linewidth=1.5)
    ax3.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Low complexity threshold')
    
    low_comp_regions = [(comp_x[i], complexity[i]) for i in range(len(complexity)) if complexity[i] < 0.5]
    if low_comp_regions:
        ax3.scatter([x for x, _ in low_comp_regions], [y for _, y in low_comp_regions],
                   c='red', s=30, zorder=5, label='Low complexity regions')
    
    ax3.set_xlabel('Position (Window Center)')
    ax3.set_ylabel('Complexity (unique/total)')
    ax3.set_title('B. Local Complexity', fontweight='bold')
    ax3.set_ylim(0, 1)
    ax3.legend(fontsize=8)
    
    # 5.4 氨基酸组成饼图
    ax4 = fig.add_subplot(gs[2, 0])
    composition = get_aa_composition(seq)
    aa_sorted = sorted(composition.items(), key=lambda x: x[1], reverse=True)
    top_aas = [aa for aa, _ in aa_sorted[:10]]
    top_vals = [v for _, v in aa_sorted[:10]]
    
    colors_pie = [prop_colors[next(p for p, aas in AA_PROPERTIES.items() if aa in aas)] 
                  for aa in top_aas]
    ax4.pie(top_vals, labels=top_aas, colors=colors_pie, autopct='%1.1f%%', startangle=90)
    ax4.set_title('C. Amino Acid Composition', fontweight='bold')
    
    # 5.5 统计摘要
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.axis('off')
    
    stats = [
        f"Sequence Length: {len(seq)}",
        f"Max Consecutive Repeat: {max_consecutive_repeat(seq)}",
        f"3-mer Diversity: {kmer_diversity(seq, 3):.3f}",
        f"Hydrophobic Content: {sum(1 for a in seq if a in 'AVILMFWY')/len(seq)*100:.1f}%",
        f"Charged Content: {sum(1 for a in seq if a in 'KRHDE')/len(seq)*100:.1f}%",
        f"Max Block Length: {get_blockiness(seq)[0]}",
    ]
    
    for i, stat in enumerate(stats):
        ax5.text(0.1, 0.9-i*0.15, stat, fontsize=12, 
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
    
    ax5.set_title('D. Sequence Statistics', fontweight='bold')
    
    plt.suptitle(f'Single Sequence Analysis: {seq[:20]}...', fontsize=14, fontweight='bold')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    return fig

# ==================== 主程序 ====================
if __name__ == "__main__":
    # 加载数据
    dfs = load_data()
    
    # 生成所有图表
    print("正在生成图表...")
    
    print("1. 综合性能概览...")
    plot_overview(dfs, 'fig1_overview.png')
    
    print("2. 序列特征分析...")
    plot_sequence_features(dfs, 'fig2_sequence_features.png')
    
    print("3. Logo风格可视化...")
    plot_sequence_logo_style(dfs, 'fig3_sequence_logo.png')
    
    print("4. 质量筛选工具...")
    create_quality_filter(dfs, 'fig4_quality_filter.png')
    
    print("5. 单序列分析示例...")
    example_seq = dfs['all'].nlargest(1, 'overall').iloc[0]['seq']
    plot_single_sequence_analysis(example_seq, 'fig5_single_seq.png')
    
    print("所有图表生成完成！")