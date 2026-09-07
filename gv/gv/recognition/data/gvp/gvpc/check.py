import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import Counter
import os

# --- 0. 用户配置区域 ---
START_POS = 40        # 起始氨基酸位置 (1-based)
WINDOW_SIZE = 35      # 窗口长度 (例如 30aa)
INPUT_JSON = r"C:\Users\r9000\Desktop\毕设（无监督聚类）\新gvp\gvpn\GvpN_sequences.json"
OUTPUT_DIR = "output_analysis"

# --- 1. 数据加载 ---
def load_sequences(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        sequences = []
        for item in data:
            if isinstance(item, dict):
                if 'sequence' in item:
                    seq = item['sequence']
                    if seq:
                        sequences.append(seq)
                elif 'representative_annotation' in item and 'sequence' in item['representative_annotation']:
                    seq = item['representative_annotation']['sequence']
                    if seq:
                        sequences.append(seq)
        print(f"✅ Loaded {len(sequences)} sequences.")
        
        # 调试：打印前几条序列长度
        if sequences:
            print(f"📏 Sequence length check (first 5): {[len(s) for s in sequences[:5]]}")
        
        return sequences
    except Exception as e:
        print(f"❌ Error loading file: {e}")
        return []

# --- 2. 核心分析逻辑 ---
def analyze_window(sequences, start_pos, window_size):
    start_idx = start_pos - 1
    pos_counters = [Counter() for _ in range(window_size)]
    fragments = []
    
    # 氨基酸理化性质字典
    properties = {
        'hydro': {'I': 4.5, 'V': 4.2, 'L': 3.8, 'F': 2.8, 'C': 2.5, 'M': 1.9, 'A': 1.8, 'G': -0.4, 'T': -0.7, 'S': -0.8, 'W': -0.9, 'Y': -1.3, 'P': -1.6, 'H': -3.2, 'E': -3.5, 'Q': -3.5, 'D': -3.5, 'N': -3.5, 'K': -3.9, 'R': -4.5},
        'charge': {'K': 1, 'R': 1, 'H': 1, 'D': -1, 'E': -1}
    }

    valid_count = 0
    for seq in sequences:
        if start_idx >= len(seq):
            continue
            
        end_idx = min(len(seq), start_idx + window_size)
        frag = seq[start_idx:end_idx]
        
        if len(frag) > 0:
            fragments.append(frag)
            valid_count += 1
            for i, aa in enumerate(frag):
                if i < window_size:
                    pos_counters[i][aa] += 1

    if valid_count == 0:
        print(f"❌ Error: No sequences found containing position {start_pos}.")
        return None

    print(f"ℹ️ Analyzing {valid_count} valid sequences...")

    avg_hydro = []
    avg_charge = []
    conservation_scores = []
    positions_actual = []

    for i in range(window_size):
        cnt = pos_counters[i]
        total = sum(cnt.values())
        
        if total == 0: continue

        # 1. 平均疏水性
        h_score = sum(properties['hydro'].get(aa, 0) * count for aa, count in cnt.items()) / total
        avg_hydro.append(h_score)
        
        # 2. 平均净电荷
        c_score = sum(properties['charge'].get(aa, 0) * count for aa, count in cnt.items()) / total
        avg_charge.append(c_score)

        # 3. 保守性
        probs = [count/total for count in cnt.values()]
        entropy = -sum(p * np.log2(p) for p in probs if p > 0)
        max_entropy = np.log2(20)
        conservation = 1 - (entropy / max_entropy)
        conservation_scores.append(conservation)
        
        positions_actual.append(start_pos + i)

    return {
        'fragments': fragments,
        'positions': positions_actual,
        'avg_hydro': avg_hydro,
        'avg_charge': avg_charge,
        'conservation': conservation_scores,
        'pos_counters': pos_counters
    }

# --- 3. 生物学意义解读 (控制台输出保留中文，方便阅读) ---
def interpret_biology(results, start_pos):
    if not results: return
        
    print("\n" + "="*40)
    print(f"🧬 生物学分析报告 (位置 {start_pos}-{start_pos+len(results['positions'])-1})")
    print("="*40)
    
    if not results['avg_hydro']: return
        
    mean_hydro = np.mean(results['avg_hydro'])
    mean_charge = np.mean(results['avg_charge'])
    
    # 简单解读
    nature = "Amphipathic"
    if mean_hydro > 1: nature = "Hydrophobic (Core/Membrane)"
    elif mean_hydro < -1: nature = "Hydrophilic (Surface/Tail)"
    
    print(f"1. 理化性质: {nature} (Avg Hydro: {mean_hydro:.2f})")
    
    charge_type = "Neutral"
    if mean_charge > 0.1: charge_type = "Positive"
    elif mean_charge < -0.1: charge_type = "Negative (Acidic Tail)"
    print(f"2. 电荷特征: {charge_type} (Avg Charge: {mean_charge:.2f})")

    if results['conservation']:
        max_conserved_idx = np.argmax(results['conservation'])
        pos = results['positions'][max_conserved_idx]
        most_common_aa = results['pos_counters'][max_conserved_idx].most_common(1)[0][0]
        print(f"3. 关键保守位点: 第 {pos} 位的 '{most_common_aa}' (保守性: {results['conservation'][max_conserved_idx]:.2f})")
    
    print("="*40 + "\n")

# --- 4. 绘图与保存 (已修改为全英文) ---
def plot_and_save(results, start_pos, window_size):
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # 样式设置
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except:
        try:
            plt.style.use('seaborn-whitegrid')
        except:
            plt.style.use('bmh')

    fig, axes = plt.subplots(3, 1, figsize=(12, 12))
    # 标题改为英文
    fig.suptitle(f'GvpC Local Feature Analysis: {start_pos}-{start_pos+window_size-1}aa', fontsize=16, fontweight='bold')

    positions = results['positions']
    
    # 1. 保守性曲线
    axes[0].plot(positions, results['conservation'], color='purple', marker='o')
    axes[0].set_ylabel('Conservation Score')
    axes[0].set_title('Sequence Conservation (Shannon Entropy)')
    axes[0].set_ylim(0, 1.1)

    # 2. 疏水性曲线
    axes[1].bar(positions, results['avg_hydro'], color='orange', alpha=0.7)
    axes[1].axhline(0, color='black', linestyle='-', linewidth=0.5)
    axes[1].set_ylabel('Hydrophobicity (KD Index)')
    axes[1].set_title('Average Hydrophobicity Profile')

    # 3. 净电荷曲线
    axes[2].bar(positions, results['avg_charge'], color='skyblue', alpha=0.7)
    axes[2].axhline(0, color='black', linestyle='-', linewidth=0.5)
    axes[2].set_ylabel('Net Charge')
    axes[2].set_xlabel('Amino Acid Position')
    axes[2].set_title('Average Net Charge Profile')

    plt.tight_layout()
    
    # 保存
    filename = f"GvpA_window_{start_pos}_{start_pos+window_size-1}.png"
    save_path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(save_path, dpi=300)
    print(f"✅ Image saved: {save_path}")
    plt.show()

    # 保存数据
    df = pd.DataFrame({
        'Position': positions,
        'Conservation': results['conservation'],
        'Hydrophobicity': results['avg_hydro'],
        'Net_Charge': results['avg_charge']
    })
    csv_path = os.path.join(OUTPUT_DIR, f"data_window_{start_pos}.csv")
    df.to_csv(csv_path, index=False)
    print(f"✅ Data saved: {csv_path}")

# --- 5. 主程序 ---
if __name__ == "__main__":
    seqs = load_sequences(INPUT_JSON)
    if seqs:
        res = analyze_window(seqs, START_POS, WINDOW_SIZE)
        if res:
            interpret_biology(res, START_POS)
            plot_and_save(res, START_POS, WINDOW_SIZE)