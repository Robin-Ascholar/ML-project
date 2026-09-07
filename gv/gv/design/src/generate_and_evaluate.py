import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import torch
import glob
from models.vae_gvp import GVAE
from models.transformer_gvp import GvpTransformer
from models.diffusion_gvp import GvpDiffusion  
from models.esm2_gvp import Esm2GVP
from data_loader import GvpDataset
from score.score_net import GvpScoreNet
import pandas as pd

# ========== 硬过滤函数（新增）==========
def hard_filter(seq: str) -> bool:
    """硬过滤：排除明显非法/低质量序列

    过滤规则：
    1. 长度 < 40 或 > 300
    2. 包含非法字符（-, X, 非标准氨基酸）
    3. 疏水性 > 60%（易聚集）
    4. 低复杂度（3连重复n-gram）
    5. 单一氨基酸占比 > 30%
    """
    if not seq or len(seq) < 40 or len(seq) > 300:
        return False

    # 检查非法字符
    valid_aas = set('ACDEFGHIKLMNPQRSTVWY')
    if any(aa not in valid_aas for aa in seq):
        return False

    # 疏水性检查
    hydrophobic = set('AVILMFWY')
    h_ratio = sum(1 for aa in seq if aa in hydrophobic) / len(seq)
    if h_ratio > 0.60:
        return False

    # 低复杂度检查（seg-like）
    for w in range(3, 6):
        for i in range(len(seq) - w * 3):
            if seq[i:i+w] == seq[i+w:i+2*w] == seq[i+2*w:i+3*w]:
                return False

    # 单一氨基酸占比检查
    from collections import Counter
    aa_counts = Counter(seq)
    max_freq = max(aa_counts.values()) / len(seq)
    if max_freq > 0.30:
        return False

    return True


def generate_candidates(model_path, model_type, dataset, num_samples=1000, 
                       species_name=None):
    """生成候选序列（带硬过滤）"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if species_name is None:
        species_name = list(dataset.species_to_idx.keys())[0]
        print(f"  自动选择物种/ID: {species_name}")

    species_idx = dataset.species_to_idx[species_name]

    if model_type == 'vae':
        model = GVAE(vocab_size=len(dataset.aa_vocab),
                    num_species=len(dataset.species_to_idx)).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        model.eval()

        with torch.no_grad():
            generated = model.generate(species=species_idx, 
                                     num_samples=num_samples,
                                     max_len=200,
                                     temperature=0.8,
                                     filter_invalid=True)  # 启用内部过滤
        generated = [g.squeeze(0) if g.dim() > 1 else g for g in generated]
        generated = [g for g in generated if g.dim() > 0]

    elif model_type == 'transformer':
        model = GvpTransformer(vocab_size=len(dataset.aa_vocab),
                              num_species=len(dataset.species_to_idx)).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        model.eval()

        generated = []
        with torch.no_grad():
            for _ in range(num_samples):
                seq_tensor = model.generate(species=species_idx, max_len=100,
                                           min_len=50,  # 最小长度控制
                                           ngram_block=4)  # n-gram阻断
                if isinstance(seq_tensor, torch.Tensor):
                    if seq_tensor.dim() > 1:
                        seq_tensor = seq_tensor.squeeze(0)
                    if seq_tensor.dim() == 0:
                        continue
                generated.append(seq_tensor)

    elif model_type == 'diffusion':
        model = GvpDiffusion(vocab_size=len(dataset.aa_vocab),
                           num_species=len(dataset.species_to_idx)).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        model.eval()

        generated = []
        with torch.no_grad():
            for _ in range(num_samples):
                sequences = model.generate(species=species_idx, 
                                         chain_A=True, 
                                         chain_C=False,
                                         max_len_A=100,
                                         batch_size=1,
                                         filter_invalid=True)  # 启用D3PM过滤
                if sequences and len(sequences) > 0:
                    _, seq_A = sequences[0]
                    if seq_A.dim() > 1:
                        seq_A = seq_A.squeeze(0)
                    if seq_A.dim() > 0:
                        generated.append(seq_A)

    elif model_type == 'esm2':
        model = Esm2GVP(
            vocab_size=len(dataset.aa_vocab),
            d_model=256,
            num_species=len(dataset.species_to_idx)
        ).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        model.eval()

        generated = []
        with torch.no_grad():
            for i in range(num_samples):
                print(f"  生成序列 {i+1}/{num_samples}...", end='\r')
                seq_tensor = model.generate(
                    species=species_idx, 
                    max_len=100,
                    temperature=0.8,
                    top_p=0.9,
                    device=device,
                    min_len=50  # 最小长度
                )
                if seq_tensor.dim() > 1:
                    seq_tensor = seq_tensor.squeeze(0)
                if seq_tensor.dim() > 0:
                    generated.append(seq_tensor)
        print()

    # 解码序列 + 硬过滤
    sequences = []
    filtered_count = 0
    for seq in generated:
        try:
            if isinstance(seq, torch.Tensor):
                if seq.dim() == 0:
                    continue
                if seq.dim() > 1:
                    seq = seq.view(-1)
                decoded = dataset.decode(seq)
            else:
                decoded = dataset.decode(torch.tensor(seq))

            # 应用硬过滤
            if hard_filter(decoded):
                sequences.append(decoded)
            else:
                filtered_count += 1
        except Exception as e:
            print(f"  解码错误，跳过该序列: {e}")
            continue

    print(f"  硬过滤: 保留 {len(sequences)}/{len(sequences)+filtered_count} 条序列")
    return sequences

def evaluate_sequences(sequences, model_name, scorer, start_id=0):
    """评估序列并返回结果DataFrame"""
    print(f"\n评估 {model_name} 的 {len(sequences)} 条序列...")
    results = []

    for i, seq in enumerate(sequences):
        try:
            if not isinstance(seq, str) or len(seq) < 10:
                print(f"  跳过无效序列 {i}: 长度={len(seq) if isinstance(seq, str) else 'N/A'}")
                continue
            score = scorer(seq)
            results.append({
                'id': f'{model_name}_gen_{start_id + i}',
                'seq': seq,
                'model': model_name,
                **score
            })
            if (i + 1) % 10 == 0:
                print(f"  已评估: {i+1}/{len(sequences)}")
        except Exception as e:
            print(f"  Error scoring {model_name} seq {i}: {e}")

    df = pd.DataFrame(results)
    if not df.empty:
        df = df.sort_values('overall', ascending=False)
    return df

def get_latest_checkpoint(pattern):
    """获取最新的检查点文件"""
    checkpoints = glob.glob(pattern)
    if not checkpoints:
        return None
    return max(checkpoints, key=os.path.getmtime)

def main():
    # 初始化
    dataset = GvpDataset('data/natural_gvp.fasta', 'data/gvpA_msa.a3m')
    scorer = GvpScoreNet()

    available_species = list(dataset.species_to_idx.keys())
    species_name = available_species[0]
    print(f"数据集包含 {len(available_species)} 个不同的序列ID")
    print(f"使用第一个ID作为条件: {species_name}")

    all_results = {}

    # ========== 1. VAE 模型 ==========
    print("\n" + "="*50)
    print("【1/4】VAE 模型")
    print("="*50)

    vae_checkpoint = 'checkpoints/vae_best.pt'
    if not os.path.exists(vae_checkpoint):
        vae_checkpoint = get_latest_checkpoint('checkpoints/vae_epoch*.pt')

    if vae_checkpoint and os.path.exists(vae_checkpoint):
        print(f"使用VAE检查点: {vae_checkpoint}")
        print("\nGenerating sequences with VAE...")
        vae_seqs = generate_candidates(vae_checkpoint, 'vae', dataset, 
                                       num_samples=100, species_name=species_name)
        print(f"  生成了 {len(vae_seqs)} 条序列")

        vae_df = evaluate_sequences(vae_seqs, 'vae', scorer, start_id=0)
        if not vae_df.empty:
            vae_df.to_csv('generated_candidates_vae.csv', index=False)
            all_results['vae'] = vae_df
            print(f"\n VAE结果已保存至: generated_candidates_vae.csv")
            print(f"   Top 3 VAE候选序列:")
            print(vae_df.head(3)[['id', 'overall', 'fold', 'conserv', 'novelty', 'toxic']].to_string())
    else:
        print("  跳过VAE（没有找到检查点）")

    # ========== 2. Transformer 模型 ==========
    print("\n" + "="*50)
    print("【2/4】Transformer 模型")
    print("="*50)

    trans_checkpoint = 'checkpoints/trans_best.pt'
    if not os.path.exists(trans_checkpoint):
        trans_checkpoint = get_latest_checkpoint('checkpoints/transformer_epoch*.pt')

    if trans_checkpoint and os.path.exists(trans_checkpoint):
        print(f"使用Transformer检查点: {trans_checkpoint}")
        print("\nGenerating sequences with Transformer...")
        trans_seqs = generate_candidates(trans_checkpoint, 'transformer', 
                                        dataset, num_samples=100, 
                                        species_name=species_name)
        print(f"  生成了 {len(trans_seqs)} 条序列")

        trans_df = evaluate_sequences(trans_seqs, 'transformer', scorer, start_id=0)
        if not trans_df.empty:
            trans_df.to_csv('generated_candidates_transformer.csv', index=False)
            all_results['transformer'] = trans_df
            print(f"\n Transformer结果已保存至: generated_candidates_transformer.csv")
            print(f"   Top 3 Transformer候选序列:")
            print(trans_df.head(3)[['id', 'overall', 'fold', 'conserv', 'novelty', 'toxic']].to_string())
    else:
        print("  跳过Transformer（没有找到检查点）")

    # ========== 3. Diffusion 模型 ==========
    print("\n" + "="*50)
    print("【3/4】Diffusion 模型")
    print("="*50)

    diffusion_checkpoint = 'checkpoints/diffusion_best.pt'
    if not os.path.exists(diffusion_checkpoint):
        diffusion_checkpoint = get_latest_checkpoint('checkpoints/diffusion_epoch*.pt')

    if diffusion_checkpoint and os.path.exists(diffusion_checkpoint):
        print(f"使用Diffusion检查点: {diffusion_checkpoint}")
        print("\nGenerating sequences with Diffusion...")
        diffusion_seqs = generate_candidates(diffusion_checkpoint, 'diffusion', 
                                            dataset, num_samples=100, 
                                            species_name=species_name)
        print(f"  生成了 {len(diffusion_seqs)} 条序列")

        diffusion_df = evaluate_sequences(diffusion_seqs, 'diffusion', scorer, start_id=0)
        if not diffusion_df.empty:
            diffusion_df.to_csv('generated_candidates_diffusion.csv', index=False)
            all_results['diffusion'] = diffusion_df
            print(f"\n Diffusion结果已保存至: generated_candidates_diffusion.csv")
            print(f"   Top 3 Diffusion候选序列:")
            print(diffusion_df.head(3)[['id', 'overall', 'fold', 'conserv', 'novelty', 'toxic']].to_string())
    else:
        print("  跳过Diffusion（没有找到检查点）")

    # ========== 4. ESM2 模型 ==========
    print("\n" + "="*50)
    print("【4/4】ESM2 模型")
    print("="*50)

    esm2_checkpoint = 'checkpoints/esm2_best.pt'
    if not os.path.exists(esm2_checkpoint):
        esm2_checkpoint = get_latest_checkpoint('checkpoints/esm2_epoch*.pt')

    if esm2_checkpoint and os.path.exists(esm2_checkpoint):
        print(f"使用ESM2检查点: {esm2_checkpoint}")
        print("\nGenerating sequences with ESM2...")
        esm2_seqs = generate_candidates(esm2_checkpoint, 'esm2', 
                                       dataset, num_samples=100, 
                                       species_name=species_name)
        print(f"  生成了 {len(esm2_seqs)} 条序列")

        esm2_df = evaluate_sequences(esm2_seqs, 'esm2', scorer, start_id=0)
        if not esm2_df.empty:
            esm2_df.to_csv('generated_candidates_esm2.csv', index=False)
            all_results['esm2'] = esm2_df
            print(f"\n ESM2结果已保存至: generated_candidates_esm2.csv")
            print(f"   Top 3 ESM2候选序列:")
            print(esm2_df.head(3)[['id', 'overall', 'fold', 'conserv', 'novelty', 'toxic']].to_string())
    else:
        print("  跳过ESM2（没有找到检查点）")

    # ========== 5. 合并所有结果 ==========
    print("\n" + "="*50)
    print("【汇总】所有模型结果对比")
    print("="*50)

    if all_results:
        combined_df = pd.concat(all_results.values(), ignore_index=True)
        combined_df = combined_df.sort_values('overall', ascending=False)
        combined_df.to_csv('generated_candidates_all_models.csv', index=False)

        print(f"\n 总计生成 {len(combined_df)} 条候选序列")
        print(f" 合并结果保存至: generated_candidates_all_models.csv")

        print("\n 各模型统计信息:")
        for model_name, df in all_results.items():
            print(f"\n  【{model_name.upper()}】")
            print(f"    序列数量: {len(df)}")
            print(f"    平均得分: {df['overall'].mean():.4f}")
            print(f"    最高得分: {df['overall'].max():.4f}")
            print(f"    最低得分: {df['overall'].min():.4f}")

        print(f"\n Top 5 候选序列（跨所有模型）:")
        print(combined_df.head(5)[['id', 'model', 'overall', 'fold', 'conserv', 'novelty', 'toxic']].to_string())

        print(f"\n 各模型最佳序列:")
        for model_name, df in all_results.items():
            best = df.iloc[0]
            print(f"  {model_name}: {best['id']} (得分: {best['overall']:.4f})")
    else:
        print("\n 警告：没有成功生成任何序列")

if __name__ == "__main__":
    main()