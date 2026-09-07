#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""深度诊断：逐行模拟 train.py 启动"""
import sys

print("=" * 60)
print("深度诊断：逐行执行 train.py 的导入")
print("=" * 60)

# 第1行: import os
print("\n[1] import os...")
import os
print("  ✅")

# 第2-3行: os.environ
print("\n[2] 设置环境变量...")
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
print("  ✅")

# 第4行: import torch
print("\n[3] import torch...")
import torch
print(f"  ✅ PyTorch {torch.__version__}")

# 第5-7行: torch.nn, optim, functional
print("\n[4] import torch.nn, optim, functional...")
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
print("  ✅")

# 第8行: DataLoader
print("\n[5] from torch.utils.data import DataLoader...")
from torch.utils.data import DataLoader
print("  ✅")

# 第9行: data_loader
print("\n[6] from data_loader import GvpDataset, collate_fn, VOCAB_SIZE...")
try:
    from data_loader import GvpDataset, collate_fn, VOCAB_SIZE
    print(f"  ✅ VOCAB_SIZE={VOCAB_SIZE}")
except Exception as e:
    print(f"  ❌ {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 第10行: vae_gvp
print("\n[7] from models.vae_gvp import GVAE, vae_loss...")
try:
    from models.vae_gvp import GVAE, vae_loss
    print("  ✅")
except Exception as e:
    print(f"  ❌ {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 第11-12行: transformer, diffusion
print("\n[8] from models.transformer_gvp import GvpTransformer...")
try:
    from models.transformer_gvp import GvpTransformer
    print("  ✅")
except Exception as e:
    print(f"  ❌ {e}")
    import traceback
    traceback.print_exc()

print("\n[9] from models.diffusion_gvp import GvpDiffusion...")
try:
    from models.diffusion_gvp import GvpDiffusion
    print("  ✅")
except Exception as e:
    print(f"  ❌ {e}")
    import traceback
    traceback.print_exc()

# 第13行: argparse
print("\n[10] import argparse...")
import argparse
print("  ✅")

# 模拟 main 块
print("\n[11] 模拟参数解析...")
parser = argparse.ArgumentParser()
parser.add_argument('--model', choices=['vae', 'transformer', 'diffusion', 'esm2'], required=True)
parser.add_argument('--fasta', default='data/natural_gvp.fasta')
parser.add_argument('--msa', default='data/gvpA_msa.a3m')
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--latent_dim', type=int, default=64)
parser.add_argument('--kl_weight', type=float, default=0.1)
parser.add_argument('--diffusion_steps', type=int, default=1000)
parser.add_argument('--max_len', type=int, default=512)
parser.add_argument('--target_len_A', type=int, default=100)
parser.add_argument('--target_len_C', type=int, default=500)

args = parser.parse_args([
    '--model', 'vae',
    '--fasta', 'data/natural_gvp.fasta',
    '--epochs', '2',
    '--batch_size', '2',
    '--max_len', '128'
])
print(f"  ✅ 参数解析成功: model={args.model}, epochs={args.epochs}")

# 模拟 train_vae
print("\n[12] 开始 train_vae 内容...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"  设备: {device}")

os.makedirs('checkpoints', exist_ok=True)
print("  ✅ checkpoints 目录")

print("\n[13] 创建数据集...")
dataset = GvpDataset(args.fasta, args.msa, max_len=args.max_len)
print(f"  ✅ 数据集: {len(dataset)} 条序列, {dataset.num_species} 物种")

print("\n[14] 创建 DataLoader...")
dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
print("  ✅ DataLoader 创建成功")

print("\n[15] 创建模型...")
model = GVAE(vocab_size=VOCAB_SIZE, num_species=dataset.num_species,
             latent_dim=args.latent_dim, target_len_A=args.target_len_A,
             target_len_C=args.target_len_C).to(device)
print(f"  ✅ 模型创建成功")

print("\n[16] 创建优化器...")
optimizer = optim.Adam(model.parameters(), lr=args.lr)
print("  ✅ 优化器创建成功")

print("\n[17] 开始训练循环（2个epoch测试）...")
for epoch in range(2):
    model.train()
    total_loss = 0
    num_batches = 0
    
    for i, batch in enumerate(dataloader):
        if i >= 2:
            break
            
        x = batch['seq'].to(device)
        species = batch['species'].to(device)
        
        optimizer.zero_grad()
        outputs, mu, logvar = model(x, species)
        
        target = x[:, 1:]
        min_len = min(outputs.size(1), target.size(1))
        outputs = outputs[:, :min_len, :]
        target = target[:, :min_len]
        
        loss, recon, kl = vae_loss(outputs, target, mu, logvar, kl_weight=args.kl_weight)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        print(f"    Epoch {epoch}, Batch {i}: Loss={loss.item():.4f}")
    
    avg_loss = total_loss / max(num_batches, 1)
    print(f"  ✅ Epoch {epoch} 完成, Avg Loss={avg_loss:.4f}")

print("\n" + "=" * 60)
print("✅ 所有步骤通过！train.py 应该可以正常运行")
print("=" * 60)