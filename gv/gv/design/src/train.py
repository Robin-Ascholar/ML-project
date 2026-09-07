import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from data_loader import GvpDataset, collate_fn, VOCAB_SIZE
from models.vae_gvp import GVAE, vae_loss
from models.transformer_gvp import GvpTransformer
from models.diffusion_gvp import GvpDiffusion
import argparse

def train_vae(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs('checkpoints', exist_ok=True)

    dataset = GvpDataset(args.fasta, args.msa, max_len=args.max_len, chain_type=args.chain_type)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, 
                           shuffle=True, collate_fn=collate_fn)

    model = GVAE(vocab_size=VOCAB_SIZE,
                 num_species=dataset.num_species,
                 latent_dim=args.latent_dim,
                 target_len_A=args.target_len_A,
                 target_len_C=args.target_len_C).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        num_batches = 0

        for batch in dataloader:
            x = batch['seq'].to(device)
            species = batch['species'].to(device)

            optimizer.zero_grad()
            outputs, mu, logvar = model(x, species)

            target = x[:, 1:]
            min_len = min(outputs.size(1), target.size(1))
            outputs = outputs[:, :min_len, :]
            target = target[:, :min_len]

            loss, recon, kl = vae_loss(outputs, target, mu, logvar, 
                                        kl_weight=args.kl_weight)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        print(f"Epoch {epoch}: Loss {avg_loss:.4f}")

        if epoch % 10 == 0:
            torch.save(model.state_dict(), f"checkpoints/vae_epoch{epoch}.pt")

    torch.save(model.state_dict(), "checkpoints/vae_final.pt")
    print("VAE训练完成！")

def train_transformer(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs('checkpoints', exist_ok=True)

    dataset = GvpDataset(args.fasta, args.msa, max_len=args.max_len, chain_type=args.chain_type)
    dataloader = DataLoader(dataset, batch_size=args.batch_size,
                           shuffle=True, collate_fn=collate_fn)

    model = GvpTransformer(vocab_size=VOCAB_SIZE,
                          num_species=dataset.num_species).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss(ignore_index=21)

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        num_batches = 0

        for batch in dataloader:
            x = batch['seq'].to(device)
            species = batch['species'].to(device)
            msa_weights = batch.get('msa_weights', None)
            if msa_weights is not None:
                msa_weights = msa_weights.to(device)

            src = x[:, :-1]
            tgt = x[:, 1:]

            logits = model(src, species, msa_weights=msa_weights)
            min_len = min(logits.size(1), tgt.size(1))
            logits = logits[:, :min_len, :]
            tgt = tgt[:, :min_len]

            loss = criterion(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1))

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        print(f"Epoch {epoch}: Loss {avg_loss:.4f}")

        if epoch % 10 == 0:
            torch.save(model.state_dict(), f"checkpoints/transformer_epoch{epoch}.pt")

    torch.save(model.state_dict(), "checkpoints/transformer_final.pt")
    print("Transformer训练完成！")

def train_diffusion(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs('checkpoints', exist_ok=True)

    dataset = GvpDataset(args.fasta, args.msa, max_len=args.max_len, chain_type=args.chain_type)
    dataloader = DataLoader(dataset, batch_size=args.batch_size,
                           shuffle=True, collate_fn=collate_fn)

    model = GvpDiffusion(vocab_size=VOCAB_SIZE,
                        num_species=dataset.num_species,
                        num_steps=args.diffusion_steps).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        num_batches = 0

        for batch in dataloader:
            x = batch['seq'].to(device)
            species = batch['species'].to(device)

            # D3PM离散扩散：随机采样时间步
            t = torch.randint(0, model.num_steps, (x.size(0),), device=device)

            # 使用新的D3PM损失
            loss, ce_loss, hydro_loss = model.p_losses(x, t, species, 
                                                       chain_ids=torch.zeros(x.size(0), dtype=torch.long, device=device))

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            if num_batches % 10 == 0:
                print(f"  Batch {num_batches}: loss={loss.item():.4f}, ce={ce_loss.item():.4f}, hydro={hydro_loss.item():.4f}")

        avg_loss = total_loss / max(num_batches, 1)
        print(f"Epoch {epoch}: Loss {avg_loss:.4f}")

        if epoch % 10 == 0:
            torch.save(model.state_dict(), f"checkpoints/diffusion_epoch{epoch}.pt")

    torch.save(model.state_dict(), "checkpoints/diffusion_final.pt")
    print("Diffusion训练完成！")

def train_esm2(args):
    from models.esm2_gvp import Esm2GVP, train_esm2_gvp 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs('checkpoints', exist_ok=True)

    dataset = GvpDataset(args.fasta, args.msa, max_len=args.max_len, chain_type=args.chain_type)
    dataloader = DataLoader(dataset, batch_size=args.batch_size,
                           shuffle=True, collate_fn=collate_fn)

    esm_model_name = args.esm_model

    model = Esm2GVP(
        vocab_size=VOCAB_SIZE,
        d_model=args.esm_d_model,
        num_species=dataset.num_species,
        esm_model_name=esm_model_name
    ).to(device)

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=0.01
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    print(f"\n{'='*50}")
    print(f"开始训练 ESM2-GVP 模型")
    print(f"ESM2模型: {esm_model_name}")
    print(f"总参数: {sum(p.numel() for p in model.parameters()):,}")
    print(f"可训练参数: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print(f"{'='*50}\n")

    for epoch in range(args.epochs):
        avg_loss, optimizer = train_esm2_gvp(
            model, dataloader, optimizer, device, epoch,
            unfreeze_esm_epoch=args.unfreeze_esm_epoch
        )

        if epoch == args.unfreeze_esm_epoch and epoch > 0:
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, 
                T_max=args.epochs - epoch
            )

        scheduler.step()
        print(f"Epoch {epoch}: Loss {avg_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.2e}")

        if epoch % 5 == 0 or epoch == args.epochs - 1:
            torch.save(model.state_dict(), f"checkpoints/esm2_epoch{epoch}.pt")

    torch.save(model.state_dict(), "checkpoints/esm2_final.pt")
    print(f"\nESM2-GVP 训练完成！")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['vae', 'transformer', 'diffusion', 'esm2'], required=True)
    parser.add_argument('--fasta', default='data', help='FASTA文件或目录路径')
    parser.add_argument('--msa', default=None, help='MSA文件或目录路径')
    parser.add_argument('--chain_type', nargs='+', default=None,
                        help='指定链类型，如 A C F；不指定则加载所有')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--latent_dim', type=int, default=64)
    parser.add_argument('--kl_weight', type=float, default=0.1)
    parser.add_argument('--diffusion_steps', type=int, default=1000)
    parser.add_argument('--max_len', type=int, default=512, help='最大序列长度')
    parser.add_argument('--target_len_A', type=int, default=100, help='GvpA目标长度')
    parser.add_argument('--target_len_C', type=int, default=500, help='GvpC目标长度')

    parser.add_argument('--esm_model', type=str, default='facebook/esm2_t12_35M_UR50D')
    parser.add_argument('--esm_d_model', type=int, default=256)
    parser.add_argument('--unfreeze_esm_epoch', type=int, default=5)

    args = parser.parse_args()

    if args.model == 'vae':
        train_vae(args)
    elif args.model == 'transformer':
        train_transformer(args)
    elif args.model == 'diffusion':
        train_diffusion(args)
    elif args.model == 'esm2':
        train_esm2(args)