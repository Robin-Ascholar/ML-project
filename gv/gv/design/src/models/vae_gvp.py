import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import torch
import torch.nn as nn
import torch.nn.functional as F
import re

class GVAE(nn.Module):
    """条件VAE，支持重复序列控制（改进版，兼容旧权重）

    修复：
    1. 累积重复惩罚（基于token频率）
    2. 动态长度控制 + 强制截断
    3. **兼容旧权重**：自动映射命名 + 缺失参数初始化
    """
    def __init__(self, vocab_size=24, embed_dim=128, hidden_dim=256, 
                 latent_dim=64, num_species=10, max_repeat=20,
                 target_len_A=100, target_len_C=500):
        super().__init__()

        self.vocab_size = vocab_size
        self.target_len_A = target_len_A
        self.target_len_C = target_len_C

        # 物种嵌入
        self.species_embed = nn.Embedding(num_species, 32)

        # 重复次数控制
        self.repeat_embed = nn.Embedding(max_repeat, 32)

        # 长度预测头
        self.length_predictor = nn.Sequential(
            nn.Linear(latent_dim + 64, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

        # 编码器
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.encoder_lstm = nn.LSTM(embed_dim + 64, hidden_dim, 
                                    num_layers=2, batch_first=True, 
                                    bidirectional=True, dropout=0.2)

        # 潜在空间
        self.fc_mu = nn.Linear(hidden_dim * 2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim * 2, latent_dim)

        # 解码器（恢复原始维度：latent_dim + 64 + embed_dim）
        self.decoder_lstm = nn.LSTM(latent_dim + 64 + embed_dim, hidden_dim, 
                                    num_layers=2, batch_first=True,
                                    dropout=0.2)

        self.fc_out = nn.Linear(hidden_dim, vocab_size)

        self.latent_dim = latent_dim

    def encode(self, x, species, repeat_count=None):
        # 条件嵌入
        sp_emb = self.species_embed(species)
        if repeat_count is None:
            rep_emb = torch.zeros(x.size(0), 32, device=x.device)
        else:
            rep_emb = self.repeat_embed(repeat_count)

        cond = torch.cat([sp_emb, rep_emb], dim=-1).unsqueeze(1).expand(-1, x.size(1), -1)

        # 编码
        emb = self.embedding(x)
        x_cond = torch.cat([emb, cond], dim=-1)

        _, (h, _) = self.encoder_lstm(x_cond)
        h = torch.cat([h[-2], h[-1]], dim=-1)

        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar, cond[:, 0, :]

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, cond, max_len=200, teacher_forcing=None,
               length_scale=1.0):
        """自回归解码（简化版，移除有问题的自注意力）

        Args:
            z: [B, latent_dim] 潜在向量
            cond: [B, 64] 条件向量
            max_len: 最大解码长度
            teacher_forcing: [B, L] 教师强制输入（训练时）
            length_scale: 长度缩放因子
        """
        batch_size = z.size(0)
        device = z.device

        # 预测目标长度
        length_input = torch.cat([z, cond], dim=-1)
        length_ratio = self.length_predictor(length_input)
        target_len = int(max_len * length_ratio.mean().item() * length_scale)
        target_len = max(40, min(target_len, max_len))  # 限制40-max_len

        # 初始输入
        input_seq = torch.full((batch_size, 1), 21, dtype=torch.long, device=device)  # <SOS>
        outputs = []
        hidden = None

        # 累积token频率（用于重复惩罚）
        token_freq = torch.zeros(batch_size, self.vocab_size, device=device)

        for t in range(target_len):
            emb = self.embedding(input_seq)  # [B, 1, E=128]

            z_exp = z.unsqueeze(1)          # [B, 1, latent_dim]
            cond_exp = cond.unsqueeze(1)    # [B, 1, 64]
            lstm_input = torch.cat([emb, z_exp, cond_exp], dim=-1)

            out, hidden = self.decoder_lstm(lstm_input, hidden)
            logits = self.fc_out(out.squeeze(1))  # [B, V]

            # === 累积重复惩罚 ===
            if len(outputs) > 0:
                # 计算当前频率分布（平滑）
                freq_penalty = (token_freq[:, :20] / (token_freq[:, :20].sum(dim=1, keepdim=True) + 1)) * 3.0
                # 对高频token施加惩罚
                logits[:, :20] -= freq_penalty

                # 特别惩罚连续相同token
                last_token = input_seq[:, 0]
                for i in range(batch_size):
                    if token_freq[i, last_token[i]] > 2:  # 已出现2次以上
                        logits[i, last_token[i]] -= 5.0  # 强惩罚

            outputs.append(logits)

            if teacher_forcing is not None:
                input_seq = teacher_forcing[:, t:t+1]
                next_token = input_seq
            else:
                probs = F.softmax(logits / 0.8, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

            # 更新token频率
            token_freq.scatter_add_(1, next_token.view(batch_size, 1), 
                                   torch.ones(batch_size, 1, device=device))

            input_seq = next_token

            # 检查EOS或强制截断
            if t >= target_len - 1:
                break

            # 早期停止：若陷入重复
            if t > 20 and self._detect_early_stop(token_freq, threshold=0.8):
                break

        return torch.stack(outputs, dim=1) if outputs else torch.zeros(batch_size, 1, self.vocab_size)

    def _detect_early_stop(self, token_freq, threshold=0.8):
        """检测是否陷入重复模式（基于频率熵）"""
        # 计算频率分布的熵
        freq_norm = token_freq / (token_freq.sum(dim=1, keepdim=True) + 1e-10)
        entropy = -(freq_norm * torch.log(freq_norm + 1e-10)).sum(dim=1)
        # 熵过低 = 分布过于集中 = 重复
        return (entropy < threshold).any()

    def forward(self, x, species, repeat_count=None):
        mu, logvar, cond = self.encode(x, species, repeat_count)
        z = self.reparameterize(mu, logvar)

        teacher_input = x[:, 1:] if x.size(1) > 1 else x
        outputs = self.decode(z, cond, max_len=teacher_input.size(1), 
                             teacher_forcing=teacher_input)

        return outputs, mu, logvar

    @torch.no_grad()
    def generate(self, species, repeat_count=None, num_samples=10, 
                 max_len=200, temperature=1.0,
                 target_length=None,
                 filter_invalid=True):
        """改进的生成：长度可控 + 质量过滤 + 重复截断

        Args:
            species: 物种索引
            repeat_count: 重复次数
            num_samples: 样本数
            max_len: 最大长度
            temperature: 采样温度
            target_length: 目标长度（None则自动预测）
            filter_invalid: 是否过滤非法序列
        """
        device = next(self.parameters()).device
        z = torch.randn(num_samples, self.latent_dim, device=device)

        species = torch.tensor([species] * num_samples, device=device)
        if repeat_count is not None:
            repeat_count = torch.tensor([repeat_count] * num_samples, device=device)

        _, _, cond = self.encode(
            torch.zeros(num_samples, 1, dtype=torch.long, device=device), 
            species, repeat_count
        )

        # 确定目标长度
        if target_length is None:
            target_length = self.target_len_A

        # 自回归生成
        batch_size = num_samples
        input_seq = torch.full((batch_size, 1), 21, dtype=torch.long, device=device)
        generated = [input_seq]
        hidden = None
        token_history = [[] for _ in range(batch_size)]
        token_freq = torch.zeros(batch_size, self.vocab_size, device=device)

        for t in range(min(max_len, target_length + 20)):
            emb = self.embedding(input_seq)
            z_exp = z.unsqueeze(1)
            cond_exp = cond.unsqueeze(1)

            lstm_input = torch.cat([emb, z_exp, cond_exp], dim=-1)
            out, hidden = self.decoder_lstm(lstm_input, hidden)
            logits = self.fc_out(out.squeeze(1)) / temperature

            # 累积重复惩罚
            if t > 0:
                freq_penalty = (token_freq[:, :20] / (token_freq[:, :20].sum(dim=1, keepdim=True) + 1)) * 3.0
                logits[:, :20] -= freq_penalty

                for i in range(batch_size):
                    if token_history[i] and len(token_history[i]) > 1:
                        last_3 = token_history[i][-3:]
                        if len(set(last_3)) == 1:  # 连续3个相同
                            logits[i, last_3[-1]] = -1e9

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1)

            for i in range(batch_size):
                token_history[i].append(next_token[i].item())
            token_freq.scatter_add_(1, next_token, torch.ones_like(next_token, dtype=torch.float))

            generated.append(next_token)
            input_seq = next_token

            # 检查EOS
            if (next_token == 23).any():
                break

            # 强制截断
            if t >= target_length - 1:
                next_token = torch.full_like(next_token, 23)
                generated.append(next_token)
                break

        result = torch.cat(generated, dim=1)

        # 后处理
        if filter_invalid:
            result = self._clean_sequence(result, device)

        return result

    def _clean_sequence(self, sequences, device):
        """清理序列：移除无效字符，截断重复模式"""
        cleaned = []
        for seq in sequences:
            tokens = seq.tolist()
            # 移除PAD(23)和无效token
            tokens = [t for t in tokens if t < 20]

            # 检测连续重复（3次以上相同5-mer）
            seq_str = ''.join([chr(65 + t) if t < 26 else 'X' for t in tokens])

            for length in range(3, 8):
                pattern = r'(.{%d})\1{2,}' % length
                match = re.search(pattern, seq_str)
                if match:
                    cut = match.start() + length
                    tokens = tokens[:cut]
                    tokens.append(23)
                    break

            cleaned.append(torch.tensor(tokens, device=device))

        # 填充
        max_len = max(len(c) for c in cleaned) if cleaned else 1
        padded = torch.full((len(sequences), max_len), 23, dtype=torch.long, device=device)
        for i, c in enumerate(cleaned):
            if len(c) > 0:
                padded[i, :len(c)] = c

        return padded

    # ========== 兼容旧权重加载（核心修复v2）==========
    def load_state_dict(self, state_dict, strict=True):
        """兼容旧权重：
        1. 将旧版 'attention' 映射到新版 'self_attn'
        2. 对新增加的参数进行初始化而非报错
        """
        new_state_dict = {}
        mapped_keys = []

        for key, value in state_dict.items():
            # 旧版命名: attention.in_proj_weight -> 新版: self_attn.in_proj_weight
            if key.startswith('attention.'):
                new_key = key.replace('attention.', 'self_attn.')
                new_state_dict[new_key] = value
                mapped_keys.append(f"{key} -> {new_key}")
            else:
                new_state_dict[key] = value

        # 获取当前模型的状态字典（用于检查缺失键）
        current_state = self.state_dict()

        # 找出新模型有但旧权重中没有的参数
        missing_in_old = set(current_state.keys()) - set(new_state_dict.keys())

        if missing_in_old:
            print(f"[权重兼容] 发现 {len(missing_in_old)} 个新参数在旧权重中不存在，将自动初始化:")
            for key in sorted(missing_in_old):
                print(f"    - {key} (shape: {current_state[key].shape})")
                # 使用当前模型的初始化值
                new_state_dict[key] = current_state[key]

        if mapped_keys:
            print(f"[权重兼容] 映射 {len(mapped_keys)} 个旧参数名:")
            for msg in mapped_keys:
                print(f"    - {msg}")

        # 调用父类方法加载（strict=False 以容忍剩余的不匹配）
        super().load_state_dict(new_state_dict, strict=False)

        if missing_in_old:
            print(f"[权重兼容] 旧权重加载完成，{len(missing_in_old)} 个新参数已用默认值初始化")
        else:
            print("[权重兼容] 旧权重加载完成，所有参数匹配")

# VAE损失函数（改进）
def vae_loss(recon_x, x, mu, logvar, kl_weight=1.0,
             length_penalty=0.001):
    """VAE损失 = 重构损失 + KL散度 + 长度正则化"""
    mask = (x != 23).float()

    recon_loss = F.cross_entropy(
        recon_x.reshape(-1, recon_x.size(-1)),  
        x.reshape(-1),                           
        reduction='none'
    )
    recon_loss = (recon_loss * mask.reshape(-1)).sum() / mask.sum()

    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / mu.size(0)

    # 长度正则化（惩罚过长序列）
    seq_lengths = mask.sum(dim=1)
    length_loss = F.relu(seq_lengths - 150).mean() * length_penalty

    return recon_loss + kl_weight * kl_loss + length_loss, recon_loss, kl_loss