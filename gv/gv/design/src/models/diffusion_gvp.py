import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class GvpDiffusion(nn.Module):
    """D3PM离散扩散模型，用于GvpA-GvpC联合生成（改进版）

    基于 Austin et al. 2021 "Structured Denoising Diffusion Models in Discrete State-Spaces"
    结合 EvoDiff 混合损失设计，加入疏水性正则化。
    """
    def __init__(self, vocab_size=24, d_model=256, num_steps=1000,
                 num_species=10, num_chains=2,
                 beta_schedule='cosine',
                 cfg_scale=2.0,
                 hydro_weight=0.5,
                 diversity_weight=0.1):
        super().__init__()

        self.vocab_size = vocab_size
        self.num_steps = num_steps
        self.num_chains = num_chains
        self.cfg_scale = cfg_scale
        self.hydro_weight = hydro_weight
        self.diversity_weight = diversity_weight

        # ========== D3PM 离散转移矩阵 ==========
        # 使用均匀转移 + 吸收态（mask token=21 作为吸收态）
        self._build_transition_mat(beta_schedule)

        # ========== 嵌入层 ==========
        self.token_embed = nn.Embedding(vocab_size, d_model)

        # 时间步嵌入
        self.time_embed = nn.Sequential(
            nn.Linear(1, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model)
        )

        # 物种和链嵌入
        self.species_embed = nn.Embedding(num_species, d_model)
        self.chain_embed = nn.Embedding(num_chains, d_model)

        # ========== 去噪网络 ==========
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=8, 
            dim_feedforward=1024,
            dropout=0.1,
            batch_first=True,
            norm_first=True
        )
        self.denoiser = nn.TransformerEncoder(encoder_layer, num_layers=6)

        self.pre_out_dropout = nn.Dropout(0.1)
        self.fc_out = nn.Linear(d_model, vocab_size)

        # ========== 疏水性正则化表 ==========
        # Kyte-Doolittle 疏水性指数（仅20种天然氨基酸）
        # A,C,D,E,F,G,H,I,K,L,M,N,P,Q,R,S,T,V,W,Y
        hydropathy_vals = torch.tensor([
            1.8,  2.5, -3.5, -3.5,  2.8, -0.4, -3.2,  4.5, -3.9,  3.8,
            1.9, -3.5, -1.6, -3.5, -4.5, -0.8, -0.7,  4.2, -0.9, -1.3,
            0.0,  0.0,  0.0,  0.0   # gap, pad, sos, eos = neutral
        ])
        self.register_buffer('hydropathy', hydropathy_vals)

        self._init_parameters()

    def _init_parameters(self):
        """Xavier初始化"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _build_transition_mat(self, schedule='cosine'):
        """构建D3PM离散转移矩阵 Q_t 和累积矩阵 ar{Q}_t"""
        V = self.vocab_size

        if schedule == 'cosine':
            steps = torch.arange(self.num_steps + 1, dtype=torch.float32)
            alphas_cumprod = torch.cos(
                ((steps / self.num_steps) + 0.008) / 1.008 * math.pi / 2
            ) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            betas = torch.clip(betas, 0.0001, 0.9999)
        else:
            betas = torch.linspace(1e-4, 0.02, self.num_steps)

        alphas = 1.0 - betas

        # 构建转移矩阵 Q_t: [T, V, V]
        # 均匀转移：以 beta_t 概率随机跳到其他 token，保留 alpha_t 概率不变
        Q = torch.zeros(self.num_steps, V, V)
        for t in range(self.num_steps):
            q = torch.full((V, V), betas[t] / (V - 1))
            q.fill_diagonal_(alphas[t])
            Q[t] = q

        # 累积转移矩阵 ar{Q}_t
        Q_bar = torch.zeros(self.num_steps, V, V)
        Q_bar[0] = Q[0]
        for t in range(1, self.num_steps):
            Q_bar[t] = Q_bar[t-1] @ Q[t]

        self.register_buffer('Q', Q)
        self.register_buffer('Q_bar', Q_bar)

        # 预计算用于采样的概率
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)

    def q_sample(self, x_0, t):
        """D3PM前向扩散：根据 ar{Q}_t 采样 x_t

        Args:
            x_0: [B, L] 原始离散序列索引
            t: [B] 时间步（或标量）
        Returns:
            x_t: [B, L] 加噪后的离散序列
        """
        if t.dim() == 0:
            t = t.unsqueeze(0).expand(x_0.size(0))

        batch_size, seq_len = x_0.shape

        # x_0 转为 one-hot: [B, L, V]
        x_0_oh = F.one_hot(x_0, self.vocab_size).float()

        # 获取对应时间步的累积转移矩阵: [B, V, V]
        q_bar = self.Q_bar[t]  # [B, V, V]

        # 计算 x_t 的分布: p(x_t | x_0) = x_0_oh @ ar{Q}_t
        # [B, L, V] @ [B, V, V] -> [B, L, V]
        prob = torch.einsum('blv,bvV->blV', x_0_oh, q_bar)

        # 从多项分布采样
        prob_flat = prob.reshape(-1, self.vocab_size)  # [B*L, V]
        x_t_flat = torch.multinomial(prob_flat, 1).squeeze(-1)
        x_t = x_t_flat.view(batch_size, seq_len)

        return x_t

    def forward(self, x_t, t, species, chain_ids, drop_cond=False):
        """去噪网络（支持CFG训练）

        Args:
            x_t: [B, L] 离散加噪序列
            t: [B] 时间步
            species: [B] 物种标签
            chain_ids: [B] 链标签
            drop_cond: 是否丢弃条件（CFG训练用）
        """
        # Token 嵌入
        x_emb = self.token_embed(x_t)  # [B, L, D]

        # 条件嵌入
        if drop_cond and self.training:
            # 训练时随机丢弃条件（10%概率）
            t_emb = torch.zeros(x_t.size(0), 1, x_emb.size(-1), device=x_t.device)
            sp_emb = torch.zeros_like(t_emb)
            ch_emb = torch.zeros_like(t_emb)
        else:
            t_emb = self.time_embed(t.float().unsqueeze(-1) / self.num_steps).unsqueeze(1)
            sp_emb = self.species_embed(species).unsqueeze(1)
            ch_emb = self.chain_embed(chain_ids).unsqueeze(1)

        h = x_emb + t_emb + sp_emb + ch_emb

        # 去噪Transformer
        h = self.denoiser(h)
        h = self.pre_out_dropout(h)
        logits = self.fc_out(h)  # [B, L, V]

        return logits

    def p_losses(self, x_0, t, species, chain_ids):
        """D3PM训练损失（混合损失：CE + 疏水正则 + 多样性正则）

        Returns:
            loss: 总损失
            ce_loss: 交叉熵损失
            hydro_loss: 疏水性正则损失
        """
        # 前向加噪
        x_t = self.q_sample(x_0, t)

        # 去噪预测
        logits = self.forward(x_t, t, species, chain_ids)

        # === 1) 标准交叉熵损失（预测 x_0）===
        ce_loss = F.cross_entropy(
            logits.reshape(-1, self.vocab_size),
            x_0.reshape(-1),
            ignore_index=21,  # 忽略 PAD
            reduction='mean'
        )

        # === 2) 疏水性正则化 ===
        # 计算预测分布的期望疏水性
        probs = F.softmax(logits, dim=-1)  # [B, L, V]
        expected_hydro = (probs * self.hydropathy.view(1, 1, -1)).sum(dim=-1)  # [B, L]

        # 惩罚极端疏水性（> 2.5 或 < -2.5）
        hydro_penalty = F.relu(expected_hydro - 2.5).mean() + F.relu(-2.5 - expected_hydro).mean()
        hydro_loss = self.hydro_weight * hydro_penalty

        # === 3) 多样性正则化 ===
        # 鼓励相邻位置的概率分布差异（避免重复）
        if logits.size(1) > 1:
            prob_diff = torch.abs(probs[:, 1:, :] - probs[:, :-1, :]).mean()
            diversity_loss = -self.diversity_weight * prob_diff
        else:
            diversity_loss = torch.tensor(0.0, device=logits.device)

        # === 总损失 ===
        loss = ce_loss + hydro_loss + diversity_loss

        return loss, ce_loss, hydro_loss

    @torch.no_grad()
    def p_sample(self, x_t, t, species, chain_ids, cfg_scale=None):
        """单步去噪采样（支持CFG）

        Args:
            x_t: [B, L] 当前离散序列
            t: [B] 时间步
        Returns:
            x_0_pred: [B, L] 预测的x_0
            x_t_prev: [B, L] 上一步的x_{t-1}
        """
        if cfg_scale is None:
            cfg_scale = self.cfg_scale

        batch_size = x_t.size(0)

        # 条件预测
        logits_cond = self.forward(x_t, t, species, chain_ids)

        # 无条件预测（CFG）
        if cfg_scale > 1.0:
            logits_uncond = self.forward(x_t, t, species, chain_ids, drop_cond=True)
            logits = logits_uncond + cfg_scale * (logits_cond - logits_uncond)
        else:
            logits = logits_cond

        # 预测 x_0 的分布
        x_0_probs = F.softmax(logits, dim=-1)  # [B, L, V]

        # 采样 x_0
        x_0_pred = torch.multinomial(
            x_0_probs.reshape(-1, self.vocab_size), 1
        ).squeeze(-1).view(batch_size, -1)

        # 计算 x_{t-1} 的分布（使用 D3PM 后验）
        # p(x_{t-1} | x_t, x_0) ∝ p(x_t | x_{t-1}) p(x_{t-1} | x_0)
        # 简化为：使用预测 x_0 重新加噪到 t-1 步
        if (t > 0).any():
            t_prev = torch.clamp(t - 1, min=0)
            x_t_prev = self.q_sample(x_0_pred, t_prev)
        else:
            x_t_prev = x_0_pred

        return x_0_pred, x_t_prev

    @torch.no_grad()
    def generate(self, species, chain_A=True, chain_C=True, 
                 max_len_A=100, max_len_C=500, batch_size=10,
                 cfg_scale=2.0,
                 temperature=1.0,
                 filter_invalid=True):
        """改进的生成：D3PM离散采样 + CFG + 后处理过滤

        Args:
            species: 物种索引
            chain_A: 是否生成GvpA
            chain_C: 是否生成GvpC
            max_len_A/C: 最大长度
            batch_size: 批量大小
            cfg_scale: CFG强度
            temperature: 采样温度
            filter_invalid: 是否过滤非法序列
        """
        device = next(self.parameters()).device
        sequences = []

        if chain_A:
            # 从全mask开始
            x = torch.full((batch_size, max_len_A), 21, 
                           dtype=torch.long, device=device)
            species_b = torch.full((batch_size,), species, device=device, dtype=torch.long)
            chain_b = torch.zeros(batch_size, dtype=torch.long, device=device)

            # 反向扩散
            for t in reversed(range(self.num_steps)):
                t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)
                _, x = self.p_sample(x, t_batch, species_b, chain_b, 
                                     cfg_scale=cfg_scale)

            # 最终解码
            final_logits = self.forward(x, 
                torch.zeros(batch_size, device=device, dtype=torch.long),
                species_b, chain_b)
            seq_A = torch.argmax(final_logits / temperature, dim=-1)

            # 后处理
            if filter_invalid:
                seq_A = self._filter_sequence(seq_A)
            sequences.append(('A', seq_A))

        if chain_C:
            x = torch.full((batch_size, max_len_C), 21,
                           dtype=torch.long, device=device)
            species_b = torch.full((batch_size,), species, device=device, dtype=torch.long)
            chain_b = torch.ones(batch_size, dtype=torch.long, device=device)

            for t in reversed(range(self.num_steps)):
                t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)
                _, x = self.p_sample(x, t_batch, species_b, chain_b,
                                     cfg_scale=cfg_scale)

            final_logits = self.forward(x,
                torch.zeros(batch_size, device=device, dtype=torch.long),
                species_b, chain_b)
            seq_C = torch.argmax(final_logits / temperature, dim=-1)

            if filter_invalid:
                seq_C = self._filter_sequence(seq_C)
            sequences.append(('C', seq_C))

        return sequences

    def _filter_sequence(self, sequences, max_repeat=4):
        """后处理：截断连续重复，过滤非法token"""
        result = []
        for seq in sequences:
            tokens = seq.tolist()
            # 过滤 pad/gap/sos
            tokens = [t for t in tokens if t < 20]  # 只保留天然氨基酸

            # 截断连续重复
            truncated = []
            repeat_count = 1
            for i, token in enumerate(tokens):
                if i > 0 and token == tokens[i-1]:
                    repeat_count += 1
                    if repeat_count <= max_repeat:
                        truncated.append(token)
                else:
                    repeat_count = 1
                    truncated.append(token)

            result.append(torch.tensor(truncated, device=seq.device))

        # 填充
        max_len = max(len(r) for r in result) if result else 1
        padded = torch.full((len(sequences), max_len), 23,
                           dtype=torch.long, device=sequences.device)
        for i, r in enumerate(result):
            if len(r) > 0:
                padded[i, :len(r)] = r

        return padded