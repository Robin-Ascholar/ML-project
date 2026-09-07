import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import torch
import torch.nn as nn
import math
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                            (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class GvpTransformer(nn.Module):
    """MSA-Conditioned Transformer for GvpA generation（改进版）

    修复：
    1. MSA条件改为可学习门控，避免过度保守
    2. 生成阶段加入n-gram重复阻断
    3. 多样性增强：动态温度衰减
    """
    def __init__(self, vocab_size=24, d_model=256, nhead=8, 
                 num_layers=6, dim_feedforward=1024, dropout=0.15,
                 num_species=10):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)

        # 物种条件
        self.species_embed = nn.Embedding(num_species, d_model)

        # MSA条件投影
        self.msa_project = nn.Sequential(
            nn.Linear(20, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout)
        )

        # MSA门控网络（新增）：让模型决定用多少MSA信息
        self.msa_gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid()
        )

        # 改进的Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        self.pre_out_dropout = nn.Dropout(dropout)
        self.fc_out = nn.Linear(d_model, vocab_size)

        self.d_model = d_model
        self._init_parameters()

    def _init_parameters(self):
        """Xavier初始化"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, species, msa_weights=None, mask=None):
        """
        src: [B, L] 输入序列
        species: [B] 物种标签
        msa_weights: [B, L, 20] MSA保守性权重（可选）
        """
        # 嵌入 + 位置编码
        x = self.embedding(src) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)

        # 添加物种条件
        sp_emb = self.species_embed(species).unsqueeze(1)
        x = x + sp_emb

        # MSA条件（门控融合，新增）
        if msa_weights is not None:
            msa_emb = self.msa_project(msa_weights)

            # 对齐长度
            if msa_emb.size(1) != x.size(1):
                if msa_emb.size(1) > x.size(1):
                    msa_emb = msa_emb[:, :x.size(1), :]
                else:
                    pad_len = x.size(1) - msa_emb.size(1)
                    msa_emb = F.pad(msa_emb, (0, 0, 0, pad_len))

            # 门控融合：gate -> 0 保留更多原始信息，gate -> 1 更多MSA
            gate_input = torch.cat([x, msa_emb], dim=-1)
            gate = self.msa_gate(gate_input)  # [B, L, D]
            x = gate * x + (1 - gate) * msa_emb  # 反向：gate小则MSA影响小

        # 因果掩码
        if mask is None:
            mask = nn.Transformer.generate_square_subsequent_mask(src.size(1), 
                                                                  device=src.device)

        output = self.transformer(x, mask=mask, is_causal=True)
        logits = self.fc_out(output)
        return logits

    @torch.no_grad()
    def generate(self, species, msa_weights=None, max_len=100, 
                 temperature=1.0, top_p=0.9,
                 repetition_penalty=1.15,
                 diversity_boost=0.05,
                 min_len=50,
                 ngram_block=4):
        """改进的自回归生成：n-gram阻断 + 动态温度 + 多样性增强

        Args:
            species: 物种索引或列表
            msa_weights: MSA权重
            max_len: 最大长度
            temperature: 采样温度
            top_p: nucleus sampling阈值
            repetition_penalty: 重复惩罚
            diversity_boost: 多样性增强系数
            min_len: 最小长度
            ngram_block: n-gram阻断窗口大小
        """
        device = next(self.parameters()).device
        batch_size = 1 if isinstance(species, int) else len(species)

        if isinstance(species, int):
            species = torch.tensor([species], device=device)

        # 从SOS开始
        generated = torch.full((batch_size, 1), 22, dtype=torch.long, device=device)
        token_logprobs = []

        for step in range(max_len):
            # 动态调整MSA权重（随生成进度衰减）
            current_msa = None
            if msa_weights is not None:
                decay = max(0.1, 1.0 - step / max_len)  # 从1.0衰减到0.1
                current_msa = msa_weights[:, :generated.size(1), :] * decay

            logits = self.forward(generated, species, msa_weights=current_msa)
            next_token_logits = logits[:, -1, :] / temperature

            # 重复惩罚
            if repetition_penalty != 1.0 and step > 0:
                for i in range(batch_size):
                    for token in set(generated[i].tolist()):
                        if token not in [21, 22, 23]:
                            next_token_logits[i, token] /= repetition_penalty

            # 多样性增强：动态温度衰减
            if diversity_boost > 0:
                # 随着生成进行，逐渐提高温度以增加多样性
                dynamic_temp = temperature * (1.0 + step / max_len * diversity_boost)
                # 修正：温度越高越多样，应除以更高的温度（压缩logits分布）
                next_token_logits = next_token_logits / dynamic_temp

            # n-gram阻断（核心修复）
            if step >= ngram_block and ngram_block > 0:
                next_token_logits = self._ngram_block(
                    next_token_logits, generated, n=ngram_block
                )

            # 最小长度控制
            if step < min_len:
                next_token_logits[:, 23] = -float('Inf')  # 禁止EOS

            # Nucleus sampling
            if top_p > 0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                next_token_logits[indices_to_remove] = -float('Inf')

            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            token_logprobs.append(torch.log(probs.gather(1, next_token) + 1e-10))

            generated = torch.cat([generated, next_token], dim=1)

            # 检查EOS
            if (next_token == 23).all() and step >= min_len:
                break

        return generated

    def _ngram_block(self, logits, generated, n=4):
        """n-gram阻断：降低已出现n-gram后续token的概率

        Args:
            logits: [B, V] 当前步logits
            generated: [B, T] 已生成序列
            n: n-gram大小
        """
        batch_size = generated.size(0)
        for i in range(batch_size):
            seq = generated[i].tolist()
            if len(seq) < n:
                continue

            # 获取最后n-1个token作为前缀
            prefix = tuple(seq[-(n-1):])

            # 在历史序列中查找匹配的n-gram前缀
            for j in range(len(seq) - n + 1):
                if tuple(seq[j:j+n-1]) == prefix:
                    # 找到匹配，降低对应第n个token的概率
                    blocked_token = seq[j + n - 1]
                    if blocked_token < logits.size(1):
                        logits[i, blocked_token] -= 5.0  # 强惩罚

        return logits


    # ========== 兼容旧权重加载（核心修复）==========
    def load_state_dict(self, state_dict, strict=True):
        """兼容旧权重：对新增加的参数（如msa_gate）进行初始化而非报错"""
        # 获取当前模型的状态字典
        current_state = self.state_dict()

        # 找出新模型有但旧权重中没有的参数
        missing_in_old = set(current_state.keys()) - set(state_dict.keys())

        new_state_dict = dict(state_dict)

        if missing_in_old:
            print(f"[权重兼容] 发现 {len(missing_in_old)} 个新参数在旧权重中不存在，将自动初始化:")
            for key in sorted(missing_in_old):
                print(f"    - {key} (shape: {list(current_state[key].shape)})")
                # 使用当前模型的初始化值
                new_state_dict[key] = current_state[key]

        # 调用父类方法加载（strict=False以容忍剩余不匹配）
        super().load_state_dict(new_state_dict, strict=False)

        if missing_in_old:
            print(f"[权重兼容] 旧权重加载完成，{len(missing_in_old)} 个新参数已用默认值初始化")
        else:
            print("[权重兼容] 旧权重加载完成，所有参数匹配")

if __name__ == "__main__":
    print("测试改进版Transformer...")
    try:
        model = GvpTransformer(vocab_size=24, num_species=5)
        print(f"模型初始化成功！参数数量: {sum(p.numel() for p in model.parameters()):,}")

        test_seq = torch.randint(0, 20, (2, 50))  # 只使用天然氨基酸
        test_species = torch.tensor([0, 1])
        output = model(test_seq, test_species)
        print(f"前向测试成功，输出形状: {output.shape}")

        # 测试生成
        generated = model.generate(species=0, max_len=20, top_p=0.9, ngram_block=3)
        print(f"生成测试成功，序列形状: {generated.shape}")
        print("所有测试通过！")

    except Exception as e:
        print(f"错误: {e}")