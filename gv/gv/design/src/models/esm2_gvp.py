import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import EsmModel, EsmTokenizer

class Esm2GVP(nn.Module):
    """ESM2 + 条件生成头，用于GVP序列生成（改进版）

    修复：
    1. 生成阶段屏蔽非法token（gap/pad/sos）
    2. 强制最小长度截断
    3. 非法token回退机制
    """
    def __init__(self, vocab_size=24, d_model=256, num_species=10, 
                 esm_model_name="facebook/esm2_t12_35M_UR50D",
                 dropout=0.15):
        super().__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model

        # 加载预训练ESM2
        print(f"Loading ESM2 model: {esm_model_name}")
        self.esm = EsmModel.from_pretrained(esm_model_name)
        self.esm_dim = self.esm.config.hidden_size

        # 默认解冻更多层（从第8层开始，共12层）
        self.freeze_esm = True
        self._freeze_esm_layers(freeze_until_layer=8)

        # 物种嵌入
        self.species_embed = nn.Embedding(num_species, 128)

        # ESM2输出投影
        self.esm_project = nn.Sequential(
            nn.Linear(self.esm_dim, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # 条件融合层
        self.condition_fusion = nn.Sequential(
            nn.Linear(d_model + 128, d_model * 2),
            nn.LayerNorm(d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model)
        )

        # 生成头
        self.generation_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, vocab_size)
        )

        # 位置编码
        self.pos_embed = nn.Embedding(2000, d_model)

        # 重复惩罚参数
        self.repetition_penalty = 1.2
        self.length_penalty = 0.01

    def _freeze_esm_layers(self, freeze_until_layer=8):
        """分层冻结ESM2"""
        total_layers = len(self.esm.encoder.layer)
        for i, layer in enumerate(self.esm.encoder.layer):
            if i < freeze_until_layer:
                for param in layer.parameters():
                    param.requires_grad = False
            else:
                for param in layer.parameters():
                    param.requires_grad = True
        print(f"ESM2: frozen layers 0-{freeze_until_layer-1}, "
              f"unfrozen {freeze_until_layer}-{total_layers-1}")

    def forward(self, input_ids, species, attention_mask=None, labels=None):
        batch_size, seq_len = input_ids.shape

        # ESM2编码
        if self.training and self.freeze_esm:
            with torch.no_grad():
                esm_outputs = self.esm(input_ids=input_ids, attention_mask=attention_mask)
        else:
            esm_outputs = self.esm(input_ids=input_ids, attention_mask=attention_mask)

        esm_hidden = esm_outputs.last_hidden_state

        # 投影
        hidden = self.esm_project(esm_hidden)

        # 位置编码
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        pos_emb = self.pos_embed(positions)
        hidden = hidden + pos_emb

        # 物种条件
        sp_emb = self.species_embed(species).unsqueeze(1).expand(-1, seq_len, -1)
        hidden = torch.cat([hidden, sp_emb], dim=-1)
        hidden = self.condition_fusion(hidden)

        # 生成logits
        logits = self.generation_head(hidden)

        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()

            loss_fct = nn.CrossEntropyLoss(ignore_index=23, label_smoothing=0.1)
            ce_loss = loss_fct(shift_logits.view(-1, self.vocab_size), shift_labels.view(-1))

            repeat_loss = self._compute_repeat_loss(shift_logits, shift_labels)
            loss = ce_loss + 0.1 * repeat_loss

        return {
            'loss': loss,
            'logits': logits,
            'hidden_states': hidden
        }

    def _compute_repeat_loss(self, logits, labels):
        """计算重复token的惩罚损失"""
        probs = F.softmax(logits, dim=-1)
        prob_diff = torch.abs(probs[:, 1:, :] - probs[:, :-1, :]).mean()
        return -prob_diff

    @torch.no_grad()
    def generate(self, species, max_len=100, temperature=1.0, top_p=0.9,
                 device='cuda', start_token=22, end_token=23,
                 repetition_penalty=1.2, min_len=50,
                 hydro_target=(0.35, 0.55),  # 目标疏水性范围
                 hydro_weight=0.3):  # 疏水性控制强度
        """改进的生成：屏蔽非法token + 最小长度控制 + 回退机制 + 疏水性控制

        Args:
            species: 物种索引或列表
            max_len: 最大生成长度
            temperature: 采样温度
            top_p: nucleus sampling阈值
            device: 计算设备
            start_token: 起始token索引
            end_token: 结束token索引
            repetition_penalty: 重复惩罚系数
            min_len: 最小长度（达到前禁止生成EOS）
            hydro_target: 目标疏水性范围 (min, max)
            hydro_weight: 疏水性控制强度 (0-1)
        """
        self.eval()

        if isinstance(species, int):
            species = torch.tensor([species], device=device)
        else:
            species = torch.tensor(species, device=device)

        batch_size = species.size(0)
        generated = torch.full((batch_size, 1), start_token, dtype=torch.long, device=device)

        # 记录已生成的token用于重复惩罚
        token_counts = torch.zeros(batch_size, self.vocab_size, device=device)

        # 疏水性指数表 (Kyte-Doolittle)
        hydropathy = torch.tensor([
            1.8, 2.5, -3.5, -3.5, 2.8, -0.4, -3.2, 4.5, -3.9, 3.8,
            1.9, -3.5, -1.6, -3.5, -4.5, -0.8, -0.7, 4.2, -0.9, -1.3,
            0.0, 0.0, 0.0, 0.0
        ], device=device)

        for step in range(max_len):
            attention_mask = torch.ones_like(generated)
            outputs = self.forward(generated, species, attention_mask=attention_mask)
            logits = outputs['logits'][:, -1, :] / temperature

            # 1) 应用重复惩罚
            if repetition_penalty != 1.0 and step > 0:
                mask = token_counts > 0
                logits[mask] = logits[mask] / repetition_penalty
                if generated.size(1) > 1:
                    last_token = generated[:, -1]
                    logits.scatter_(1, last_token.unsqueeze(1), 
                                   logits.gather(1, last_token.unsqueeze(1)) / 2.0)

            # 2) 应用非法token屏蔽
            logits[:, 20] = -float('Inf')
            logits[:, 21] = -float('Inf')
            logits[:, 22] = -float('Inf')

            # 3) 最小长度控制
            if step < min_len:
                logits[:, end_token] = -float('Inf')

            # 4) 疏水性动态控制（新增）
            if step > 5 and hydro_weight > 0:
                current_seq = generated[:, 1:]
                current_hydro = torch.zeros(batch_size, device=device)
                for b in range(batch_size):
                    valid_tokens = current_seq[b][current_seq[b] < 20]
                    if len(valid_tokens) > 0:
                        current_hydro[b] = hydropathy[valid_tokens].mean()

                for b in range(batch_size):
                    if current_hydro[b] > hydro_target[1]:
                        hydro_mask = hydropathy[:20] > 2.0
                        logits[b, :20][hydro_mask] -= hydro_weight * 3.0
                    elif current_hydro[b] < hydro_target[0]:
                        hydro_mask = hydropathy[:20] < -1.0
                        logits[b, :20][hydro_mask] -= hydro_weight * 3.0

            # 5) Nucleus (top-p) sampling
            if top_p > 0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                logits[indices_to_remove] = -float('Inf')

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # 6) 非法token回退机制
            illegal_mask = (next_token >= 20) & (next_token != end_token)
            if illegal_mask.any():
                fallback_logits = logits.clone()
                fallback_logits[:, :20] = logits[:, :20]
                fallback_logits[:, 20:] = -float('Inf')
                fallback_probs = F.softmax(fallback_logits, dim=-1)
                fallback_token = torch.multinomial(fallback_probs, num_samples=1)
                next_token[illegal_mask] = fallback_token[illegal_mask]

            token_counts.scatter_add_(1, next_token, torch.ones_like(next_token, dtype=torch.float))
            generated = torch.cat([generated, next_token], dim=1)

            if step >= min_len and (next_token == end_token).all():
                break

        return generated

    def unfreeze_esm(self, unfreeze_layers=None):
        """解冻ESM2参数进行微调"""
        self.freeze_esm = False

        if unfreeze_layers is None:
            for param in self.esm.parameters():
                param.requires_grad = True
        else:
            total_layers = len(self.esm.encoder.layer)
            for i, layer in enumerate(self.esm.encoder.layer):
                if i >= total_layers - unfreeze_layers:
                    for param in layer.parameters():
                        param.requires_grad = True

        self.esm_lr = 1e-5
        self.head_lr = 1e-4

        print(f"ESM2 unfrozen. Trainable parameters: "
              f"{sum(p.numel() for p in self.parameters() if p.requires_grad):,}")


def train_esm2_gvp(model, dataloader, optimizer, device, epoch, 
                   unfreeze_esm_epoch=3, max_grad_norm=1.0):
    """训练ESM2-GVP模型（改进版）"""
    model.train()

    if epoch == unfreeze_esm_epoch and model.freeze_esm:
        model.unfreeze_esm(unfreeze_layers=4)
        optimizer = torch.optim.AdamW([
            {'params': [p for p in model.esm.parameters() if p.requires_grad], 
             'lr': model.esm_lr, 'weight_decay': 0.01},
            {'params': [p for n, p in model.named_parameters() 
                       if not n.startswith('esm') and p.requires_grad], 
             'lr': model.head_lr, 'weight_decay': 0.1}
        ])

    total_loss = 0
    num_batches = 0

    for batch in dataloader:
        x = batch['seq'].to(device)
        species = batch['species'].to(device)
        attention_mask = (x != 23).long()

        optimizer.zero_grad()

        outputs = model(
            input_ids=x,
            species=species,
            attention_mask=attention_mask,
            labels=x
        )

        loss = outputs['loss']
        if loss is not None:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

    return total_loss / max(num_batches, 1), optimizer