import os
import glob
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import torch
from torch.utils.data import Dataset, DataLoader
from Bio import SeqIO, AlignIO
import numpy as np

# ========== 统一词汇表定义 ==========
AA_VOCAB = {aa: i for i, aa in enumerate("ACDEFGHIKLMNPQRSTVWY-")}
AA_VOCAB['<PAD>'] = 21
AA_VOCAB['<SOS>'] = 22
AA_VOCAB['<EOS>'] = 23
VOCAB_SIZE = 24


class GvpDataset(Dataset):
    """GVP序列数据集，支持多FASTA/多MSA批量加载"""
    def __init__(self, fasta_file, msa_file=None, max_len=512, chain_type=None):
        self.sequences = []
        self.species = []
        self.chain_types = []
        self.msa_weights_list = []
        self.msa_len_list = []
        self.max_len = max_len
        
        self.aa_vocab = AA_VOCAB.copy()
        self.idx_to_aa = {v: k for k, v in self.aa_vocab.items()}
        
        # ---- 1. 收集所有 fasta 文件 ----
        fasta_files = []
        if os.path.isdir(fasta_file):
            for ext in ['*.fasta', '*.fa']:
                fasta_files.extend(sorted(glob.glob(os.path.join(fasta_file, ext))))
            if not fasta_files:
                raise ValueError(f"目录 {fasta_file} 下未找到 .fasta 或 .fa 文件")
        else:
            if not os.path.exists(fasta_file):
                raise FileNotFoundError(f"找不到 FASTA 文件: {fasta_file}")
            fasta_files = [fasta_file]
        
        # ---- 2. 逐个 fasta 处理 ----
        for fpath in fasta_files:
            fname = os.path.splitext(os.path.basename(fpath))[0]  # e.g. "GvpA"
            
            # 从文件名推断链类型（最可靠）
            file_chain = None
            if fname.upper().startswith('GVP') and len(fname) > 3:
                file_chain = fname[3:].upper()  # 'A', 'C', 'F', ...
            
            # 自动匹配 MSA
            current_msa_file = self._resolve_msa_file(fpath, msa_file)
            
            msa_w = None
            msa_len = 0
            if current_msa_file and os.path.exists(current_msa_file):
                try:
                    msa_w = self._compute_msa_weights(current_msa_file)
                    msa_len = msa_w.size(0)
                except Exception as e:
                    print(f"[警告] 无法加载 MSA {current_msa_file}: {e}")
            
            # 读取该 fasta
            for record in SeqIO.parse(fpath, "fasta"):
                seq = str(record.seq)
                header = record.id
                
                # 解析物种（header 只用来取物种）
                sp = self._parse_species(header)
                
                # 链类型：优先文件名，fallback 从 header 解析
                current_chain = file_chain if file_chain is not None else self._parse_chain_from_header(header)
                if current_chain is None:
                    current_chain = 'UNKNOWN'
                
                # 过滤
                if chain_type is not None:
                    allowed = [chain_type.upper()] if isinstance(chain_type, str) else [c.upper() for c in chain_type]
                    if current_chain not in allowed:
                        continue
                
                if len(seq) >= max_len - 2:
                    continue
                
                self.sequences.append(seq)
                self.species.append(sp)
                self.chain_types.append(current_chain)
                self.msa_weights_list.append(msa_w)
                self.msa_len_list.append(msa_len)
        
        # ---- 3. 物种编码 ----
        unique_species = sorted(list(set(self.species)))
        self.species_to_idx = {sp: i for i, sp in enumerate(unique_species)}
        self.num_species = len(unique_species)
        
        print(f"[Dataset] 共加载 {len(self.sequences)} 条序列，"
              f"链类型分布: {self._chain_stats()}")
    
    def _parse_species(self, header):
        """从 header 解析物种名"""
        # UniProt: sp|P33957.1|GVPM2_METJA -> METJA
        if header.startswith('sp|') or header.startswith('tr|'):
            parts = header.split('|')[-1].split('_')
            return parts[-1] if len(parts) > 1 else 'unknown'
        
        # NCBI: WP_010869123.1 -> unknown（无物种信息）
        if header.startswith('WP_') or header.startswith('YP_'):
            return 'unknown'
        
        # 自定义: GvpA_Halobacterium -> Halobacterium
        parts = header.split('_')
        if len(parts) >= 2:
            return '_'.join(parts[1:])
        
        return 'unknown'
    
    def _parse_chain_from_header(self, header):
        """从 header 解析链类型（fallback）"""
        # UniProt: sp|P33957.1|GVPM2_METJA -> M
        if header.startswith('sp|') or header.startswith('tr|'):
            last = header.split('|')[-1]  # GVPM2_METJA
            chain_part = last.split('_')[0]  # GVPM2
            if chain_part.upper().startswith('GVP') and len(chain_part) > 3:
                c = chain_part[3:].upper()
                return c[0] if len(c) > 1 else c  # M2 -> M
        
        # 自定义: GvpA_Halobacterium -> A
        parts = header.split('_')
        if len(parts) >= 1:
            chain_part = parts[0]
            if chain_part.upper().startswith('GVP') and len(chain_part) > 3:
                return chain_part[3:].upper()
        
        return None
    
    def _resolve_msa_file(self, fpath, msa_file):
        fname = os.path.splitext(os.path.basename(fpath))[0]
        if msa_file is None:
            dir_name = os.path.dirname(fpath) or '.'
            candidate = os.path.join(dir_name, f"{fname}_msa.a3m")
            return candidate if os.path.exists(candidate) else None
        elif os.path.isdir(msa_file):
            candidate = os.path.join(msa_file, f"{fname}_msa.a3m")
            return candidate if os.path.exists(candidate) else None
        else:
            return msa_file if os.path.exists(msa_file) else None
    
    def _chain_stats(self):
        from collections import Counter
        return dict(Counter(self.chain_types))
    
    def _compute_msa_weights(self, msa_file):
        msa = AlignIO.read(msa_file, "fasta")
        weights = []
        for i in range(msa.get_alignment_length()):
            col = [str(rec.seq)[i] for rec in msa if str(rec.seq)[i] != '-']
            counts = torch.zeros(20)
            for aa in col:
                if aa in self.aa_vocab and self.aa_vocab[aa] < 20:
                    counts[self.aa_vocab[aa]] += 1
            if counts.sum() > 0:
                counts = counts / counts.sum()
            weights.append(counts)
        return torch.stack(weights)
    
    def encode(self, seq):
        indices = [self.aa_vocab['<SOS>']]
        for aa in seq:
            indices.append(self.aa_vocab.get(aa, self.aa_vocab['-']))
        indices.append(self.aa_vocab['<EOS>'])
        return torch.tensor(indices, dtype=torch.long)
    
    def decode(self, indices):
        seq = []
        for idx in indices:
            if isinstance(idx, torch.Tensor):
                idx = idx.item()
            aa = self.idx_to_aa.get(idx, '-')
            if aa in ['<PAD>', '<SOS>', '<EOS>']:
                continue
            seq.append(aa)
        return ''.join(seq)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq = self.sequences[idx]
        x = self.encode(seq)
        species_idx = self.species_to_idx[self.species[idx]]
        
        msa_w = torch.zeros(self.max_len, 20)
        weights = self.msa_weights_list[idx]
        if weights is not None:
            seq_len = min(len(x), self.msa_len_list[idx], self.max_len)
            msa_w[:seq_len] = weights[:seq_len]
        
        return {
            'seq': x,
            'species': torch.tensor(species_idx, dtype=torch.long),
            'length': len(x),
            'chain': self.chain_types[idx],
            'msa_weights': msa_w
        }


def collate_fn(batch):
    seqs = [item['seq'] for item in batch]
    species = torch.stack([item['species'] for item in batch])
    lengths = [item['length'] for item in batch]
    msa_weights = torch.stack([item['msa_weights'] for item in batch])
    
    max_len = max(lengths)
    padded = torch.full((len(batch), max_len), 21, dtype=torch.long)
    masks = torch.zeros(len(batch), max_len, dtype=torch.bool)
    
    for i, seq in enumerate(seqs):
        padded[i, :len(seq)] = seq
        masks[i, :len(seq)] = True
    
    return {
        'seq': padded,
        'mask': masks,
        'species': species,
        'lengths': lengths,
        'msa_weights': msa_weights[:, :max_len, :]
    }


if __name__ == "__main__":
    dataset = GvpDataset(r"D:\毕设\1\data", msa_file=None, max_len=512, chain_type=None)
    loader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
    
    for batch in loader:
        print("Batch seq shape:", batch['seq'].shape)
        print("Chains:", [dataset.chain_types[i] for i in range(len(batch['seq']))])
        break