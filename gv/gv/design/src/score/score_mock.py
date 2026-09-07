# score/score_mock.py - 增强Mock评分器
import random

def fold_score(seq):
    """模拟折叠评分：基于序列长度和氨基酸组成"""
    length = len(seq)
    if 60 <= length <= 160:
        base = 0.85
    elif 40 <= length <= 200:
        base = 0.70
    else:
        base = 0.50

    hydrophobic = 'AILMFWV'
    h_count = sum(1 for aa in seq if aa in hydrophobic)
    h_ratio = h_count / len(seq) if seq else 0
    if 0.4 <= h_ratio <= 0.6:
        h_bonus = 0.10
    else:
        h_bonus = 0.0

    score = min(0.99, base + h_bonus + random.uniform(-0.05, 0.05))
    return round(score, 2)

def assembly_score(seq_A, seq_B=None):
    """模拟组装评分"""
    if seq_B is None:
        return round(random.uniform(0.6, 0.95), 2)
    # 基于序列互补性的简单启发
    pos_A = sum(seq_A.count(aa) for aa in 'KRH')
    neg_A = sum(seq_A.count(aa) for aa in 'DE')
    pos_B = sum(seq_B.count(aa) for aa in 'KRH') if seq_B else 0
    neg_B = sum(seq_B.count(aa) for aa in 'DE') if seq_B else 0
    complement = min(pos_A, neg_B) + min(neg_A, pos_B)
    total = max(len(seq_A), len(seq_B)) if seq_B else len(seq_A)
    score = min(0.95, 0.6 + (complement / total) * 0.3) if total > 0 else 0.5
    return round(score + random.uniform(-0.05, 0.05), 2)

def conserv_score(seq):
    """模拟保守性评分"""
    return round(random.uniform(0.5, 0.95), 2)

def max_identity(seq):
    """模拟新颖度"""
    return round(random.uniform(0.2, 0.6), 2)

def toxic_score(seq):
    """模拟毒性评分"""
    return round(random.uniform(0.7, 0.95), 2)