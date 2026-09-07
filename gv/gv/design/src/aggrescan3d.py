def run_a3d(sequence, **kwargs):
    """Mock Aggrescan3D prediction - 统一返回格式"""
    # 简单的启发式：富含疏水/带电残基 → 更高聚集倾向（更高score）
    hydrophobic = set('AILMFWVY')
    charged = set('KRHDE')
    h_count = sum(1 for aa in sequence if aa in hydrophobic)
    c_count = sum(1 for aa in sequence if aa in charged)

    # A3D_score: 正值=易聚集，负值=稳定（与score_toxic.py的期望一致）
    # 简单启发：疏水多+带电少 → 易聚集
    length = len(sequence) if sequence else 1
    h_ratio = h_count / length
    c_ratio = c_count / length

    # 映射到典型A3D范围 (-20 ~ +20)
    score = (h_ratio - c_ratio) * 20 + (length / 100) * 2
    score = max(-20, min(20, score))

    return {
        "A3D_score": score,           # 与score_toxic.py期望的key一致
        "aggregation_prone_regions": [],
        "total_score": score,
        "score": score,               # 兼容旧代码
    }