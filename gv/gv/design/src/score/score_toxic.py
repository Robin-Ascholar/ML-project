# 毒性/可溶解性安全评分
import os
import warnings

# 尝试导入DeepLoc-2，失败则使用mock
try:
    from deeploc2.predict import predict as dl2_predict
    DEEPLOC_AVAILABLE = True
except ImportError:
    DEEPLOC_AVAILABLE = False
    warnings.warn("DeepLoc-2未安装，使用mock可溶性预测")

# 尝试导入Aggrescan3D
try:
    from aggrescan3d import run_a3d
except ImportError:
    try:
        from .aggrescan3d import run_a3d
    except ImportError:
        # 完全mock
        def run_a3d(sequence, **kwargs):
            hydrophobic = set('AILMFWVY')
            h_count = sum(1 for aa in sequence if aa in hydrophobic)
            length = len(sequence) if sequence else 1
            score = (h_count / length) * 20 - 10
            return {"A3D_score": score, "aggregation_prone_regions": [], "total_score": score}

def _mock_solubility(seq):
    """基于序列特征的mock可溶性预测"""
    # 疏水残基比例高 → 可溶性低
    hydrophobic = set('AILMFWVY')
    h_ratio = sum(1 for aa in seq if aa in hydrophobic) / len(seq) if seq else 0
    # 理想可溶性蛋白：疏水性30-50%
    if 0.3 <= h_ratio <= 0.5:
        return 0.85
    elif h_ratio < 0.3:
        return 0.75  # 太亲水也可能不稳定
    else:
        return max(0.3, 0.9 - h_ratio)

def toxic_score(seq: str) -> float:
    """
    毒性/安全性评分 (0-1, 越高越安全)
    结合：可溶性 + 聚集倾向
    """
    # 1. 可溶性预测
    if DEEPLOC_AVAILABLE:
        try:
            device = "cuda:0" if os.environ.get('CUDA_VISIBLE_DEVICES') else "cpu"
            df = dl2_predict([seq], device=device)
            sol = float(df["soluble"].iloc[0])
        except Exception as e:
            warnings.warn(f"DeepLoc-2预测失败: {e}，使用mock")
            sol = _mock_solubility(seq)
    else:
        sol = _mock_solubility(seq)

    # 2. 聚集倾向 (A3D_score: 负值=稳定，正值=易聚集)
    try:
        agg_result = run_a3d(seq)
        agg_score = agg_result.get('A3D_score', agg_result.get('score', 0))
        # 映射到概率: score <= -10 → 0%聚集, score >= 10 → 100%聚集
        agg_prob = max(0.0, min(1.0, (agg_score + 10) / 20))
    except Exception as e:
        warnings.warn(f"Aggrescan3D失败: {e}，使用mock")
        agg_prob = 0.3

    # 加权: 60%可溶性 + 40%抗聚集能力
    final_score = 0.6 * sol + 0.4 * (1 - agg_prob)
    return min(0.99, max(0.0, final_score))