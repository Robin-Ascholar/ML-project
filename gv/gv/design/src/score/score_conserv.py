# 保守性：MSA恢复度（增强版，支持超出MSA范围的序列）
from Bio import AlignIO
import os
import warnings

# MSA路径：优先环境变量，其次相对路径
MSA_FILE = os.environ.get('GVP_MSA',
            os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                         "..", "data", "gvpA_msa.a3m"))

# 氨基酸理化性质分组（用于超出MSA范围的保守性评估）
AA_GROUPS = {
    'hydrophobic': set('AVILMFWY'),      # 疏水
    'polar': set('STNQ'),                 # 极性
    'positive': set('KRH'),               # 正电
    'negative': set('DE'),                # 负电
    'special': set('CGP'),                # 特殊（半胱氨酸、甘氨酸、脯氨酸）
}

def _get_aa_group(aa):
    """获取氨基酸所属理化性质组"""
    for group, members in AA_GROUPS.items():
        if aa in members:
            return group
    return 'other'

def _physicochemical_similarity(aa1, aa2):
    """计算两个氨基酸的理化性质相似度 (0-1)"""
    if aa1 == aa2:
        return 1.0
    g1, g2 = _get_aa_group(aa1), _get_aa_group(aa2)
    if g1 == g2:
        return 0.5  # 同组但不同氨基酸
    # 疏水-极性转换惩罚
    if (g1 == 'hydrophobic' and g2 in ['polar', 'positive', 'negative']) or \
       (g2 == 'hydrophobic' and g1 in ['polar', 'positive', 'negative']):
        return 0.1
    return 0.2  # 其他转换

def conserv_score(seq: str) -> float:
    """
    计算设计序列的保守性评分（增强版）

    原理:
      1. MSA覆盖区域: 检查每个残基是否出现在天然MSA对应列中
      2. MSA外区域: 基于氨基酸理化性质相似性评估

    返回: 0-1, 越高越保守
    """
    # 检查文件存在性
    if not os.path.exists(MSA_FILE):
        warnings.warn(f"MSA文件不存在: {MSA_FILE}，使用理化性质相似性评分")
        return _physicochemical_conserv_score(seq)

    try:
        msa = AlignIO.read(MSA_FILE, "fasta")
    except Exception as e:
        warnings.warn(f"MSA读取失败: {e}，使用理化性质相似性评分")
        return _physicochemical_conserv_score(seq)

    n_cols = msa.get_alignment_length()
    seq_len = len(seq)

    # MSA覆盖区域的保守性
    msa_matches = 0
    compare_len = min(seq_len, n_cols)

    for i in range(compare_len):
        col = set(msa[:, i]) - {'-'}
        if seq[i] in col:
            msa_matches += 1

    msa_score = msa_matches / compare_len if compare_len > 0 else 0.5

    # 超出MSA范围的部分，使用理化性质评估
    if seq_len > n_cols:
        extra_len = seq_len - n_cols
        extra_score = 0

        # 获取MSA最后一列的氨基酸分布作为参考
        last_col = set(msa[:, n_cols-1]) - {'-'}

        for i in range(n_cols, seq_len):
            aa = seq[i]
            # 与MSA最后一列的残基进行理化性质比较
            similarities = [_physicochemical_similarity(aa, ref_aa) for ref_aa in last_col]
            extra_score += max(similarities) if similarities else 0.5

        extra_score = extra_score / extra_len if extra_len > 0 else 0.5

        # 加权综合: MSA部分权重更高
        total_score = (msa_score * compare_len + extra_score * extra_len * 0.5) / seq_len

        warnings.warn(f"设计序列长度({seq_len})超过MSA列数({n_cols})，"
                     f"MSA覆盖部分保守性={msa_score:.3f}，"
                     f"超出部分理化保守性={extra_score:.3f}，"
                     f"综合保守性={total_score:.3f}")
        return total_score

    return msa_score


def _physicochemical_conserv_score(seq: str) -> float:
    """无MSA时的fallback：基于理化性质连续性评估"""
    if len(seq) < 2:
        return 0.5

    # 评估相邻残基的理化性质连续性
    continuity = 0
    for i in range(len(seq) - 1):
        g1, g2 = _get_aa_group(seq[i]), _get_aa_group(seq[i+1])
        # 疏水-疏水或极性-极性连续更"保守"
        if g1 == g2:
            continuity += 1.0
        elif (g1 == 'hydrophobic' and g2 in ['polar', 'positive', 'negative']) or \
             (g2 == 'hydrophobic' and g1 in ['polar', 'positive', 'negative']):
            continuity += 0.2  # 疏水-亲水转换惩罚
        else:
            continuity += 0.5

    return continuity / (len(seq) - 1)