# 新颖度：与天然库最大identity
import subprocess, tempfile, os, shutil, warnings

# 数据库路径：优先环境变量，其次相对路径，最后绝对路径
DB = os.environ.get('GVP_DB', 
            os.environ.get('NOVEL_DB', 
                os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                             "..", "data", "natural_gvp.fasta")))

def _find_mmseqs():
    """查找mmseqs可执行文件"""
    mmseqs = shutil.which("mmseqs")
    if mmseqs:
        return mmseqs
    # 常见安装路径
    for path in ['/usr/local/bin/mmseqs', '/usr/bin/mmseqs', 
                 os.path.expanduser('~/mmseqs2/mmseqs')]:
        if os.path.isfile(path):
            return path
    return None

def max_identity(seq: str) -> float:
    """
    计算设计序列与天然库的最大identity
    返回: 1.0 - best_identity/100  (值越大越新颖)
    """
    mmseqs = _find_mmseqs()

    # 如果没有mmseqs，使用简单的字符串匹配fallback
    if mmseqs is None:
        warnings.warn("mmseqs未找到，使用启发式新颖度评分")
        return _heuristic_novelty(seq)

    # 检查数据库文件
    if not os.path.exists(DB):
        warnings.warn(f"数据库文件不存在: {DB}，使用启发式评分")
        return _heuristic_novelty(seq)

    tmpdir = tempfile.mkdtemp()
    try:
        # 写入查询序列
        query_path = os.path.join(tmpdir, "query.fasta")
        with open(query_path, 'w') as f:
            f.write(f">query\n{seq}\n")

        aln_path = os.path.join(tmpdir, "aln.m8")
        tmp_db = os.path.join(tmpdir, "tmp_mmseqs")

        cmd = [
            mmseqs, "easy-search", 
            query_path, DB, aln_path, tmp_db,
            "--max-seqs", "1000", 
            "-e", "1e-3",
            "--threads", "1"
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            warnings.warn(f"mmseqs搜索失败: {result.stderr}，使用启发式评分")
            return _heuristic_novelty(seq)

        # 解析最好的identity
        best = 0.0
        if os.path.exists(aln_path):
            with open(aln_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 3:
                        try:
                            identity = float(parts[2])
                            best = max(best, identity)
                        except ValueError:
                            continue

        # 映射: 0-20% identity → 0.8-1.0 新颖度
        #       20-50% → 0.5-0.8
        #       50-100% → 0.0-0.5
        novelty = 1.0 - best / 100.0
        return round(max(0.0, min(1.0, novelty)), 3)

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def _heuristic_novelty(seq: str) -> float:
    """启发式新颖度：基于序列特征估计"""
    # 理想GvpA序列的特征
    # 过于"标准"的序列（理想长度+理想疏水性）→ 可能已知
    # 偏离标准的 → 可能新颖
    length = len(seq)

    # 长度偏离度 (理想60-100aa)
    if 60 <= length <= 100:
        len_score = 0.3  # 标准长度，可能已知
    elif 40 <= length <= 120:
        len_score = 0.5
    else:
        len_score = 0.8  # 异常长度，更可能新颖

    # 氨基酸组成偏离
    aa_counts = {aa: seq.count(aa) for aa in set(seq)}
    max_freq = max(aa_counts.values()) / len(seq) if seq else 0
    # 单一氨基酸过多 → 人工设计痕迹 → 新颖
    comp_score = min(1.0, max_freq * 3)

    # 重复模式
    has_repeat = any(seq[i:i+3] == seq[i+3:i+6] for i in range(len(seq)-6))
    repeat_score = 0.7 if has_repeat else 0.3

    # 综合
    novelty = 0.4 * len_score + 0.3 * comp_score + 0.3 * repeat_score
    return round(novelty, 3)