# 统一评分封装 - 增强版
import os
import warnings

# 尝试多种导入方式，逐级fallback
def _try_import_real_scorers():
    """尝试导入真实评分器，返回是否成功"""
    global fold_score, assembly_score, conserv_score, max_identity, toxic_score

    # 尝试1: 包内相对导入（作为包使用时）
    try:
        from .score_fold import fold_score
        from .score_assembly import assembly_score
        from .score_conserv import conserv_score
        from .score_novel import max_identity
        from .score_toxic import toxic_score
        return True
    except ImportError:
        pass

    # 尝试2: 同级绝对导入（作为脚本运行时）
    try:
        from score_fold import fold_score
        from score_assembly import assembly_score
        from score_conserv import conserv_score
        from score_novel import max_identity
        from score_toxic import toxic_score
        return True
    except ImportError:
        pass

    # 尝试3: 原始文件名导入
    try:
        from score_fold import fold_score
        from score_assembly import assembly_score
        from score_conserv import conserv_score
        from score_novel import max_identity
        from score_toxic import toxic_score
        return True
    except ImportError:
        pass

    # 尝试4: fallback到mock
    try:
        from .score_mock import fold_score, assembly_score, conserv_score, max_identity, toxic_score
        warnings.warn("使用Mock评分器（所有外部工具未安装）")
        return True
    except ImportError:
        from score_mock import fold_score, assembly_score, conserv_score, max_identity, toxic_score
        warnings.warn("使用Mock评分器（所有外部工具未安装）")
        return True

# 执行导入
_try_import_real_scorers()

class GvpScoreNet:
    def __init__(self, weights=None):
        self.w = weights or {
            "fold": 0.3,
            "assembly": 0.3,
            "conserv": 0.15,
            "novelty": 0.15,
            "toxic": 0.1
        }

    def __call__(self, seq_A: str, seq_B: str = None):
        """
        seq_A: 生成的 GvpA
        seq_B: 可选 GvpC（如评估组装性）
        """
        # 输入验证
        if not isinstance(seq_A, str) or len(seq_A) == 0:
            warnings.warn("无效输入序列，返回默认评分")
            return {"fold": 0.5, "conserv": 0.5, "novelty": 0.5,
                    "toxic": 0.5, "assembly": 0.5, "overall": 0.5}

        try:
            scores = {
                "fold": fold_score(seq_A),
                "conserv": conserv_score(seq_A),
                "novelty": max_identity(seq_A),
                "toxic": toxic_score(seq_A)
            }

            # 组装性评分
            # 修复：即使seq_B为None，也调用单链组装评估
            if seq_B and isinstance(seq_B, str) and len(seq_B) > 0:
                asm_result = assembly_score(seq_A, seq_B)
            else:
                asm_result = assembly_score(seq_A, None)

            # 兼容返回dict或float的assembly_score
            if isinstance(asm_result, dict):
                scores["assembly"] = asm_result.get("score", 0.5)
            else:
                scores["assembly"] = float(asm_result)

            overall = sum(self.w[k] * scores[k] for k in self.w)
            scores["overall"] = overall
            return scores

        except Exception as e:
            warnings.warn(f"评分错误 ({seq_A[:20]}...): {e}")
            return {"fold": 0.5, "conserv": 0.5, "novelty": 0.5,
                    "toxic": 0.5, "assembly": 0.5, "overall": 0.5}

# 使用示例
if __name__ == "__main__":
    evaluator = GvpScoreNet()
    seq = "MSEQINQRIQRLASGIKLANLLFSGIAISAISAISAIVILGVLIQYF"
    print(evaluator(seq))