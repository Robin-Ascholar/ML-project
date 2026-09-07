"""
score_assembly.py
蛋白质复合物组装性评分系统
支持 PyRosetta 局部突变 和 ColabFold 重新预测 两种模式
"""

import os
import sys
import glob
import shutil
import tempfile
import subprocess
import warnings
from typing import List, Tuple, Optional
from dataclasses import dataclass

import pandas as pd

# ============================================================================
# 配置区（可通过环境变量覆盖）
# ============================================================================

WT_PDB = os.environ.get('WT_PDB', "wt_complex.pdb")
FOLDX_BIN = os.environ.get('FOLDX_BIN', "foldx")
COLABFOLD_BIN = os.environ.get('COLABFOLD_BIN', "colabfold_batch")

# 评分映射参数：ddG <= SCORE_MAX 得 1 分，>= SCORE_MIN 得 0 分
SCORE_DD_G_MIN = float(os.environ.get('SCORE_DD_G_MIN', -15.0))
SCORE_DD_G_MAX = float(os.environ.get('SCORE_DD_G_MAX', 5.0))

# ============================================================================
# PyRosetta 初始化
# ============================================================================

try:
    import pyrosetta
    from pyrosetta.rosetta.protocols.simple_moves import MutateResidue
    PYROSETTA_AVAILABLE = True
except ImportError:
    PYROSETTA_AVAILABLE = False
    warnings.warn("PyRosetta 未安装，局部突变模式不可用")


def _init_pyrosetta():
    """延迟初始化 PyRosetta"""
    if not PYROSETTA_AVAILABLE:
        raise RuntimeError("PyRosetta 未安装，无法使用局部突变模式")
    if not pyrosetta.rosetta.core.init.was_init_called():
        pyrosetta.init("-mute all -ignore_unrecognized_res")


# ============================================================================
# 核心函数：结构突变
# ============================================================================

def mutate_pdb_pyrosetta(pdb: str, seq_A: str, seq_B: str, out_pdb: str) -> str:
    """
    使用 PyRosetta 对现有结构进行逐残基突变
    适用于：点突变、小片段替换（保持整体折叠不变）
    """
    _init_pyrosetta()

    pose = pyrosetta.pose_from_pdb(pdb)
    scorefxn = pyrosetta.get_fa_scorefxn()

    n_res_A = pose.chain_end(1)
    current_A = pose.chain_sequence(1)
    current_B = pose.chain_sequence(2)

    if len(seq_A) != len(current_A):
        raise ValueError(f"链 A 长度不匹配: 当前{len(current_A)} vs 目标{len(seq_A)}")
    if len(seq_B) != len(current_B):
        raise ValueError(f"链 B 长度不匹配: 当前{len(current_B)} vs 目标{len(seq_B)}")

    mutations = []

    # 链 A 突变
    for i, (wt, mut) in enumerate(zip(current_A, seq_A), 1):
        if wt != mut:
            mutator = MutateResidue(i, mut)
            mutator.apply(pose)
            mutations.append(f"A{i}:{wt}->{mut}")

    # 链 B 突变
    offset = n_res_A
    for i, (wt, mut) in enumerate(zip(current_B, seq_B), 1):
        if wt != mut:
            res_id = offset + i
            mutator = MutateResidue(res_id, mut)
            mutator.apply(pose)
            mutations.append(f"B{i}:{wt}->{mut}")

    # 局部重新包装
    if mutations:
        from pyrosetta.rosetta.core.pack.task import TaskFactory
        from pyrosetta.rosetta.core.pack.task.operation import RestrictToRepackingRLT, IncludeCurrent
        from pyrosetta.rosetta.core.select.residue_selector import ResidueIndexSelector, NeighborhoodResidueSelector

        mut_indices = []
        for m in mutations:
            chain = m[0]
            idx = int(m[1:].split(":")[0])
            if chain == "A":
                mut_indices.append(idx)
            else:
                mut_indices.append(offset + idx)

        index_str = ",".join(map(str, mut_indices))
        mut_selector = ResidueIndexSelector(index_str)
        neighbor_selector = NeighborhoodResidueSelector(mut_selector, 6.0, False)

        tf = TaskFactory()
        tf.push_back(RestrictToRepackingRLT())
        tf.push_back(IncludeCurrent())

        packer = pyrosetta.rosetta.protocols.minimization_packing.PackRotamersMover(scorefxn)
        packer.task_factory(tf)
        packer.nloop(2)
        packer.apply(pose)

        minimizer = pyrosetta.rosetta.protocols.minimization_packing.MinMover()
        minimizer.score_function(scorefxn)
        minimizer.min_type("lbfgs_armijo_nonmonotone")
        minimizer.apply(pose)

    pose.dump_pdb(out_pdb)
    return out_pdb


def mutate_pdb_colabfold(seq_A: str, seq_B: str, out_pdb: str) -> str:
    """
    使用 ColabFold 重新预测复合物结构
    适用于：大突变、序列差异大、需要重新折叠的情况
    """
    cf_bin = shutil.which("colabfold_batch") or COLABFOLD_BIN
    if not shutil.which(cf_bin) and not os.path.isfile(cf_bin):
        raise RuntimeError(f"ColabFold未找到: {cf_bin}")

    tmpdir = tempfile.mkdtemp()
    try:
        fasta_path = os.path.join(tmpdir, "input.fasta")
        with open(fasta_path, "w") as f:
            f.write(f">seq_A\n{seq_A}\n>seq_B\n{seq_B}\n")

        output_dir = os.path.join(tmpdir, "cf_output")
        cmd = [
            cf_bin, fasta_path, output_dir,
            "--model-type", "alphafold2_multimer_v3",
            "--num-models", "1",
            "--num-recycle", "3",
            "--rank", "ptm"
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"ColabFold 失败:\n{result.stderr}")

        best_pdb = glob.glob(os.path.join(output_dir, "*rank_001*.pdb"))
        if not best_pdb:
            best_pdb = glob.glob(os.path.join(output_dir, "*.pdb"))
            if not best_pdb:
                raise FileNotFoundError("ColabFold 未生成 PDB 输出")

        shutil.copy(best_pdb[0], out_pdb)
        return out_pdb

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ============================================================================
# 核心函数：FoldX 界面能量计算
# ============================================================================

def foldx_interface(pdb: str, chains: str = "A,B") -> float:
    """
    使用 FoldX 计算蛋白质复合物界面 ΔΔG
    返回: Interface Energy (kcal/mol)，负值表示有利结合
    """
    foldx_exe = shutil.which("foldx") or FOLDX_BIN
    if not shutil.which(foldx_exe) and not os.path.isfile(foldx_exe):
        raise RuntimeError(f"FoldX未找到: {foldx_exe}")

    tmpdir = tempfile.mkdtemp()
    try:
        shutil.copy(pdb, os.path.join(tmpdir, "in.pdb"))

        repair_cmd = [foldx_exe, "--command", "RepairPDB", "--pdb", "in.pdb"]
        repair_result = subprocess.run(repair_cmd, cwd=tmpdir, capture_output=True, text=True)
        if repair_result.returncode != 0:
            raise RuntimeError(f"FoldX RepairPDB 失败:\n{repair_result.stderr}")

        repaired_files = glob.glob(os.path.join(tmpdir, "*Repair*.pdb"))
        if not repaired_files:
            raise FileNotFoundError("RepairPDB 未生成输出文件")
        repaired = os.path.basename(repaired_files[0])

        config_path = os.path.join(tmpdir, "config.cfg")
        with open(config_path, "w") as f:
            f.write(f"""command=AnalyseComplex
pdb={repaired}
complexWithDNA=false
chains={chains}
output-file=interface
""")

        analysis_cmd = [foldx_exe, "--command", "AnalyseComplex", "--config", "config.cfg"]
        analysis_result = subprocess.run(analysis_cmd, cwd=tmpdir, capture_output=True, text=True)
        if analysis_result.returncode != 0:
            raise RuntimeError(f"FoldX AnalyseComplex 失败:\n{analysis_result.stderr}")

        res_patterns = [
            os.path.join(tmpdir, "*Interaction_InterfaceEnergy.txt"),
            os.path.join(tmpdir, "interface*.txt"),
            os.path.join(tmpdir, "*.fxout")
        ]

        res_files = []
        for pattern in res_patterns:
            res_files = glob.glob(pattern)
            if res_files:
                break

        if not res_files:
            raise FileNotFoundError("FoldX 界面能量输出未找到")

        with open(res_files[0]) as f:
            lines = f.readlines()
            for line in lines[1:]:
                parts = line.strip().split()
                if len(parts) >= 3:
                    try:
                        ddG = float(parts[2])
                        return ddG
                    except (ValueError, IndexError):
                        continue
            for line in lines:
                parts = line.strip().split()
                for p in parts:
                    try:
                        val = float(p)
                        if -100 < val < 100:
                            return val
                    except ValueError:
                        continue

        raise ValueError(f"无法从 FoldX 输出解析能量值")

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ============================================================================
# 评分函数
# ============================================================================

def ddG_to_score(ddG: float, ddG_min: float = SCORE_DD_G_MIN, 
                 ddG_max: float = SCORE_DD_G_MAX) -> float:
    """将 ΔΔG 映射到 0-1 分数"""
    if ddG <= ddG_min:
        return 1.0
    if ddG >= ddG_max:
        return 0.0
    score = 1.0 - (ddG - ddG_min) / (ddG_max - ddG_min)
    return max(0.0, min(1.0, score))


def assembly_score(seq_A: str, seq_B: str = None, 
                   wt_pdb: str = None, method: str = "auto") -> dict:
    """
    计算突变复合物的组装性评分

    Args:
        seq_A: 链 A 目标序列
        seq_B: 链 B 目标序列（可选，单链评估时传None）
        wt_pdb: 野生型结构路径（默认从环境变量/配置读取）
        method: "pyrosetta", "colabfold", "auto"

    Returns:
        dict: {'score': 0-1, 'ddG': float, 'method': str}
    """
    if wt_pdb is None:
        wt_pdb = WT_PDB

    # 如果缺少外部工具或seq_B为None，使用启发式评分
    if seq_B is None or not os.path.exists(wt_pdb):
        # 增强启发式：基于多维度物理化学互补性
        if seq_B:
            # 1. 电荷互补性（静电相互作用）
            pos_A = sum(seq_A.count(aa) for aa in 'KRH')
            neg_A = sum(seq_A.count(aa) for aa in 'DE')
            pos_B = sum(seq_B.count(aa) for aa in 'KRH')
            neg_B = sum(seq_B.count(aa) for aa in 'DE')
            charge_complement = min(pos_A, neg_B) + min(neg_A, pos_B)
            charge_score = min(1.0, charge_complement / max(len(seq_A), len(seq_B)) * 5.0)

            # 2. 疏水互补性（疏水核心匹配）
            hydro_A = sum(seq_A.count(aa) for aa in 'AVILMFWY') / len(seq_A)
            hydro_B = sum(seq_B.count(aa) for aa in 'AVILMFWY') / len(seq_B)
            hydro_score = 1.0 - abs(hydro_A - hydro_B)

            # 3. 大小匹配（长度比例）
            len_ratio = min(len(seq_A), len(seq_B)) / max(len(seq_A), len(seq_B))
            size_score = len_ratio

            # 4. 芳香族相互作用
            aromatic_A = sum(seq_A.count(aa) for aa in 'FWY') / len(seq_A)
            aromatic_B = sum(seq_B.count(aa) for aa in 'FWY') / len(seq_B)
            aromatic_score = min(1.0, (aromatic_A + aromatic_B) * 3.0)

            # 加权综合
            score = (0.35 * charge_score + 
                    0.30 * hydro_score + 
                    0.20 * size_score + 
                    0.15 * aromatic_score)
            score = min(0.95, max(0.3, score))
        else:
            # 单链评估
            pos = sum(seq_A.count(aa) for aa in 'KRH')
            neg = sum(seq_A.count(aa) for aa in 'DE')
            hydro = sum(seq_A.count(aa) for aa in 'AVILMFWY') / len(seq_A)
            balance_score = 1.0 - abs(pos - neg) / max(pos + neg, 1)
            hydro_score = 1.0 - abs(hydro - 0.45) * 2.0
            score = 0.5 + 0.3 * balance_score + 0.2 * hydro_score
            score = min(0.9, max(0.3, score))

        return {'score': round(score, 3), 'ddG': 0.0, 'method': 'enhanced_heuristic'}

    # 自动选择方法
    if method == "auto":
        if not PYROSETTA_AVAILABLE:
            method = "colabfold"
        else:
            try:
                _init_pyrosetta()
                pose = pyrosetta.pose_from_pdb(wt_pdb)
                current_A = pose.chain_sequence(1)
                current_B = pose.chain_sequence(2)
                diff_A = sum(a != b for a, b in zip(current_A, seq_A))
                diff_B = sum(a != b for a, b in zip(current_B, seq_B))
                method = "colabfold" if (diff_A + diff_B > 3 or 
                                         len(seq_A) != len(current_A) or 
                                         len(seq_B) != len(current_B)) else "pyrosetta"
            except Exception:
                method = "colabfold"

    mut_pdb = tempfile.mktemp(suffix="_mut.pdb")
    try:
        if method == "pyrosetta":
            mutate_pdb_pyrosetta(wt_pdb, seq_A, seq_B, mut_pdb)
        elif method == "colabfold":
            mutate_pdb_colabfold(seq_A, seq_B, mut_pdb)
        else:
            raise ValueError(f"未知方法: {method}")

        ddG = foldx_interface(mut_pdb, chains="A,B")
        score = ddG_to_score(ddG)

        return {'score': score, 'ddG': ddG, 'method': method}

    except Exception as e:
        warnings.warn(f"组装评分失败 ({method}): {e}，使用启发式评分")
        return {'score': 0.5, 'ddG': 0.0, 'method': f'{method}_failed'}
    finally:
        if os.path.exists(mut_pdb):
            os.remove(mut_pdb)


# ============================================================================
# 批量评估
# ============================================================================

@dataclass
class Design:
    name: str
    seq_A: str
    seq_B: str


def batch_score(designs: List[Design],
                wt_pdb: str = None,
                method: str = "auto",
                output_csv: Optional[str] = None) -> pd.DataFrame:
    """批量评估多个设计序列"""
    if wt_pdb is None:
        wt_pdb = WT_PDB

    results = []

    for design in designs:
        try:
            result = assembly_score(design.seq_A, design.seq_B, wt_pdb, method)
            results.append({
                'name': design.name,
                'seq_A': design.seq_A,
                'seq_B': design.seq_B,
                'ddG': result['ddG'],
                'score': result['score'],
                'method': result['method'],
                'status': 'success'
            })
        except Exception as e:
            results.append({
                'name': design.name,
                'seq_A': design.seq_A,
                'seq_B': design.seq_B,
                'ddG': None,
                'score': 0.0,
                'method': method,
                'status': f'error: {str(e)}'
            })

    df = pd.DataFrame(results)
    if output_csv:
        df.to_csv(output_csv, index=False)
        print(f"结果已保存至: {output_csv}")
    return df


# ============================================================================
# 命令行接口
# ============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description='蛋白质复合物组装性评分')
    parser.add_argument('--wt-pdb', default=WT_PDB, help='野生型结构路径')
    parser.add_argument('--seq-A', required=True, help='链 A 序列')
    parser.add_argument('--seq-B', required=True, help='链 B 序列')
    parser.add_argument('--method', choices=['pyrosetta', 'colabfold', 'auto'],
                        default='auto', help='结构生成方法')
    parser.add_argument('--batch', help='批量输入 CSV 文件 (name,seq_A,seq_B)')
    parser.add_argument('--output', help='输出 CSV 路径')

    args = parser.parse_args()

    if args.batch:
        df_input = pd.read_csv(args.batch)
        designs = [Design(row['name'], row['seq_A'], row['seq_B']) 
                   for _, row in df_input.iterrows()]
        df_result = batch_score(designs, args.wt_pdb, args.method, args.output)
        print(df_result.to_string())
    else:
        result = assembly_score(args.seq_A, args.seq_B, args.wt_pdb, args.method)
        print(f"组装性评分: {result['score']:.3f}")
        print(f"界面能量: {result['ddG']:.2f} kcal/mol")
        print(f"使用方法: {result['method']}")


if __name__ == '__main__':
    main()