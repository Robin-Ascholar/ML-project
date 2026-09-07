#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import platform
import sqlite3
import subprocess
import importlib.metadata as metadata


# =========================================================
# 可修改路径
# =========================================================
MAFFT_PATH = r"C:\Users\r9000\Desktop\mafft-win\mafft.bat"

# 如果 PyMOL 已加入环境变量，可以写 "pymol"
# 如果没有加入环境变量，可以改成类似：
# r"C:\Program Files\PyMOL\PyMOLWin.exe"
PYMOL_EXE = "pymol"


# =========================================================
# 期望版本，可按论文表格填写
# =========================================================
EXPECTED = {
    "操作系统": "Windows 10，版本号10.0.26200",
    "Python": "3.11.15",
    "PyTorch": "2.12.0.dev20260408+cu128",
    "CUDA": "12.8，GPU可用",
    "ESM / fair-esm": "2.0.0",
    "ESM-2": "esm2_t12_35M_UR50D；esm2_t30_150M_UR50D；esm2_t33_650M_UR50D",
    "NumPy": "2.4.4",
    "Pandas": "2.3.3",
    "scikit-learn": "1.8.0",
    "UMAP": "0.5.11",
    "HDBSCAN": "v7.526",
    "MAFFT": "当前Python环境未检测到命令行版本",
    "PyMOL": "3.1.0",
    "SQLite": "3.53.0",
    "Streamlit": "1.56.0",
}


# =========================================================
# 通用工具函数
# =========================================================
def get_package_version(package_name):
    try:
        return metadata.version(package_name)
    except metadata.PackageNotFoundError:
        return "未安装或当前环境不可见"


def run_cmd(cmd):
    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=True,
            text=True,
            timeout=20
        )
        output = (result.stdout or result.stderr or "").strip()
        return output if output else "命令已执行，但无版本输出"
    except Exception as e:
        return f"检测失败：{e}"


def check_status(actual, expected):
    if not expected:
        return "未设置期望值"

    if actual == expected:
        return "一致"

    if str(expected) in str(actual) or str(actual) in str(expected):
        return "基本一致"

    if "未安装" in str(actual) or "检测失败" in str(actual) or "未检测到" in str(actual):
        return "未检测到"

    return "需核对"


# =========================================================
# 各项版本检测
# =========================================================
def check_os():
    return f"{platform.system()} {platform.release()}，版本号{platform.version()}"


def check_python():
    return platform.python_version()


def check_pytorch_and_cuda():
    try:
        import torch

        torch_version = torch.__version__
        cuda_version = torch.version.cuda

        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            cuda_info = f"{cuda_version}，GPU可用，设备：{gpu_name}"
        else:
            cuda_info = f"{cuda_version}，GPU不可用"

        return torch_version, cuda_info

    except Exception as e:
        return f"检测失败：{e}", f"检测失败：{e}"


def check_fair_esm():
    version = get_package_version("fair-esm")

    if version == "未安装或当前环境不可见":
        version = get_package_version("esm")

    return version


def check_mafft():
    if not os.path.exists(MAFFT_PATH):
        return f"未找到MAFFT路径：{MAFFT_PATH}"

    cmd = f'"{MAFFT_PATH}" --version'
    return run_cmd(cmd)


def check_pymol():
    # 方法1：尝试从Python包导入
    try:
        import pymol
        try:
            version = pymol.cmd.get_version()[0]
            return version
        except Exception:
            pass
    except Exception:
        pass

    # 方法2：尝试命令行调用
    cmd = f'"{PYMOL_EXE}" -cq -d "from pymol import cmd; print(cmd.get_version()[0]); quit"'
    output = run_cmd(cmd)

    if "检测失败" in output or "不是内部或外部命令" in output:
        return "当前Python环境未检测到PyMOL命令行版本"

    return output


def check_sqlite():
    return sqlite3.sqlite_version


def main():
    torch_version, cuda_info = check_pytorch_and_cuda()

    actual = {
        "操作系统": check_os(),
        "Python": check_python(),
        "PyTorch": torch_version,
        "CUDA": cuda_info,
        "ESM / fair-esm": check_fair_esm(),
        "ESM-2": "esm2_t12_35M_UR50D；esm2_t30_150M_UR50D；esm2_t33_650M_UR50D",
        "NumPy": get_package_version("numpy"),
        "Pandas": get_package_version("pandas"),
        "scikit-learn": get_package_version("scikit-learn"),
        "UMAP": get_package_version("umap-learn"),
        "HDBSCAN": get_package_version("hdbscan"),
        "MAFFT": check_mafft(),
        "PyMOL": check_pymol(),
        "SQLite": check_sqlite(),
        "Streamlit": get_package_version("streamlit"),
    }

    purpose = {
        "操作系统": "实验运行环境",
        "Python": "数据处理与模型调用",
        "PyTorch": "ESM-2模型推理",
        "CUDA": "模型推理加速",
        "ESM / fair-esm": "ESM-2模型调用环境",
        "ESM-2": "片段嵌入提取",
        "NumPy": "数值计算",
        "Pandas": "表格数据整理",
        "scikit-learn": "聚类质量评价指标计算",
        "UMAP": "嵌入降维与可视化",
        "HDBSCAN": "密度聚类分析",
        "MAFFT": "多序列比对",
        "PyMOL": "结构渲染与二级结构统计",
        "SQLite": "候选片段结果存储",
        "Streamlit": "数据库原型界面展示",
    }

    rows = []

    for name in EXPECTED.keys():
        rows.append({
            "项目": name,
            "检测版本": actual.get(name, "未检测"),
            "期望版本": EXPECTED.get(name, ""),
            "用途": purpose.get(name, ""),
            "状态": check_status(actual.get(name, ""), EXPECTED.get(name, "")),
        })

    print("\n主要实验环境与软件版本检测结果")
    print("=" * 120)

    for row in rows:
        print(f"{row['项目']:<18} | {row['检测版本']:<45} | {row['状态']}")

    print("=" * 120)

    try:
        import pandas as pd

        df = pd.DataFrame(rows)
        out_path = "environment_version_check.csv"
        df.to_csv(out_path, index=False, encoding="utf-8-sig")

        print(f"\n检测结果已保存：{os.path.abspath(out_path)}")

    except Exception as e:
        print(f"\n未生成CSV文件：{e}")


if __name__ == "__main__":
    main()
