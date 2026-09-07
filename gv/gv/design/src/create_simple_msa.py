import os
import glob
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq

def create_simple_msa(input_file, output_dir=None):
    """
    读取单个 FASTA，截断到最短序列长度，输出为 _msa.a3m
    """
    sequences = list(SeqIO.parse(input_file, "fasta"))
    if not sequences:
        print(f"[跳过] {input_file} 为空文件")
        return

    # 截断到相同长度
    min_len = min(len(rec.seq) for rec in sequences)
    aligned = []
    for rec in sequences:
        seq_str = str(rec.seq)[:min_len]
        aligned.append(SeqRecord(Seq(seq_str), id=rec.id, description=""))

    # 构造输出文件名：GvpC.fasta -> GvpC_msa.a3m
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"{base_name}_msa.a3m")
    else:
        # 默认和原文件同目录
        dir_name = os.path.dirname(input_file)
        output_file = os.path.join(dir_name, f"{base_name}_msa.a3m")

    SeqIO.write(aligned, output_file, "fasta")
    print(f"[完成] {os.path.basename(input_file)} -> {os.path.basename(output_file)} "
          f"(序列数: {len(aligned)}, 对齐长度: {min_len})")


def batch_process(input_dir, output_dir=None):
    """
    批量处理目录下所有 .fasta 文件
    """
    # 支持 .fasta 和 .fa 后缀
    patterns = [os.path.join(input_dir, "*.fasta"), os.path.join(input_dir, "*.fa")]
    files = []
    for p in patterns:
        files.extend(glob.glob(p))

    if not files:
        print(f"在 {input_dir} 下未找到 .fasta 或 .fa 文件")
        return

    print(f"共找到 {len(files)} 个 FASTA 文件，开始处理...\n")
    for f in sorted(files):
        create_simple_msa(f, output_dir)
    print("\n全部处理完毕！")


if __name__ == "__main__":
    # ==================== 修改这里 ====================
    # 你的数据目录路径（Windows 用原始字符串 r"..." 或双反斜杠）
    INPUT_DIR = r"D:\毕设\1\data"
    
    # 输出目录：None 表示和输入同目录；也可以改成 r"D:\毕设\1\output"
    OUTPUT_DIR = None
    # =================================================

    batch_process(INPUT_DIR, OUTPUT_DIR)