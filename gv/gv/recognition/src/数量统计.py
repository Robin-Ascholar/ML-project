import os
import re
from collections import defaultdict

# ================= 配置区域 =================

TARGET_FOLDER = r"C:\Users\r9000\Desktop\毕业设计\data\提取结果_按Gvp分类"

# 合并后的原始序列文件
MERGED_OUTPUT = "jia_all_combined.fasta"

# CD-HIT 去冗余后的文件
DEDUP_OUTPUT = r"C:\Users\r9000\Desktop\毕业设计\去重后\jia_95nr.fasta"

# 拆分后的输出文件夹
SPLIT_OUTPUT_DIR = "jia_split_results"

# 统计结果表
COUNT_OUTPUT = "sequence_count_summary.csv"

# ===========================================


def extract_gene_name(filepath):
    """
    从文件路径中提取 GVP 基因名。
    优先从文件名提取，例如 gvpA_data.fasta -> gvpa；
    如果文件名没有，则从父文件夹名提取。
    """
    filename = os.path.basename(filepath)
    parent_dir = os.path.basename(os.path.dirname(filepath))

    name_base = os.path.splitext(filename)[0]
    match = re.search(r"(gvp[a-z])", name_base, re.IGNORECASE)
    if match:
        return match.group(1).lower()

    match_dir = re.search(r"(gvp[a-z])", parent_dir, re.IGNORECASE)
    if match_dir:
        return match_dir.group(1).lower()

    return f"unknown_{parent_dir}"


def parse_gene_from_header(header_line):
    """
    从 FASTA 标题行中解析 gene=xxx 标签。
    """
    gene = "unknown"

    parts = header_line.strip().split("|")
    for part in parts:
        if part.startswith("gene="):
            gene = part.replace("gene=", "").strip()
            break

    return gene


def scan_recursive_and_merge(root_folder, output_file):
    """
    递归扫描目标文件夹下所有 FASTA 文件，并合并为一个总 FASTA 文件。
    同时在每条序列标题行中加入 gene 和 src 标签。
    """
    if not os.path.exists(root_folder):
        print(f"❌ 错误：找不到文件夹：{root_folder}")
        return False

    print(f"🔍 开始递归扫描：{root_folder}")

    fasta_files = []

    for dirpath, dirnames, filenames in os.walk(root_folder):
        for filename in filenames:
            if filename.lower().endswith((".fasta", ".fa", ".fas", ".faa")):
                full_path = os.path.join(dirpath, filename)
                fasta_files.append(full_path)

    if not fasta_files:
        print(f"❌ 未找到任何 FASTA 文件。")
        return False

    print(f"✅ 共发现 {len(fasta_files)} 个 FASTA 文件。")
    print("📌 开始合并并添加 gene 标签...\n")

    total_seqs = 0
    file_stats = []

    with open(output_file, "w", encoding="utf-8") as f_out:
        for f_path in sorted(fasta_files):
            rel_path = os.path.relpath(f_path, root_folder)
            gene_tag = extract_gene_name(f_path)

            count = 0

            try:
                with open(f_path, "r", encoding="utf-8", errors="ignore") as f_in:
                    for line in f_in:
                        line = line.rstrip("\n")

                        if line.startswith(">"):
                            original_desc = line.strip()[1:]
                            new_header = f">{original_desc}|gene={gene_tag}|src={rel_path}"
                            f_out.write(new_header + "\n")
                            count += 1
                        else:
                            f_out.write(line + "\n")

                print(f"  📄 [{gene_tag}] {rel_path} -> {count} 条")
                total_seqs += count
                file_stats.append((gene_tag, rel_path, count))

            except Exception as e:
                print(f"  ⚠️ 读取失败：{f_path}")
                print(f"     原因：{e}")

    print("\n" + "-" * 60)
    print("🎉 原始 FASTA 合并完成")
    print(f"原始序列总数：{total_seqs:,}")
    print(f"合并输出文件：{output_file}")
    print("-" * 60)

    return True


def count_fasta_by_gene(input_file):
    """
    统计 FASTA 文件中的序列总数，并按 gene 类型统计。
    适用于 jia_all_combined.fasta 和 jia_all_nr.fasta。
    """
    if not os.path.exists(input_file):
        print(f"❌ 文件不存在：{input_file}")
        return 0, {}

    total = 0
    gene_stats = defaultdict(int)

    with open(input_file, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith(">"):
                total += 1
                gene = parse_gene_from_header(line)
                gene_stats[gene] += 1

    return total, dict(gene_stats)


def print_gene_stats(title, total, gene_stats):
    """
    打印按基因类型统计结果。
    """
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)
    print(f"序列总数：{total:,}")
    print("\n按 GVP 类型统计：")

    for gene, count in sorted(gene_stats.items(), key=lambda x: x[1], reverse=True):
        print(f"  - {gene}: {count:,} 条")


def compare_raw_and_dedup(raw_file, dedup_file):
    """
    对比原始合并序列和 CD-HIT 去冗余后的序列数量。
    """
    raw_total, raw_stats = count_fasta_by_gene(raw_file)
    dedup_total, dedup_stats = count_fasta_by_gene(dedup_file)

    print("\n" + "=" * 80)
    print("📊 原始序列与 CD-HIT 去冗余后序列数量对比")
    print("=" * 80)

    print(f"原始合并序列总数：{raw_total:,}")
    print(f"CD-HIT 去冗余后序列总数：{dedup_total:,}")

    if raw_total > 0:
        removed = raw_total - dedup_total
        keep_ratio = dedup_total / raw_total * 100
        remove_ratio = removed / raw_total * 100

        print(f"去除冗余序列数：{removed:,}")
        print(f"去冗余后保留比例：{keep_ratio:.2f}%")
        print(f"冗余序列去除比例：{remove_ratio:.2f}%")

    print("\n按 GVP 类型统计：")
    print(f"{'GVP类型':<15}{'原始数量':>15}{'去冗余后数量':>18}{'去除数量':>15}{'保留比例':>15}")

    all_genes = sorted(set(raw_stats.keys()) | set(dedup_stats.keys()))

    rows = []

    for gene in all_genes:
        raw_count = raw_stats.get(gene, 0)
        dedup_count = dedup_stats.get(gene, 0)
        removed_count = raw_count - dedup_count

        if raw_count > 0:
            ratio = dedup_count / raw_count * 100
            ratio_text = f"{ratio:.2f}%"
        else:
            ratio_text = "-"

        print(
            f"{gene:<15}"
            f"{raw_count:>15,}"
            f"{dedup_count:>18,}"
            f"{removed_count:>15,}"
            f"{ratio_text:>15}"
        )

        rows.append([gene, raw_count, dedup_count, removed_count, ratio_text])

    save_count_summary(rows)

    return raw_total, dedup_total, raw_stats, dedup_stats


def save_count_summary(rows):
    """
    保存统计结果为 CSV，方便整理到论文表格中。
    """
    try:
        with open(COUNT_OUTPUT, "w", encoding="utf-8-sig") as f:
            f.write("GVP类型,原始数量,去冗余后数量,去除数量,保留比例\n")
            for row in rows:
                f.write(",".join(map(str, row)) + "\n")

        print(f"\n📁 统计结果已保存：{COUNT_OUTPUT}")

    except Exception as e:
        print(f"\n⚠️ 统计结果保存失败：{e}")


def split_by_gene(input_file, out_dir):
    """
    按 gene 类型拆分 FASTA 文件。
    推荐对 CD-HIT 去冗余后的 jia_all_nr.fasta 进行拆分。
    """
    if not os.path.exists(input_file):
        print(f"❌ 输入文件不存在，无法拆分：{input_file}")
        return

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # 清空旧的拆分结果，避免重复追加
    for filename in os.listdir(out_dir):
        if filename.endswith(".fasta"):
            os.remove(os.path.join(out_dir, filename))

    print(f"\n✂️ 正在按 GVP 类型拆分：{input_file}")
    print(f"输出文件夹：{out_dir}")

    stats = defaultdict(int)
    current_record = []
    current_gene = "unknown"

    def write_record(gene, record_lines):
        if not record_lines:
            return

        safe_name = re.sub(r"[^\w\-_]", "_", gene)
        out_path = os.path.join(out_dir, f"{safe_name}.fasta")

        with open(out_path, "a", encoding="utf-8") as f_out:
            for rec_line in record_lines:
                f_out.write(rec_line)

        stats[safe_name] += 1

    with open(input_file, "r", encoding="utf-8", errors="ignore") as f_in:
        for line in f_in:
            if line.startswith(">"):
                write_record(current_gene, current_record)

                current_record = [line]
                current_gene = parse_gene_from_header(line)
            else:
                current_record.append(line)

        write_record(current_gene, current_record)

    print("\n📊 拆分统计：")
    for gene, count in sorted(stats.items(), key=lambda x: x[1], reverse=True):
        print(f"  - {gene}: {count:,} 条")

    print(f"\n🎉 拆分完成，结果保存在：{out_dir}")


if __name__ == "__main__":
    print("=" * 80)
    print("GVP 蛋白序列合并、CD-HIT 去冗余数量统计与拆分脚本")
    print("=" * 80)

    # 步骤 1：扫描原始 FASTA 文件并合并
    merge_success = scan_recursive_and_merge(TARGET_FOLDER, MERGED_OUTPUT)

    if not merge_success:
        print("❌ 合并失败，程序终止。")
        exit()

    # 步骤 2：统计原始合并文件数量
    raw_total, raw_stats = count_fasta_by_gene(MERGED_OUTPUT)
    print_gene_stats("📊 原始合并序列统计", raw_total, raw_stats)

    # 步骤 3：提示运行 CD-HIT
    print("\n" + "=" * 80)
    print("💡 下一步：运行 CD-HIT 去冗余")
    print("=" * 80)
    print("请在终端中运行下面这条命令：\n")
    print(f"cd-hit -i {MERGED_OUTPUT} -o {DEDUP_OUTPUT} -c 1.0 -n 5 -T 8 -d 0")

    print("\n参数说明：")
    print("-c 1.0 ：完全一致性去冗余")
    print("-n 5   ：适用于较高相似性阈值")
    print("-T 8   ：使用 8 个线程")
    print("-d 0   ：保留完整 FASTA 标题信息")

    # 步骤 4：如果去冗余文件已经存在，则自动统计对比
    if os.path.exists(DEDUP_OUTPUT):
        print(f"\n✅ 检测到 CD-HIT 去冗余文件：{DEDUP_OUTPUT}")

        compare_raw_and_dedup(MERGED_OUTPUT, DEDUP_OUTPUT)

        resp = input(f"\n是否按 GVP 类型拆分 {DEDUP_OUTPUT}？(y/n): ").strip().lower()

        if resp == "y":
            split_by_gene(DEDUP_OUTPUT, SPLIT_OUTPUT_DIR)
        else:
            print("已跳过拆分步骤。")

    else:
        print(f"\n⚠️ 当前未检测到去冗余文件：{DEDUP_OUTPUT}")
        print("请先运行上面的 CD-HIT 命令。")
        print("CD-HIT 运行完成后，再次运行本脚本，即可自动统计去冗余后的序列数量。")

    print("\n" + "=" * 80)
    print("程序运行结束")
    print("=" * 80)
