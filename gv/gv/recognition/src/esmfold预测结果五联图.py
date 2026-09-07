#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import re
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.image as mpimg


# =========================================================
# 输入路径配置
# =========================================================
BASE_DIR=Path(
    r"C:\Users\r9000\Desktop\毕设（无监督聚类）"
    r"\Table_4_11_ESMFold_API_Structure_Results\rendered_png"
)

FIGURE_TASKS=[
    {
        "gvp":"GvpA",
        "region":"25-55",
        "input_dir":BASE_DIR/"GvpA"/"region_25_55",
        "output_name":"GvpA_region_25_55_five_panel"
    },
    {
        "gvp":"GvpA",
        "region":"16-46",
        "input_dir":BASE_DIR/"GvpA"/"region_16_46",
        "output_name":"GvpA_region_16_46_five_panel"
    },
    {
        "gvp":"GvpC",
        "region":"76-111",
        "input_dir":BASE_DIR/"GvpC"/"region_76_111",
        "output_name":"GvpC_region_76_111_five_panel"
    },
    {
        "gvp":"GvpC",
        "region":"102-137",
        "input_dir":BASE_DIR/"GvpC"/"region_102_137",
        "output_name":"GvpC_region_102_137_five_panel"
    },
]


# =========================================================
# 输出目录：全部保存到论文用图
# =========================================================
OUTPUT_DIR=Path(r"C:\Users\r9000\Desktop\毕业设计\论文用图")
OUTPUT_DIR.mkdir(parents=True,exist_ok=True)


# =========================================================
# 工具函数
# =========================================================
def natural_key(path):
    name=path.name
    return [
        int(x) if x.isdigit() else x.lower()
        for x in re.split(r"(\d+)",name)
    ]


def collect_images(input_dir):
    exts=["*.png","*.jpg","*.jpeg","*.tif","*.tiff"]
    files=[]

    for ext in exts:
        files.extend(input_dir.glob(ext))

    files=sorted(files,key=natural_key)

    if len(files)<5:
        raise ValueError(
            f"{input_dir}中图片数量不足5张，当前为{len(files)}张"
        )

    if len(files)>5:
        print(
            f"⚠️{input_dir}中图片数量超过5张，"
            f"将按文件名排序取前5张"
        )

    return files[:5]


def make_five_panel(task):
    input_dir=task["input_dir"]
    gvp=task["gvp"]
    region=task["region"]
    output_name=task["output_name"]

    image_files=collect_images(input_dir)

    plt.rcParams["font.family"]="Times New Roman"
    plt.rcParams["axes.linewidth"]=0.8
    plt.rcParams["pdf.fonttype"]=42
    plt.rcParams["ps.fonttype"]=42

    fig,axes=plt.subplots(
        1,
        5,
        figsize=(15,3.2),
        dpi=600,
        facecolor="white"
    )

    panel_labels=["A","B","C","D","E"]

    for i,ax in enumerate(axes):
        img=mpimg.imread(image_files[i])
        ax.imshow(img)
        ax.axis("off")

        ax.text(
            0.02,
            0.96,
            panel_labels[i],
            transform=ax.transAxes,
            fontsize=14,
            fontweight="bold",
            va="top",
            ha="left",
            color="black",
            bbox=dict(
                facecolor="white",
                edgecolor="none",
                alpha=0.75,
                pad=1.5
            )
        )

    fig.suptitle(
        f"{gvp} candidate region {region}",
        fontsize=15,
        fontweight="bold",
        y=0.98
    )

    plt.subplots_adjust(
        left=0.01,
        right=0.99,
        top=0.86,
        bottom=0.02,
        wspace=0.02
    )

    png_path=OUTPUT_DIR/f"{output_name}.png"
    pdf_path=OUTPUT_DIR/f"{output_name}.pdf"

    fig.savefig(
        png_path,
        dpi=600,
        bbox_inches="tight",
        pad_inches=0.03
    )

    fig.savefig(
        pdf_path,
        bbox_inches="tight",
        pad_inches=0.03
    )

    plt.close(fig)

    print(f"✅已保存PNG:{png_path}")
    print(f"✅已保存PDF:{pdf_path}")


# =========================================================
# 主程序
# =========================================================
def main():
    for task in FIGURE_TASKS:
        print("\n"+"="*90)
        print(f"正在生成:{task['gvp']} {task['region']}")
        print(f"输入目录:{task['input_dir']}")
        make_five_panel(task)

    print("\n🎉全部五联图生成完成")
    print(f"输出目录:{OUTPUT_DIR}")


if __name__=="__main__":
    main()
