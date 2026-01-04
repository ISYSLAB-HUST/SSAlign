#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import numpy as np
import matplotlib

# ===== headless 环境：必须在 import pyplot 之前设置 backend =====
if os.environ.get("DISPLAY", "") == "":
    matplotlib.use("Agg")   # 集群/无GUI

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from brokenaxes import brokenaxes

# Optional: scienceplots style
try:
    import scienceplots  # noqa: F401
    plt.style.use(['science', 'no-latex', 'grid'])
except Exception:
    pass

# Fonts（集群没有也能出图，只是会 fallback）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Noto Sans CJK SC', 'WenQuanYi Micro Hei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

matplotlib.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 9,
    'axes.titlesize': 10,
    'axes.labelsize': 9,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'legend.title_fontsize': 9,
    'mathtext.fontset': 'stix',
    'figure.dpi': 600,
})


def thousands_formatter(x, pos):
    return f'{int(x / 1000)}' if x >= 1000 else str(int(x))


def parse_npz_name(npz_path: str):
    """
    从文件名中解析:
      cumsum_(score_measure)_sorted_by_(sort_mode).npz
    """
    base = os.path.basename(npz_path)
    m = re.search(r'cumsum_(.*?)_sorted_by_(.*?)\.npz$', base)
    if m:
        return m.group(1), m.group(2)
    return "score", "unknown"


def plot_cumsum_brokenaxes_5tools(data_file_path: str, begin_points: int, max_points: int):
    data = np.load(data_file_path, allow_pickle=False)
    print("[INFO] npz keys:", data.files)

    # ===== 只认你现在统一的 key =====
    required = ["foldseek", "ssalign", "ssalign_prefilter_1000", "ssalign_prefilter_2000"]
    missing = [k for k in required if k not in data.files]
    if missing:
        raise RuntimeError(f"npz 缺少 key: {missing}. 现有 keys={data.files}")

    foldseek_cumsum = data['foldseek'][begin_points: max_points]
    ssalign_cumsum = data['ssalign'][begin_points:max_points]
    ssalign_prefilter_1000_cumsum = data['ssalign_prefilter_1000'][begin_points:max_points]
    ssalign_prefilter_2000_cumsum = data['ssalign_prefilter_2000'][begin_points:max_points]

    # ===== x 轴（你原代码里 x 没定义，会报错）=====
    x_foldseek = np.arange(begin_points, begin_points + len(foldseek_cumsum))
    x = np.arange(begin_points, begin_points + len(ssalign_cumsum))

    score_measure, sort_mode = parse_npz_name(data_file_path)
    print("[INFO] score_measure:", score_measure)
    print("[INFO] sort_mode:", sort_mode)

    # ===== y 断轴范围：按你的旧设置保留 =====
    if "RMSD" in score_measure or "rmsd" in score_measure:
        ylims = ((0, 1), (75000, 260000))   # RMSD 用
        y_label = f'Cumulative RMSD (×10$^3$)'
    else:
        ylims = ((0, 1), (15000, 85000))    # Avg_TM-score 用
        y_label = f'Cumulative Avg_TM-score (×10$^3$)'

    bax = brokenaxes(
        xlims=((0, 1), (begin_points, max_points)),
        ylims=ylims,
        hspace=0.1,
        wspace=0.1,
        diag_color='k',
    )

    colors = {
        'Foldseek': '#5BB85B',
        'SSAlign': '#6D3E99',
        'SSAlign-prefilter-1000': '#448DFC',
        'SSAlign-prefilter-2000': '#000000',
    }

    # ===== 绘图 =====
    bax.plot(x_foldseek, foldseek_cumsum, label='Foldseek', color=colors['Foldseek'], linestyle='-')
    bax.plot(x, ssalign_cumsum, label='SSAlign', color=colors['SSAlign'], linestyle='-')
    bax.plot(x, ssalign_prefilter_1000_cumsum, label='SSAlign-prefilter-1000',
             color=colors['SSAlign-prefilter-1000'], linestyle='-')
    bax.plot(x, ssalign_prefilter_2000_cumsum, label='SSAlign-prefilter-2000',
             color=colors['SSAlign-prefilter-2000'], linestyle='-')

    bax.set_title(f'sorted by {sort_mode}', pad=12)
    bax.set_xlabel('top hits (×10$^3$)')
    bax.set_ylabel(y_label)

    for ax in bax.axs:
        ax.xaxis.set_major_formatter(FuncFormatter(thousands_formatter))
        ax.yaxis.set_major_formatter(FuncFormatter(thousands_formatter))
        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                     ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontname('Times New Roman')

    # brokenaxes 推荐用它自带 legend
    try:
        bax.legend(loc='lower right', fontsize=7, frameon=True, framealpha=0.8)
    except Exception:
        pass

    out_png = f'{data_file_path}_brokenaxes.png'
    plt.savefig(out_png, dpi=600, bbox_inches='tight')
    plt.close()
    print(f"[DONE] saved -> {out_png}")

"""
ssalign 和 ssalign_prefilter_1000 过于相近了。就不绘制 ssalign_prefilter_1000
"""
def _slice_with_x(arr: np.ndarray, begin_points: int, max_points: int):
    end = min(max_points, len(arr))
    y = arr[begin_points:end]
    x = np.arange(begin_points, begin_points + len(y))
    return x, y


def plot_cumsum_brokenaxes(data_file_path: str, begin_points: int, max_points: int):
    data = np.load(data_file_path, allow_pickle=False)
    print("[INFO] npz keys:", data.files)

    required = ["foldseek", "ssalign", "ssalign_prefilter_2000"]
    missing = [k for k in required if k not in data.files]
    if missing:
        raise RuntimeError(f"npz 缺少 key: {missing}. 现有 keys={data.files}")

    # ✅ 每个方法各自 slice，并各自生成 x
    x_foldseek, foldseek_cumsum = _slice_with_x(data["foldseek"], begin_points, max_points)
    x_ssalign,  ssalign_cumsum  = _slice_with_x(data["ssalign"], begin_points, max_points)
    x_p2000,    ssalign_prefilter_2000_cumsum = _slice_with_x(data["ssalign_prefilter_2000"], begin_points, max_points)

    score_measure, sort_mode = parse_npz_name(data_file_path)
    print("[INFO] score_measure:", score_measure)
    print("[INFO] sort_mode:", sort_mode)

    y_label = 'Cumulative Avg_TM-score (×10$^3$)'

    bax = brokenaxes(
        xlims=((0, 1), (begin_points, max_points)),
        hspace=0.1,
        wspace=0.1,
        diag_color='k',
    )

    colors = {
        'Foldseek': '#5BB85B',
        'SSAlign': '#6D3E99',
        'SSAlign-prefilter-2000': '#000000',
    }

    # ✅ 每条线用各自的 x
    bax.plot(x_foldseek, foldseek_cumsum, label='Foldseek', color=colors['Foldseek'], linestyle='-')
    bax.plot(x_ssalign,  ssalign_cumsum,  label='SSAlign',  color=colors['SSAlign'], linestyle='-')
    bax.plot(x_p2000,    ssalign_prefilter_2000_cumsum,
             label='SSAlign-prefilter-2000', color=colors['SSAlign-prefilter-2000'], linestyle='-')

    bax.set_title(f'sorted by {sort_mode}', pad=12)
    bax.set_xlabel('top hits (×10$^3$)')
    bax.set_ylabel(y_label)

    for ax in bax.axs:
        ax.xaxis.set_major_formatter(FuncFormatter(thousands_formatter))
        ax.yaxis.set_major_formatter(FuncFormatter(thousands_formatter))
        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                     ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontname('Times New Roman')

    out_png = f'{data_file_path}_brokenaxes.png'
    plt.savefig(out_png, dpi=600, bbox_inches='tight')
    plt.close()
    print(f"[DONE] saved -> {out_png}")


"""
   用于 绘制 各种方法的差集累计得分图，即论文中Figure3
   为了图像清晰， begin_points,max_points 应该按照实际选择的 except_data_file_path 长度选择
   建议 begin_points = 0， max_points=6k 比较清晰
"""
def plot_except_cumsum_no_brokenaxes_5tools(except_data_file_path,begin_points,max_points):
    data = np.load(except_data_file_path)

    print(data.files)  # 会列出所有保存的数组名称
    """
    只有前10k，因为再爱多就不明显了
    ['ssalign_except_foldseek_cumsum', 'foldseek_except_ssalign_cumsum', 'ssalign_prefilter_1000_except_foldseek_cumsum', 'ssalign_prefilter_2000_foldseek_cumsum', 'foldseek_except_ssalign_prefilter_1000_cumsum', 'foldseek_except_ssalign_prefilter_2000_cumsum']
   
    55092
    69417
    55101
    126595
    69441
    48568
    """

    ssalign_except_foldseek_cumsum = data['ssalign_except_foldseek_cumsum'][begin_points:max_points]
    foldseek_except_ssalign_cumsum = data['foldseek_except_ssalign_cumsum'][begin_points:max_points]
    ssalign_prefilter_1000_except_foldseek_cumsum = data['ssalign_prefilter_1000_except_foldseek_cumsum'][begin_points:max_points]
    ssalign_prefilter_2000_except_foldseek_cumsum = data['ssalign_prefilter_2000_foldseek_cumsum'][begin_points:max_points]
    foldseek_except_ssalign_prefilter_1000_cumsum = data['foldseek_except_ssalign_prefilter_1000_cumsum'][begin_points:max_points]
    foldseek_except_ssalign_prefilter_2000_cumsum = data['foldseek_except_ssalign_prefilter_2000_cumsum'][begin_points:max_points]

    print(len(ssalign_except_foldseek_cumsum))
    print(len(foldseek_except_ssalign_cumsum))
    print(len(ssalign_prefilter_1000_except_foldseek_cumsum))
    print(len(ssalign_prefilter_2000_except_foldseek_cumsum))
    print(len(foldseek_except_ssalign_prefilter_1000_cumsum))
    print(len(foldseek_except_ssalign_prefilter_2000_cumsum))

    colors = {
        'ssalign_except_foldseek_cumsum' : '#6D3E99',
        'foldseek_except_ssalign_cumsum': '#5BB85B',
        'ssalign_prefilter_1000_except_foldseek_cumsum': '#448DFC',
        'ssalign_prefilter_2000_except_foldseek_cumsum':'#000000',
        'foldseek_except_ssalign_prefilter_1000_cumsum':'#FAAF30',
        'foldseek_except_ssalign_prefilter_2000_cumsum':'#F47D1F',
    }




    def thousands_formatter(x, pos):
        return f'{int(x / 1000)}' if x >= 1000 else str(int(x))

    match = re.search(r'cumsum_(.*?)_sorted_by_(.*?)\.npz', except_data_file_path)
    if match:
        score_measure = match.group(1)  # 'avg_TM-score'
        sort_mode = match.group(2)  # 'method_measure'
        print("得分指标:", score_measure)
        print("排序方式:", sort_mode)
    else:
        print("Pattern not matched!")
    # print(sort_mode)
    # print(score_measure)

    for label, y_data in zip(
            ['ssalign_except_foldseek_cumsum','foldseek_except_ssalign_cumsum','ssalign_prefilter_1000_except_foldseek_cumsum' ,'ssalign_prefilter_2000_except_foldseek_cumsum' ,'foldseek_except_ssalign_prefilter_1000_cumsum' ,'foldseek_except_ssalign_prefilter_2000_cumsum' ,],
            [ssalign_except_foldseek_cumsum ,foldseek_except_ssalign_cumsum ,ssalign_prefilter_1000_except_foldseek_cumsum ,ssalign_prefilter_2000_except_foldseek_cumsum ,foldseek_except_ssalign_prefilter_1000_cumsum ,foldseek_except_ssalign_prefilter_2000_cumsum ,]
    ):
        x_data = range(1,len(y_data)+1)
        plt.plot(
            x_data, y_data,
            label=label,
            color=colors[label],
            linestyle='-',
            solid_capstyle='round',

        )

    # 标题和标签
    plt.title(f'sorted by {sort_mode}', pad=12)
    plt.xlabel('top hits (×10$^3$)')
    plt.ylabel(f'Cumulative {score_measure} (×10$^3$)')

    ax = plt.gca()
    ax.xaxis.set_major_formatter(FuncFormatter(thousands_formatter))
    ax.yaxis.set_major_formatter(FuncFormatter(thousands_formatter))

    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                 ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontname('Times New Roman')


    # Save and show
    plt.savefig(f'/{except_data_file_path}_no_brokenaxes_except.png', dpi=600, bbox_inches='tight')
    plt.show()


def plot_except_cumsum_no_brokenaxes(except_data_file_path, begin_points, max_points):
    data = np.load(except_data_file_path)

    print(data.files)  # 会列出所有保存的数组名称
    """
    ['ssalign_except_foldseek_cumsum', 'foldseek_except_ssalign_cumsum', , 'ssalign_prefilter_2000_foldseek_cumsum', , 'foldseek_except_ssalign_prefilter_2000_cumsum']

    55092
    69417
    126595
    48568
    """

    ssalign_except_foldseek_cumsum = data['ssalign_except_foldseek_cumsum'][begin_points:max_points]
    foldseek_except_ssalign_cumsum = data['foldseek_except_ssalign_cumsum'][begin_points:max_points]

    ssalign_prefilter_2000_except_foldseek_cumsum = data['ssalign_prefilter_2000_foldseek_cumsum'][
                                                    begin_points:max_points]
    foldseek_except_ssalign_prefilter_2000_cumsum = data['foldseek_except_ssalign_prefilter_2000_cumsum'][
                                                    begin_points:max_points]

    print(len(ssalign_except_foldseek_cumsum))
    print(len(foldseek_except_ssalign_cumsum))
    print(len(ssalign_prefilter_2000_except_foldseek_cumsum))
    print(len(foldseek_except_ssalign_prefilter_2000_cumsum))

    colors = {
        'ssalign_except_foldseek_cumsum': '#6D3E99',
        'foldseek_except_ssalign_cumsum': '#5BB85B',
        'ssalign_prefilter_2000_except_foldseek_cumsum': '#000000',
        'foldseek_except_ssalign_prefilter_2000_cumsum': '#F47D1F',
    }

    def thousands_formatter(x, pos):
        return f'{int(x / 1000)}' if x >= 1000 else str(int(x))

    match = re.search(r'cumsum_(.*?)_sorted_by_(.*?)\.npz', except_data_file_path)
    if match:
        score_measure = match.group(1)  # 'avg_TM-score'
        sort_mode = match.group(2)  # 'method_measure'
        print("得分指标:", score_measure)
        print("排序方式:", sort_mode)
    else:
        print("Pattern not matched!")
    # print(sort_mode)
    # print(score_measure)

    for label, y_data in zip(
            ['ssalign_except_foldseek_cumsum', 'foldseek_except_ssalign_cumsum',
              'ssalign_prefilter_2000_except_foldseek_cumsum',
             'foldseek_except_ssalign_prefilter_2000_cumsum', ],
            [ssalign_except_foldseek_cumsum, foldseek_except_ssalign_cumsum,
             ssalign_prefilter_2000_except_foldseek_cumsum,
              foldseek_except_ssalign_prefilter_2000_cumsum]
    ):
        x_data = range(1, len(y_data) + 1)
        plt.plot(
            x_data, y_data,
            label=label,
            color=colors[label],
            linestyle='-',
            solid_capstyle='round',

        )

    # 标题和标签
    plt.title(f'sorted by {sort_mode}', pad=12)
    plt.xlabel('top hits (×10$^3$)')
    plt.ylabel(f'Cumulative {score_measure} (×10$^3$)')

    ax = plt.gca()
    ax.xaxis.set_major_formatter(FuncFormatter(thousands_formatter))
    ax.yaxis.set_major_formatter(FuncFormatter(thousands_formatter))

    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                 ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontname('Times New Roman')

    # Save and show
    plt.savefig(f'/{except_data_file_path}_no_brokenaxes_except.png', dpi=600, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":

    data_file_path = "/SSAlign/afdb50/test_pdb/benchmark/cumsumNpz/AFDB50_dim_512_cumsum_avg_TM-score_sorted_by_avg_TM-score.npz"
    plot_cumsum_brokenaxes(data_file_path, 0, 150000)

    data_file_path = "/SSAlign/afdb50/test_pdb/benchmark/cumsumNpz/AFDB50_dim_512_cumsum_avg_TM-score_sorted_by_method_measure.npz"
    plot_cumsum_brokenaxes(data_file_path, 0, 150000)

    data_file_path = "/SSAlign/afdb50/test_pdb/benchmark/cumsumNpz/AFDB50_dim_512_cumsum_RMSD_sorted_by_avg_TM-score.npz"
    plot_cumsum_brokenaxes(data_file_path, 0, 150000)
    #
    data_file_path = "/SSAlign/afdb50/test_pdb/benchmark/cumsumNpz/AFDB50_dim_512_cumsum_RMSD_sorted_by_method_measure.npz"
    plot_cumsum_brokenaxes(data_file_path, 0, 150000)


    except_data_file_path = "/SSAlign/afdb50/test_pdb/benchmark/cumsumNpz/AFDB50_dim_512_except_cumsum_avg_TM-score_sorted_by_avg_TM-score.npz"
    plot_except_cumsum_no_brokenaxes(except_data_file_path,0,120000)

    except_data_file_path = "/SSAlign/afdb50/test_pdb/benchmark/cumsumNpz/AFDB50_dim_512_except_cumsum_avg_TM-score_sorted_by_method_measure.npz"
    plot_except_cumsum_no_brokenaxes(except_data_file_path,0,120000)

    except_data_file_path = "/SSAlign/afdb50/test_pdb/benchmark/cumsumNpz/AFDB50_dim_512_except_cumsum_RMSD_sorted_by_avg_TM-score.npz"
    plot_except_cumsum_no_brokenaxes(except_data_file_path,0,120000)

    except_data_file_path = "/SSAlign/afdb50/test_pdb/benchmark/cumsumNpz/AFDB50_dim_512_except_cumsum_RMSD_sorted_by_method_measure.npz"
    plot_except_cumsum_no_brokenaxes(except_data_file_path,0,120000)

