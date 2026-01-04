import os
import re
import numpy as np
import matplotlib

# ===== headless 环境：必须在 import pyplot 之前设置 backend =====
if os.environ.get("DISPLAY", "") == "":
    matplotlib.use("Agg")  # 集群/无GUI

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




def plot_cumsum_no_brokenaxes_5tools(except_data_file_path, begin_points, max_points):
    data = np.load(except_data_file_path)
    print(data.files)


    ssalign = data['ssalign'][begin_points:max_points]
    foldseek = data['foldseek'][begin_points:max_points]
    ssalign_prefilter_2000 = data['ssalign_prefilter_2000'][begin_points:max_points]





    colors = {
        'foldseek': '#5BB85B',
        'ssalign': '#6D3E99',
        'ssalign_prefilter_2000': '#000000',
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
            ['ssalign', 'foldseek',
             'ssalign_prefilter_2000'],
            [ssalign, foldseek,
              ssalign_prefilter_2000,
             ]
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
    plt.savefig(f'/{except_data_file_path}_no_brokenaxes.png', dpi=600, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":

    data_file_path = "/SSAlign/afdb50/test_pdb/benchmark/cumsumNpz/AFDB50_dim_512_cumsum_avg_TM-score_sorted_by_avg_TM-score.npz"
    plot_cumsum_no_brokenaxes_5tools(data_file_path, 0, 150000)

    data_file_path = "/SSAlign/afdb50/test_pdb/benchmark/cumsumNpz/AFDB50_dim_512_cumsum_avg_TM-score_sorted_by_method_measure.npz"
    plot_cumsum_no_brokenaxes_5tools(data_file_path, 0, 150000)

    data_file_path = "/SSAlign/afdb50/test_pdb/benchmark/cumsumNpz/AFDB50_dim_512_cumsum_RMSD_sorted_by_avg_TM-score.npz"
    plot_cumsum_no_brokenaxes_5tools(data_file_path, 0, 150000)
    #
    data_file_path = "/SSAlign/afdb50/test_pdb/benchmark/cumsumNpz/AFDB50_dim_512_cumsum_RMSD_sorted_by_method_measure.npz"
    plot_cumsum_no_brokenaxes_5tools(data_file_path, 0, 150000)
