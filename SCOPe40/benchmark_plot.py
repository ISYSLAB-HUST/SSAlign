import os

import matplotlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scienceplots
from matplotlib.ticker import FuncFormatter
from sklearn.metrics import auc, precision_recall_curve


plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体为黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
plt.rcParams['font.family'] = 'DejaVu Sans'

matplotlib.use('TkAgg')  # 或 'Qt5Agg'

plt.style.use(['science', 'no-latex','grid'])

matplotlib.rcParams.update({
    'font.family': 'serif',        # 主字体类型
    'font.serif': 'Times New Roman',  # 明确指定衬线字体
    'font.size': 9,                # 基础字号
    'axes.titlesize': 10,          # 标题字号
    'axes.labelsize': 9,           # 坐标轴标签字号
    'xtick.labelsize': 8,          # X轴刻度字号
    'ytick.labelsize': 8,          # Y轴刻度字号
    'legend.fontsize': 8,          # 图例字号
    'legend.title_fontsize': 9,    # 图例标题字号
    'mathtext.fontset': 'stix',    # 数学公式字体
    'figure.dpi': 600,             # 输出分辨率
})


def plot_firstFP(true_level, dim):

    data_sets = {
        "Foldseek": np.load("../benchmarkData/SCOPe40/cumsumNpz/foldseek_firstFP_accuracy_results05.npz"),
        "TM-align": np.load("../benchmarkData/SCOPe40/cumsumNpz/tmalign_firstFP_accuracy_results05.npz"),
        "SSAlign": np.load(f"../benchmarkData/SCOPe40/cumsumNpz/dim_{dim}_ssalign_firstFP_accuracy_results05.npz"),
        "SSAlign-prefilter-500": np.load(f"../benchmarkData/SCOPe40/cumsumNpz/dim_{dim}_ssalign-prefilter-500_firstFP_accuracy_results05.npz"),
        "SSAlign-prefilter-1000": np.load(f"../benchmarkData/SCOPe40/cumsumNpz/dim_{dim}_ssalign-prefilter-1000_firstFP_accuracy_results05.npz")
    }
    # 下面绘图测试
    colors = {
        'Foldseek': '#5BB85B',
        'TM-align': '#EEA235',
        'SSAlign': '#6D3E99',
        'SSAlign-prefilter-500': '#448DFC',
        'SSAlign-prefilter-1000': '#000000',
    }


    markers = {
        'Foldseek': '*',  # 五角星
        'TM-align': '+',  # 加号
        'SSAlign': 'x',  # 叉号
        'SSAlign-prefilter-500': '1',  # 三脚架↓
        'SSAlign-prefilter-1000': '2',  # 三脚架↑
    }

    # 横坐标（0到1，步长0.01）
    x = np.arange(0, 1.01, 0.01)

    # 统一标记参数
    markersize = 4
    marker_step = len(x) // 20  # 100/20=5

    def thousands105_formatter(x, pos):
        return f'{int(x / 100000)}' if x >= 1000 else str(int(x))


    # 绘制每个数据集
    # 绘制每个数据集
    for name, data in data_sets.items():

        # 下面是只是用个数
        y = data[f"{true_level}"]

        plt.plot(x, y,
                 color=colors[name],
                 label=name,
                 linestyle='-',
                 linewidth=1.5,
                 marker=markers[name],
                 markevery=marker_step,
                 markersize=markersize,
                 solid_capstyle='round')

    plt.title(f'{true_level}', pad=12)
    plt.xlabel('Fraction of queries')
    plt.ylabel('Hits up to the 1st FP (×10$^5$)')
    plt.xticks(fontsize=11)
    plt.yticks(fontsize=11)


    ax = plt.gca()

    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                 ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontname('Times New Roman')

    ax.yaxis.set_major_formatter(FuncFormatter(thousands105_formatter))

    # 调整布局并保存
    plt.tight_layout()
    plt.savefig(f'../benchmarkData/SCOPe40/benchmark/{dim}_{true_level}_sensitivity_1st.png',dpi=600, bbox_inches='tight')


def plot_PR(true_level,dim):

    data_all_true = np.load("../benchmarkData/SCOPe40/cumsumNpz/tmalign_true_counts_results05.npz")
    total_true = {
        'family': data_all_true['family_true'].sum(),
        'superfamily': data_all_true['superfamily_true'].sum(),
        'folderror': data_all_true['folderror'].sum()
    }

    print(total_true['family'])


    tools = {
        "Foldseek": np.load("../benchmarkData/SCOPe40/cumsumNpz/foldseek_firstFP_accuracy_results05.npz"),
        "TM-align": np.load("../benchmarkData/SCOPe40/cumsumNpz/tmalign_firstFP_accuracy_results05.npz"),
        "SSAlign": np.load(f"../benchmarkData/SCOPe40/cumsumNpz/dim_{dim}_ssalign_firstFP_accuracy_results05.npz"),
        "SSAlign-prefilter-500": np.load(f"../benchmarkData/SCOPe40/cumsumNpz/dim_{dim}_ssalign-prefilter-500_firstFP_accuracy_results05.npz"),
        "SSAlign-prefilter-1000": np.load(f"../benchmarkData/SCOPe40/cumsumNpz/dim_{dim}_ssalign-prefilter-1000_firstFP_accuracy_results05.npz")
    }

    # 计算准确率和召回率
    results = {}
    for name, data in tools.items():
        results[name] = {
            'precision': data[f'{true_level}'] / data['query_sum'],
            'recall': data[f'{true_level}'] / total_true['family']
        }

    colors = {
        'Foldseek': '#5BB85B',
        'TM-align': '#EEA235',
        'SSAlign': '#6D3E99',
        'SSAlign-prefilter-500': '#448DFC',
        'SSAlign-prefilter-1000': '#000000',
    }

    markers = {
        'Foldseek': '*',  # 五角星
        'TM-align': '+',  # 加号
        'SSAlign': 'x',  # 叉号
        'SSAlign-prefilter-500': '1',  # 三脚架↓
        'SSAlign-prefilter-1000': '2',  # 三脚架↑
    }

    markersize = 4
    marker_step = 100 // 20  # 100/20=5
    # 绘制PR曲线
    for name in tools.keys():
        plt.plot(results[name]['recall'],
                 results[name]['precision'],
                 label=name,
                 color=colors[name],
                 linewidth=1.5,
                 marker=markers[name],
                 markevery=marker_step,
                 markersize=markersize,
                 solid_capstyle='round')

        print(f"{name}工具的：recall",results[name]['recall'].max())

    # 图表美化
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'{true_level}', pad=12)
    plt.xticks(fontsize=11)
    plt.yticks(fontsize=11)

    plt.grid(True, linestyle='--', alpha=0.5)

    # 保存图像
    plt.savefig(f'../benchmarkData/SCOPe40/benchmark/{dim}_{true_level}_PR_curve.png', dpi=600,bbox_inches='tight')
    plt.close()

    print(f"PR曲线图已保存为 {dim}_{true_level}_PR_curve.png")





def sklean_PR_auc(dim):
    data_all_true = np.load("../benchmarkData/SCOPe40/cumsumNpz/tmalign_true_counts_results05.npz")
    total_true = {
        'family': data_all_true['family_true'].sum(),
        'superfamily': data_all_true['superfamily_true'].sum(),
        'folderror': data_all_true['folderror'].sum()
    }



    tools = {
        "Foldseek": np.load("../benchmarkData/SCOPe40/cumsumNpz/foldseek_firstFP_accuracy_results05.npz"),
        "TM-align": np.load("../benchmarkData/SCOPe40/cumsumNpz/tmalign_firstFP_accuracy_results05.npz"),
        "SSAlign": np.load(f"../benchmarkData/SCOPe40/cumsumNpz/dim_{dim}_ssalign_firstFP_accuracy_results05.npz"),
        "SSAlign-prefilter-500": np.load(f"../benchmarkData/SCOPe40/cumsumNpz/dim_{dim}_ssalign-prefilter-500_firstFP_accuracy_results05.npz"),
        "SSAlign-prefilter-1000": np.load(f"../benchmarkData/SCOPe40/cumsumNpz/dim_{dim}_ssalign-prefilter-1000_firstFP_accuracy_results05.npz")
    }

    # 计算准确率和召回率
    results = {}
    for name, data in tools.items():
        results[name] = {
            'precision_family': data['family_true'] / data['query_sum'],
            'precision_superfamily': data['superfamily_true'] / data['query_sum'],
            'recall_family': data['family_true'] / total_true['family'],
            'recall_superfamily': data['superfamily_true'] / total_true['superfamily']
        }


    for tool_name, data in results.items():
        # 构造模拟标签和概率（假设负样本为0）
        family_auc = auc(results[tool_name]['recall_family'], results[tool_name]['precision_family'])
        superfamily_auc = auc(results[tool_name]['recall_superfamily'], results[tool_name]['precision_superfamily'])

        print(f"{tool_name},family层面的曲线下面积是 : " ,family_auc)
        print(f"{tool_name},superfamily层面的曲线下面积是 : " ,superfamily_auc)



def sklean_firstFP_auc():
    data_sets = {
        "Foldseek": np.load("../benchmarkData/SCOPe40/cumsumNpz/foldseek_firstFP_accuracy_results05.npz"),
        "TM-align": np.load("../benchmarkData/SCOPe40/cumsumNpz/tmalign_firstFP_accuracy_results05.npz"),
        "SSAlign": np.load(f"../benchmarkData/SCOPe40/cumsumNpz/dim_{dim}_ssalign_firstFP_accuracy_results05.npz"),
        "SSAlign-prefilter-500": np.load(f"../benchmarkData/SCOPe40/cumsumNpz/dim_{dim}_ssalign-prefilter-500_firstFP_accuracy_results05.npz"),
        "SSAlign-prefilter-1000": np.load(f"../benchmarkData/SCOPe40/cumsumNpz/dim_{dim}_ssalign-prefilter-1000_firstFP_accuracy_results05.npz")
    }

    # 横坐标（0到1，步长0.01）
    x = np.arange(0, 1.01, 0.01)

    for name, data in data_sets.items():
        y_family = data["family_true"] / data["query_sum"]
        y_superfamily = data["superfamily_true"] / data["query_sum"]

        family_auc = auc(x, y_family)
        superfamily_auc = auc(x,y_superfamily)

        print(f"{name},family层面的曲线下面积是 : ", family_auc)
        print(f"{name},superfamily层面的曲线下面积是 : ", superfamily_auc)






if __name__=="__main__":

    for dim in [1280, 512, 256, 128, 64]:
        plot_firstFP("family_true", dim)
        plot_firstFP("superfamily_true", dim)

        plot_PR("family_true", dim)
        plot_PR("superfamily_true", dim)

        sklean_PR_auc()
        sklean_firstFP_auc()


