"""

利用累计和 测试绘图风格

下面是几种方法的差集  的 10k 的累计和
except_cumsum_avg_TM-score_sorted_by_avg_TM-score.npz
except_cumsum_avg_TM-score_sorted_by_method_measure.npz
except_cumsum_RMSD_sorted_by_avg_TM-score.npz
except_cumsum_RMSD_sorted_by_method_measure.npz


下面是几种方法收集的 200k 的 累计和
cumsum_avg_TM-score_sorted_by_avg_TM-score.npz
cumsum_avg_TM-score_sorted_by_method_measure.npz
cumsum_RMSD_sorted_by_avg_TM-score.npz
cumsum_RMSD_sorted_by_method_measure.npz


"""
import re

from brokenaxes import brokenaxes
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import scienceplots
from matplotlib.ticker import FuncFormatter
from matplotlib.patches import ConnectionPatch
from matplotlib import rcParams

plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体为黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
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

"""
plot_cumsum 和 plot_cumsum_brokenaxes用于 绘制 累计得分图，即论文中Figure2
两个函数的不同在于是否需要断轴

    为了更加集中显示差距，选择从 各个方法的 [begin_points,max_points] 区间来显示
    建议 begin_points = 40000， max_points=55000 比较清晰
"""


def plot_cumsum(data_file_path,begin_points,max_points):
    data = np.load(data_file_path)

    print(data.files)  # 会列出所有保存的数组名称

    """
        foldseek=foldseek_cumsum,
        tmalign=tmalign_cumsum,
        ssalign=ssalign_cumsum,
        ssalign_prefilter_1000=ssalign_prefilter_1000_cumsum,
        ssalign_prefilter_2000=ssalign_prefilter_2000_cumsum,
    """
    foldseek_cumsum = data['foldseek'][begin_points:min(52974, max_points)]
    tmalign_cumsum = data['tmalign'][begin_points:max_points]
    ssalign_cumsum = data['ssalign'][begin_points:max_points]
    ssalign_prefilter_1000_cumsum = data['ssalign_prefilter_1000'][begin_points:max_points]
    ssalign_prefilter_2000_cumsum = data['ssalign_prefilter_2000'][begin_points:max_points]

    print(len(foldseek_cumsum))
    print(len(tmalign_cumsum))
    print(len(ssalign_cumsum))
    print(len(ssalign_prefilter_1000_cumsum))
    print(len(ssalign_prefilter_2000_cumsum))

    # 创建x轴数据 (1到max_points)

    x_foldseek = np.arange(begin_points, begin_points + len(foldseek_cumsum))
    x =  np.arange(begin_points, begin_points + len(tmalign_cumsum))

    # 设置图形大小
    fig, ax = plt.subplots(figsize=(6,6),dpi=600)
    colors = {
        'Foldseek': '#5BB85B',
        'TM-align': '#EEA235',
        'SSAlign': '#6D3E99',
        'SSAlign-prefilter-1000': '#448DFC',
        'SSAlign-prefilter-2000': '#000000',
    }
    markers = {
        'Foldseek': 'o',  # 圆圈
        'TM-align': 's',  # 方形
        'SSAlign': '^',  # 上三角
        'SSAlign-prefilter-1000': 'D',  # 菱形
        'SSAlign-prefilter-2000': '*',  # 星号
    }

    markersize = 2 # 统一标记大小
    # 绘制折线（每隔 n 个点显示一个标记以避免重叠）
    marker_step = max(1, (max_points-begin_points) // 20)  # 动态调整标记密度
    for label, y_data in zip(
            ['Foldseek', 'TM-align', 'SSAlign', 'SSAlign-prefilter-1000', 'SSAlign-prefilter-2000'],
            [foldseek_cumsum, tmalign_cumsum, ssalign_cumsum, ssalign_prefilter_1000_cumsum,
             ssalign_prefilter_2000_cumsum]
    ):
        x_data = x_foldseek if label == 'Foldseek' else x
        ax.plot(
            x_data, y_data,
            label=label,
            color=colors[label],
            linestyle='-',
            solid_capstyle='round',
        )


    def thousands_formatter(x, pos):
        return f'{int(x / 1000)}' if x >= 1000 else str(int(x))

    ax.xaxis.set_major_formatter(FuncFormatter(thousands_formatter))
    ax.yaxis.set_major_formatter(FuncFormatter(thousands_formatter))

    # 标题和标签
    ax.set_title('sorted by Avg_TM-score', pad=12)
    ax.set_xlabel('top hits (×10$^3$)')
    ax.set_ylabel('Cumulative Av_TM-score (×10$^3$)')

    # 强制所有文本使用Times New Roman
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                 ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontname('Times New Roman')


    legend = ax.legend(
        fontsize=7,
        frameon=True,
        framealpha=0.8,
        loc='lower right',  # 根据数据分布调整位置
        handlelength=1.5,
    )
    for text in legend.get_texts():
        text.set_fontname('Times New Roman')

    # 坐标轴范围和刻度
    ax.set_xlim(begin_points, max_points)
    ax.set_ylim(32500, max(foldseek_cumsum.max(), tmalign_cumsum.max()) * 1.05)  # 留 5% 空白

    # 网格线（SciencePlots 默认虚线）
    plt.grid(True, linestyle=':', alpha=0.3)

    # 紧凑布局并保存
    plt.tight_layout()

    plt.savefig(f'../benchmarkData/SwissProt/benchmark/{data_file_path}_no_brokenaxes.png', dpi=600, bbox_inches='tight')
    plt.close()



def plot_cumsum_brokenaxes(data_file_path,begin_points,max_points):
    data = np.load(data_file_path)

    print(data.files)  # 会列出所有保存的数组名称

    # 访问各个数组

    """
        foldseek=foldseek_cumsum,
        tmalign=tmalign_cumsum,
        ssalign=ssalign_cumsum,
        ssalign_prefilter_1000=ssalign_prefilter_1000_cumsum,
        ssalign_prefilter_2000=ssalign_prefilter_2000_cumsum,
    """
    foldseek_cumsum = data['foldseek'][begin_points:min(52974, max_points)]
    tmalign_cumsum = data['tmalign'][begin_points:max_points]
    ssalign_cumsum = data['ssalign'][begin_points:max_points]
    ssalign_prefilter_1000_cumsum = data['ssalign_prefilter_1000'][begin_points:max_points]
    ssalign_prefilter_2000_cumsum = data['ssalign_prefilter_2000'][begin_points:max_points]


    # 创建x轴数据 (1到max_points)
    x_foldseek = np.arange(begin_points, begin_points + len(foldseek_cumsum))
    x =  np.arange(begin_points, begin_points + len(tmalign_cumsum))

    if __name__ == '__main__':
        max_score = max(max(foldseek_cumsum.max(), tmalign_cumsum.max()),ssalign_cumsum.max(),ssalign_prefilter_1000_cumsum.max())

    # 设置图形大小
    bax = brokenaxes(
        xlims = ((0, 1), (begin_points, max_points)),
        # ylims = ((0, 1), (19500, 46000)),  # 纵轴断裂 tmscore使用
        ylims = ((0, 1), (75000, 260000)),  # 纵轴断裂  RMSD使用
        hspace=0.1,  # 水平断裂间距
        wspace=0.1,  # 垂直断裂间距
        diag_color='k',  # 断裂线颜色
        # figsize=(6, 6)
    )

    colors = {
        'Foldseek': '#5BB85B',
        'TM-align': '#EEA235',
        'SSAlign': '#6D3E99',
        'SSAlign-prefilter-1000': '#448DFC',
        'SSAlign-prefilter-2000': '#000000',
    }

    for label, y_data in zip(
            ['Foldseek', 'TM-align', 'SSAlign', 'SSAlign-prefilter-1000', 'SSAlign-prefilter-2000'],
            [foldseek_cumsum, tmalign_cumsum, ssalign_cumsum, ssalign_prefilter_1000_cumsum,
             ssalign_prefilter_2000_cumsum]
    ):
        x_data = x_foldseek if label == 'Foldseek' else x
        bax.plot(
            x_data, y_data,
            label=label,
            color=colors[label],
            linestyle='-',
            solid_capstyle='round',
        )

    def thousands_formatter(x, pos):
        return f'{int(x / 1000)}' if x >= 1000 else str(int(x))



    match = re.search(r'cumsum_(.*?)_sorted_by_(.*?)\.npz', data_file_path)
    if match:
        score_measure = match.group(1)  # 'avg_TM-score'
        sort_mode = match.group(2)  # 'method_measure'
        print("得分指标:", score_measure)
        print("排序方式:", sort_mode)
    else:
        print("Pattern not matched!")



    print(sort_mode)
    print(score_measure)

    # 标题和标签
    bax.set_title(f'sorted by {sort_mode}', pad=12)
    bax.set_xlabel('top hits (×10$^3$)')
    bax.set_ylabel(f'Cumulative {score_measure} (×10$^3$)')

    # 强制所有文本使用Times New Roman
    for ax in bax.axs:
        ax.xaxis.set_major_formatter(FuncFormatter(thousands_formatter))
        ax.yaxis.set_major_formatter(FuncFormatter(thousands_formatter))
        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                     ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontname('Times New Roman')



    # 8. 保存和显示
    plt.savefig(f'../benchmarkData/SwissProt/benchmark/{data_file_path}_brokenaxes.png', dpi=600, bbox_inches='tight')
    plt.close()




"""
   用于 绘制 各种方法的差集累计得分图，即论文中Figure3
   为了图像清晰， begin_points,max_points 应该按照实际选择的 except_data_file_path 长度选择
   建议 begin_points = 0， max_points=6k 比较清晰
"""
def plot_except_cumsum_no_brokenaxes(except_data_file_path,begin_points,max_points):
    data = np.load(except_data_file_path)

    print(data.files)  # 会列出所有保存的数组名称
    """
    只有前10k，因为再多就不明显了
    ['ssalign_except_foldseek_cumsum', 'foldseek_except_ssalign_cumsum', 'ssalign_prefilter_1000_except_foldseek_cumsum', 'ssalign_prefilter_2000_except_foldseek_cumsum', 'foldseek_except_ssalign_prefilter_1000_cumsum', 'foldseek_except_ssalign_prefilter_2000_cumsum']
    10000
    4889
    10000
    10000
    5687
    3679
    """

    ssalign_except_foldseek_cumsum = data['ssalign_except_foldseek_cumsum'][begin_points:min(52974, max_points)]
    foldseek_except_ssalign_cumsum = data['foldseek_except_ssalign_cumsum'][begin_points:max_points]
    ssalign_prefilter_1000_except_foldseek_cumsum = data['ssalign_prefilter_1000_except_foldseek_cumsum'][begin_points:max_points]
    ssalign_prefilter_2000_except_foldseek_cumsum = data['ssalign_prefilter_2000_except_foldseek_cumsum'][begin_points:max_points]
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
    plt.savefig(f'../benchmarkData/SwissProt/benchmark/{except_data_file_path}_no_brokenaxes_except.png', dpi=600, bbox_inches='tight')
    plt.show()




if __name__=="__main__":

    data_file_path = "../benchmarkData/SwissProt/cumsumNpz/dim_1280__cumsum_avg_TM-score_sorted_by_avg_TM-score.npz"
    data_file_path = "../benchmarkData/SwissProt/cumsumNpz/dim_1280_cumsum_avg_TM-score_sorted_by_method_measure.npz"
    data_file_path = "../benchmarkData/SwissProt/cumsumNpz/dim_1280_cumsum_RMSD_sorted_by_avg_TM-score.npz"
    data_file_path = "../benchmarkData/SwissProt/cumsumNpz/dim_1280_cumsum_RMSD_sorted_by_method_measure.npz"

    plot_cumsum_brokenaxes(data_file_path,40000,55000)


    data_except_file_path = "../benchmarkData/SwissProt/cumsumNpz/dim_1280_except_cumsum_avg_TM-score_sorted_by_avg_TM-score.npz"
    data_except_file_path = "../benchmarkData/SwissProt/cumsumNpz/dim_1280_except_cumsum_avg_TM-score_sorted_by_method_measure.npz"
    data_except_file_path = "../benchmarkData/SwissProt/cumsumNpz/dim_1280_except_cumsum_RMSD_sorted_by_avg_TM-score.npz"
    data_except_file_path = "../benchmarkData/SwissProt/cumsumNpz/dim_1280_except_cumsum_RMSD_sorted_by_method_measure.npz"

    plot_except_cumsum_no_brokenaxes(data_except_file_path,0,6000)



