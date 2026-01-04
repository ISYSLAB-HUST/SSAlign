import numpy as np
import pandas as pd
import argparse

"""
为了计算 累计得分，先按照不同排序方式，生成npz，便于后续调用
"""


"""
dim,cos_threshold 维度和阈值
sort_mode,score_measure 排序方式和得分指标
max_points,point_step 绘图的个数和步长
"""


def benchmark_cumsum_score(dim, cos_threshold, sort_mode, score_measure, max_points):
    with open("100filenames.txt", "r") as f:
        basenames = [line.strip() for line in f if line.strip()]  # 去掉空行和多余空格

    """
    sort_mode 排序方式: 
        "avg_TM-score"
        "method_measure"

    score_measure  得分指标:
        "avg_TM-score"
        "RMSD"

    这100个文件，foldseek的结果个数 52974
    ssalign是  100*1000
    """

    # 这些就是 100 个测试文件，各个方法的结果  累计在一起
    merged_foldseek = pd.DataFrame()

    merged_tmalign = pd.DataFrame()

    merged_ssalign = pd.DataFrame()

    merged_ssalign_prefilter_1000 = pd.DataFrame()
    merged_ssalign_prefilter_2000 = pd.DataFrame()

    for basename in basenames:
        foldseek_file_path = f"../benchmarkData/SwissProt/foldseek/{basename}.result"

        tmalign_file_path = f"../benchmarkData/SwissProt/tmalign/{basename}.result"

        ssalign_file_path = f"../benchmarkData/SwissProt/SSAlign/SVD{dim}/ssalign/{basename}.result"

        ssalign_prefilter_file_path = f"../benchmarkData/SwissProt/SSAlign/SVD1280/ssalign_prefilter/{basename}.result"

        df_foldseek = pd.read_csv(foldseek_file_path)
        merged_foldseek = pd.concat([merged_foldseek, df_foldseek], ignore_index=True)

        df_tmalign = pd.read_csv(tmalign_file_path)
        # 只筛选平均分数 >= 0.5 的
        df_tmalign_filtered = df_tmalign[(df_tmalign['TM-Score1'] + df_tmalign['TM-Score2']) >= 1]
        merged_tmalign = pd.concat([merged_tmalign, df_tmalign_filtered], ignore_index=True)

        df_ssalign = pd.read_csv(ssalign_file_path)
        merged_ssalign = pd.concat([merged_ssalign, df_ssalign], ignore_index=True)

        df_ssalign_prefilter = pd.read_csv(ssalign_prefilter_file_path)
        df_ssalign_prefilter_sorted = df_ssalign_prefilter.sort_values(by='Cosine_Similarity', ascending=False)
        df_ssalign_prefilter_sorted['Avg_TM_Score'] = (df_ssalign_prefilter_sorted['TM-Score1'] +
                                                       df_ssalign_prefilter_sorted['TM-Score2']) / 2
        df_ssalign_prefilter_sorted_1000 = df_ssalign_prefilter_sorted.head(1000)
        df_ssalign_prefilter_sorted_2000 = df_ssalign_prefilter_sorted.head(2000)

        merged_ssalign_prefilter_1000 = pd.concat([merged_ssalign_prefilter_1000, df_ssalign_prefilter_sorted_1000],
                                                  ignore_index=True)
        merged_ssalign_prefilter_2000 = pd.concat([merged_ssalign_prefilter_2000, df_ssalign_prefilter_sorted_2000],
                                                  ignore_index=True)

    merged_foldseek['Avg_TM_Score'] = (merged_foldseek['TM-Score1'] + merged_foldseek['TM-Score2']) / 2
    merged_tmalign['Avg_TM_Score'] = (merged_tmalign['TM-Score1'] + merged_tmalign['TM-Score2']) / 2

    # 按照  avg_TM-score 排序
    if sort_mode == "avg_TM-score":
        merged_foldseek_sorted = merged_foldseek.sort_values(by='Avg_TM_Score', ascending=False)

        merged_tmalign_sorted = merged_tmalign.sort_values(by='Avg_TM_Score', ascending=False)

        merged_ssalign_sorted = merged_ssalign.sort_values(by='Avg_TM_Score', ascending=False)

        merged_ssalign_prefilter_1000_sorted = merged_ssalign_prefilter_1000.sort_values(by='Avg_TM_Score',
                                                                                         ascending=False)
        merged_ssalign_prefilter_2000_sorted = merged_ssalign_prefilter_2000.sort_values(by='Avg_TM_Score',
                                                                                         ascending=False)

    # 按照方法的指标排序 ： cosine E-value tmscore
    if sort_mode == "method_measure":
        # foldseek 按照 E-value排序
        merged_foldseek['E-value'] = merged_foldseek['FoldSeek_Metrics'].apply(
            lambda x: float(x.split()[8]))  # 取第9个元素（E-value），并且  float(x.split()[8]) 可以 把字符串 改为科学计数法形式

        merged_foldseek_sorted = merged_foldseek.sort_values(by='E-value', ascending=True)

        # tmalign还是按照得分排序
        merged_tmalign_sorted = merged_tmalign.sort_values(by='Avg_TM_Score', ascending=False)

        """
        ssalign，先按照 cosine，再按照 saligner排序
        1. 分数大于阈值的，按照cosine排序
        2. 分数小于阈值的，按照saligner排序
        最终汇总
        """
        merged_ssalign_part1 = merged_ssalign[merged_ssalign['Cosine_Similarity'] >= cos_threshold]
        merged_ssalign_part2 = merged_ssalign[merged_ssalign['Cosine_Similarity'] < cos_threshold]

        merged_ssalign_part1_sorted = merged_ssalign_part1.sort_values(by='Cosine_Similarity', ascending=False)
        merged_ssalign_part2_sorted = merged_ssalign_part2.sort_values(by='Score', ascending=False)

        merged_ssalign_sorted = pd.concat([merged_ssalign_part1_sorted, merged_ssalign_part2_sorted])

        # 预过滤阶段，按照 cosine排序
        merged_ssalign_prefilter_1000_sorted = merged_ssalign_prefilter_1000.sort_values(by='Cosine_Similarity',
                                                                                         ascending=False)
        merged_ssalign_prefilter_2000_sorted = merged_ssalign_prefilter_2000.sort_values(by='Cosine_Similarity',
                                                                                         ascending=False)

    foldseek_scores = []
    tmalign_scores = []
    ssalign_scores = []
    ssalign_prefilter_1000_scores = []
    ssalign_prefilter_2000_scores = []

    # 得分指标
    if score_measure == "avg_TM-score":
        foldseek_scores = merged_foldseek_sorted['Avg_TM_Score'].values

        tmalign_scores = merged_tmalign_sorted['Avg_TM_Score'].values

        ssalign_scores = merged_ssalign_sorted['Avg_TM_Score'].values

        ssalign_prefilter_1000_scores = merged_ssalign_prefilter_1000_sorted['Avg_TM_Score'].values
        ssalign_prefilter_2000_scores = merged_ssalign_prefilter_2000_sorted['Avg_TM_Score'].values

    if score_measure == "RMSD":
        foldseek_scores = merged_foldseek_sorted['RMSD'].values

        tmalign_scores = merged_tmalign_sorted['RMSD'].values

        ssalign_scores = merged_ssalign_sorted['RMSD'].values

        ssalign_prefilter_1000_scores = merged_ssalign_prefilter_1000_sorted['RMSD'].values
        ssalign_prefilter_2000_scores = merged_ssalign_prefilter_2000_sorted['RMSD'].values

    """

    计算前向累计和，并且绘图
    """

    # 对每个数组截取前max_points条（不足的保留全部）
    foldseek_scores = foldseek_scores[:max_points]
    tmalign_scores = tmalign_scores[:max_points]
    ssalign_scores = ssalign_scores[:max_points]
    ssalign_prefilter_1000_scores = ssalign_prefilter_1000_scores[:max_points]
    ssalign_prefilter_2000_scores = ssalign_prefilter_2000_scores[:max_points]

    # 计算累计和
    foldseek_cumsum = np.cumsum(foldseek_scores)
    tmalign_cumsum = np.cumsum(tmalign_scores)
    ssalign_cumsum = np.cumsum(ssalign_scores)
    ssalign_prefilter_1000_cumsum = np.cumsum(ssalign_prefilter_1000_scores)
    ssalign_prefilter_2000_cumsum = np.cumsum(ssalign_prefilter_2000_scores)

    np.savez(
        f"../benchmarkData/SwissProt/cumsumNpz/dim_{dim}_cumsum_{score_measure}_sorted_by_{sort_mode}.npz",
        foldseek=foldseek_cumsum,
        tmalign=tmalign_cumsum,
        ssalign=ssalign_cumsum,
        ssalign_prefilter_1000=ssalign_prefilter_1000_cumsum,
        ssalign_prefilter_2000=ssalign_prefilter_2000_cumsum,
    )


"""
各个方法检索结果的差集

"""


def benckmark_except_cumsum_score(dim, cos_threshold, sort_mode, score_measure, max_points):
    with open("100filenames.txt", "r") as f:
        basenames = [line.strip() for line in f if line.strip()]  # 去掉空行和多余空格

    merged_ssalign_except_foldseek = pd.DataFrame()  # ssalign有，foldseek无

    merged_foldseek_except_ssalign = pd.DataFrame()  # foldseek有，ssalign无

    # ssalign_prefilter 有，foldseek无
    merged_ssalign_prefilter_1000_except_foldseek = pd.DataFrame()
    merged_ssalign_prefilter_2000_except_foldseek = pd.DataFrame()

    # foldseek有，ssalign_prefilter_1000无
    merged_foldseek_except_ssalign_prefilter_1000 = pd.DataFrame()
    merged_foldseek_except_ssalign_prefilter_2000 = pd.DataFrame()

    for basename in basenames:
        foldseek_file_path = f"../benchmarkData/SwissProt/foldseek/{basename}.result"

        tmalign_file_path = f"../benchmarkData/SwissProt/tmalign/{basename}.result"

        ssalign_file_path = f"../benchmarkData/SwissProt/SSAlign/SVD{dim}/ssalign/{basename}.result"

        ssalign_prefilter_file_path = f"../benchmarkData/SwissProt/SSAlign/SVD1280/ssalign_prefilter/{basename}.result"

        df_foldseek = pd.read_csv(foldseek_file_path)

        df_tmalign = pd.read_csv(tmalign_file_path)
        # 只筛选平均分数 >= 0.5 的
        df_tmalign_filtered = df_tmalign[(df_tmalign['TM-Score1'] + df_tmalign['TM-Score2']) >= 1]

        df_ssalign = pd.read_csv(ssalign_file_path)

        df_ssalign_prefilter = pd.read_csv(ssalign_prefilter_file_path)
        df_ssalign_prefilter_sorted = df_ssalign_prefilter.sort_values(by='Cosine_Similarity', ascending=False)
        df_ssalign_prefilter_sorted['Avg_TM_Score'] = (df_ssalign_prefilter_sorted['TM-Score1'] +
                                                       df_ssalign_prefilter_sorted['TM-Score2']) / 2
        df_ssalign_prefilter_sorted_1000 = df_ssalign_prefilter_sorted.head(1000)
        df_ssalign_prefilter_sorted_2000 = df_ssalign_prefilter_sorted.head(2000)

        # 下面就提取差集
        ssalign_except_foldseek = df_ssalign[~df_ssalign['File1'].isin(df_foldseek['File1'])]
        foldseek_except_ssalign = df_foldseek[~df_foldseek['File1'].isin(df_ssalign['File1'])]

        ssalign_prefilter_1000_except_foldseek = df_ssalign_prefilter_sorted_1000[
            ~df_ssalign_prefilter_sorted_1000['File1'].isin(df_foldseek['File1'])]
        ssalign_prefilter_2000_except_foldseek = df_ssalign_prefilter_sorted_2000[
            ~df_ssalign_prefilter_sorted_2000['File1'].isin(df_foldseek['File1'])]

        # foldseek有，ssalign_prefilter_1000无
        foldseek_except_ssalign_prefilter_1000 = df_foldseek[
            ~df_foldseek['File1'].isin(df_ssalign_prefilter_sorted_1000['File1'])]
        foldseek_except_ssalign_prefilter_2000 = df_foldseek[
            ~df_foldseek['File1'].isin(df_ssalign_prefilter_sorted_2000['File1'])]

        # 合并
        merged_ssalign_except_foldseek = pd.concat([merged_ssalign_except_foldseek, ssalign_except_foldseek],
                                                   ignore_index=True)
        merged_foldseek_except_ssalign = pd.concat([merged_foldseek_except_ssalign, foldseek_except_ssalign],
                                                   ignore_index=True)

        merged_ssalign_prefilter_1000_except_foldseek = pd.concat(
            [merged_ssalign_prefilter_1000_except_foldseek, ssalign_prefilter_1000_except_foldseek], ignore_index=True)
        merged_ssalign_prefilter_2000_except_foldseek = pd.concat(
            [merged_ssalign_prefilter_2000_except_foldseek, ssalign_prefilter_2000_except_foldseek], ignore_index=True)

        merged_foldseek_except_ssalign_prefilter_1000 = pd.concat(
            [merged_foldseek_except_ssalign_prefilter_1000, foldseek_except_ssalign_prefilter_1000], ignore_index=True)
        merged_foldseek_except_ssalign_prefilter_2000 = pd.concat(
            [merged_foldseek_except_ssalign_prefilter_2000, foldseek_except_ssalign_prefilter_2000], ignore_index=True)

    merged_foldseek_except_ssalign['Avg_TM_Score'] = (merged_foldseek_except_ssalign['TM-Score1'] +
                                                      merged_foldseek_except_ssalign['TM-Score2']) / 2
    merged_foldseek_except_ssalign_prefilter_1000['Avg_TM_Score'] = (merged_foldseek_except_ssalign_prefilter_1000[
                                                                         'TM-Score1'] +
                                                                     merged_foldseek_except_ssalign_prefilter_1000[
                                                                         'TM-Score2']) / 2
    merged_foldseek_except_ssalign_prefilter_2000['Avg_TM_Score'] = (merged_foldseek_except_ssalign_prefilter_2000[
                                                                         'TM-Score1'] +
                                                                     merged_foldseek_except_ssalign_prefilter_2000[
                                                                         'TM-Score2']) / 2

    # 排序
    if sort_mode == "avg_TM-score":
        merged_ssalign_except_foldseek_sorted = merged_ssalign_except_foldseek.sort_values(by='Avg_TM_Score',
                                                                                           ascending=False)
        merged_foldseek_except_ssalign_sorted = merged_foldseek_except_ssalign.sort_values(by='Avg_TM_Score',
                                                                                           ascending=False)

        merged_ssalign_prefilter_1000_except_foldseek_sorted = merged_ssalign_prefilter_1000_except_foldseek.sort_values(
            by='Avg_TM_Score', ascending=False)
        merged_ssalign_prefilter_2000_except_foldseek_sorted = merged_ssalign_prefilter_2000_except_foldseek.sort_values(
            by='Avg_TM_Score', ascending=False)

        merged_foldseek_except_ssalign_prefilter_1000_sorted = merged_foldseek_except_ssalign_prefilter_1000.sort_values(
            by='Avg_TM_Score', ascending=False)
        merged_foldseek_except_ssalign_prefilter_2000_sorted = merged_foldseek_except_ssalign_prefilter_2000.sort_values(
            by='Avg_TM_Score', ascending=False)

    if sort_mode == "method_measure":
        """
            ssalign，先按照 cosine，再按照 saligner排序
            1. 分数大于阈值的，按照cosine排序
            2. 分数小于阈值的，按照saligner排序
            3. 最终汇总
        """
        merged_ssalign_except_foldseek_part1 = merged_ssalign_except_foldseek[
            merged_ssalign_except_foldseek['Cosine_Similarity'] >= cos_threshold]
        merged_ssalign_except_foldseek_part2 = merged_ssalign_except_foldseek[
            merged_ssalign_except_foldseek['Cosine_Similarity'] < cos_threshold]
        merged_ssalign_except_foldseek_part1_sorted = merged_ssalign_except_foldseek_part1.sort_values(
            by='Cosine_Similarity', ascending=False)
        merged_ssalign_except_foldseek_part2_sorted = merged_ssalign_except_foldseek_part2.sort_values(by='Score',
                                                                                                       ascending=False)

        merged_ssalign_except_foldseek_sorted = pd.concat(
            [merged_ssalign_except_foldseek_part1_sorted, merged_ssalign_except_foldseek_part2_sorted])

        # foldseek 按照 E-value排序
        merged_foldseek_except_ssalign['E-value'] = merged_foldseek_except_ssalign['FoldSeek_Metrics'].apply(
            lambda x: float(x.split()[8]))  # 取第9个元素（E-value），并且  float(x.split()[8]) 可以 把字符串 改为科学计数法形式
        merged_foldseek_except_ssalign_sorted = merged_foldseek_except_ssalign.sort_values(by='E-value', ascending=True)

        # ssalign-prefilter直接按照cosine排序即可
        merged_ssalign_prefilter_1000_except_foldseek_sorted = merged_ssalign_prefilter_1000_except_foldseek.sort_values(
            by='Avg_TM_Score', ascending=False)
        merged_ssalign_prefilter_2000_except_foldseek_sorted = merged_ssalign_prefilter_2000_except_foldseek.sort_values(
            by='Avg_TM_Score', ascending=False)

        # foldseek 按照 E-value排序
        merged_foldseek_except_ssalign_prefilter_1000['E-value'] = merged_foldseek_except_ssalign_prefilter_1000[
            'FoldSeek_Metrics'].apply(
            lambda x: float(x.split()[8]))  # 取第9个元素（E-value），并且  float(x.split()[8]) 可以 把字符串 改为科学计数法形式
        merged_foldseek_except_ssalign_prefilter_1000_sorted = merged_foldseek_except_ssalign_prefilter_1000.sort_values(
            by='E-value', ascending=True)

        merged_foldseek_except_ssalign_prefilter_2000['E-value'] = merged_foldseek_except_ssalign_prefilter_2000[
            'FoldSeek_Metrics'].apply(
            lambda x: float(x.split()[8]))  # 取第9个元素（E-value），并且  float(x.split()[8]) 可以 把字符串 改为科学计数法形式
        merged_foldseek_except_ssalign_prefilter_2000_sorted = merged_foldseek_except_ssalign_prefilter_2000.sort_values(
            by='E-value', ascending=True)

    ssalign_except_foldseek_scores = []
    foldseek_except_ssalign_scores = []

    ssalign_prefilter_1000_except_foldseek_scores = []
    ssalign_prefilter_2000_except_foldseek_scores = []

    foldseek_except_ssalign_prefilter_1000_scores = []
    foldseek_except_ssalign_prefilter_2000_scores = []

    if score_measure == "avg_TM-score":
        ssalign_except_foldseek_scores = merged_ssalign_except_foldseek_sorted['Avg_TM_Score'].values
        foldseek_except_ssalign_scores = merged_foldseek_except_ssalign_sorted['Avg_TM_Score'].values

        ssalign_prefilter_1000_except_foldseek_scores = merged_ssalign_prefilter_1000_except_foldseek_sorted[
            'Avg_TM_Score'].values
        ssalign_prefilter_2000_except_foldseek_scores = merged_ssalign_prefilter_2000_except_foldseek_sorted[
            'Avg_TM_Score'].values

        foldseek_except_ssalign_prefilter_1000_scores = merged_foldseek_except_ssalign_prefilter_1000_sorted[
            'Avg_TM_Score'].values
        foldseek_except_ssalign_prefilter_2000_scores = merged_foldseek_except_ssalign_prefilter_2000_sorted[
            'Avg_TM_Score'].values

    if score_measure == "RMSD":
        ssalign_except_foldseek_scores = merged_ssalign_except_foldseek_sorted['RMSD'].values
        foldseek_except_ssalign_scores = merged_foldseek_except_ssalign_sorted['RMSD'].values

        ssalign_prefilter_1000_except_foldseek_scores = merged_ssalign_prefilter_1000_except_foldseek_sorted[
            'RMSD'].values
        ssalign_prefilter_2000_except_foldseek_scores = merged_ssalign_prefilter_2000_except_foldseek_sorted[
            'RMSD'].values

        foldseek_except_ssalign_prefilter_1000_scores = merged_foldseek_except_ssalign_prefilter_1000_sorted[
            'RMSD'].values
        foldseek_except_ssalign_prefilter_2000_scores = merged_foldseek_except_ssalign_prefilter_2000_sorted[
            'RMSD'].values

    """
        计算前向累计和，并且绘图
    """
    # 对每个数组截取前max_points条（不足的保留全部）
    ssalign_except_foldseek_scores = ssalign_except_foldseek_scores[:max_points]
    foldseek_except_ssalign_scores = foldseek_except_ssalign_scores[:max_points]
    ssalign_prefilter_1000_except_foldseek_scores = ssalign_prefilter_1000_except_foldseek_scores[:max_points]
    ssalign_prefilter_2000_except_foldseek_scores = ssalign_prefilter_2000_except_foldseek_scores[:max_points]
    foldseek_except_ssalign_prefilter_1000_scores = foldseek_except_ssalign_prefilter_1000_scores[:max_points]
    foldseek_except_ssalign_prefilter_2000_scores = foldseek_except_ssalign_prefilter_2000_scores[:max_points]

    # 计算累计和
    ssalign_except_foldseek_cumsum = np.cumsum(ssalign_except_foldseek_scores)
    foldseek_except_ssalign_cumsum = np.cumsum(foldseek_except_ssalign_scores)
    ssalign_prefilter_1000_except_foldseek_cumsum = np.cumsum(ssalign_prefilter_1000_except_foldseek_scores)
    ssalign_prefilter_2000_except_foldseek_cumsum = np.cumsum(ssalign_prefilter_2000_except_foldseek_scores)
    foldseek_except_ssalign_prefilter_1000_cumsum = np.cumsum(foldseek_except_ssalign_prefilter_1000_scores)
    foldseek_except_ssalign_prefilter_2000_cumsum = np.cumsum(foldseek_except_ssalign_prefilter_2000_scores)

    np.savez(
        f"../benchmarkData/SwissProt/cumsumNpz/dim_{dim}_except_cumsum_{score_measure}_sorted_by_{sort_mode}.npz",
        ssalign_except_foldseek_cumsum=ssalign_except_foldseek_cumsum,
        foldseek_except_ssalign_cumsum=foldseek_except_ssalign_cumsum,
        ssalign_prefilter_1000_except_foldseek_cumsum=ssalign_prefilter_1000_except_foldseek_cumsum,
        ssalign_prefilter_2000_except_foldseek_cumsum=ssalign_prefilter_2000_except_foldseek_cumsum,
        foldseek_except_ssalign_prefilter_1000_cumsum=foldseek_except_ssalign_prefilter_1000_cumsum,
        foldseek_except_ssalign_prefilter_2000_cumsum=foldseek_except_ssalign_prefilter_2000_cumsum,
    )


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="生成基准测试图")

    parser.add_argument("--except_mode", type=int, required=True)

    parser.add_argument("--dim", type=int, required=True)
    parser.add_argument("--cosine", type=float, required=True)

    parser.add_argument("--sort_mode", type=str, required=True,
                        choices=["avg_TM-score", "method_measure"],
                        help="排序方式: 'avg_TM-score' 或 'method_measure'")
    parser.add_argument("--score_measure", type=str, required=True,
                        choices=["avg_TM-score", "RMSD"],
                        help="得分指标: 'avg_TM-score' 或 'RMSD'")
    parser.add_argument("--max_points", type=int, default=200000,
                        help="最大数据点数")

    return parser.parse_args()


def main():
    args = parse_args()

    if args.except_mode == 0:  # 累计得分
        # 调用你的绘图函数
        benchmark_cumsum_score(args.dim, args.cosine,
                             args.sort_mode,
                             args.score_measure,
                             args.max_points,
                             )
    else:  # 差集得分
        benckmark_except_cumsum_score(args.dim, args.cosine,
                                    args.sort_mode,
                                    args.score_measure,
                                    args.max_points,
                                    )


if __name__ == "__main__":
    """
        sort_mode 排序方式: 
            "avg_TM-score"
            "method_measure"

        score_measure  得分指标:
            "avg_TM-score"
            "RMSD"

        这100个文件，foldseek的结果个数 52974
        ssalign是  100*1000
        """
    main()

    """
    for example, you can run :
        --max_points 200000 （是因为除了TM-Align的检索结果个数可能大于200000,其他都不会）
    
    
        # 1280维度
        nohup python foldseek_benchmark_cumsum_score.py --except_mode 0 --dim 1280 --cosine 0.2 --sort_mode avg_TM-score --score_measure avg_TM-score --max_points 200000 &
        nohup python foldseek_benchmark_cumsum_score.py --except_mode 0 --dim 1280 --cosine 0.2 --sort_mode avg_TM-score --score_measure RMSD --max_points 200000 &
        
        nohup python foldseek_benchmark_cumsum_score.py --except_mode 0 --dim 1280 --cosine 0.2 --sort_mode method_measure --score_measure avg_TM-score --max_points 200000 &
        nohup python foldseek_benchmark_cumsum_score.py --except_mode 0 --dim 1280 --cosine 0.2 --sort_mode method_measure --score_measure RMSD --max_points 200000 &
        
        nohup python foldseek_benchmark_cumsum_score.py --except_mode 1 --dim 1280 --cosine 0.2 --sort_mode avg_TM-score --score_measure avg_TM-score --max_points 10000 &
        nohup python foldseek_benchmark_cumsum_score.py --except_mode 1 --dim 1280 --cosine 0.2 --sort_mode avg_TM-score --score_measure RMSD --max_points 10000 &
        
        nohup python foldseek_benchmark_cumsum_score.py --except_mode 1 --dim 1280 --cosine 0.2 --sort_mode method_measure --score_measure avg_TM-score --max_points 10000 &
        nohup python foldseek_benchmark_cumsum_score.py --except_mode 1 --dim 1280 --cosine 0.2 --sort_mode method_measure --score_measure RMSD --max_points 10000 &
        
    
        # 512维度
        nohup python foldseek_benchmark_cumsum_score.py --except_mode 0 --dim 512 --cosine 0.3 --sort_mode avg_TM-score --score_measure avg_TM-score --max_points 200000 &
        nohup python foldseek_benchmark_cumsum_score.py --except_mode 0 --dim 512 --cosine 0.3 --sort_mode avg_TM-score --score_measure RMSD --max_points 200000 &
        
        nohup python foldseek_benchmark_cumsum_score.py --except_mode 0 --dim 512 --cosine 0.3 --sort_mode method_measure --score_measure avg_TM-score --max_points 200000 &
        nohup python foldseek_benchmark_cumsum_score.py --except_mode 0 --dim 512 --cosine 0.3 --sort_mode method_measure --score_measure RMSD --max_points 200000 &
        
        nohup python foldseek_benchmark_cumsum_score.py --except_mode 1 --dim 512 --cosine 0.3 --sort_mode avg_TM-score --score_measure avg_TM-score --max_points 10000 &
        nohup python foldseek_benchmark_cumsum_score.py --except_mode 1 --dim 512 --cosine 0.3 --sort_mode avg_TM-score --score_measure RMSD --max_points 10000 &
        
        nohup python foldseek_benchmark_cumsum_score.py --except_mode 1 --dim 512 --cosine 0.3 --sort_mode method_measure --score_measure avg_TM-score --max_points 10000 &
        nohup python foldseek_benchmark_cumsum_score.py --except_mode 1 --dim 512 --cosine 0.3 --sort_mode method_measure --score_measure RMSD --max_points 10000 &
        
        
        # 256维度
        nohup python foldseek_benchmark_cumsum_score.py --except_mode 0 --dim 256 --cosine 0.45 --sort_mode avg_TM-score --score_measure avg_TM-score --max_points 200000 &
        nohup python foldseek_benchmark_cumsum_score.py --except_mode 0 --dim 256 --cosine 0.45 --sort_mode avg_TM-score --score_measure RMSD --max_points 200000 &
        
        nohup python foldseek_benchmark_cumsum_score.py --except_mode 0 --dim 256 --cosine 0.45 --sort_mode method_measure --score_measure avg_TM-score --max_points 200000 &
        nohup python foldseek_benchmark_cumsum_score.py --except_mode 0 --dim 256 --cosine 0.45 --sort_mode method_measure --score_measure RMSD --max_points 200000 &
        
        nohup python foldseek_benchmark_cumsum_score.py --except_mode 1 --dim 256 --cosine 0.45 --sort_mode avg_TM-score --score_measure avg_TM-score --max_points 10000 &
        nohup python foldseek_benchmark_cumsum_score.py --except_mode 1 --dim 256 --cosine 0.45 --sort_mode avg_TM-score --score_measure RMSD --max_points 10000 &
        
        nohup python foldseek_benchmark_cumsum_score.py --except_mode 1 --dim 256 --cosine 0.45 --sort_mode method_measure --score_measure avg_TM-score --max_points 10000 &
        nohup python foldseek_benchmark_cumsum_score.py --except_mode 1 --dim 256 --cosine 0.45 --sort_mode method_measure --score_measure RMSD --max_points 10000 &
        
        
        
        # 128维度
        nohup python foldseek_benchmark_cumsum_score.py --except_mode 0 --dim 128 --cosine 0.6 --sort_mode avg_TM-score --score_measure avg_TM-score --max_points 200000 &
        nohup python foldseek_benchmark_cumsum_score.py --except_mode 0 --dim 128 --cosine 0.6 --sort_mode avg_TM-score --score_measure RMSD --max_points 200000 &
        
        nohup python foldseek_benchmark_cumsum_score.py --except_mode 0 --dim 128 --cosine 0.6 --sort_mode method_measure --score_measure avg_TM-score --max_points 200000 &
        nohup python foldseek_benchmark_cumsum_score.py --except_mode 0 --dim 128 --cosine 0.6 --sort_mode method_measure --score_measure RMSD --max_points 200000 &
        
        nohup python foldseek_benchmark_cumsum_score.py --except_mode 1 --dim 128 --cosine 0.6 --sort_mode avg_TM-score --score_measure avg_TM-score --max_points 10000 &
        nohup python foldseek_benchmark_cumsum_score.py --except_mode 1 --dim 128 --cosine 0.6 --sort_mode avg_TM-score --score_measure RMSD --max_points 10000 &
        
        nohup python foldseek_benchmark_cumsum_score.py --except_mode 1 --dim 128 --cosine 0.6 --sort_mode method_measure --score_measure avg_TM-score --max_points 10000 &
        nohup python foldseek_benchmark_cumsum_score.py --except_mode 1 --dim 128 --cosine 0.6 --sort_mode method_measure --score_measure RMSD --max_points 10000 &
        
        
        
        # 64维度
        nohup python foldseek_benchmark_cumsum_score.py --except_mode 0 --dim 64 --cosine 0.7 --sort_mode avg_TM-score --score_measure avg_TM-score --max_points 200000 &
        nohup python foldseek_benchmark_cumsum_score.py --except_mode 0 --dim 64 --cosine 0.7 --sort_mode avg_TM-score --score_measure RMSD --max_points 200000 &
        
        nohup python foldseek_benchmark_cumsum_score.py --except_mode 0 --dim 64 --cosine 0.7 --sort_mode method_measure --score_measure avg_TM-score --max_points 200000 &
        nohup python foldseek_benchmark_cumsum_score.py --except_mode 0 --dim 64 --cosine 0.7 --sort_mode method_measure --score_measure RMSD --max_points 200000 &
        
        nohup python foldseek_benchmark_cumsum_score.py --except_mode 1 --dim 64 --cosine 0.7 --sort_mode avg_TM-score --score_measure avg_TM-score --max_points 10000 &
        nohup python foldseek_benchmark_cumsum_score.py --except_mode 1 --dim 64 --cosine 0.7 --sort_mode avg_TM-score --score_measure RMSD --max_points 10000 &
        
        nohup python foldseek_benchmark_cumsum_score.py --except_mode 1 --dim 64 --cosine 0.7 --sort_mode method_measure --score_measure avg_TM-score --max_points 10000 &
        nohup python foldseek_benchmark_cumsum_score.py --except_mode 1 --dim 64 --cosine 0.7 --sort_mode method_measure --score_measure RMSD --max_points 10000 &
    """










