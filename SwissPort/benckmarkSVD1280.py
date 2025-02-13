import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline

"""
下面仅仅对1280维度的结果做测试

"""
def overlap_faiss_foldseek(basenames,topk):


    results = []

    for basename in basenames:
        faiss_file_path = f"../data/result/Swissport/SSAlign/SVD1280/{basename}.result"
        foldseek_file_path = f"../data/result/Swissport/foldseek/{basename}.result"
        tmalign_file_path = f"../data/result/Swissport/tmalign/{basename}.result"

        # faiss 结果
        df_faiss = pd.read_csv(faiss_file_path)
        df_faiss_sorted_topk = df_faiss.sort_values(by='Cosine_Similarity', ascending=False).head(topk)
        faiss_set = set(df_faiss_sorted_topk['File1'].tolist())

        # foldseek结果
        df_foldseek = pd.read_csv(foldseek_file_path)
        foldseek_set = set(df_foldseek['File1'].tolist())

        # tmalign结果
        df_tmalign = pd.read_csv(tmalign_file_path)
        df_tmalign['Avg_TM_Score'] = (df_tmalign['TM-Score1']+df_tmalign['TM-Score2'])/2
        df_tmalign_filtered = df_tmalign[df_tmalign['Avg_TM_Score']>=0.5]
        tmalign_set = set(df_tmalign_filtered['File1'].tolist())


        common_set_faiss_foldseek = faiss_set.intersection(foldseek_set)

        common_set_faiss_tmalign = faiss_set.intersection(tmalign_set)

        common_set_foldseek_tmalign = foldseek_set.intersection(tmalign_set)




        # 将每个文件的结果记录到 results 列表中
        results.append({
            'basename': basename,
            'faiss_count': len(faiss_set),
            'foldseek_count': len(foldseek_set),
            'tmalign_count': len(tmalign_set),
            'common_set_faiss_foldseek_count': len(common_set_faiss_foldseek),
            'common_set_faiss_tmalign_count': len(common_set_faiss_tmalign),
            'common_set_foldseek_tmalign': len(common_set_foldseek_tmalign),
        })




    # 创建 DataFrame
    df_results = pd.DataFrame(results)

    save_path = f"../data/result/Swissport/SSAlign/SVD1280/benckmark/overlap_faiss{topk}_foldseek.csv"

    # 将结果保存到 CSV 文件
    df_results.to_csv(save_path , index=False)




"""
ssalign(2000+0.2+1000)的结果和foldseek的重合率

foldseek搜索的总数为：52974
commenset的总数为：48085
foldseek搜索的总数为：92.77250685554547

"""
def overlap_ssalign_foldseek(basenames):


    sim = 0
    results = []

    for basename in basenames:
        ssalign_file_path = f"../data/result/Swissport/SSAlign/SVD1280/ssalign/{basename}.result"
        foldseek_file_path = f"../data/result/Swissport/foldseek/{basename}.result"
        tmalign_file_path = f"../data/result/Swissport/tmalign/{basename}.result"

        # faiss结果
        df_ssalign = pd.read_csv(ssalign_file_path)
        ssalign_set = set(df_ssalign['File1'].tolist())

        # foldseek 结果
        df_foldseek = pd.read_csv(foldseek_file_path)
        foldseek_set = set(df_foldseek['File1'].tolist())

        # tmalign结果
        df_tmalign = pd.read_csv(tmalign_file_path)
        df_tmalign['Avg_TM_Score'] = (df_tmalign['TM-Score1'] + df_tmalign['TM-Score2'])/2
        df_tmalign_filtered = df_tmalign[df_tmalign['Avg_TM_Score'] >= 0.5]
        tmalign_set = set(df_tmalign_filtered['File1'].tolist())



        common_set_ssalign_foldseek = ssalign_set.intersection(foldseek_set)

        common_set_ssalign_tmalign = ssalign_set.intersection(tmalign_set)

        common_set_foldseek_tmalign = foldseek_set.intersection(tmalign_set)

        # 将每个文件的结果记录到 results 列表中
        results.append({
            'basename': basename,
            'ssalign_count': len(ssalign_set),
            'foldseek_count': len(foldseek_set),
            'tmalign_count': len(tmalign_set),
            'common_set_ssalign_foldseek_count': len(common_set_ssalign_foldseek),
            'common_set_ssalign_tmalign_count': len(common_set_ssalign_tmalign),
            'common_set_foldseek_tmalign': len(common_set_foldseek_tmalign),
        })




    save_path = f"../data/result/Swissport/SSAlign/SVD1280/benckmark/overlap_ssalign_foldseek.csv"
    df_results = pd.DataFrame(results)
    # 将结果保存到 CSV 文件
    df_results.to_csv(save_path, index=False)




"""
ssalign 和 foldseek的差集
差集的好坏

mode 排序方式  
    "avgtmscore":
    "method_measure":
    
measure  指标：
    "avgtmscore":
    "rmsd":
    
"""
def ssalign_more_foldseek(basenames,end,step,mode,measure):

    # 收集两个差集
    # ssalign有，foldseek无
    all_only_ssalign = pd.DataFrame()
    # foldseek有，ssalign无
    all_only_foldseek = pd.DataFrame()

    # faiss 有，foldseek无
    all_only_fiass1000 = pd.DataFrame()
    all_only_fiass2000 = pd.DataFrame()
    all_only_fiass4000 = pd.DataFrame()
    all_only_fiass8000 = pd.DataFrame()

    # foldseek有，faiss无
    all_only_foldseek_faiss1000 = pd.DataFrame()
    all_only_foldseek_faiss2000 = pd.DataFrame()
    all_only_foldseek_faiss4000 = pd.DataFrame()
    all_only_foldseek_faiss8000 = pd.DataFrame()

    for basename in basenames:
        ssalign_file_path = f"../data/result/Swissport/SSAlign/SVD1280/ssalign/{basename}.result"
        foldseek_file_path = f"../data/result/Swissport/foldseek/{basename}.result"
        # faiiss 的路径 ，注意只有 TM-Score1,TM-Score2,
        faiss_file_path = f"../data/result/Swissport/SSAlign/SVD1280/{basename}.result"

        # ssalign结果
        df_ssalign = pd.read_csv(ssalign_file_path)

        # foldseek 结果
        df_foldseek = pd.read_csv(foldseek_file_path)
        df_foldseek['Avg_TM_Score'] = (df_foldseek['TM-Score1'] + df_foldseek['TM-Score2'] )/2

        # faiss结果
        df_faiss = pd.read_csv(faiss_file_path)
        df_faiss_sorted = df_faiss.sort_values(by='Cosine_Similarity',ascending=False)
        df_faiss_sorted['Avg_TM_Score'] = (df_faiss_sorted['TM-Score1'] + df_faiss_sorted['TM-Score2'] )/ 2

        df_faiss_1000 = df_faiss_sorted.head(1000)
        df_faiss_2000 = df_faiss_sorted.head(2000)
        df_faiss_4000 = df_faiss_sorted.head(4000)
        df_faiss_8000 = df_faiss_sorted.head(8000)


        # 按照 File1 提取差集的列
        only_ssalign = df_ssalign[~df_ssalign['File1'].isin(df_foldseek['File1'])]
        only_foldseek = df_foldseek[~df_foldseek['File1'].isin(df_ssalign['File1'])]

        only_fiass1000 = df_faiss_1000[~df_faiss_1000['File1'].isin(df_foldseek['File1'])]
        only_fiass2000 = df_faiss_2000[~df_faiss_2000['File1'].isin(df_foldseek['File1'])]
        only_fiass4000 = df_faiss_4000[~df_faiss_4000['File1'].isin(df_foldseek['File1'])]
        only_fiass8000 = df_faiss_8000[~df_faiss_8000['File1'].isin(df_foldseek['File1'])]

        only_foldseek_faiss1000 = df_foldseek[~df_foldseek['File1'].isin(df_faiss_1000['File1'])]
        only_foldseek_faiss2000 = df_foldseek[~df_foldseek['File1'].isin(df_faiss_2000['File1'])]
        only_foldseek_faiss4000 = df_foldseek[~df_foldseek['File1'].isin(df_faiss_4000['File1'])]
        only_foldseek_faiss8000 = df_foldseek[~df_foldseek['File1'].isin(df_faiss_8000['File1'])]



        # 合并到总结果
        all_only_ssalign = pd.concat([all_only_ssalign, only_ssalign], ignore_index=True)
        all_only_foldseek = pd.concat([all_only_foldseek, only_foldseek], ignore_index=True)

        all_only_fiass1000 = pd.concat([all_only_fiass1000,only_fiass1000], ignore_index=True)
        all_only_fiass2000 = pd.concat([all_only_fiass2000,only_fiass2000], ignore_index=True)
        all_only_fiass4000 = pd.concat([all_only_fiass4000,only_fiass4000], ignore_index=True)
        all_only_fiass8000 = pd.concat([all_only_fiass8000,only_fiass8000], ignore_index=True)

        all_only_foldseek_faiss1000 = pd.concat([all_only_foldseek_faiss1000,only_foldseek_faiss1000],ignore_index=True)
        all_only_foldseek_faiss2000 = pd.concat([all_only_foldseek_faiss2000,only_foldseek_faiss2000],ignore_index=True)
        all_only_foldseek_faiss4000 = pd.concat([all_only_foldseek_faiss4000,only_foldseek_faiss4000],ignore_index=True)
        all_only_foldseek_faiss8000 = pd.concat([all_only_foldseek_faiss8000,only_foldseek_faiss8000],ignore_index=True)




    if mode == "avgtmscore":
        # 排序 ，就按照 tmscore排吧
        all_only_ssalign_sorted = all_only_ssalign.sort_values(by='Avg_TM_Score', ascending=False)
        all_only_foldseek_sorted = all_only_foldseek.sort_values(by='Avg_TM_Score', ascending=False)
        all_only_fiass1000_sorted = all_only_fiass1000.sort_values(by='Avg_TM_Score', ascending=False)
        all_only_fiass2000_sorted = all_only_fiass2000.sort_values(by='Avg_TM_Score', ascending=False)
        all_only_fiass4000_sorted = all_only_fiass4000.sort_values(by='Avg_TM_Score', ascending=False)
        all_only_fiass8000_sorted = all_only_fiass8000.sort_values(by='Avg_TM_Score', ascending=False)

        all_only_foldseek_faiss1000_sorted = all_only_foldseek_faiss1000.sort_values(by='Avg_TM_Score', ascending=False)
        all_only_foldseek_faiss2000_sorted = all_only_foldseek_faiss2000.sort_values(by='Avg_TM_Score', ascending=False)
        all_only_foldseek_faiss4000_sorted = all_only_foldseek_faiss4000.sort_values(by='Avg_TM_Score', ascending=False)
        all_only_foldseek_faiss8000_sorted = all_only_foldseek_faiss8000.sort_values(by='Avg_TM_Score', ascending=False)


    if mode == "method_measure":

        # 如果按照自己的指标来排序呢
        all_only_ssalign_none = all_only_ssalign[all_only_ssalign['Score'].isna()]
        all_only_ssalign_sorted_none = all_only_ssalign_none.sort_values(by='Cosine_Similarity',ascending=False) # 降序排列

        all_only_ssalign_not_none = all_only_ssalign[all_only_ssalign['Score'].notna()]
        all_only_ssalign_sorted_not_none = all_only_ssalign_not_none.sort_values(by='Score',ascending=False)

        all_only_ssalign_sorted = pd.concat([all_only_ssalign_sorted_none,all_only_ssalign_sorted_not_none])

        """"""
        all_only_foldseek['E-value'] = all_only_foldseek['FoldSeek_Metrics'].apply(
            lambda x: float(x.split()[8])
        )

        all_only_foldseek_sorted = all_only_foldseek.sort_values(by='E-value',ascending=True) # E-value 升序排列


        all_only_fiass1000_sorted = all_only_fiass1000.sort_values(by='Cosine_Similarity', ascending=False)
        all_only_fiass2000_sorted = all_only_fiass2000.sort_values(by='Cosine_Similarity', ascending=False)
        all_only_fiass4000_sorted = all_only_fiass4000.sort_values(by='Cosine_Similarity', ascending=False)
        all_only_fiass8000_sorted = all_only_fiass8000.sort_values(by='Cosine_Similarity', ascending=False)

        all_only_foldseek_faiss1000['E-value'] = all_only_foldseek_faiss1000['FoldSeek_Metrics'].apply(lambda x: float(x.split()[8]))
        all_only_foldseek_faiss2000['E-value'] = all_only_foldseek_faiss2000['FoldSeek_Metrics'].apply(lambda x: float(x.split()[8]))
        all_only_foldseek_faiss4000['E-value'] = all_only_foldseek_faiss4000['FoldSeek_Metrics'].apply(lambda x: float(x.split()[8]))
        all_only_foldseek_faiss8000['E-value'] = all_only_foldseek_faiss8000['FoldSeek_Metrics'].apply(lambda x: float(x.split()[8]))


        all_only_foldseek_faiss1000_sorted = all_only_foldseek_faiss1000.sort_values(by='E-value',ascending=True)
        all_only_foldseek_faiss2000_sorted = all_only_foldseek_faiss2000.sort_values(by='E-value',ascending=True)
        all_only_foldseek_faiss4000_sorted = all_only_foldseek_faiss4000.sort_values(by='E-value',ascending=True)
        all_only_foldseek_faiss8000_sorted = all_only_foldseek_faiss8000.sort_values(by='E-value',ascending=True)



    # print(all_only_foldseek) # [4889 rows x 9 columns]
    # print(all_only_foldseek.describe())
    #
    # print(all_only_ssalign) # [51915 rows x 7 columns]
    # print(all_only_ssalign.describe())


    print(all_only_foldseek['Avg_TM_Score'].sum())
    print(all_only_ssalign_sorted.head(4889)['Avg_TM_Score'].sum())

    print(all_only_foldseek['RMSD'].sum())
    print(all_only_ssalign_sorted.head(4889)['RMSD'].sum())


    if measure == "avgtmscore":
        ssalign_scores = all_only_ssalign_sorted['Avg_TM_Score'].values
        foldseek_scores = all_only_foldseek_sorted['Avg_TM_Score'].values

        only_fiass1000_scores = all_only_fiass1000_sorted['Avg_TM_Score'].values
        only_fiass2000_scores = all_only_fiass2000_sorted['Avg_TM_Score'].values
        only_fiass4000_scores = all_only_fiass4000_sorted['Avg_TM_Score'].values
        only_fiass8000_scores = all_only_fiass8000_sorted['Avg_TM_Score'].values

        only_foldseek_faiss1000_scores = all_only_foldseek_faiss1000_sorted['Avg_TM_Score'].values
        only_foldseek_faiss2000_scores = all_only_foldseek_faiss2000_sorted['Avg_TM_Score'].values
        only_foldseek_faiss4000_scores = all_only_foldseek_faiss4000_sorted['Avg_TM_Score'].values
        only_foldseek_faiss8000_scores = all_only_foldseek_faiss8000_sorted['Avg_TM_Score'].values

    if measure == "rmsd":
        ssalign_scores = all_only_ssalign_sorted['RMSD'].values
        foldseek_scores = all_only_foldseek_sorted['RMSD'].values

        only_fiass1000_scores = all_only_fiass1000_sorted['RMSD'].values
        only_fiass2000_scores = all_only_fiass2000_sorted['RMSD'].values
        only_fiass4000_scores = all_only_fiass4000_sorted['RMSD'].values
        only_fiass8000_scores = all_only_fiass8000_sorted['RMSD'].values

        only_foldseek_faiss1000_scores = all_only_foldseek_faiss1000_sorted['RMSD'].values
        only_foldseek_faiss2000_scores = all_only_foldseek_faiss2000_sorted['RMSD'].values
        only_foldseek_faiss4000_scores = all_only_foldseek_faiss4000_sorted['RMSD'].values
        only_foldseek_faiss8000_scores = all_only_foldseek_faiss8000_sorted['RMSD'].values

    topk_values  = range(0,end,step)

    cumulative_ssalign_scores = [ssalign_scores[:k].sum() for k in topk_values]
    cumulative_foldseek_scores = [foldseek_scores[:k].sum() for k in topk_values]

    cumulative_only_fiass1000_scores = [only_fiass1000_scores[:k].sum() for k in topk_values]
    cumulative_only_fiass2000_scores = [only_fiass2000_scores[:k].sum() for k in topk_values]
    cumulative_only_fiass4000_scores = [only_fiass4000_scores[:k].sum() for k in topk_values]
    cumulative_only_fiass8000_scores = [only_fiass8000_scores[:k].sum() for k in topk_values]

    cumulative_only_foldseek_faiss1000_scores = [only_foldseek_faiss1000_scores[:k].sum() for k in topk_values]
    cumulative_only_foldseek_faiss2000_scores = [only_foldseek_faiss2000_scores[:k].sum() for k in topk_values]
    cumulative_only_foldseek_faiss4000_scores = [only_foldseek_faiss4000_scores[:k].sum() for k in topk_values]
    cumulative_only_foldseek_faiss8000_scores = [only_foldseek_faiss8000_scores[:k].sum() for k in topk_values]


    # 用 make_interp_spline 生成平滑的曲线
    def smooth_curve(x, y, num_points=1000):
        spline = make_interp_spline(x, y, k=3)  # k=3 是三次样条曲线
        x_smooth = np.linspace(min(x), max(x), num_points)
        y_smooth = spline(x_smooth)
        return x_smooth, y_smooth

    # 平滑后的曲线
    x_smooth, y_smooth_ssalign = smooth_curve(topk_values, cumulative_ssalign_scores)
    _, y_smooth_foldseek = smooth_curve(topk_values, cumulative_foldseek_scores)
    _, y_smooth_fiass1000 = smooth_curve(topk_values, cumulative_only_fiass1000_scores)
    _, y_smooth_fiass2000 = smooth_curve(topk_values, cumulative_only_fiass2000_scores)
    _, y_smooth_fiass4000 = smooth_curve(topk_values, cumulative_only_fiass4000_scores)
    _, y_smooth_fiass8000 = smooth_curve(topk_values, cumulative_only_fiass8000_scores)
    _, y_smooth_foldseek_faiss1000 = smooth_curve(topk_values, cumulative_only_foldseek_faiss1000_scores)
    _, y_smooth_foldseek_faiss2000 = smooth_curve(topk_values, cumulative_only_foldseek_faiss2000_scores)
    _, y_smooth_foldseek_faiss4000 = smooth_curve(topk_values, cumulative_only_foldseek_faiss4000_scores)
    _, y_smooth_foldseek_faiss8000 = smooth_curve(topk_values, cumulative_only_foldseek_faiss8000_scores)


    # 绘制平滑的曲线图
    plt.figure(figsize=(10, 6))

    # 绘制每条平滑曲线
    # plt.plot(x_smooth / 1000, y_smooth_ssalign / 1000, label='only_SSAlign', color='blue', linestyle='-', linewidth=2)
    plt.plot(x_smooth / 1000, y_smooth_ssalign / 1000 , label='SSAlign', color='blue', linestyle='-', linewidth=2)
    plt.plot(x_smooth / 1000, y_smooth_foldseek / 1000, label='FoldSeek', color='green', linestyle='--', linewidth=2)

    plt.plot(x_smooth / 1000, y_smooth_fiass1000 / 1000, label='ssalign-prefilter-1000', color='red', linestyle='-',linewidth=2)
    plt.plot(x_smooth / 1000, y_smooth_fiass2000 / 1000, label='ssalign-prefilter-2000', color='purple', linestyle='--',linewidth=2)

    plt.plot(x_smooth / 1000, y_smooth_foldseek_faiss1000 / 1000, label='Foldseek(ssalign-prefilter-1000)', color='cyan',linestyle='-', linewidth=2)
    plt.plot(x_smooth / 1000, y_smooth_foldseek_faiss2000 / 1000, label='Foldseek(ssalign-prefilter-2000)', color='magenta', linestyle='--', linewidth=2)

    # 添加图表标题和标签
    plt.title('Cumulative Scores vs. Top-K Values')
    plt.xlabel('Top-hits (K)')
    plt.ylabel(f'Cumulative {measure} score (K)')

    # 显示图例
    plt.legend()
    # 显示网格
    plt.grid(True)

    plt.savefig(f"more_{measure}_gtalign_{mode}_{end}_{step}.png")





"""
foldseek搜索的总数为：52974
commenset的总数为：48085

ssalign 搜索的是 100 * 1000 = 100000
所以参考gtalign来对比，是一个截断，最就是多  top100000

5K一步吧，是整体数据哦
mode1 是按照自己的指标排序
mode2 是按照tmsore排序
"""
def gtalign_ssalign_foldseek_faiss_mode1(basenames,end,step,measure):

    merged_ssalign =  pd.DataFrame()

    # 100个文件的 foldseek 结果，全部保存在一起 52974
    merged_foldseek = pd.DataFrame()

    merged_tmalign = pd.DataFrame()

    merged_faiss_1000 = pd.DataFrame()
    merged_faiss_2000 = pd.DataFrame()
    merged_faiss_4000 = pd.DataFrame()
    merged_faiss_8000 = pd.DataFrame()

    for basename in basenames:
        ssalign_file_path = f"../data/result/Swissport/SSAlign/SVD1280/ssalign/{basename}.result"
        faiss_file_path = f"../data/result/Swissport/SSAlign/SVD1280/{basename}.result"
        foldseek_file_path = f"../data/result/Swissport/foldseek/{basename}.result"
        tmalign_file_path = f"../data/result/Swissport/tmalign/{basename}.result"

        # faiiss 的路径 ，注意只有 TM-Score1,TM-Score2,

        df_ssalign = pd.read_csv(ssalign_file_path)
        merged_ssalign = pd.concat([merged_ssalign,df_ssalign],ignore_index=True)

        df_foldseek = pd.read_csv(foldseek_file_path)
        merged_foldseek = pd.concat([merged_foldseek, df_foldseek], ignore_index=True)

        # print(basename)
        df_tmalign = pd.read_csv(tmalign_file_path)
        df_tmalign_filtered = df_tmalign[(df_tmalign['TM-Score1'] + df_tmalign['TM-Score2']) >= 1 ]
        merged_tmalign = pd.concat([merged_tmalign,df_tmalign_filtered], ignore_index=True)

        # faiss结果
        df_faiss = pd.read_csv(faiss_file_path)
        df_faiss_sorted = df_faiss.sort_values(by='Cosine_Similarity',ascending=False)
        df_faiss_sorted['Avg_TM_Score'] = (df_faiss_sorted['TM-Score1'] + df_faiss_sorted['TM-Score2'] )/ 2

        df_faiss_sorted_1000 = df_faiss_sorted.head(1000)
        df_faiss_sorted_2000 = df_faiss_sorted.head(2000)
        df_faiss_sorted_4000 = df_faiss_sorted.head(4000)
        df_faiss_sorted_8000 = df_faiss_sorted.head(8000)

        merged_faiss_1000 = pd.concat([merged_faiss_1000,df_faiss_sorted_1000], ignore_index=True)
        merged_faiss_2000 = pd.concat([merged_faiss_2000,df_faiss_sorted_2000], ignore_index=True)
        merged_faiss_4000 = pd.concat([merged_faiss_4000,df_faiss_sorted_4000], ignore_index=True)
        merged_faiss_8000 = pd.concat([merged_faiss_8000,df_faiss_sorted_8000], ignore_index=True)


    # ssalign怎么可以按照   'Avg_TM_Score' 排序呢？？？ 这个是后面补充的啊   应该是先按 'Cosine_Similarity'，再按照 Score
    # merged_ssalign_sorted = merged_ssalign.sort_values(by='Avg_TM_Score', ascending=False)
    df_score_none = merged_ssalign[merged_ssalign['Score'].isna()]
    df_score_none_sorted = df_score_none.sort_values(by='Cosine_Similarity', ascending=False)

    # 2. 将 Score 不为 None 的行提取出来并按 Score 排序
    df_score_not_none = merged_ssalign[merged_ssalign['Score'].notna()]
    df_score_not_none_sorted = df_score_not_none.sort_values(by='Score', ascending=False)

    # 3. 合并这两部分，并进行最终的排序
    merged_ssalign_sorted = pd.concat([df_score_none_sorted, df_score_not_none_sorted])



    # foldseek 怎么可以按照 'Avg_TM_Score' 排序呢？？？ 这个是后面补充的啊  应该按照 E-value排序
    merged_foldseek['Avg_TM_Score'] =  (merged_foldseek['TM-Score1']+ merged_foldseek['TM-Score2'])/2
    merged_foldseek['E-value'] = merged_foldseek['FoldSeek_Metrics'].apply(
        lambda x: float(x.split()[8]))  # 取第9个元素（E-value），并且  float(x.split()[8]) 可以 把字符串 改为科学计数法形式

    # merged_foldseek_sorted = merged_foldseek.sort_values(by='Avg_TM_Score', ascending=False)
    merged_foldseek_sorted = merged_foldseek.sort_values(by='E-value', ascending=True)



    merged_tmalign['Avg_TM_Score'] = (merged_tmalign['TM-Score1'] + merged_tmalign['TM-Score2']) / 2
    merged_tmalign_sorted = merged_tmalign.sort_values(by='Avg_TM_Score', ascending=False)


    # faiss处理
    merged_faiss_1000_sorted = merged_faiss_1000.sort_values(by='Cosine_Similarity',ascending=False)
    merged_faiss_2000_sorted = merged_faiss_2000.sort_values(by='Cosine_Similarity',ascending=False)
    merged_faiss_4000_sorted = merged_faiss_4000.sort_values(by='Cosine_Similarity',ascending=False)
    merged_faiss_8000_sorted = merged_faiss_8000.sort_values(by='Cosine_Similarity',ascending=False)

    # print(merged_ssalign_sorted)
    # print(merged_foldseek_sorted)
    # print(merged_tmalign_sorted)

    """
    绘图
    """
    if measure == "avgtmscore":
        # 获取 Avg_TM_Score 列
        ssalign_scores = merged_ssalign_sorted['Avg_TM_Score'].values
        foldseek_scores = merged_foldseek_sorted['Avg_TM_Score'].values
        tmalign_scores = merged_tmalign_sorted['Avg_TM_Score'].values

        faiss_1000_scores = merged_faiss_1000_sorted['Avg_TM_Score'].values
        faiss_2000_scores = merged_faiss_2000_sorted['Avg_TM_Score'].values
        faiss_4000_scores = merged_faiss_4000_sorted['Avg_TM_Score'].values
        faiss_8000_scores = merged_faiss_8000_sorted['Avg_TM_Score'].values

    if measure == "rmsd":
        ssalign_scores = merged_ssalign_sorted['RMSD'].values
        foldseek_scores = merged_foldseek_sorted['RMSD'].values
        tmalign_scores = merged_tmalign_sorted['RMSD'].values

        faiss_1000_scores = merged_faiss_1000_sorted['RMSD'].values
        faiss_2000_scores = merged_faiss_2000_sorted['RMSD'].values
        faiss_4000_scores = merged_faiss_4000_sorted['RMSD'].values
        faiss_8000_scores = merged_faiss_8000_sorted['RMSD'].values


    # 横坐标范围：从 0 到 100000，步长为 5000
    # topk_values = range(0, 100001, 5000)
    topk_values = range(0, end, step)

    # 累积计算
    cumulative_ssalign_scores = [ssalign_scores[:k].sum() for k in topk_values]
    cumulative_foldseek_scores = [foldseek_scores[:k].sum() for k in topk_values]
    cumulative_tmalign_scores = [tmalign_scores[:k].sum() for k in topk_values]

    cumulative_faiss_1000_scores = [faiss_1000_scores[:k].sum() for k in topk_values]
    cumulative_faiss_2000_scores = [faiss_2000_scores[:k].sum() for k in topk_values]
    cumulative_faiss_4000_scores = [faiss_4000_scores[:k].sum() for k in topk_values]
    cumulative_faiss_8000_scores = [faiss_8000_scores[:k].sum() for k in topk_values]

    # 用 make_interp_spline 生成平滑的曲线
    def smooth_curve(x, y, num_points=1000):
        spline = make_interp_spline(x, y, k=3)  # k=3 是三次样条曲线
        x_smooth = np.linspace(min(x), max(x), num_points)
        y_smooth = spline(x_smooth)
        return x_smooth, y_smooth

    # 平滑后的曲线
    x_smooth, y_smooth_ssalign = smooth_curve(topk_values, cumulative_ssalign_scores)
    _, y_smooth_foldseek = smooth_curve(topk_values, cumulative_foldseek_scores)
    _, y_smooth_tmalign = smooth_curve(topk_values, cumulative_tmalign_scores)
    _, y_smooth_faiss_1000 = smooth_curve(topk_values, cumulative_faiss_1000_scores)
    _, y_smooth_faiss_2000 = smooth_curve(topk_values, cumulative_faiss_2000_scores)
    _, y_smooth_faiss_4000 = smooth_curve(topk_values, cumulative_faiss_4000_scores)
    _, y_smooth_faiss_8000 = smooth_curve(topk_values, cumulative_faiss_8000_scores)

    # 绘制平滑的曲线图
    plt.figure(figsize=(10, 6))

    # 绘制每条平滑曲线
    plt.plot(x_smooth / 1000, y_smooth_ssalign / 1000, label='SSAlign', color='blue', linestyle='-', linewidth=2)
    plt.plot(x_smooth / 1000, y_smooth_foldseek / 1000, label='FoldSeek', color='green', linestyle='--', linewidth=2)
    plt.plot(x_smooth / 1000, y_smooth_tmalign / 1000, label='Tmalign', color='red', linestyle='-.', linewidth=2)
    plt.plot(x_smooth / 1000, y_smooth_faiss_1000 / 1000, label='ssalign-prefilter-1000', color='purple', linestyle=':',linewidth=2)
    plt.plot(x_smooth / 1000, y_smooth_faiss_2000 / 1000, label='ssalign-prefilter-2000', color='orange', linestyle='-',linewidth=2)

    # 添加图表标题和标签
    plt.title('Cumulative Scores vs. Top-K Values')
    plt.xlabel('Top-hits (K)')
    plt.ylabel('Cumulative Score (K)')

    # 显示图例
    plt.legend()
    # 显示网格
    plt.grid(True)

    plt.savefig(f"{measure}_gtalign_method_measure_{end}_{step}.png")

    plt.close()


"""
这里 按照 tmscore来对比
"""
def gtalign_ssalign_foldseek_faiss_mode2(basenames,end,step,measure):

    merged_ssalign =  pd.DataFrame()

    # 100个文件的 foldseek 结果，全部保存在一起 52974
    merged_foldseek = pd.DataFrame()

    merged_tmalign = pd.DataFrame()

    merged_faiss_1000 = pd.DataFrame()
    merged_faiss_2000 = pd.DataFrame()
    merged_faiss_4000 = pd.DataFrame()
    merged_faiss_8000 = pd.DataFrame()

    for basename in basenames:
        ssalign_file_path = f"../data/result/Swissport/SSAlign/SVD1280/ssalign/{basename}.result"
        faiss_file_path = f"../data/result/Swissport/SSAlign/SVD1280/{basename}.result"
        foldseek_file_path = f"../data/result/Swissport/foldseek/{basename}.result"
        tmalign_file_path = f"../data/result/Swissport/tmalign/{basename}.result"

        df_ssalign = pd.read_csv(ssalign_file_path)
        merged_ssalign = pd.concat([merged_ssalign,df_ssalign],ignore_index=True)

        df_foldseek = pd.read_csv(foldseek_file_path)
        merged_foldseek = pd.concat([merged_foldseek, df_foldseek], ignore_index=True)

        # print(basename)
        df_tmalign = pd.read_csv(tmalign_file_path)
        df_tmalign_filtered = df_tmalign[(df_tmalign['TM-Score1'] + df_tmalign['TM-Score2']) >= 1 ]
        merged_tmalign = pd.concat([merged_tmalign,df_tmalign_filtered], ignore_index=True)

        # faiss结果
        df_faiss = pd.read_csv(faiss_file_path)
        df_faiss_sorted = df_faiss.sort_values(by='Cosine_Similarity',ascending=False)
        df_faiss_sorted['Avg_TM_Score'] = (df_faiss_sorted['TM-Score1'] + df_faiss_sorted['TM-Score2'] )/ 2

        df_faiss_sorted_1000 = df_faiss_sorted.head(1000)
        df_faiss_sorted_2000 = df_faiss_sorted.head(2000)
        df_faiss_sorted_4000 = df_faiss_sorted.head(4000)
        df_faiss_sorted_8000 = df_faiss_sorted.head(8000)

        merged_faiss_1000 = pd.concat([merged_faiss_1000,df_faiss_sorted_1000], ignore_index=True)
        merged_faiss_2000 = pd.concat([merged_faiss_2000,df_faiss_sorted_2000], ignore_index=True)
        merged_faiss_4000 = pd.concat([merged_faiss_4000,df_faiss_sorted_4000], ignore_index=True)
        merged_faiss_8000 = pd.concat([merged_faiss_8000,df_faiss_sorted_8000], ignore_index=True)


    # 按照   'Avg_TM_Score' 排序
    merged_ssalign_sorted = merged_ssalign.sort_values(by='Avg_TM_Score', ascending=False)



    # foldseek 怎么可以按照 'Avg_TM_Score' 排序呢？？？ 这个是后面补充的啊  应该按照 E-value排序
    merged_foldseek['Avg_TM_Score'] =  (merged_foldseek['TM-Score1']+ merged_foldseek['TM-Score2'])/2
    merged_foldseek_sorted = merged_foldseek.sort_values(by='Avg_TM_Score', ascending=False)



    merged_tmalign['Avg_TM_Score'] = (merged_tmalign['TM-Score1'] + merged_tmalign['TM-Score2']) / 2
    merged_tmalign_sorted = merged_tmalign.sort_values(by='Avg_TM_Score', ascending=False)


    # faiss处理
    merged_faiss_1000_sorted = merged_faiss_1000.sort_values(by='Avg_TM_Score',ascending=False)
    merged_faiss_2000_sorted = merged_faiss_2000.sort_values(by='Avg_TM_Score',ascending=False)
    merged_faiss_4000_sorted = merged_faiss_4000.sort_values(by='Avg_TM_Score',ascending=False)
    merged_faiss_8000_sorted = merged_faiss_8000.sort_values(by='Avg_TM_Score',ascending=False)

    # print(merged_ssalign_sorted)
    # print(merged_foldseek_sorted)
    # print(merged_tmalign_sorted)

    """
    绘图
    """
    if measure=="avgtmscore":
    # 获取 Avg_TM_Score 列
        ssalign_scores = merged_ssalign_sorted['Avg_TM_Score'].values
        foldseek_scores = merged_foldseek_sorted['Avg_TM_Score'].values
        tmalign_scores = merged_tmalign_sorted['Avg_TM_Score'].values

        faiss_1000_scores = merged_faiss_1000_sorted['Avg_TM_Score'].values
        faiss_2000_scores = merged_faiss_2000_sorted['Avg_TM_Score'].values
        faiss_4000_scores = merged_faiss_4000_sorted['Avg_TM_Score'].values
        faiss_8000_scores = merged_faiss_8000_sorted['Avg_TM_Score'].values

    if measure == "rmsd":
        ssalign_scores = merged_ssalign_sorted['RMSD'].values
        foldseek_scores = merged_foldseek_sorted['RMSD'].values
        tmalign_scores = merged_tmalign_sorted['RMSD'].values

        faiss_1000_scores = merged_faiss_1000_sorted['RMSD'].values
        faiss_2000_scores = merged_faiss_2000_sorted['RMSD'].values
        faiss_4000_scores = merged_faiss_4000_sorted['RMSD'].values
        faiss_8000_scores = merged_faiss_8000_sorted['RMSD'].values



    # 横坐标范围：从 0 到 100000，步长为 5000
    topk_values = range(0, end, step)
    # topk_values = range(0, 50000, 1000)

    # 累积计算
    cumulative_ssalign_scores = [ssalign_scores[:k].sum() for k in topk_values]
    cumulative_foldseek_scores = [foldseek_scores[:k].sum() for k in topk_values]
    cumulative_tmalign_scores = [tmalign_scores[:k].sum() for k in topk_values]

    cumulative_faiss_1000_scores = [faiss_1000_scores[:k].sum() for k in topk_values]
    cumulative_faiss_2000_scores = [faiss_2000_scores[:k].sum() for k in topk_values]
    cumulative_faiss_4000_scores = [faiss_4000_scores[:k].sum() for k in topk_values]
    cumulative_faiss_8000_scores = [faiss_8000_scores[:k].sum() for k in topk_values]

    # 用 make_interp_spline 生成平滑的曲线
    def smooth_curve(x, y, num_points=1000):
        spline = make_interp_spline(x, y, k=3)  # k=3 是三次样条曲线
        x_smooth = np.linspace(min(x), max(x), num_points)
        y_smooth = spline(x_smooth)
        return x_smooth, y_smooth

    # 平滑后的曲线
    x_smooth, y_smooth_ssalign = smooth_curve(topk_values, cumulative_ssalign_scores)
    _, y_smooth_foldseek = smooth_curve(topk_values, cumulative_foldseek_scores)
    _, y_smooth_tmalign = smooth_curve(topk_values, cumulative_tmalign_scores)
    _, y_smooth_faiss_1000 = smooth_curve(topk_values, cumulative_faiss_1000_scores)
    _, y_smooth_faiss_2000 = smooth_curve(topk_values, cumulative_faiss_2000_scores)
    _, y_smooth_faiss_4000 = smooth_curve(topk_values, cumulative_faiss_4000_scores)
    _, y_smooth_faiss_8000 = smooth_curve(topk_values, cumulative_faiss_8000_scores)

    # 绘制平滑的曲线图
    plt.figure(figsize=(10, 6))

    # 绘制每条平滑曲线
    # plt.plot(x_smooth / 1000, y_smooth_ssalign / 1000, label='SSAlign', color='blue', linestyle='-', linewidth=2)
    plt.plot(x_smooth / 1000, y_smooth_ssalign/ 1000, label='SSAlign', color='blue', linestyle='-', linewidth=2)
    plt.plot(x_smooth / 1000, y_smooth_foldseek / 1000 , label='FoldSeek', color='green', linestyle='--', linewidth=2)
    plt.plot(x_smooth / 1000, y_smooth_tmalign / 1000 , label='Tmalign', color='red', linestyle='-.', linewidth=2)
    plt.plot(x_smooth / 1000, y_smooth_faiss_1000 / 1000 , label='ssalign-prefilter-1000', color='purple', linestyle=':',linewidth=2)
    plt.plot(x_smooth / 1000, y_smooth_faiss_2000 / 1000 , label='ssalign-prefilter-2000', color='orange', linestyle='-',linewidth=2)

    # 添加图表标题和标签
    plt.title('Cumulative Scores vs. Top-K Values')
    plt.xlabel('Top-hits (K)')
    plt.ylabel('Cumulative Score (K)')

    # 显示图例
    plt.legend()
    # 显示网格
    plt.grid(True)

    plt.savefig(f"{measure}_gtalign_avgtmscore_{end}_{step}.png")

    plt.close()







def benckmark_svd1280():

    with open("/data/wanglei_workspace/ssalign/SaprotProject/Saprot/SaProt/re_Linearliftting/100filenames.txt","r") as f:
        basenames = [line.strip() for line in f if line.strip()]  # 去掉空行和多余空格




    """
    mode 排序方式  
    "avgtmscore":
    "method_measure":
    
measure  指标：
    "avgtmscore":
    "rmsd":
    """
    # ssalign_more_foldseek(basenames,5000,50,"avgtmscore","avgtmscore")
    # ssalign_more_foldseek(basenames,100000,1000,"avgtmscore","avgtmscore")
    # ssalign_more_foldseek(basenames,800000,8000,"avgtmscore","avgtmscore")
    #
    # ssalign_more_foldseek(basenames,5000,50,"avgtmscore","rmsd")
    # ssalign_more_foldseek(basenames,100000,1000,"avgtmscore","rmsd")
    # ssalign_more_foldseek(basenames,800000,8000,"avgtmscore","rmsd")
    #
    #
    # ssalign_more_foldseek(basenames, 5000, 50, "method_measure", "avgtmscore")
    # ssalign_more_foldseek(basenames, 100000, 1000, "method_measure", "avgtmscore")
    # ssalign_more_foldseek(basenames, 800000, 8000, "method_measure", "avgtmscore")
    #
    # ssalign_more_foldseek(basenames, 5000, 50, "method_measure", "rmsd")
    # ssalign_more_foldseek(basenames, 100000, 1000, "method_measure", "rmsd")
    # ssalign_more_foldseek(basenames, 800000, 8000, "method_measure", "rmsd")


    gtalign_ssalign_foldseek_faiss_mode1(basenames,50000,500,"avgtmscore")
    gtalign_ssalign_foldseek_faiss_mode1(basenames,100000,1000,"avgtmscore")
    gtalign_ssalign_foldseek_faiss_mode1(basenames,200000,2000,"avgtmscore")
    gtalign_ssalign_foldseek_faiss_mode1(basenames,800000,8000,"avgtmscore")

    gtalign_ssalign_foldseek_faiss_mode1(basenames,50000,500,"rmsd")
    gtalign_ssalign_foldseek_faiss_mode1(basenames,100000,1000,"rmsd")
    gtalign_ssalign_foldseek_faiss_mode1(basenames,200000,2000,"rmsd")
    gtalign_ssalign_foldseek_faiss_mode1(basenames,800000,8000,"rmsd")



    gtalign_ssalign_foldseek_faiss_mode2(basenames, 50000, 500,"avgtmscore")
    gtalign_ssalign_foldseek_faiss_mode2(basenames, 100000, 1000,"avgtmscore")
    gtalign_ssalign_foldseek_faiss_mode2(basenames, 200000, 2000,"avgtmscore")
    gtalign_ssalign_foldseek_faiss_mode2(basenames, 800000, 8000,"avgtmscore")

    gtalign_ssalign_foldseek_faiss_mode2(basenames, 50000, 500,"rmsd")
    gtalign_ssalign_foldseek_faiss_mode2(basenames, 100000, 1000,"rmsd")
    gtalign_ssalign_foldseek_faiss_mode2(basenames, 200000, 2000,"rmsd")
    gtalign_ssalign_foldseek_faiss_mode2(basenames, 800000, 8000,"rmsd")


    #
    # overlap_ssalign_foldseek(basenames)
    # overlap_faiss_foldseek(basenames,1000)
    # overlap_faiss_foldseek(basenames,2000)
    # overlap_faiss_foldseek(basenames,4000)
    # overlap_faiss_foldseek(basenames,8000)

