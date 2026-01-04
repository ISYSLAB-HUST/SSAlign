import os

import pandas as pd
import numpy as np




"""
    计算 “Hits up to the 1st FP” 所需数据
"""
def acc_firstFP(tool_name, basenames, dim=1280, cos_threshold=0.2):
    merged_data = []

    if tool_name == "foldseek":
        for basename in basenames:
            foldseek_file_path = f"../benchmarkData/SCOPe40/foldseek/new05/{basename}.result"
            df_foldseek = pd.read_csv(foldseek_file_path)
            df_foldseek['E-value'] = df_foldseek['FoldSeek_Metrics'].apply(lambda x: float(x.split()[8]))
            df_foldseek_sorted = df_foldseek.sort_values(by='E-value', ascending=True)  # E-value按照升序排序！
            merged_data.append(df_foldseek_sorted)

    if tool_name == "tmalign":
        for basename in basenames:
            tmalign_file_path = f"../benchmarkData/SCOPe40/tmalign/new05/{basename}.result"
            df_tmalign = pd.read_csv(tmalign_file_path)
            df_tmalign_sorted = df_tmalign.sort_values(by='Avg_TM_Score', ascending=False)
            merged_data.append(df_tmalign_sorted)

    # 参数是 --prefilter_target 2000  --final_target 1000
    if tool_name == "ssalign":
        for basename in basenames:
            ssalign_file_path = f"../benchmarkData/SCOPe40/SSAlign/SVD{dim}/new05/{basename}.result"
            df_ssalign_all = pd.read_csv(ssalign_file_path)

            # 1. 按 Cosine_Similarity 降序取前 2000 条
            df_ssalign_prefilter_2000 = df_ssalign_all.sort_values(by='Cosine_Similarity', ascending=False).head(2000)
            # 2. 筛选 Cosine_Similarity > 0.2 的部分
            df_cosine_filtered = df_ssalign_prefilter_2000[df_ssalign_prefilter_2000['Cosine_Similarity'] >= cos_threshold]
            # 3. 判断是否 ≥ 1000 条
            if len(df_cosine_filtered) >= 1000:
                df_ssalign = df_cosine_filtered.head(1000)  # 直接取前 1000 条
            else:
                # 先取所有 Cosine_Similarity > 0.2 的
                df_ssalign = df_cosine_filtered.copy()
                # 剩余需要补充的数量
                remaining = 1000 - len(df_ssalign)
                # 从 df_ssalign_prefilter_2000 中排除已选的部分，并按 Score 降序补充
                df_remaining = df_ssalign_prefilter_2000[~df_ssalign_prefilter_2000.index.isin(df_ssalign.index)]
                df_remaining_sorted = df_remaining.sort_values(by='Score', ascending=False).head(remaining)
                # 合并两部分
                df_ssalign = pd.concat([df_ssalign, df_remaining_sorted])
            merged_data.append(df_ssalign)

    if tool_name == "ssalign-prefilter-500":
        for basename in basenames:
            ssalign_file_path = f"../benchmarkData/SCOPe40/SSAlign/SVD{dim}/new05/{basename}.result"
            df_ssalign_all = pd.read_csv(ssalign_file_path)
            df_ssalign_prefilter_500 = df_ssalign_all.sort_values(by='Cosine_Similarity', ascending=False).head(500)
            merged_data.append(df_ssalign_prefilter_500)

    if tool_name == "ssalign-prefilter-1000":
        for basename in basenames:
            ssalign_file_path = f"../benchmarkData/SCOPe40/SSAlign/SVD{dim}/new05/{basename}.result"
            df_ssalign_all = pd.read_csv(ssalign_file_path)
            df_ssalign_prefilter_1000 = df_ssalign_all.sort_values(by='Cosine_Similarity', ascending=False).head(1000)
            merged_data.append(df_ssalign_prefilter_1000)

    """
    此时就获取到了所有测试的结果，接下来就应该按照 查询比例，计算正确率
    直到第一个错误？ 
        family,superfamily,folderror

    (df['folderror'] == 1).idxmax()

    """
    print(f"数据收集完毕，{len(merged_data)}")

    family_true = []  # 直到第一个错误的 正确的个数
    superfamily_true = []
    query_sum = []  # 查询的总个数

    for query_rate in np.arange(0, 1.01, 0.01):
        print(f"处理查询比例{query_rate}")
        family_true_count = 0
        superfamily_true_count = 0
        query_sum_count = 0

        for df in merged_data:
            query_length = max(1, int(len(df) * query_rate))
            query_sum_count += query_length

            # 1. 先取出前query_rate个
            df_rate = df.head(query_length)

            # 2. 找到第一个 folderror=1 的索引（如果存在）
            if (df_rate['folderror'] == 1).any():  # 检查是否存在 folderror=1
                df_result = df_rate.iloc[:df_rate['folderror'].eq(1).idxmax()]

            else:
                df_result = df_rate  # 如果没有 folderror=1，返回全部 df_rate

            family_true_count += df_result['family'].eq(1).sum()
            superfamily_true_count += df_result['superfamily'].eq(1).sum()

        family_true.append(family_true_count)
        superfamily_true.append(superfamily_true_count)
        query_sum.append(query_sum_count)

        print(family_true)
        print(superfamily_true)
        print(query_sum)

    np.savez(
        f"../benchmarkData/SCOPe40/cumsumNpz/dim_{dim}_{tool_name}_firstFP_accuracy_results05.npz",
        family_true=np.array(family_true),
        superfamily_true=np.array(superfamily_true),
        query_sum=np.array(query_sum)
    )




"""
    统计结果正确的个数
    绘制PR曲线所需
"""
def acc_PR(tool_name, basenames, dim=1280, cos_threshold=0.2):
    merged_data = []


    if tool_name == "foldseek":
        for basename in basenames:
            foldseek_file_path = f"../benchmarkData/SCOPe40/foldseek/new05/{basename}.result"
            df_foldseek = pd.read_csv(foldseek_file_path)
            df_foldseek['E-value'] = df_foldseek['FoldSeek_Metrics'].apply(lambda x: float(x.split()[8]))
            df_foldseek_sorted = df_foldseek.sort_values(by='E-value', ascending=True)  # E-value按照升序排序！
            merged_data.append(df_foldseek_sorted)

    if tool_name == "tmalign":
        for basename in basenames:
            tmalign_file_path =  f"../benchmarkData/SCOPe40/tmalign/new05/{basename}.result"
            df_tmalign = pd.read_csv(tmalign_file_path)
            df_tmalign_sorted = df_tmalign.sort_values(by='Avg_TM_Score', ascending=False)
            merged_data.append(df_tmalign_sorted)

    if tool_name == "ssalign":
        for basename in basenames:
            ssalign_file_path = f"../benchmarkData/SCOPe40/SSAlign/SVD{dim}/new05/{basename}.result"
            df_ssalign_all = pd.read_csv(ssalign_file_path)
            # 1. 按 Cosine_Similarity 降序取前 2000 条
            df_ssalign_prefilter_2000 = df_ssalign_all.sort_values(by='Cosine_Similarity', ascending=False).head(2000)
            # 2. 筛选 Cosine_Similarity > 0.2 的部分
            df_cosine_filtered = df_ssalign_prefilter_2000[df_ssalign_prefilter_2000['Cosine_Similarity'] >= cos_threshold]

            df_ssalign = pd.DataFrame()
            # 3. 判断是否 ≥ 1000 条
            if len(df_cosine_filtered) >= 1000:
                df_ssalign = df_cosine_filtered.head(1000)  # 直接取前 1000 条
            else:
                # 先取所有 Cosine_Similarity > 0.2 的
                df_ssalign = df_cosine_filtered.copy()
                # 剩余需要补充的数量
                remaining = 1000 - len(df_ssalign)
                # 从 df_ssalign_prefilter_2000 中排除已选的部分，并按 Score 降序补充
                df_remaining = df_ssalign_prefilter_2000[~df_ssalign_prefilter_2000.index.isin(df_ssalign.index)]
                df_remaining_sorted = df_remaining.sort_values(by='Score', ascending=False).head(remaining)
                # 合并两部分
                df_ssalign = pd.concat([df_ssalign, df_remaining_sorted])
            merged_data.append(df_ssalign)

    if tool_name == "ssalign-prefilter-500":
        for basename in basenames:
            ssalign_file_path = f"../benchmarkData/SCOPe40/SSAlign/SVD{dim}/new05/{basename}.result"
            df_ssalign_all = pd.read_csv(ssalign_file_path)
            df_ssalign_prefilter_500 = df_ssalign_all.sort_values(by='Cosine_Similarity', ascending=False).head(500)
            merged_data.append(df_ssalign_prefilter_500)

    if tool_name == "ssalign-prefilter-1000":
        for basename in basenames:
            ssalign_file_path = f"../benchmarkData/SCOPe40/SSAlign/SVD{dim}/new05/{basename}.result"
            df_ssalign_all = pd.read_csv(ssalign_file_path)
            df_ssalign_prefilter_1000 = df_ssalign_all.sort_values(by='Cosine_Similarity', ascending=False).head(1000)
            merged_data.append(df_ssalign_prefilter_1000)


    print(f"数据收集完毕，{len(merged_data)}")

    family_true = []
    superfamily_true = []
    query_sum = []  # 查询的总个数

    for query_rate in np.arange(0, 1.01, 0.01):
        print(f"处理查询比例{query_rate}")
        family_true_count = 0
        superfamily_true_count = 0
        query_sum_count = 0

        for df in merged_data:
            query_length = max(1, int(len(df) * query_rate))
            query_sum_count += query_length
            # 1. 先取出前query_rate个
            df_rate = df.head(query_length)

            family_true_count += df_rate['family'].eq(1).sum()
            superfamily_true_count += df_rate['superfamily'].eq(1).sum()

        family_true.append(family_true_count)
        superfamily_true.append(superfamily_true_count)
        query_sum.append(query_sum_count)

        print(family_true)
        print(superfamily_true)
        print(query_sum)

    np.savez(
        f"../benchmarkData/SCOPe40/cumsumNpz/dim_{dim}_{tool_name}_PR_accuracy_results05.npz",
        family_true=np.array(family_true),
        superfamily_true=np.array(superfamily_true),
        query_sum=np.array(query_sum)

    )

"""
    由于我们使用TM-align进行了全对全检索，所以此处正确的个数（（家族/超家族）正确：同一个家族/超家族 、或者 >= 0.3），可以作为召回率分母
"""
def acc_true_count(basenames):

    family_true_count_list = []
    superfamily_true_count_list = []
    folderror_list = []

    for basename in basenames:
        tmalign_file_path = f"../benchmarkData/SCOPe40/tmalign/new05/{basename}.result"


        df_tmalign = pd.read_csv(tmalign_file_path)
        # 统计字段为1的个数
        family_count = len(df_tmalign[df_tmalign['family'] == 1])
        superfamily_count = len(df_tmalign[df_tmalign['superfamily'] == 1])
        folderror_count = len(df_tmalign[df_tmalign['folderror'] == 1])

        family_true_count_list.append(family_count)
        superfamily_true_count_list.append(superfamily_count)
        folderror_list.append(folderror_count)


    np.savez(
        "../benchmarkData/SCOPe40/cumsumNpz/tmalign_true_counts_results05.npz",
        family_true=family_true_count_list,
        superfamily_true=superfamily_true_count_list,
        folderror = folderror_list
    )

    data = np.load("tmalign_true_counts_results05.npz")
    family_counts = data["family_true"]
    superfamily_counts = data["superfamily_true"]
    folderror_counts = data["folderror"]

    print("Family counts:", family_counts)
    print("Family length:", len(family_counts))
    print("Family avg:", np.mean(family_counts))

    print("Superfamily counts:", superfamily_counts)
    print("folderror counts:", folderror_counts)




if __name__=="__main__":
    basenames = []
    file_dir = "../pdbData/pdb/SCOPe40"

    for root, dirs, files in os.walk(file_dir):
        for file in files:
            basenames.append(file)
    # 13个特殊文件
    foldseek_empty_files = [
        "d1dpjb_",
        "d2e74d2",
        "d1q90g_",
        "d1q90a3",
        "d1xoua_",
        "d1rzhh2",
        "d1ehkc_",
        "d1l2pa_",
        "d2ciob_",
        "d1q90m_",
        "d1ehkb2",
        "d1jb0x_",
        "d1jb0m_"]

    foldseek_basenames = [file for file in basenames if file not in foldseek_empty_files]

    #############################foldseek######################################
    acc_firstFP(tool_name="foldseek",basenames=foldseek_basenames)
    acc_PR(tool_name="foldseek",basenames=foldseek_basenames)

    #############################tmalign######################################
    acc_firstFP(tool_name="tmalign",basenames=foldseek_basenames)
    acc_PR(tool_name="tmalign",basenames=foldseek_basenames)

    dim2cos = {1280: 0.2, 512: 0.3, 256: 0.45, 128: 0.6, 64: 0.7}

    for dim in [1280, 512, 256, 128, 64]:
        cos_th = dim2cos[dim]

        #############################ssalign######################################
        acc_firstFP(tool_name="ssalign", basenames=foldseek_basenames, dim=dim, cos_threshold=cos_th)
        acc_PR(tool_name="ssalign", basenames=foldseek_basenames, dim=dim, cos_threshold=cos_th)

        #############################ssalign-prefilter-500######################################
        acc_firstFP(tool_name="ssalign-prefilter-500", basenames=foldseek_basenames, dim=dim, cos_threshold=cos_th)
        acc_PR(tool_name="ssalign-prefilter-500", basenames=foldseek_basenames, dim=dim, cos_threshold=cos_th)

        #############################ssalign-prefilter-1000######################################
        acc_firstFP(tool_name="ssalign-prefilter-1000", basenames=foldseek_basenames, dim=dim, cos_threshold=cos_th)
        acc_PR(tool_name="ssalign-prefilter-1000", basenames=foldseek_basenames, dim=dim, cos_threshold=cos_th)

