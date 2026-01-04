import pandas as pd


"""
    ssalign 
    ssalign_prefilter
    foldseek
    tmalign
    overlap
"""

def overlap_ssalign_prefilter_foldseek(basenames,topk,dim):
    results = []

    for basename in basenames:
        ssalign_prefilter_file_path = f"../benchmarkData/SwissProt/SSAlign/SVD{dim}/ssalign_prefilter/{basename}.result"
        foldseek_file_path = f"../benchmarkData/SwissProt/foldseek/{basename}.result"
        tmalign_file_path = f"../benchmarkData/SwissProt/tmalign/{basename}.result"

        # ssalign_prefilter 结果
        df_ssalign_prefilter = pd.read_csv(ssalign_prefilter_file_path)
        df_ssalign_prefilter_sorted_topk = df_ssalign_prefilter.sort_values(by='Cosine_Similarity', ascending=False).head(topk)
        ssalign_prefilter_set = set(df_ssalign_prefilter_sorted_topk['File1'].tolist())

        # foldseek结果
        df_foldseek = pd.read_csv(foldseek_file_path)
        foldseek_set = set(df_foldseek['File1'].tolist())

        # tmalign结果
        df_tmalign = pd.read_csv(tmalign_file_path)
        df_tmalign['Avg_TM_Score'] = (df_tmalign['TM-Score1'] + df_tmalign['TM-Score2'])/2
        df_tmalign_filtered = df_tmalign[df_tmalign['Avg_TM_Score'] >= 0.5]
        tmalign_set = set(df_tmalign_filtered['File1'].tolist())

        common_set_ssalign_prefilter_foldseek = ssalign_prefilter_set.intersection(foldseek_set)

        common_set_faiss_tmalign = ssalign_prefilter_set.intersection(tmalign_set)

        common_set_foldseek_tmalign = foldseek_set.intersection(tmalign_set)

        # 将每个文件的结果记录到 results 列表中
        results.append({
            'basename': basename,
            'ssalign_prefilter_count': len(ssalign_prefilter_set),
            'foldseek_count': len(foldseek_set),
            'tmalign_count': len(tmalign_set),
            'common_set_ssalign_prefilter_foldseek_count': len(common_set_ssalign_prefilter_foldseek),
            'common_set_ssalign_prefilter_tmalign_count': len(common_set_faiss_tmalign),
            'common_set_foldseek_tmalign': len(common_set_foldseek_tmalign),
        })

        # 创建 DataFrame
    df_results = pd.DataFrame(results)

    save_path = f"../benchmarkData/SwissPort/benchmark/{dim}_overlap_ssalign_prefilter_{topk}_foldseek.csv"

    # 将结果保存到 CSV 文件
    df_results.to_csv(save_path, index=False)


def overlap_ssalign_foldseek(basenames,dim):
    results = []

    for basename in basenames:
        ssalign_file_path = f"../benchmarkData/SwissProt/SSAlign/SVD{dim}/ssalign/{basename}.result"
        foldseek_file_path = f"../benchmarkData/SwissProt/foldseek/{basename}.result"
        tmalign_file_path = f"../benchmarkData/SwissProt/tmalign/{basename}.result"

        # ssalign 结果
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

    save_path = f"../benchmarkData/SCOPe40/benchmark/{dim}_overlap_ssalign_foldseek.csv"
    df_results = pd.DataFrame(results)
    # 将结果保存到 CSV 文件
    df_results.to_csv(save_path, index=False)




