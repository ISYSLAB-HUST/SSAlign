"""
定义：
    （家族/超家族）正确：同一个家族/超家族 、或者 >= 0.3
    错误：不同fold 且 分数 < 0.5

1. 增加标签
    作用：对“某个命中项 file1”判定三件事：
    family=1/0：是否属于同 family 集合，或者 Avg_TM_Score >= 0.5
    superfamily=1/0：是否属于同 superfamily 集合，或者 Avg_TM_Score >= 0.5
    folderror=1/0：是否视为“折叠级错误”。核心逻辑是：不在同 fold 且 Avg_TM_Score < 0.5 才算 fold-error；如果跨 fold 但 TM 分数 ≥ 0.5，则不会记为 fold-error。

作用：分别读取该 query 的 foldseek / tmalign / ssalign 结果文件，对每一行调用 tp_fp(...)，把 family/superfamily/folderror 三列写回到 “new05” 目录下的新结果文件里。


2. 整理npz
    作用：读取 tmalign/new05/{basename}.result，统计每个 query 里 family==1 / superfamily==1 / folderror==1 的数量，并保存成 tmalign_true_counts_results05.npz。这个 npz 之后会被画 PR 曲线时当作 召回率分母（total true）。

"""

import csv
import os
import pandas as pd


def find_lookup(basename):
    tsv_file = "scop_lookup.fix.tsv"

    with open(tsv_file, mode='r') as file:
        reader = csv.reader(file, delimiter='\t')

        for row in reader:
            if len(row) != 2:
                continue  # 跳过无效行

            domain_id, scop_code = row
            if domain_id == basename:
                # return scop_code

                levels = scop_code.split('.')
                result = {
                    "class": levels[0],
                    "fold": ".".join(levels[:2]) if len(levels) > 1 else None,
                    "superfamily": ".".join(levels[:3]) if len(levels) > 2 else None,
                    "family": ".".join(levels[:4]) if len(levels) > 3 else None
                }
                return result


"""
获得级别
"""


def get_scop_levels(scopecode):
    # 将 SCOP 层级分解为不同层级（例如：'a.1.1.1' -> ['a', 'a.1', 'a.1.1', 'a.1.1.1']）
    levels = scopecode.split('.')
    return {
        "class": levels[0],
        "fold": ".".join(levels[:2]) if len(levels) > 1 else None,
        "superfamily": ".".join(levels[:3]) if len(levels) > 2 else None,
        "family": ".".join(levels[:4]) if len(levels) > 3 else None
    }


"""
    筛选所有同一 family 的文件
"""


def group_files_by_family(basename):
    tsv_file = "scop_lookup.fix.tsv"

    df = pd.read_csv(tsv_file, sep='\t', header=None, names=['File', 'SCOP_Level'])

    # 找到 basename 对应的 行
    file_scop_level = df[df['File'] == basename]['SCOP_Level']

    if file_scop_level.empty:
        return []

    file_scop = get_scop_levels(file_scop_level.iloc[0])
    family = file_scop['family']  # 获取 family 层级

    # 筛选所有同一 family 的文件
    same_family_files = []

    for index, row in df.iterrows():
        current_scop = get_scop_levels(row['SCOP_Level'])
        # 比较 family 层级
        if current_scop['family'] == family:
            same_family_files.append(row['File'])

    return same_family_files


"""
    筛选所有同一 superfamily 的文件
"""


def group_files_by_superfamily(basename):
    tsv_file = "scop_lookup.fix.tsv"

    df = pd.read_csv(tsv_file, sep='\t', header=None, names=['File', 'SCOP_Level'])

    # 找到 basename 对应的 行
    file_scop_level = df[df['File'] == basename]['SCOP_Level']

    if file_scop_level.empty:
        return []

    file_scop = get_scop_levels(file_scop_level.iloc[0])
    superfamily = file_scop['superfamily']  # 获取 superfamily 层级

    # 筛选所有同一 superfamily 的文件
    same_superfamily_files = []

    for index, row in df.iterrows():
        current_scop = get_scop_levels(row['SCOP_Level'])
        # 比较 superfamily 层级
        if current_scop['superfamily'] == superfamily:
            same_superfamily_files.append(row['File'])

    return same_superfamily_files


"""
    筛选所有同一 fold 的文件
"""


def group_same_fold_files(basename):
    tsv_file = "scop_lookup.fix.tsv"
    # 读取 TSV 文件
    df = pd.read_csv(tsv_file, sep='\t', header=None, names=['File', 'SCOP_Level'])

    # 找到 basename 对应的 行
    file_scop_level = df[df['File'] == basename]['SCOP_Level']

    if file_scop_level.empty:
        return []

    # 获取该文件的 SCOP 层级，并提取折叠层级
    file_scop = get_scop_levels(file_scop_level.iloc[0])
    fold = file_scop['fold']  # 获取 fold 层级

    # 筛选所有同一 fold 的文件
    same_fold_files = []

    for index, row in df.iterrows():
        current_scop = get_scop_levels(row['SCOP_Level'])
        # 比较 fold 层级
        if current_scop['fold'] == fold:
            same_fold_files.append(row['File'])

    return same_fold_files


def tp_fp(file, same_family_files, same_superfamily_files, same_fold_files, avg_tmscore):
    # [,,] 三个值，分别表示 family、superfamily 是否正确，是否错误
    result = []

    # 家族正确：同一个家族 或 分数 >= 0.5
    is_family_correct = (avg_tmscore >= 0.5) or (file in same_family_files)
    result.append(1 if is_family_correct else 0)

    # 超家族正确：同一个超家族 或 分数 >= 0.5
    is_superfamily_correct = (avg_tmscore >= 0.5) or (file in same_superfamily_files)
    result.append(1 if is_superfamily_correct else 0)

    # 错误：不同折叠 且 分数 < 0.3
    is_error = (file not in same_fold_files) and (avg_tmscore < 0.5)
    result.append(1 if is_error else 0)

    return result


"""
根据 正确、错误
补充一下csv表格，就是补充两列： family,superfamily,error

family  表示family级别是否正确: 同一个家族 、或者 >= 0.5
superfamily  表示family级别是否正确: 同一个超家族 、或者 >= 0.5
error   不同fold 且 分数 < 0.5
"""
def add_csv_tp_fp(basename, dim):
    # 找到家族、超家族、折叠
    same_family_files = group_files_by_family(basename)
    same_superfamily_files = group_files_by_superfamily(basename)
    same_fold_files = group_same_fold_files(basename)

    ###############################foldseek###############################################

    foldseek_file_path = f"../benchmarkData/SCOPe40/foldseek/{basename}.result"

    df_foldseek = pd.read_csv(foldseek_file_path)
    df_foldseek['Avg_TM_Score'] = (df_foldseek['TM-Score1'] + df_foldseek['TM-Score2']) / 2
    df_foldseek[["family", "superfamily", "folderror"]] = None

    # 应用函数并拆分结果
    results_foldseek = df_foldseek.apply(
        lambda row: tp_fp(row["File1"], same_family_files, same_superfamily_files, same_fold_files,
                          row["Avg_TM_Score"]),
        axis=1,
        result_type="expand")

    # 将结果赋值到对应列
    df_foldseek[["family", "superfamily", "folderror"]] = results_foldseek

    df_foldseek.to_csv(f"../benchmarkData/SCOPe40/foldseek/new05/{basename}.result", index=False)

    ##################################tmalign############################################

    tmalign_file_path = f"../benchmarkData/SCOPe40/tmalign/{basename}.result"

    df_tmalign = pd.read_csv(tmalign_file_path)
    df_tmalign['Avg_TM_Score'] = (df_tmalign['TM-Score1'] + df_tmalign['TM-Score2']) / 2
    df_tmalign[["family", "superfamily", "folderror"]] = None
    # 应用函数并拆分结果
    results_tmalign = df_tmalign.apply(
        lambda row: tp_fp(os.path.basename(row["File1"]), same_family_files, same_superfamily_files, same_fold_files,
                          row["Avg_TM_Score"]),
        axis=1,
        result_type="expand")
    # 将结果赋值到对应列
    df_tmalign[["family", "superfamily", "folderror"]] = results_tmalign

    df_tmalign.to_csv(f"../benchmarkData/SCOPe40/tmalign/new05/{basename}.result", index=False)

        #######################################ssalign-prefilter#######################################

    for dim in [1280, 512, 256, 128, 64]:
        # 因为 SSAlign 使用参数 faiss_topk=2000, final_number=1000，并且最后的测试中使用的是 SSAlign_Prefilter-500 和  SSAlign_Prefilter-1000，所以这里直接使用 SSAlign结果即可

        # ssalign_file_path = f"../benchmarkData/SCOPe40/SSAlign/SVD{dim}/ssalign_prefilter/{basename}.resault"
        #
        # df_ssalign = pd.read_csv(ssalign_file_path)
        # df_ssalign[["family", "superfamily", "folderror"]] = None
        # results_ssalign = df_ssalign.apply(
        #     lambda row: tp_fp(row["File1"], same_family_files, same_superfamily_files, same_fold_files,
        #                       row["Avg_TM_Score"]),
        #     axis=1,
        #     result_type="expand")
        # # 将结果赋值到对应列
        # df_ssalign[["family", "superfamily", "folderror"]] = results_ssalign
        #
        # df_ssalign.to_csv(f"../benchmarkData/SCOPe40/SSAlign/SVD{dim}/ssalign_prefilter/new05/{basename}.resault",
        #                   index=False)

        #######################################ssalign#######################################

        ssalign_file_path = f"../benchmarkData/SCOPe40/SSAlign/SVD{dim}/ssalign/{basename}.resault"

        df_ssalign = pd.read_csv(ssalign_file_path)
        df_ssalign[["family", "superfamily", "folderror"]] = None
        results_ssalign = df_ssalign.apply(
            lambda row: tp_fp(row["File1"], same_family_files, same_superfamily_files, same_fold_files,
                              row["Avg_TM_Score"]),
            axis=1,
            result_type="expand")
        # 将结果赋值到对应列
        df_ssalign[["family", "superfamily", "folderror"]] = results_ssalign

        df_ssalign.to_csv(f"../benchmarkData/SCOPe40/SSAlign/SVD{dim}/ssalign/new05/{basename}.resault",
                          index=False)








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

    count = 1
    for basename in foldseek_basenames:
        add_csv_tp_fp(basename)
        print(f"蛋白质{basename}，处理完毕: {count}/{len(foldseek_basenames)}")
        count += 1