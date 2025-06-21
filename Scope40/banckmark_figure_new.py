"""
正确：同一个家族/超家族 、或者 >= 0.3
错误：不同fold 且 分数 < 0.3


后来改成了 0.5
文件就是原来的后面加上了 05

"""
import csv
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import argparse
import multiprocessing
from functools import partial

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


def group_files_by_family(basename):
    tsv_file = "scop_lookup.fix.tsv"

    df = pd.read_csv(tsv_file, sep='\t', header=None, names=['File', 'SCOP_Level'])

    # 找到 basename 对应的 superfamily
    file_scop_level = df[df['File'] == basename]['SCOP_Level']

    if file_scop_level.empty:
            return []

    file_scop = get_scop_levels(file_scop_level.iloc[0])
    family = file_scop['family']  # 获取 family 层级

    # 筛选所有同一 family 的文件
    same_family_files = []

    for index, row in df.iterrows():
        current_scop = get_scop_levels(row['SCOP_Level'])
        # 比较 superfamily 层级
        if current_scop['family'] == family:
            same_family_files.append(row['File'])

    return same_family_files


def group_files_by_superfamily(basename):
    tsv_file = "scop_lookup.fix.tsv"

    df = pd.read_csv(tsv_file, sep='\t', header=None, names=['File', 'SCOP_Level'])

    # 找到 basename 对应的 superfamily
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


def group_same_fold_files(basename):

    tsv_file = "scop_lookup.fix.tsv"
    # 读取 TSV 文件
    df = pd.read_csv(tsv_file, sep='\t', header=None, names=['File', 'SCOP_Level'])

    # 找到 basename 对应的 SCOP 层级
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




"""
/data2/zxc_data/foldseek_database/foldseek_database/Scope40/gitdata/
foldseek SSAlign  tmalign

foldseek结果 
    /data2/zxc_data/foldseek_database/foldseek_database/Scope40/gitdata/foldseek$ head d2vqei1.result
        File2是自己
        File1,File2,TM-Score1,TM-Score2,Aligned Length,RMSD,Seq_ID,FoldSeek_Metrics
        d2vqei1,d2vqei1,1.0,1.0,127,0.0,1.0,1.000 127 0 0 1 127 1 127 1.337E-28 1076

tmalign结果
    /data2/zxc_data/foldseek_database/foldseek_database/Scope40/gitdata/tmalign$ head d2vqei1.result
    File2是自己
    File1,File2,TM-Score1,TM-Score2,Aligned Length,RMSD,Seq_ID
    /data/foldseek_database/Scope40/pdb/d2fnba_,/data/foldseek_database/Scope40/pdb/d2vqei1,0.27033,0.32548,53,3.68,0.17

ssalign结果

(base) xuchaozhang@ubuntu:/data2/zxc_data/foldseek_database/foldseek_database/Scope40/gitdata/SSAlign/SVD1280$ cat d2vqei1.result | wc -l
8001
(base) xuchaozhang@ubuntu:/data2/zxc_data/foldseek_database/foldseek_database/Scope40/gitdata/SSAlign/SVD1280$ cat bio_d2vqei1_lower_global.csv | wc -l
8001

(base) xuchaozhang@ubuntu:/data2/zxc_data/foldseek_database/foldseek_database/Scope40/gitdata/SSAlign/SVD1280$ head d2vqei1.result
File2是自己
File1,File2,TM-Score1,TM-Score2,Aligned Length,RMSD,Seq_ID,Cosine_Similarity
d2vqei1,d2vqei1,1.0,1.0,127,0.0,1.0,0.9999998211860657

下面就是加了Score的
(base) xuchaozhang@ubuntu:/data2/zxc_data/foldseek_database/foldseek_database/Scope40/gitdata/SSAlign/SVD1280$ head bio_d2vqei1_lower_global.csv 
File1,Aligned Length,RMSD,Seq_ID,length_squeue,Avg_TM_Score,Cosine_Similarity,Length,Identity,Similarity,Gaps,Score
d2fnba_,53,3.68,0.17,95,0.297905,0.020725542679429,162,24/162 (14.814814814814813%),60/162 (37.03703703703704%),102/162 (62.96296296296296%),18.5




"""

def tp_fp(file,same_family_files,same_superfamily_files,same_fold_files,avg_tmscore):
    # same_family_files = group_files_by_family(file1)
    # same_superfamily_files = group_files_by_superfamily(file1)
    # same_fold_files = group_same_fold_files(file1)

    # [,,] 三个值，分别表示 family、superfamily是否正确，是否错误
    result=[]

    # 家族正确：同一个家族 或 分数 >= 0.3
    is_family_correct = (avg_tmscore >= 0.5) or (file in same_family_files)
    result.append(1 if is_family_correct else 0)

    # 超家族正确：同一个超家族 或 分数 >= 0.3
    is_superfamily_correct = (avg_tmscore >= 0.5) or (file in same_superfamily_files)
    result.append(1 if is_superfamily_correct else 0)

    # 错误：不同折叠 且 分数 < 0.3
    is_error = (file not in same_fold_files) and (avg_tmscore < 0.5)
    result.append(1 if is_error else 0)

    return result







"""
根据 正确、错误
补充一下csv表格，就是补充两列： family,superfamily,error

family  表示family级别是否正确: 同一个家族 、或者 >= 0.3
superfamily  表示family级别是否正确: 同一个超家族 、或者 >= 0.3
error   不同fold 且 分数 < 0.3
"""
def add_csv_tp_fp(basename,dim):

    # for basename in basenames:
    # foldseek_file_path = f"/data2/zxc_data/foldseek_database/foldseek_database/Scope40/gitdata/foldseek/{basename}.result"
    # tmalign_file_path = f"/data2/zxc_data/foldseek_database/foldseek_database/Scope40/gitdata/tmalign/{basename}.result"

    ssalign_file_path = f"/data2/zxc_data/foldseek_database/foldseek_database/Scope40/gitdata/SSAlign/SVD{dim}/bio_{basename}_lower_global.csv"

    # 找到家族、超家族、折叠
    same_family_files = group_files_by_family(basename)
    same_superfamily_files = group_files_by_superfamily(basename)
    same_fold_files = group_same_fold_files(basename)


    # df_foldseek = pd.read_csv(foldseek_file_path)
    # df_foldseek['Avg_TM_Score'] = (df_foldseek['TM-Score1'] + df_foldseek['TM-Score2'])/2
    # df_foldseek[["family", "superfamily", "folderror"]] = None
    #
    # # 应用函数并拆分结果
    # results_foldseek = df_foldseek.apply(
    #     lambda row: tp_fp(row["File1"],same_family_files,same_superfamily_files,same_fold_files, row["Avg_TM_Score"]),
    #     axis=1,
    #     result_type="expand")
    #
    # # 将结果赋值到对应列
    # df_foldseek[["family", "superfamily", "folderror"]] = results_foldseek
    #
    # df_foldseek.to_csv(f"/data2/zxc_data/foldseek_database/foldseek_database/Scope40/gitdata/foldseek/new05/{basename}.result",index=False)
    #
    # ##############################################################################
    #
    # df_tmalign = pd.read_csv(tmalign_file_path)
    # df_tmalign['Avg_TM_Score'] = (df_tmalign['TM-Score1'] + df_tmalign['TM-Score2']) / 2
    # df_tmalign[["family", "superfamily", "folderror"]] = None
    # # 应用函数并拆分结果
    # results_tmalign = df_tmalign.apply(
    #     lambda row: tp_fp(os.path.basename(row["File1"]),same_family_files,same_superfamily_files,same_fold_files, row["Avg_TM_Score"]),
    #     axis=1,
    #     result_type="expand")
    # # 将结果赋值到对应列
    # df_tmalign[["family", "superfamily", "folderror"]] = results_tmalign
    #
    # df_tmalign.to_csv(f"/data2/zxc_data/foldseek_database/foldseek_database/Scope40/gitdata/tmalign/new05/{basename}.result",index=False)

    ##############################################################################

    df_ssalign = pd.read_csv(ssalign_file_path)
    df_ssalign[["family", "superfamily", "folderror"]] = None
    results_ssalign = df_ssalign.apply(
        lambda row: tp_fp(row["File1"],same_family_files,same_superfamily_files,same_fold_files, row["Avg_TM_Score"]),
        axis=1,
        result_type="expand")
    # 将结果赋值到对应列
    df_ssalign[["family", "superfamily", "folderror"]] = results_ssalign

    df_ssalign.to_csv(f"/data2/zxc_data/foldseek_database/foldseek_database/Scope40/gitdata/SSAlign/SVD{dim}/new05/bio_{basename}_lower_global.csv", index=False)






"""
现在，正确的定义变了
每个文件大概有多少个正确的呢？
这个应该看tmalign啊

"""
def acc_true_count(basenames):

    family_true_count_list = []
    superfamily_true_count_list = []
    folderror_list = []

    for basename in basenames:
        # tmalign_file_path = f"/data2/zxc_data/foldseek_database/foldseek_database/Scope40/gitdata/tmalign/new/{basename}.result"
        tmalign_file_path = f"/data2/zxc_data/foldseek_database/foldseek_database/Scope40/gitdata/tmalign/new05/{basename}.result"


        df_tmalign = pd.read_csv(tmalign_file_path)

        # 统计family字段为1的个数
        family_count = len(df_tmalign[df_tmalign['family'] == 1])
        superfamily_count = len(df_tmalign[df_tmalign['superfamily'] == 1])
        folderror_count = len(df_tmalign[df_tmalign['folderror'] == 1])

        family_true_count_list.append(family_count)
        superfamily_true_count_list.append(superfamily_count)
        folderror_list.append(folderror_count)


    np.savez(
        "tmalign_true_counts_results05.npz",
        family_true=family_true_count_list, # 平均3000多个啊
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


def acc_tmscore05_true_count(basenames):

    true_count_list = []


    for basename in basenames:
        tmalign_file_path = f"/data2/zxc_data/foldseek_database/foldseek_database/Scope40/gitdata/tmalign/new/{basename}.result"

        df_tmalign = pd.read_csv(tmalign_file_path)

        # 统计family字段为1的个数
        count = len(df_tmalign[df_tmalign['Avg_TM_Score'] >= 0.5])

        true_count_list.append(count)

    np.savez("tmalign_tmscore05_counts_results.npz",tmscore05 = true_count_list)

    data = np.load("tmalign_tmscore05_counts_results.npz")
    data05 = data["tmscore05"]
    print(f"一共有这么多文件：",len(data05))
    print(f"tmscore >= 0.5的，平均每个文件有：",np.mean(data05))
    """
    一共有这么多文件： 11198
tmscore >= 0.5的，平均每个文件有： 60.76915520628684

tmscore >= 0.3的，平均每个文件有：2903 
    """





def acc_true_count_load():
    data = np.load("tmalign_true_counts_results.npz")
    family_counts = data["family_true"]
    superfamily_counts = data["superfamily_true"]
    folderror_counts = data["folderror"]

    print("Family counts:", family_counts)
    print("Family length:", len(family_counts))
    print("Family avg:",np.mean(family_counts))


    print("Superfamily counts:", superfamily_counts)
    print("folderror counts:", folderror_counts)

    """
Family counts: [2407 3791 1380 ... 3845 3674 3799]
Family length: 11198
Family avg: 2902.4532059296303

60


Superfamily counts: [2408 3791 1382 ... 3845 3674 3799]
folderror counts: [8803 7420 9829 ... 7366 7528 7412]
"""






if __name__=="__main__":

    parser = argparse.ArgumentParser(description="添加正确、错误指标")
    parser.add_argument("--dim", type=int,
                        help="ssalign的维度，因为其他的都弄好了")
    args = parser.parse_args()

    basenames = []
    file_dir = "/data2/zxc_data/foldseek_database/foldseek_database/Scope40/pdb"


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


        add_csv_tp_fp(basename,args.dim)
        print(f"蛋白质{basename}，处理完毕: {count}/{len(foldseek_basenames)}")
        count += 1


    # # acc_true_count(foldseek_basenames)


    # acc_true_count_load()

    # acc_tmscore05_true_count(foldseek_basenames)

    # acc_true_count(foldseek_basenames)


