"""
true ：same family/superfamily || avg_tmscore >= 0.5
false：diff fold && avg_tmscore < 0.5

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
                continue  # 

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

"""
def get_scop_levels(scopecode):
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

    file_scop_level = df[df['File'] == basename]['SCOP_Level']

    if file_scop_level.empty:
            return []

    file_scop = get_scop_levels(file_scop_level.iloc[0])
    family = file_scop['family']  # 

    same_family_files = []

    for index, row in df.iterrows():
        current_scop = get_scop_levels(row['SCOP_Level'])
        if current_scop['family'] == family:
            same_family_files.append(row['File'])

    return same_family_files


def group_files_by_superfamily(basename):
    tsv_file = "scop_lookup.fix.tsv"

    df = pd.read_csv(tsv_file, sep='\t', header=None, names=['File', 'SCOP_Level'])

    file_scop_level = df[df['File'] == basename]['SCOP_Level']

    if file_scop_level.empty:
            return []

    file_scop = get_scop_levels(file_scop_level.iloc[0])
    superfamily = file_scop['superfamily'] 

    same_superfamily_files = []

    for index, row in df.iterrows():
        current_scop = get_scop_levels(row['SCOP_Level'])
        if current_scop['superfamily'] == superfamily:
            same_superfamily_files.append(row['File'])

    return same_superfamily_files


def group_same_fold_files(basename):

    tsv_file = "scop_lookup.fix.tsv"
    df = pd.read_csv(tsv_file, sep='\t', header=None, names=['File', 'SCOP_Level'])

    file_scop_level = df[df['File'] == basename]['SCOP_Level']

    if file_scop_level.empty:
        return []

    file_scop = get_scop_levels(file_scop_level.iloc[0])
    fold = file_scop['fold']  

    same_fold_files = []

    for index, row in df.iterrows():
        current_scop = get_scop_levels(row['SCOP_Level'])
       
        if current_scop['fold'] == fold:
            same_fold_files.append(row['File'])

    return same_fold_files




"""

"""

def tp_fp(file,same_family_files,same_superfamily_files,same_fold_files,avg_tmscore):
    # same_family_files = group_files_by_family(file1)
    # same_superfamily_files = group_files_by_superfamily(file1)
    # same_fold_files = group_same_fold_files(file1)

    result=[]

    is_family_correct = (avg_tmscore >= 0.5) or (file in same_family_files)
    result.append(1 if is_family_correct else 0)

    is_superfamily_correct = (avg_tmscore >= 0.5) or (file in same_superfamily_files)
    result.append(1 if is_superfamily_correct else 0)

    is_error = (file not in same_fold_files) and (avg_tmscore < 0.5)
    result.append(1 if is_error else 0)

    return result







"""

"""
def add_csv_tp_fp(basename,dim):

    # for basename in basenames:
    # foldseek_file_path = f"/data2/zxc_data/foldseek_database/foldseek_database/Scope40/gitdata/foldseek/{basename}.result"
    # tmalign_file_path = f"/data2/zxc_data/foldseek_database/foldseek_database/Scope40/gitdata/tmalign/{basename}.result"

    ssalign_file_path = f"/data2/zxc_data/foldseek_database/foldseek_database/Scope40/gitdata/SSAlign/SVD{dim}/bio_{basename}_lower_global.csv"

    same_family_files = group_files_by_family(basename)
    same_superfamily_files = group_files_by_superfamily(basename)
    same_fold_files = group_same_fold_files(basename)


    # df_foldseek = pd.read_csv(foldseek_file_path)
    # df_foldseek['Avg_TM_Score'] = (df_foldseek['TM-Score1'] + df_foldseek['TM-Score2'])/2
    # df_foldseek[["family", "superfamily", "folderror"]] = None
    #
    # results_foldseek = df_foldseek.apply(
    #     lambda row: tp_fp(row["File1"],same_family_files,same_superfamily_files,same_fold_files, row["Avg_TM_Score"]),
    #     axis=1,
    #     result_type="expand")
    #
    # df_foldseek[["family", "superfamily", "folderror"]] = results_foldseek
    #
    # df_foldseek.to_csv(f"/data2/zxc_data/foldseek_database/foldseek_database/Scope40/gitdata/foldseek/new05/{basename}.result",index=False)
    #
    # ##############################################################################
    #
    # df_tmalign = pd.read_csv(tmalign_file_path)
    # df_tmalign['Avg_TM_Score'] = (df_tmalign['TM-Score1'] + df_tmalign['TM-Score2']) / 2
    # df_tmalign[["family", "superfamily", "folderror"]] = None

    # results_tmalign = df_tmalign.apply(
    #     lambda row: tp_fp(os.path.basename(row["File1"]),same_family_files,same_superfamily_files,same_fold_files, row["Avg_TM_Score"]),
    #     axis=1,
    #     result_type="expand")

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
    df_ssalign[["family", "superfamily", "folderror"]] = results_ssalign

    df_ssalign.to_csv(f"/data2/zxc_data/foldseek_database/foldseek_database/Scope40/gitdata/SSAlign/SVD{dim}/new05/bio_{basename}_lower_global.csv", index=False)






"""

"""
def acc_true_count(basenames):

    family_true_count_list = []
    superfamily_true_count_list = []
    folderror_list = []

    for basename in basenames:
        # tmalign_file_path = f"/data2/zxc_data/foldseek_database/foldseek_database/Scope40/gitdata/tmalign/new/{basename}.result"
        tmalign_file_path = f"/data2/zxc_data/foldseek_database/foldseek_database/Scope40/gitdata/tmalign/new05/{basename}.result"


        df_tmalign = pd.read_csv(tmalign_file_path)

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

        count = len(df_tmalign[df_tmalign['Avg_TM_Score'] >= 0.5])

        true_count_list.append(count)

    np.savez("tmalign_tmscore05_counts_results.npz",tmscore05 = true_count_list)

    data = np.load("tmalign_tmscore05_counts_results.npz")
    data05 = data["tmscore05"]
    print(f"一共有这么多文件：",len(data05))
    print(f"tmscore >= 0.5的，平均每个文件有：",np.mean(data05))
    """

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


