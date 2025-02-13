import csv
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import argparse
from Scope40_workflow import ssalign
from scipy.interpolate import make_interp_spline
from multiprocessing import Pool

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
    family = file_scop['family']  # 获取 superfamily 层级

    # print(superfamily)

    # 筛选所有同一 superfamily 的文件
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
    family = file_scop['family']  # 获取 superfamily 层级

    # print(superfamily)

    # 筛选所有同一 superfamily 的文件
    same_superfamily_files = []

    for index, row in df.iterrows():
        current_scop = get_scop_levels(row['SCOP_Level'])
        # 比较 superfamily 层级
        if current_scop['family'] == family:
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




def PR_foldseek(basenames,mode):
    all_recall = []
    all_precision = []


    for basename in basenames:

        # 同一超家族 ---- 正确TP
        if mode == 'family':
            same_superfamily_files = group_files_by_family(basename)
        if mode == 'superfamily':
            same_superfamily_files = group_files_by_superfamily(basename)



        #  File1,File2,TM-Score1,TM-Score2,Aligned Length,RMSD,Seq_ID,FoldSeek_Metrics
        foldseek_file_path = f"../data/result/Scope40/foldseek/{basename}.result"
        df_foldseek = pd.read_csv(foldseek_file_path)

        # print(f"foldseek查询到了：{len(df_foldseek)}")

        all_right = len(same_superfamily_files)

        # print(f"同一个超家族的一共有：{all_right}")

        recall = []
        precision = []

        TP = 0

        count = 0
        # 每一个都判断，更新 recall 和  precision
        for index, row in df_foldseek.iterrows():
            count += 1
            file1 = row['File1']  #
            if file1 in same_superfamily_files:  # 查询到的同一超家族，那就是正确
                TP += 1

            recall.append(TP/all_right)
            precision.append(TP/count)

        all_recall.append(recall)
        all_precision.append(precision)


def PR_ssalign(basenames,dim,cos_threshold,mode):
    all_recall = []
    all_precision = []
    for basename in basenames:
        if mode == 'family':
            same_superfamily_files = group_files_by_family(basename)
        if mode == 'superfamily':
            same_superfamily_files = group_files_by_superfamily(basename)

        df_ssalign = ssalign(dim,basename,2000,cos_threshold,500)

        all_right = len(same_superfamily_files)

        recall = []
        precision = []

        TP = 0
        count = 0
        # 每一个都判断，更新 recall 和  precision
        for index, row in df_ssalign.iterrows():
            count += 1
            file1 = row['File1']  #
            if file1 in same_superfamily_files:  # 查询到的同一超家族，那就是正确
                TP += 1

            recall.append(TP / all_right)
            precision.append(TP / count)

        all_recall.append(recall)
        all_precision.append(precision)

    return all_recall, all_precision

def PR_tmalign(basenames,mode):
    all_recall = []
    all_precision = []

    for basename in basenames:
        if mode == 'family':
            same_superfamily_files = group_files_by_family(basename)
        if mode == 'superfamily':
            same_superfamily_files = group_files_by_superfamily(basename)

        #  File1,File2,TM-Score1,TM-Score2,Aligned Length,RMSD,Seq_ID,FoldSeek_Metrics
        tmalign_file_path = f"../data/result/Scope40/tmalign/{basename}.result"
        df_tmalign = pd.read_csv(tmalign_file_path)

        df_tmalign['Avg_TM_Score'] = (df_tmalign['TM-Score1'] + df_tmalign['TM-Score2'])/2

        df_tmalign_sorted = df_tmalign.sort_values(by='Avg_TM_Score',ascending=False)

        all_right = len(same_superfamily_files)

        recall = []
        precision = []

        TP = 0
        count = 0

        for index, row in df_tmalign_sorted.iterrows():
            count += 1
            File1 = row['File1']  # 假设 'File1' 是列名，根据需要调整
            file1 = os.path.basename(File1)
            if file1 in same_superfamily_files:
                TP += 1
            recall.append(TP / all_right)
            precision.append(TP / count)

        all_recall.append(recall)
        all_precision.append(precision)

    return all_recall, all_precision



def PR_prefilter(basenames,dim,topk,mode):
    all_recall = []
    all_precision = []

    for basename in basenames:
        if mode == 'family':
            same_superfamily_files = group_files_by_family(basename)
        if mode == 'superfamily':
            same_superfamily_files = group_files_by_superfamily(basename)

        fiass_file_path = f"../data/result/Scope40/SSAlign/SVD{dim}/bio_{basename}_lower_global.csv"

        df_faiss = pd.read_csv(fiass_file_path)

        df_faiss_fraction = df_faiss.sort_values(by='Cosine_Similarity',ascending=False).head(topk)


        all_right = len(same_superfamily_files)
        recall = []
        precision = []
        TP = 0
        count = 0

        for index, row in df_faiss_fraction.iterrows():
            count += 1
            file1 = row['File1']
            if file1 in same_superfamily_files:
                TP += 1
            recall.append(TP / all_right)
            precision.append(TP / count)

        all_recall.append(recall)
        all_precision.append(precision)

    return all_recall, all_precision


def calculate_average_recall_precision(all_recall, all_precision):
    # 找到最大长度
    max_length = max(len(recall) for recall in all_recall)

    # 对 recall 和 precision 列表进行填充
    padded_recall = []
    for recall in all_recall:
        # 复制原始列表
        padded = recall.copy()
        # 计算需要填充的数量
        while len(padded) < max_length:
            padded.append(padded[-1])  # 用最后一个值填充
        padded_recall.append(padded)

    padded_precision = []
    for precision in all_precision:
        # 复制原始列表
        padded = precision.copy()
        # 计算需要填充的数量
        while len(padded) < max_length:
            padded.append(padded[-1])  # 用最后一个值填充
        padded_precision.append(padded)

    # print(padded_recall)
    # print(padded_precision)

    # 计算平均值
    avg_recall = np.mean(padded_recall, axis=0)
    avg_precision = np.mean(padded_precision, axis=0)

    return avg_recall, avg_precision


def benckamrk_main_1(dim,cos_threshold,mode):
    basenames = []
    file_dir = "../data/pdb/Scope40"
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            basenames.append(file)

            # 这13个，foldseek结果为空
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

    foldseek_recall, foldseek_precision = PR_foldseek(foldseek_basenames,mode)
    for i in range(0, 13):
        foldseek_recall.append([0])
        foldseek_precision.append([0])
    foldseek_avg_recall, foldseek_avg_precision = calculate_average_recall_precision(foldseek_recall,foldseek_precision)

    ssalign_recall, ssalign_precision = PR_ssalign(basenames, dim, cos_threshold,mode)
    ssalign_avg_recall, ssalign_avg_precision = calculate_average_recall_precision(ssalign_recall, ssalign_precision)

    tmalign_recall, tmalign_precision = PR_tmalign(basenames,mode)
    tmalign_avg_recall, tmalign_avg_precision = calculate_average_recall_precision(tmalign_recall, tmalign_precision)


    faiss_recall, faiss_precision = PR_prefilter(basenames, dim, 500,mode)
    faiss_avg_recall, faiss_avg_precision = calculate_average_recall_precision(faiss_recall, faiss_precision)

    plt.figure(figsize=(10, 6), dpi=600)  # 设置高分辨率

    # 绘制折线图
    plt.plot(foldseek_avg_recall, foldseek_avg_precision, color='green', label='Foldseek ', linewidth=1)
    plt.plot(ssalign_avg_recall, ssalign_avg_precision, color='blue', label='SSAlign ', linewidth=1)
    plt.plot(tmalign_avg_recall, tmalign_avg_precision, color='red', label='Tmalign ', linewidth=1)
    plt.plot(faiss_avg_recall, faiss_avg_precision, color='orange', label='ssalign-prefilter-500 ', linewidth=1)

    # 添加标签、标题和图例
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Family', fontsize=14)
    plt.legend(loc='best', fontsize=12, frameon=True)  # 设置图例位置为最优，添加边框

    plt.tight_layout()  # 自动调整布局避免文字被截断
    plt.savefig(f"PR_{mode}.png",dpi=600)
    plt.legend()
    plt.close()






def firstFP_foldseek(basenames,query_fraction,mode):


    sum_sen = 0

    for basename in basenames:

        # 同一超家族 ---- 正确TP
        if mode == 'family':
            same_superfamily_files = group_files_by_family(basename)
        if mode == 'superfamily':
            same_superfamily_files = group_files_by_superfamily(basename)
        # 同一折叠
        same_fold_files = group_same_fold_files(basename)

        #  File1,File2,TM-Score1,TM-Score2,Aligned Length,RMSD,Seq_ID,FoldSeek_Metrics
        foldseek_file_path = f"../data/result/Scope40/foldseek/{basename}.result"
        df_foldseek = pd.read_csv(foldseek_file_path)

        # 取出 前 query_fraction 比例
        df_foldseek_fraction = df_foldseek.head(int(len(df_foldseek)*query_fraction))


        TP = [] #  TP,   同一超家族中的匹配项是正确的
        FP = []  # FP    不同折叠之间的匹配是错误的

        # 有几个foldseek是无法查询的
        if len(df_foldseek_fraction) == 0:
            sum_sen += 0
        else:
            # 只选出是同一个折叠的！
            for index, row in df_foldseek_fraction.iterrows():
                file1 = row['File1']  # 假设 'File1' 是列名，根据需要调整
                if file1 not in same_fold_files: # 查询到的不是同一折叠的 ，直接退出了，只算这之前的数据。所以一定是同一个折叠

                    break # 遇到第一个就直接退出 1st FP。写在这里还是前面？
                # 那么下面就全是同一个折叠的
                if file1 in same_superfamily_files: # 查询到的同一超家族，TP
                    TP.append(file1)
                else:
                    FP.append(file1)

            if len(TP) == 0:
                sen = 0
            else:
                sen = len(TP) / (len(TP)+len(FP))
                # sen = len(TP) / len(df_foldseek_fraction)

            sum_sen += sen

    return sum_sen/len(basenames)


def firstFP_ssalign(basenames,dim,cos_threshold,query_fraction,mode):

    sum_sen = 0

    for basename in basenames:

        if mode == 'family':
            same_superfamily_files = group_files_by_family(basename)
        if mode == 'superfamily':
            same_superfamily_files = group_files_by_superfamily(basename)
        # 同一折叠
        same_fold_files = group_same_fold_files(basename)

        # ssalign(dim,basename,faiss_topk,cos_threshold,final_number)
        df_ssalign = ssalign(dim,basename,2000,cos_threshold,500)

        # 取出 前 query_fraction 比例
        df_ssalign_fraction = df_ssalign.head(int(len(df_ssalign)*query_fraction))


        TP = [] #  TP,   同一超家族中的匹配项是正确的
        FP = []  # FP    不同折叠之间的匹配是错误的

        # 只选出是同一个折叠的！
        for index, row in df_ssalign_fraction.iterrows():
            file1 = row['File1']  # 假设 'File1' 是列名，根据需要调整
            if file1 not in same_fold_files: # 查询到的不是同一折叠的 ，直接退出了，只算这之前的数据。所以一定是同一个折叠
                break # 遇到第一个就直接退出 1st FP。写在这里还是前面？
            # 那么下面就全是同一个折叠的
            if file1 in same_superfamily_files: # 查询到的同一超家族，TP
                TP.append(file1)
            else:
                FP.append(file1)

        if len(TP) == 0:
            sen = 0
        else:
            sen = len(TP) / (len(TP)+len(FP))
            # sen = len(TP) / len(df_ssalign_fraction)

        sum_sen += sen

    return sum_sen/len(basenames)






def firstFP_tmalign(basenames,query_fraction,mode):
    sum_sen = 0

    for basename in basenames:

        # 同一超家族 ---- 正确TP
        if mode == 'family':
            same_superfamily_files = group_files_by_family(basename)
        if mode == 'superfamily':
            same_superfamily_files = group_files_by_superfamily(basename)        # 同一折叠

        same_fold_files = group_same_fold_files(basename)

        #  File1,File2,TM-Score1,TM-Score2,Aligned Length,RMSD,Seq_ID,FoldSeek_Metrics

        tmalign_file_path = f"../data/result/Scope40/tmalign/{basename}.result"
        df_tmalign = pd.read_csv(tmalign_file_path)

        df_tmalign['Avg_TM_Score'] = (df_tmalign['TM-Score1'] + df_tmalign['TM-Score2'])/2


        df_tmalign_sorted = df_tmalign.sort_values(by='Avg_TM_Score',ascending=False)

        # 取出 前 query_fraction 比例
        df_tmalign_fraction = df_tmalign_sorted.head(int(len(df_tmalign_sorted)*query_fraction))


        TP = [] #  TP,   同一超家族中的匹配项是正确的
        FP = []  # FP    不同折叠之间的匹配是错误的


        # 只选出是同一个折叠的！
        for index, row in df_tmalign_fraction.iterrows():
            File1 = row['File1']  # 假设 'File1' 是列名，根据需要调整
            file1 = os.path.basename(File1)
            if file1 not in same_fold_files: # 查询到的不是同一折叠的 ，直接退出了，只算这之前的数据。所以一定是同一个折叠

                break # 遇到第一个就直接退出 1st FP。写在这里还是前面？
            # 那么下面就全是同一个折叠的
            if file1 in same_superfamily_files: # 查询到的同一超家族，TP
                TP.append(file1)
            else:
                FP.append(file1)

        if len(TP) == 0:
            sen = 0
        else:
            sen = len(TP) / (len(TP)+len(FP))
            # sen = len(TP) / len(df_tmalign_fraction)

        sum_sen += sen

    return sum_sen/len(basenames)



def firstFP_faiss(basenames,dim,topk,query_fraction,mode):
    sum_sen = 0

    for basename in basenames:

        # 同一超家族 ---- 正确TP
        if mode == 'family':
            same_superfamily_files = group_files_by_family(basename)
        if mode == 'superfamily':
            same_superfamily_files = group_files_by_superfamily(basename)
        # 同一折叠
        same_fold_files = group_same_fold_files(basename)

        fiass_file_path = f"../data/result/Scope40/SSAlign/SVD{dim}/bio_{basename}_lower_global.csv"

        df_faiss = pd.read_csv(fiass_file_path)

        df_faiss_fraction = df_faiss.sort_values(by='Cosine_Similarity',ascending=False).head(int(topk*query_fraction))

        TP = []  # TP,   同一超家族中的匹配项是正确的
        FP = []  # FP    不同折叠之间的匹配是错误的

        # 只选出是同一个折叠的！
        for index, row in df_faiss_fraction.iterrows():
            file1 = row['File1']  # 假设 'File1' 是列名，根据需要调整
            if file1 not in same_fold_files:  # 查询到的不是同一折叠的 ，直接退出了，只算这之前的数据。所以一定是同一个折叠
                break  # 遇到第一个就直接退出 1st FP。写在这里还是前面？
            # 那么下面就全是同一个折叠的
            if file1 in same_superfamily_files:  # 查询到的同一超家族，TP
                TP.append(file1)
            else:
                FP.append(file1)

        if len(TP) == 0:
            sen = 0
        else:
            sen = len(TP) / (len(TP) + len(FP))
            # sen = len(TP) / len(df_faiss_fraction)

        sum_sen += sen


    return sum_sen / len(basenames)




def benckamrk_main_2(dim,cos_threshold,mode):

    basenames = []
    file_dir = "../data/pdb/Scope40/"

    for root, dirs, files in os.walk(file_dir):
        for file in files:
            basenames.append(file)
    foldseek_results = []
    ssalign_results = []
    tmalign_results = []
    faiss500_result = []
    params_result = []

    params = np.arange(0, 1.01, 0.01)
    for param in params:
        foldseek_results.append(firstFP_foldseek(basenames, param,mode))
        ssalign_results.append(firstFP_ssalign(basenames, dim, cos_threshold, param,mode))
        tmalign_results.append(firstFP_tmalign(basenames, param,mode))
        faiss500_result.append(firstFP_faiss(basenames, dim, 500, param,mode))

    #  修改列表的第一个元素为1
    foldseek_results[0] = 1
    ssalign_results[0] = 1
    tmalign_results[0] = 1
    faiss500_result[0] = 1

    plt.figure(figsize=(10, 6), dpi=600)  # 设置高分辨率

    # 绘制折线图
    plt.plot(params, foldseek_results, color='green', label='Foldseek', linewidth=1)
    plt.plot(params, ssalign_results, color='blue', label='SSAlign', linewidth=1)
    plt.plot(params, tmalign_results, color='red', label='Tmalign', linewidth=1)
    plt.plot(params, faiss500_result, color='orange', label='SSAlign-prefilter-500', linewidth=1)

    # 添加标签、标题和图例
    plt.xlabel('Fraction of queries')
    plt.ylabel('Sensitvity up to the 1st FP')
    plt.title('Family', fontsize=14)
    plt.legend(loc='best', fontsize=12, frameon=True)  # 设置图例位置为最优，添加边框
    plt.grid(True, linestyle='--', alpha=0.5)  # 添加网格线以提升可读性
    plt.ylim(0, 1)
    plt.tight_layout()  # 自动调整布局避免文字被截断
    plt.savefig(f"AOC_{mode}.png",dpi=600)
    plt.legend()
    plt.close()


if __name__=="__main__":
    benckamrk_main_1(1280,0.2,'family')

