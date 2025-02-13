import os
import numpy as np
import pandas as pd
import csv
import faiss


def load_vectors_from_file_and_queryvector(vector_file_path,index, random_filenames,dim):
    """
    index 是每一行中 向量的索引  这里是1  (whitening之后的格式，和原来的文件有一点差别：省略掉了seq)

    情况一：还是取第33层来处理
    从 /data/foldseek_database/sp_cif_file/swissprot_cif_v4_files_results文件中加载文件名和向量


    index是指 向量的位置  因为之前是  文件名,seq,向量
                        whitening之后是   文件名,向量
    """
    all_filenames=[]
    all_vectors=[]
    query_vectors = []
    query_filenames = []


    with open(vector_file_path, 'r') as f:
        for line in f:
            # 解析文件每一行的内容 [文件名,序列,[向量数据]

            #也可以这样 ： parts = line.strip().split(",", 1)  # 仅分割一次，以避免向量数据被错误分割
            parts = line.strip().split(",")
            filename = parts[0]  # 文件名,绝对路径
            base_filename = os.path.basename(filename)  # 提取文件名

            # print(base_filename)

            # sequence = parts[1]  # 提取序列
            vector_str = ",".join(parts[index:])  # 获取向量字符串
            vector = np.array(eval(vector_str))[:,:dim]  # 将字符串转换为numpy array,并且指定维度

            all_filenames.append(filename)
            all_vectors.append(vector)

            # 如果当前文件名在查询列表中，则保存为查询向量
            if base_filename in random_filenames:
                # print(filename)
                # print(vector)
                query_filenames.append(filename)
                query_vectors.append(vector)

    # 转换为numpy数组，方便FAISS使用
    all_vectors = np.vstack(all_vectors).astype('float32')# FAISS要求float32类型
    query_vectors = np.vstack(query_vectors).astype('float32')


    return all_filenames, all_vectors, query_filenames,query_vectors



def build_faiss_index_IP_gpu(vectors,gpu_id,dim):
    """
    构建FAISS  IP 索引并添加向量

    ---- 由于是归一化之后的，所以就是内积
    """

    # d = SVD1280  # 向量的维度

    index = faiss.IndexFlatIP(dim)  # 使用 内积 构建平坦索引

    # # 将索引移动到 GPU（这里假设使用 GPU 0）
    res = faiss.StandardGpuResources()  # 初始化 GPU 资源

    index_gpu = faiss.index_cpu_to_gpu(res, gpu_id, index)
    #
    # # 将数据添加到 GPU 索引
    #
    index_gpu.add(vectors)  # 添加所有向量

    return index_gpu

def search_similar_vectors(index, query_vector, top_k):

    # 查询最相似的top_k个向量
    D, I = index.search(query_vector, top_k)
    return D, I  # 返回距离和索引

def save_to_file(output_file, result):
    """
    保存到文件中
    """
    with open(output_file, 'a') as f:

        # 将vector转换为列表形式存储
        # print(f"正在处理{result[0]}")
         f.write(f"{result[0]},{result[1]}\n")


def faiss_align_vector(index, filenames, query_vector, top_k, result_file_path):
    # index.nprobe = 50  #
    distances, indices = search_similar_vectors(index, query_vector, top_k)

    for i in range(top_k):
        # 写结果，相对路径
        file_name = os.path.basename(filenames[indices[0][i]])

        print(distances[0][i].dtype)
        print(distances[0][i])
        save_to_file(result_file_path,(file_name,distances[0][i]))


def load_squeue(file_path,fullname_list,target_fullname,charset):

    target_sequence_data = {}

    sequence_data = {}

    with open(file_path, 'r') as f:
        for line in f:
            # 解析文件每一行的内容 [文件名,序列,[向量数据]
            parts = line.strip().split(",")
            filename = parts[0]  # 文件名,绝对路径
            sequence = parts[1]  # 序列


            if charset == "upper":
                # 只保留大写字母部分
                sequence_upper = ''.join([char for char in sequence if char.isupper()])
            if charset == "lower":
                # 只保留3Di部分
                sequence_upper = ''.join([char.upper() for char in sequence if char.islower()])
            if charset == "all":
                # 全部保存，序列+3Di
                sequence_upper = ''.join([char.upper() for char in sequence])

            # 如果是目标文件，单独保存
            if filename == target_fullname:
                target_sequence_data[filename] = sequence_upper
                #(filename)
            # 其他文件根据列表进行分组
            if filename in fullname_list:
                sequence_data[filename] = sequence_upper
                #print(filename)

            # sequence = parts[1]
    return target_sequence_data,sequence_data


def add_prefilter_tmscore(faiss_file,tmalign_file,output_file,basename,dim):

    tmalign_dict = {}
    try:
        with open(tmalign_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader)  # 跳过表头
            for row in reader:
                file1_name = row[0].split("/")[-1]  # 提取 File1 的文件名
                file2_name = row[1].split("/")[-1]  # 提取 File2 的文件名
                metrics = row[2:]  # 提取 TM-align 指标
                tmalign_dict[file1_name] = metrics
    except Exception as e:
        print(f"读取 TM-align 文件时出错: {e}")
        return

    # 读取 FoldSeek 文件并补充 TM-align 数据
    merged_data = []

    try:
        with open(faiss_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) < 2:
                    continue
                file_name = parts[0]  # Faiss 文件名
                cosine_similarity = parts[1]  # Cosine Similarity

                # 获取 TM-align 数据
                tmalign_metrics = tmalign_dict.get(file_name, ["N/A"] * 5)

                # 合并数据
                merged_row = [file_name,basename ] + tmalign_metrics + [cosine_similarity]
                merged_data.append(merged_row)
    except Exception as e:
        print(f"读取 Faiss 文件时出错: {e}")
        return

  # 写入合并后的数据到输出文件
    try:
        with open(output_file, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            header = ["File1", "File2", "TM-Score1", "TM-Score2", "Aligned Length", "RMSD", "Seq_ID", "Cosine_Similarity"]
            writer.writerow(header)
            writer.writerows(merged_data)
        print(f"成功生成文件: {output_file}")
    except Exception as e:
        print(f"写入输出文件时出错: {e}")


"""
和tmalign结果对应，补充分数
"""
def add_foldseek_tmscore(foldseek_file,tmalign_file,output_file,basename):

    # 读取 TM-align 文件，创建 {File2: (TM-Score1, TM-Score2, Aligned Length, RMSD, Seq_ID)} 映射
    tmalign_dict = {}
    try:
        with open(tmalign_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader)  # 跳过表头
            for row in reader:
                file1_name = row[0].split("/")[-1]  # 提取 File2 的文件名
                metrics = row[2:]  # 提取 TM-align 指标
                tmalign_dict[file1_name] = metrics
    except Exception as e:
        print(f"读取 TM-align 文件时出错: {e}")
        return

    # 读取 FoldSeek 文件并补充 TM-align 数据
    merged_data = []
    try:
        with open(foldseek_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                file1_name = parts[0]  # FoldSeek 第一列
                file2_name = parts[1]  # FoldSeek 第二列
                foldseek_metrics = " ".join(parts[2:])  # FoldSeek 指标

                # 获取 TM-align 数据
                tmalign_metrics = tmalign_dict.get(file2_name, ["N/A"] * 5)

                # 合并数据
                merged_row = [ file2_name,file1_name] + tmalign_metrics + [foldseek_metrics]
                merged_data.append(merged_row)
    except Exception as e:
        print(f"读取 FoldSeek 文件时出错: {e}")

    # 写入合并后的数据到输出文件
    try:
        with open(output_file, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            header = ["File1", "File2", "TM-Score1", "TM-Score2", "Aligned Length", "RMSD", "Seq_ID", "FoldSeek_Metrics"]
            writer.writerow(header)
            writer.writerows(merged_data)
        print(f"成功生成文件: {output_file}")
    except Exception as e:
        print(f"写入输出文件时出错: {e}")