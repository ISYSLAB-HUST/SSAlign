from utils.foldseek_util import get_struc_seq
from utils.esm_loader import load_esm_saprot
import torch
import numpy as np
import csv
import os
import pandas as pd
from whitening.whitening_model import WhiteningModel

"""
将scope40数据库中11211个pdb文件转为
使用foldseek_util.py来计算3Di序列 ---- 其实是combined_seq
然后再使用Sport计算出它的向量

全部保存在一个文件中，格式：
    文件名,combined_seq,vector
"""

scope40_dir = "../data/pdb/Scope40/"  # 原始文件
Saport_model_path = "../models/SaProt_650M_AF2.pt"  # SaProt模型的路径
foldseek_path = "../bin/foldseek"

cuda_device = "cuda:6"

model, alphabet = load_esm_saprot(Saport_model_path)  # 加载模型
model = model.to(cuda_device)
batch_converter = alphabet.get_batch_converter()  # 获取批量转换器

def list_all_files(directory):
    file_list = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            full_path = os.path.join(root, file)
            file_list.append(full_path)
    # print(file_list)
    return file_list


def file_to_combined_seq(file_full_path):
    """
    使用foldseek_util.py来计算3Di序列 ---- 其实是combined_seq
    """

    # print(f"正在处理{file_full_path}")
    parsed_seqs = get_struc_seq(foldseek_path, file_full_path, plddt_mask=False)

    key, (seq, foldseek_seq, combined_seq) = next(iter(parsed_seqs.items()))

    #seq, foldseek_seq, combined_seq = parsed_seqs["A"]  # 只提取A链
    return combined_seq


def combined_seq_to_vector(file_full_path, combined_seq):
    """
    将combined_seq序列 使用Saprot转为 为向量表示
    """
    data = [(file_full_path, combined_seq)]

    # 禁用科学计数法
    torch.set_printoptions(sci_mode=False, threshold=5000)

    # 使用批量转化器，将蛋白质序列转为模型的所需的输入格式
    batch_labels, batch_strs, batch_tokens = batch_converter(data)

    with torch.no_grad():  # 禁止梯度计算
        batch_tokens = batch_tokens.to(cuda_device)
        results = model(batch_tokens, repr_layers=[33], return_contacts=False)

    # 提取第 33 层的输出表示，并且处理，计算平均值，获取句向量
    token_representations = results["representations"][33][:, 1:-1, :].mean(1)

    np.set_printoptions(suppress=True, threshold=5000)  # 禁用科学计数法

    return token_representations.cpu().numpy()



def process_file(file_full_path):

    # 这里可能出现。会出现 Saport和foldseek无法处理某个文件，直接标记为error
    try:
        # 获取combined_seq
        combined_seq = file_to_combined_seq(file_full_path)
        # foldseek处理之后，
        # 氨基酸序列：缺失氨基酸的位置会使用X来表示
        # 3Di序列也可能出现 x
        # 全部使用 # 来代替
        combined_seq = combined_seq.replace("X", "#").replace("x", "#")

        # 获取vector
        vector = combined_seq_to_vector(file_full_path, combined_seq)
        return file_full_path, combined_seq, vector
    except Exception as e:
        print(f"文件处理错误：{file_full_path}：{e}")
        return file_full_path,"error","error"


def save_to_file(output_file, result):
    """
    将处理后的结果（文件名、combined_seq、vector）保存到文件中。
    """
    with open(output_file, 'a') as f:
        combined_seq,vector = result[1],result[2]

        if combined_seq == "error" or (isinstance(vector,str) and vector == "error"):
            f.write(f"{result[0]},error,error\n")
        # 将vector转换为列表形式存储
        # print(f"处理完成 {result[0]}")
        else:
            f.write(f"{result[0]},{result[1]},{result[2].tolist()}\n")


def main(full_dir, output_file):
    # 列出所有的文件
    all_files = list_all_files(full_dir)

    #all_files = all_files[:]

    for file_full_path in all_files:

        file_full_path, combined_seq, vector = process_file(file_full_path)
        save_to_file(output_file, (file_full_path, combined_seq, vector))



def whitening():

    # 初始化 WhiteningProcessor
    vector_index = 2  # 向量在数据中的索引（例如第三列）
    batch_size = 1000  # 每次处理的批次大小
    mu_filename = "../data/result/Scope40/scope40_whitening_mu.npy"
    W_filename = "../data/result/Scope40/scope40_whitening_W.npy"

    processor = WhiteningModel(vector_index, batch_size, mu_filename, W_filename)

    # 文件路径
    input_file = '../data/result/Scope40/scope40_vector_results'
    output_file = '../data/result/Scope40/scope40_vector_results_whitening'

    # 处理文件
    processor.process_file_incremental(input_file, output_file)





if __name__ == "__main__":

    output_file = "../data/result/Scope40/scope40_vector_results"

    main(scope40_dir, output_file)

    whitening()