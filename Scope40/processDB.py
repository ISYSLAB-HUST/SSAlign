from utils.foldseek_util import get_struc_seq
from utils.esm_loader import load_esm_saprot
import torch
import numpy as np
import csv
import os
import pandas as pd
from whitening.whitening_model import WhiteningModel

"""
    filename,combined_seq,vector
"""

scope40_dir = "../data/pdb/Scope40/"  # 
Saport_model_path = "../models/SaProt_650M_AF2.pt"  # 
foldseek_path = "../bin/foldseek"

cuda_device = "cuda:6"

model, alphabet = load_esm_saprot(Saport_model_path)  # 
model = model.to(cuda_device)
batch_converter = alphabet.get_batch_converter()  # 

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
    foldseek_util.py combined_seq
    """

    # print(f"正在处理{file_full_path}")
    parsed_seqs = get_struc_seq(foldseek_path, file_full_path, plddt_mask=False)

    key, (seq, foldseek_seq, combined_seq) = next(iter(parsed_seqs.items()))

    #seq, foldseek_seq, combined_seq = parsed_seqs["A"]  # 
    return combined_seq


def combined_seq_to_vector(file_full_path, combined_seq):
    """
    """
    data = [(file_full_path, combined_seq)]

    # 
    torch.set_printoptions(sci_mode=False, threshold=5000)

    # 
    batch_labels, batch_strs, batch_tokens = batch_converter(data)

    with torch.no_grad():  # 
        batch_tokens = batch_tokens.to(cuda_device)
        results = model(batch_tokens, repr_layers=[33], return_contacts=False)

    token_representations = results["representations"][33][:, 1:-1, :].mean(1)

    np.set_printoptions(suppress=True, threshold=5000)  # 

    return token_representations.cpu().numpy()



def process_file(file_full_path):

    try:
        # 
        combined_seq = file_to_combined_seq(file_full_path)
        #
        combined_seq = combined_seq.replace("X", "#").replace("x", "#")

        # 
        vector = combined_seq_to_vector(file_full_path, combined_seq)
        return file_full_path, combined_seq, vector
    except Exception as e:
        print(f"文件处理错误：{file_full_path}：{e}")
        return file_full_path,"error","error"


def save_to_file(output_file, result):
    """
    """
    with open(output_file, 'a') as f:
        combined_seq,vector = result[1],result[2]

        if combined_seq == "error" or (isinstance(vector,str) and vector == "error"):
            f.write(f"{result[0]},error,error\n")
        # 
        else:
            f.write(f"{result[0]},{result[1]},{result[2].tolist()}\n")


def main(full_dir, output_file):
    # 
    all_files = list_all_files(full_dir)

    #all_files = all_files[:]

    for file_full_path in all_files:

        file_full_path, combined_seq, vector = process_file(file_full_path)
        save_to_file(output_file, (file_full_path, combined_seq, vector))



def whitening():

    # 
    vector_index = 2  # 
    batch_size = 1000  #
    mu_filename = "../data/result/Scope40/scope40_whitening_mu.npy"
    W_filename = "../data/result/Scope40/scope40_whitening_W.npy"

    processor = WhiteningModel(vector_index, batch_size, mu_filename, W_filename)

    # 
    input_file = '../data/result/Scope40/scope40_vector_results'
    output_file = '../data/result/Scope40/scope40_vector_results_whitening'

    # 处理文件
    processor.process_file_incremental(input_file, output_file)





if __name__ == "__main__":

    output_file = "../data/result/Scope40/scope40_vector_results"

    main(scope40_dir, output_file)

    whitening()
