from utils.foldseek_util import get_struc_seq
from utils.esm_loader import load_esm_saprot
import torch
import numpy as np
import csv
import os
import pandas as pd
from whitening.whitening_model import WhiteningModel

"""
Convert 11,211 pdb files in the scope40 database to
Use foldseek_util.py to calculate the 3Di sequence ---- which is actually combined_seq
Then use Sport to calculate its vector

Save everything in a single file in the following format:
    File name, combined_seq, vector
"""

scope40_dir = "../data/pdb/Swissport/"
Saport_model_path = "../models/SaProt_650M_AF2.pt"
foldseek_path = "../bin/foldseek"

cuda_device = "cuda:6"

model, alphabet = load_esm_saprot(Saport_model_path)
model = model.to(cuda_device)
batch_converter = alphabet.get_batch_converter()

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
    use foldseek_util.py
    """

    # print(f"正在处理{file_full_path}")
    parsed_seqs = get_struc_seq(foldseek_path, file_full_path, plddt_mask=False)

    key, (seq, foldseek_seq, combined_seq) = next(iter(parsed_seqs.items()))

    #seq, foldseek_seq, combined_seq = parsed_seqs["A"]
    return combined_seq


def combined_seq_to_vector(file_full_path, combined_seq):
    """
    Use Saprot to convert the combined_seq sequence into a vector representation.
    """
    data = [(file_full_path, combined_seq)]

    torch.set_printoptions(sci_mode=False, threshold=5000)

    batch_labels, batch_strs, batch_tokens = batch_converter(data)

    with torch.no_grad():  
        batch_tokens = batch_tokens.to(cuda_device)
        results = model(batch_tokens, repr_layers=[33], return_contacts=False)

    token_representations = results["representations"][33][:, 1:-1, :].mean(1)

    np.set_printoptions(suppress=True, threshold=5000) 

    return token_representations.cpu().numpy()



def process_file(file_full_path):

    # This may occur here. Saport and foldseek may be unable to process a certain file, which will be directly marked as an error.    try:
        # 获取combined_seq
        combined_seq = file_to_combined_seq(file_full_path)
        # After foldseek processing,
        # amino acid sequence: missing amino acids are represented by X
        # 3Di sequences may also appear as x
        # all replaced with #
        combined_seq = combined_seq.replace("X", "#").replace("x", "#")

        # 获取vector
        vector = combined_seq_to_vector(file_full_path, combined_seq)
        return file_full_path, combined_seq, vector
    except Exception as e:
        print(f"文件处理错误：{file_full_path}：{e}")
        return file_full_path,"error","error"


def save_to_file(output_file, result):
    """
    Save the processed results (file name, combined_seq, vector) to a file.
    """
    with open(output_file, 'a') as f:
        combined_seq,vector = result[1],result[2]

        if combined_seq == "error" or (isinstance(vector,str) and vector == "error"):
            f.write(f"{result[0]},error,error\n")
        # print(f"处理完成 {result[0]}")
        else:
            f.write(f"{result[0]},{result[1]},{result[2].tolist()}\n")


def main(full_dir, output_file):
    all_files = list_all_files(full_dir)

    #all_files = all_files[:]

    for file_full_path in all_files:

        file_full_path, combined_seq, vector = process_file(file_full_path)
        save_to_file(output_file, (file_full_path, combined_seq, vector))



def whitening():

    # Initialize WhiteningProcessor
    vector_index = 2  # Vector index in the data (e.g., third column)
    batch_size = 1000  # Batch size for each processing
    mu_filename = "../data/result/Swissport/sp_whitening_mu.npy"
    W_filename = "../data/result/Swissport/sp_whitening_W.npy"



    processor = WhiteningModel(vector_index, batch_size, mu_filename, W_filename)

    input_file = '../data/result/Swissport/swissprot_cif_v4_files_results'
    output_file = '../data/result/Swissport/swissprot_cif_v4_files_results_whitening'

    processor.process_file_incremental(input_file, output_file)





if __name__ == "__main__":

    output_file = '../data/result/Swissport/swissprot_cif_v4_files_results'

    main(scope40_dir, output_file)

    whitening()
