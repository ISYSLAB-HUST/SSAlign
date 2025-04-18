import ast
import numpy as np
import torch
import faiss
import os
import pandas as pd
import argparse
import multiprocessing
from multiprocessing import Pool
from utils.esm_loader import load_esm_saprot
from utils.foldseek_util import get_struc_seq
from build_indexDB import get_protein_by_index_batch,get_protein_by_index_batch_ok
from saligner import saligner # 预编译的模块
import time
from concurrent.futures import ProcessPoolExecutor

# 常量定义
FOLDSEEK_PATH = "../bin/foldseek"
SAPROT_MODEL_PATH = "/home/xuchaozhang/ssalign/ssalign/SaprotProject/models/SaProt_650M_AF2.pt"
CUDA_DEVICE = "cuda:1"
MU_FILENAME = "/home/xuchaozhang/ssalign/ssalign/SaprotProject/Saprot/SaProt/whitening/data2/zxc_data/afdb50_combined_fasta/whitening/whitening_mu.npy"
W_FILENAME = "/home/xuchaozhang/ssalign/ssalign/SaprotProject/Saprot/SaProt/whitening/data2/zxc_data/afdb50_combined_fasta/whitening/whitening_W.npy"
LOOKUP_FILE = './ssalign_afdb50_combined_seq.lookup'
INDEX_FILE = './ssalign_afdb50_combined_seq.index'
SEQ_FILE = './ssalign_afdb50_combined_seq'

def generate_3di_sequences(file_full_path_list):
    """生成3Di序列"""
    foldseek_seqs = {}
    for file_full_path in file_full_path_list:
        parsed_seqs = get_struc_seq(FOLDSEEK_PATH, file_full_path, plddt_mask=False)
        key, (seq, foldseek_seq, combined_seq) = next(iter(parsed_seqs.items()))
        foldseek_seqs[file_full_path] = (foldseek_seq, combined_seq)
    return foldseek_seqs

def generate_saprot_embeddings(file_full_path_list,cuda_device,batch_size = 20):

    time1 = time.time()
    """使用SaProt生成蛋白质嵌入"""
    model, alphabet = load_esm_saprot(SAPROT_MODEL_PATH)
    model = model.to(cuda_device)
    batch_converter = alphabet.get_batch_converter()

    time2 = time.time()
    print(f"模型加载，耗费时间：{time2 - time1}")

    """ 上面是模型加载花费时间 """

    # foldseek_seqs = generate_3di_sequences(file_full_path_list)

    # 3di 序列生成
    foldseek_seqs = {}
    for file_full_path in file_full_path_list:
        parsed_seqs = get_struc_seq(FOLDSEEK_PATH, file_full_path, plddt_mask=False)
        key, (seq, foldseek_seq, combined_seq) = next(iter(parsed_seqs.items()))
        foldseek_seqs[file_full_path] = (foldseek_seq, combined_seq)

    # 批量生成嵌入
    data = [(file_full_path, foldseek_seqs[file_full_path][1]) for file_full_path in file_full_path_list]

    for batch_start in range(0, len(data), batch_size):
        batch_end = batch_start + batch_size
        batch_data = data[batch_start:batch_end]

        # batch_file_paths, batch_seqs = zip(*batch_data)

        # 将数据转换为模型输入格式
        batch_labels, batch_strs, batch_tokens = batch_converter(batch_data)
        batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)

        # 将数据移动到 GPU
        with torch.no_grad():
            batch_tokens = batch_tokens.to(cuda_device)
            results = model(batch_tokens, repr_layers=[33], return_contacts=False)

        # 提取特征表示
        token_representations = results["representations"][33]
        for i, tokens_len in enumerate(batch_lens):
            sequence_rep = token_representations[i, 1:tokens_len - 1].mean(0)
            avg_representation = sequence_rep.unsqueeze(0)
            file_full_path = batch_data[i][0]
            foldseek_seq, _ = foldseek_seqs[file_full_path]
            foldseek_seqs[file_full_path] = (foldseek_seq, avg_representation.cpu().numpy())

    time3 = time.time()
    print(f"嵌入生成，花费：{time3 - time2}")

    return foldseek_seqs




def load_mu_and_W():
    """加载白化所需的均值和矩阵"""
    mu = np.load(MU_FILENAME)
    W = np.load(W_FILENAME)
    return mu, W

def apply_whitening(mu,W,foldseek_seqs):
    """对嵌入进行白化处理"""
    # mu, W = load_mu_and_W()
    for file_full_path, (foldseek_seq, saprot_embedding) in foldseek_seqs.items():
        X_centered = saprot_embedding - mu
        whitened_embedding = np.dot(X_centered, W)
        whitened_embedding = np.array(whitened_embedding, dtype=np.float32)
        faiss.normalize_L2(whitened_embedding)
        foldseek_seqs[file_full_path] = (foldseek_seq, whitened_embedding)
    return foldseek_seqs

def search_faiss(index, dim, query_vectors, prefilter_target):
    """使用FAISS进行相似性搜索"""
    
    #index = faiss.read_index(faiss_file)
    
    #res = faiss.StandardGpuResources()  # 默认GPU资源
    #gpu_id = 1
    #index = faiss.index_cpu_to_gpu(res,gpu_id,index)


    query_vectors = np.ascontiguousarray(query_vectors[:, :dim], dtype=np.float32)  # TODO
    
    print(f"Query vectors shape: {query_vectors.shape}")

    #distances, indices = index.search(query_vectors[:, :dim], prefilter_target)
    
    distances, indices = index.search(query_vectors, prefilter_target)

    return distances, indices+1



def run_saligner(query_squeue, prefilter_threshold, max_target, prefilter_results_pdb, saligner_pdb):
    
    # time1 = time.time()

    """运行Saligner进行序列比对"""

    df_prefilter = pd.DataFrame(prefilter_results_pdb, columns=["Protein_Name", "prefilter_score"])
    
    # df_prefilter_sorted = df_prefilter.sort_values(by="prefilter_score", ascending=True)
    df_prefilter_sorted = df_prefilter.sort_values(by="prefilter_score", ascending=False)

    df_prefilter_sorted['saligner_score'] = np.nan
   

    time2 = time.time()

    # print(f"查询相关信息完毕了,cost :{time2 - time1}")

    saligner_result = []

    for protein_name, prefilter_score, sequence in saligner_pdb:
        d3i_sequence = ''.join(filter(str.islower, sequence))  # 获取小写部分
        d3i_sequence = d3i_sequence.upper()  # 将小写部分转换为大写

        saligner_score = saligner(query_squeue, d3i_sequence)

        # saligner_result.append((protein_name, prefilter_score,saligner_score, d3i_sequence))
        saligner_result.append((protein_name, prefilter_score, saligner_score))


    time3 = time.time()
    # print(f"SAligner比对完毕了,cost: {time3-time2}")

    # 将结果转为DataFrame
    df_saligner = pd.DataFrame(saligner_result, columns=["Protein_Name", "prefilter_score", "saligner_score"])
    df_saligner_sorted = df_saligner.sort_values(by="saligner_score", ascending=False).head(max_target - prefilter_threshold)
    final_df = pd.concat([df_prefilter_sorted, df_saligner_sorted], ignore_index=True)
    return final_df





def process_file(i, file_full_path, foldseek_seqs, prefilter_threshold, max_target,prefilter_results_pdb, saligner_pdb):
    foldseek_seq, _ = foldseek_seqs[file_full_path]

    # run_saligner(query_squeue, prefilter_threshold, max_target, prefilter_results_pdb, saligner_pdb):
    final_df = run_saligner(foldseek_seq, prefilter_threshold, max_target, prefilter_results_pdb, saligner_pdb)
    output_path = f"./test_pdb/ssalign_result/{os.path.basename(file_full_path)}.ssalign"
    final_df.to_csv(output_path, index=False)
    #print(f"处理完{file_full_path}了")



def process_chunk(mu, W, foldseek_seqs, chunk_keys):
    """处理一个数据块"""
    chunk = {key: foldseek_seqs[key] for key in chunk_keys}  # 直接从 foldseek_seqs 中提取子字典
    return apply_whitening(mu, W, chunk)



def main(query_pdb_name_list, dim, prefilter_target, prefilter_threshold, max_target, num_processes, num_gpus):

    time0 = time.time()
    """主函数"""
    # faiss_index_file = f"/data2/zxc_data/afdb50_combined_fasta/faiss_index/other/IVFPQ/afdb50_{dim}_IVFPQ_faiss.faiss"

    faiss_index_file = f"/data2/zxc_data/afdb50_combined_fasta/faiss_index/million/afdb50_{dim}_IVFPQ_faiss.faiss"

    file_full_path_list = [
        f"/home/xuchaozhang/ssalign/ssalign/SaprotProject/Saprot/SaProt/time_benckmark/test_pdb/pdb/{query_pdb_name}.cif"
        for query_pdb_name in query_pdb_name_list
    ]


    time1 = time.time()

    print(f"参数解析 ，花费 : {time1 - time0}")

    # 从 lookup 文件获取蛋白质名
    lookup_dict = {}
    with open(LOOKUP_FILE, 'r') as lookup_file:
        for line in lookup_file:
            seq_num, name = line.strip().split('\t')
            lookup_dict[int(seq_num)] = name

    # 从 index 文件获取序列的起始和结束位置
    index_dict = {}
    with open(INDEX_FILE, 'r') as index_file:
        for line in index_file:
            seq_num, start, end = map(int, line.strip().split())
            index_dict[seq_num] = (start, end)

    time2 = time.time()

    index = faiss.read_index(faiss_index_file)
   
    #res = faiss.StandardGpuResources()  # 默认 GPU 资源
    #gpu_id = 0
    #index = faiss.index_cpu_to_gpu(res, gpu_id, index)

    print(f"faiss索引加载到gpu，完毕{time.time() - time2}")

    mu, W = load_mu_and_W()

    print(f"前期预加载 ，花费 : {time2-time1}")

    start_time = time.time()

    # 生成SaProt嵌入
    # foldseek_seqs = generate_saprot_embeddings(file_full_path_list)
    avg_len = len(file_full_path_list) // num_gpus
    file_parts = [file_full_path_list[i * avg_len:(i + 1) * avg_len] for i in range(num_gpus)]
    with ProcessPoolExecutor(max_workers=num_gpus) as executor:
        futures = []
        for i in range(num_gpus):
            cuda_device = f'cuda:{i}'  # 分配到不同的 GPU
            futures.append(executor.submit(generate_saprot_embeddings, file_parts[i], cuda_device))

        # 获取所有的结果
        results = [future.result() for future in futures]

        # 合并所有 GPU 的结果
    foldseek_seqs = {}
    for result in results:
        foldseek_seqs.update(result)


    time3 = time.time()

    # 白化处理

    whitened_foldseek_seqs = apply_whitening(mu, W,foldseek_seqs)

    #keys = list(foldseek_seqs.keys())
    #chunk_size = len(keys) // num_processes
    #chunks = [keys[i:i + chunk_size] for i in range(0, len(keys), chunk_size)]

    # 使用多进程处理
    #with multiprocessing.Pool(processes=num_processes) as pool:
    #    results = pool.starmap(
    #        process_chunk,
    #        [(mu, W, foldseek_seqs, chunk) for chunk in chunks]
    #    )

    # 合并结果
    #whitened_foldseek_seqs = {}
    #for result in results:
    #    whitened_foldseek_seqs.update(result)


    time4 = time.time()

    print(f"Whitening applied.cost:{time4 - time3})")

    # 准备FAISS查询向量
    query_vectors = np.array([whitened_embedding.flatten() for _, (_, whitened_embedding) in whitened_foldseek_seqs.items()]).astype(np.float32)
   # print(f"Query vectors shape: {query_vectors.shape}")

    #query_vectors = np.ascontiguousarray(query_vectors)
    #print(f"Query vectors shape: {query_vectors.shape}")

    # FAISS搜索
    distances, indices = search_faiss(index, dim, query_vectors, prefilter_target)
    

    time5 = time.time()

    print(f"FAISS search completed.cost: {time5 - time4}")

    
    all_remaining_results = []

    for i in range(len(file_full_path_list)):
        all_remaining_results.append(list(zip(indices[i], distances[i])))

    all_prefilter_results_pdb, all_saligner_pdb = get_protein_by_index_batch_ok(lookup_dict,index_dict, SEQ_FILE, all_remaining_results,prefilter_threshold)

    timea = time.time()
    print(f"查询相关信息 : {timea - time5}")

    """
    下面使用多线程
    """
    with multiprocessing.Pool(processes=num_processes) as pool:
        # 使用enumerate来传递每个文件的索引和路径
        pool.starmap(process_file, [(i, file_full_path, foldseek_seqs, prefilter_threshold, max_target,all_prefilter_results_pdb[i], all_saligner_pdb[i])
                                    for i, file_full_path in enumerate(file_full_path_list)])





    # for i, file_full_path in enumerate(file_full_path_list):
    #     foldseek_seq, _ = foldseek_seqs[file_full_path]
    #     remaining_results = list(zip(indices[i], distances[i]))
    #     final_df = run_saligner(foldseek_seq, remaining_results, prefilter_threshold, max_target, num_processes,lookup_dict,index_dict)
    #     final_df.to_csv(f"./test_pdb/{os.path.basename(file_full_path)}.ssalign", index=False)
    #     print(f"处理完{file_full_path}了")

    time6 = time.time()

    print(f"Saligner完成，总共花费  {time6 - time5}")

    print(f"总共花费:{time6 - start_time}")

def parse_list_from_file(file_path):
    """从文件中读取文件名列表"""
    with open(file_path, 'r') as f:
        return [line.strip() for line in f if line.strip()]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='args for ssalign')
    parser.add_argument('--query_file_list_file', type=str, help="Path to a file containing a list of query files, one per line")
    parser.add_argument('--dim', type=int, help="Dimension of embeddings")
    parser.add_argument('--prefilter_target', type=int, help="Number of prefilter targets")
    parser.add_argument('--prefilter_threshold', type=int, help="Prefilter threshold")
    parser.add_argument('--max_target', type=int, help="Maximum number of targets")
    parser.add_argument('--num_processes', type=int, default=multiprocessing.cpu_count(), help="Number of processes to use")
    parser.add_argument('--num_gpus', type=int, default=1, help="Number of processes to use")
    args = parser.parse_args()

    query_file_list = parse_list_from_file(args.query_file_list_file)


    main(query_file_list, args.dim, args.prefilter_target, args.prefilter_threshold, args.max_target, args.num_processes,args.num_gpus)

