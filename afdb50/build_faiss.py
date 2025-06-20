import faiss
import numpy as np
import glob
import ast  # 解析字符串为列表
import os
import json  # 用于存储蛋白质名称
import logging
import argparse


# 配置索引参数
def create_faiss_index(dim, index_type, index_path, name_path, batch_size=100000, nlist=100, m=16, nbits=8):
    """
    Create a Faiss index, 
    dim : the dimension to be used, 
    index_type : the type of index, 
    index_path : the path to the index file,
    name_path : the storage path for protein names, 
    batch_size : the size of each processed batch, 
    nlist : the number of clusters,
    m : the number of subspaces for product quantization,  
    nbits : the number of bits per subspace.。
    """

    if index_type == 'IVFFlat':
        quantizer = faiss.IndexFlatIP(dim)  
        index = faiss.IndexIVFFlat(quantizer, dim, nlist)  
    elif index_type == 'IVFPQ':
        quantizer = faiss.IndexFlatIP(dim)  
        index = faiss.IndexIVFPQ(quantizer, dim, nlist, m, nbits)  
    else:
        raise ValueError("Invalid index type. Choose either 'IVFFlat' or 'IVFPQ'.")

    protein_names = []  

    file_paths = [f"/data2/zxc_data/afdb50_combined_fasta/whitening_vector/split_fasta_{i}_vector_whitend.fasta" for i in range(1, 41)]

    for file_id, file_path in enumerate(file_paths):
        logging.info(f"正在处理文件 {file_id + 1}/{len(file_paths)}: {file_path}")

        with open(file_path, "r") as f:
            lines = f.readlines()

        embeddings = []

        for i in range(0, len(lines), 2):  # 
            name = lines[i].strip()[1:]  # 
            embedding_str = lines[i + 1].strip()
            embedding = np.array(ast.literal_eval(embedding_str), dtype=np.float32)  #

            embedding = embedding[:dim]

            if dim == 1280:
                protein_names.append(name)  # 

            embeddings.append(embedding)

            # 
            if len(embeddings) >= batch_size:
                batch_data = np.array(embeddings, dtype=np.float32)

                faiss.normalize_L2(batch_data)

                if not index.is_trained:
                    index.train(batch_data)
                index.add(batch_data)  # 
                embeddings.clear()  # 

        # 
        if embeddings:
            batch_data = np.array(embeddings, dtype=np.float32)
            faiss.normalize_L2(batch_data)
            index.add(batch_data)

    logging.info("保存索引...")
    faiss.write_index(index, index_path)

    if dim == 1280:
        with open(name_path, "w") as f:
            json.dump(protein_names, f)
        logging.info("蛋白质名称已保存！")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Faiss索引构建")
    parser.add_argument('--dim', type=int, required=True, help='维度')
    parser.add_argument('--index_type', type=str, choices=['IVFFlat', 'IVFPQ'], required=True, help='索引类型')
    args = parser.parse_args()

    # 
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(f"/data2/zxc_data/afdb50_combined_fasta/faiss_index/other/{args.index_type}/{args.dim}_{args.index_type}.log"),
            logging.StreamHandler()
        ]
    )

    # 
    index_path = f"/data2/zxc_data/afdb50_combined_fasta/faiss_index/other/{args.index_type}/afdb50_{args.dim}_{args.index_type}_faiss.faiss"
    name_path = f"/data2/zxc_data/afdb50_combined_fasta/faiss_index/other/{args.index_type}/afdb50_names.json"

    create_faiss_index(args.dim, args.index_type, index_path, name_path)
