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
    创建Faiss索引，dim为要使用的维度，index_type为索引类型，index_path为索引文件路径，
    name_path为蛋白质名称存储路径，batch_size为每次处理的批次大小，nlist为簇的数量，
    m为乘积量化的子空间数，nbits为每个子空间的位数。
    """

    # 选择不同的索引类型
    if index_type == 'IVFFlat':
        quantizer = faiss.IndexFlatIP(dim)  # 扁平内积索引作为量化器
        index = faiss.IndexIVFFlat(quantizer, dim, nlist)  # 倒排索引 + 扁平存储
    elif index_type == 'IVFPQ':
        quantizer = faiss.IndexFlatIP(dim)  # 扁平内积索引作为量化器
        index = faiss.IndexIVFPQ(quantizer, dim, nlist, m, nbits)  # 倒排索引 + 乘积量化
    else:
        raise ValueError("Invalid index type. Choose either 'IVFFlat' or 'IVFPQ'.")

    protein_names = []  # 用于存储蛋白质名称

    # 获取所有 FASTA 文件
    file_paths = [f"/data2/zxc_data/afdb50_combined_fasta/whitening_vector/split_fasta_{i}_vector_whitend.fasta" for i in range(1, 41)]

    for file_id, file_path in enumerate(file_paths):
        logging.info(f"正在处理文件 {file_id + 1}/{len(file_paths)}: {file_path}")

        # 读取 FASTA 文件
        with open(file_path, "r") as f:
            lines = f.readlines()

        embeddings = []

        for i in range(0, len(lines), 2):  # 每两个为一组（蛋白质名称 + 向量）
            name = lines[i].strip()[1:]  # 去掉 '>'
            embedding_str = lines[i + 1].strip()
            embedding = np.array(ast.literal_eval(embedding_str), dtype=np.float32)  # 解析字符串为数组

            # 取前 dim 维作为实际维度（降维处理）
            embedding = embedding[:dim]

            # 仅在 dim == 1280 时，保存蛋白质名称
            if dim == 1280:
                protein_names.append(name)  # 存储蛋白质名称

            embeddings.append(embedding)

            # 分批处理，防止内存溢出
            if len(embeddings) >= batch_size:
                batch_data = np.array(embeddings, dtype=np.float32)

                # 使用 Faiss 的归一化方法
                faiss.normalize_L2(batch_data)

                # 训练索引并添加数据
                if not index.is_trained:
                    index.train(batch_data)
                index.add(batch_data)  # 添加数据
                embeddings.clear()  # 清空列表释放内存

        # 处理剩余数据
        if embeddings:
            batch_data = np.array(embeddings, dtype=np.float32)
            faiss.normalize_L2(batch_data)
            index.add(batch_data)

    logging.info("保存索引...")
    faiss.write_index(index, index_path)

    # 仅在 dim == 1280 时，保存蛋白质名称
    if dim == 1280:
        with open(name_path, "w") as f:
            json.dump(protein_names, f)
        logging.info("蛋白质名称已保存！")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Faiss索引构建")
    parser.add_argument('--dim', type=int, required=True, help='维度')
    parser.add_argument('--index_type', type=str, choices=['IVFFlat', 'IVFPQ'], required=True, help='索引类型')
    args = parser.parse_args()

    # 设置日志记录
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(f"/data2/zxc_data/afdb50_combined_fasta/faiss_index/other/{args.index_type}/{args.dim}_{args.index_type}.log"),
            logging.StreamHandler()
        ]
    )

    # 索引文件和名称文件路径
    index_path = f"/data2/zxc_data/afdb50_combined_fasta/faiss_index/other/{args.index_type}/afdb50_{args.dim}_{args.index_type}_faiss.faiss"
    name_path = f"/data2/zxc_data/afdb50_combined_fasta/faiss_index/other/{args.index_type}/afdb50_names.json"

    # 调用函数创建 Faiss 索引
    create_faiss_index(args.dim, args.index_type, index_path, name_path)
