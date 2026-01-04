import os
import numpy as np
import pandas as pd
import csv
import faiss

def load_query_vectors_from_npz(embedding_npz_path, query_names, dim):
    """
    embedding_npz_path: SwissProt_id_embedding.npz  (ids, embeddings)
    query_names: 例如 ["AF-xxx.cif", ...]
    dim: 取前 dim 维

    返回:
      ids 数据库内的蛋白质名字
      query_vectors  和query_names匹配的嵌入
    """
    data = np.load(embedding_npz_path, allow_pickle=True)
    ids = [os.path.basename(str(x)) for x in data["ids"]]          # 与索引下标一致
    embs = data["embeddings"]                                      # shape [N, D]

    name2pos = {name: i for i, name in enumerate(ids)}

    vecs = []
    for name in query_names:
        qid = os.path.basename(str(name)).strip()
        if not qid:
            continue
        pos = name2pos.get(qid)
        if pos is None:
            print(f"[WARN] not found in npz ids: {qid}")
            continue
        vecs.append(embs[pos, :dim])

    query_vectors = np.ascontiguousarray(np.vstack(vecs), dtype=np.float32)
    return ids, query_vectors









def faiss_align_vector(index, db_names, query_vector, top_k, result_file_path):
    """
    index: faiss index (CPU 或 GPU 都行)
    db_names: 与 index 下标一致的名字列表（来自 npz["ids"]）
    query_vector: shape [1, dim]
    输出: 每行 "File1,Cosine_Similarity"
    """
    distances, indices = index.search(query_vector, top_k)

    with open(result_file_path, "w", encoding="utf-8") as f:
        for i in range(top_k):
            hit = db_names[int(indices[0][i])]
            score = float(distances[0][i])
            f.write(f"{hit},{score}\n")





def add_prefilter_tmscore(faiss_file,tmalign_file,output_file,basename):

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

    # 读取 prefilter 结果文件并补充 TM-align 数据
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