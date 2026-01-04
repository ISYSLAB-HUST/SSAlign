import numpy as np

from utils.SSAlign_utils import load_query_vectors_from_npz, faiss_align_vector, add_prefilter_tmscore
from SAligner.example import use_pairalign
import faiss
import os
import pandas as pd

"""

三种工具的运行

全队全搜索，

"""


"""
dim  1280 512 256 128 64
"""
def SSAlign_prefilter_workflow(dim, gpu_id):

    embedding_npz = "../models/SSAlignDB/SCOPe40/SCOPe40_id_embedding.npz"

    # 0) 读 query 名单
    file_dir = "../pdbData/pdb/SCOPe40/"
    basenames = []
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            basenames.append(file)

    # 1) NPZ: 得到 db_names(与索引下标一致) + query_vectors(只取名单里的)
    db_names, query_vectors = load_query_vectors_from_npz(embedding_npz, basenames, dim)

    # 2) normalize query（索引当初是 IP + normalize，则 query 必须 normalize）
    faiss.normalize_L2(query_vectors)

    # 3) 读索引并搬到 GPU
    faiss_index_file = f"../models/SSAlignDB/SCOPe40/SCOPe40_IndexFlatIP_{dim}_faiss.index"
    index_cpu = faiss.read_index(faiss_index_file)

    res = faiss.StandardGpuResources()
    index = faiss.index_cpu_to_gpu(res, gpu_id, index_cpu)

    # 4) ssalign prefilter 输出
    out_dir = f"../benchmarkData/SCOPe40/SSAlign/SVD{dim}/ssalign_prefilter"
    os.makedirs(out_dir, exist_ok=True)

    for qid, qv in zip(basenames, query_vectors):
        out_path = f"{out_dir}/{qid}.result"
        faiss_align_vector(index, db_names, qv.reshape(1, -1), 8000, out_path)

    # 5) 合并 TM-align 指标（原逻辑保持不变）
    for qid in basenames:
        faiss_file = f"{out_dir}/{qid}.result"
        tmalign_file = f"../benchmarkData/SCOPe40/tmalign/{qid}.result"
        add_prefilter_tmscore(faiss_file, tmalign_file, faiss_file, qid, dim)




"""
SSAlign_prefilter -----  SAligner
"""
def SSAlign_workflow(dim, faiss_topk, cos_threshold, final_number):
    seq_npz = "./models/SSAlignDB/SCOPe40/SCOPe40_id_Seq.npz"

    # 1) 一次性加载 id->seq 映射（用 basename 做 key，和你结果文件 File1/查询名保持一致）
    data = np.load(seq_npz, allow_pickle=True)
    ids = [os.path.basename(str(x)) for x in data["ids"]]
    seqs = [str(s) for s in data["seqs"]]
    id2seq = dict(zip(ids, seqs))

    def pick_seq(sequence: str, charset: str):
        if charset == "upper":
            return "".join([c for c in sequence if c.isupper()])
        if charset == "lower":
            return "".join([c.upper() for c in sequence if c.islower()])
        if charset == "all":
            return "".join([c.upper() for c in sequence])
        return sequence

    pre_dir = f"./benchmarkData/SCOPe40/SSAlign/SVD{dim}/ssalign_prefilter"
    out_dir = f"./benchmarkData/SCOPe40/SSAlign/SVD{dim}/ssalign"
    os.makedirs(out_dir, exist_ok=True)

    # 2) 遍历所有 query（用 prefilter 的结果文件名来驱动）
    for basename in ids:
        csv_file_path = f"{pre_dir}/{basename}.result"
        if not os.path.exists(csv_file_path):
            continue

        df = pd.read_csv(csv_file_path)
        df_sorted = df.sort_values(by="Cosine_Similarity", ascending=False).head(faiss_topk)

        df_sorted["Avg_TM_Score"] = df_sorted.apply(lambda r: (r["TM-Score1"] + r["TM-Score2"]) / 2, axis=1)

        # 3) 取 query 序列（这里沿用你之前的 'lower'：只用 3Di 部分）
        if basename not in id2seq:
            continue
        target_sequence = pick_seq(id2seq[basename], "lower")

        comparison_results = []
        for _, row in df_sorted.iterrows():
            file1 = str(row["File1"])  # 候选
            if file1 not in id2seq:
                continue

            cand_seq = pick_seq(id2seq[file1], "lower")
            result = use_pairalign(target_sequence, cand_seq)

            comparison_results.append({
                "File1": file1,
                "Aligned Length": row["Aligned Length"],
                "RMSD": row["RMSD"],
                "Seq_ID": row["Seq_ID"],
                "length_squeue": len(cand_seq),
                "Avg_TM_Score": row["Avg_TM_Score"],
                "Cosine_Similarity": row["Cosine_Similarity"],
                "Score": result["Score"],
            })

        df_results = pd.DataFrame(comparison_results)

        # 4) （阈值以上按 Cos，以下按 Score）
        df_top = df_results.sort_values(by="Cosine_Similarity", ascending=False).head(faiss_topk)
        df_first = df_top[df_top["Cosine_Similarity"] >= cos_threshold]

        if len(df_first) >= final_number:
            df_all = df_first.sort_values(by="Cosine_Similarity", ascending=False).head(final_number)
        else:
            df_less = df_top[df_top["Cosine_Similarity"] < cos_threshold]
            df_second = df_less.sort_values(by="Score", ascending=False).head(final_number - len(df_first))
            df_all = pd.concat([df_first, df_second], ignore_index=True)

        df_all.to_csv(f"{out_dir}/{basename}.result", index=False)

    return True


if __name__=="__main__":
    """
        All-against-all performance evaluation on the SCOPe40 dataset.
        SSAlign
    """
    SSAlign_prefilter_workflow(1280, 1)
    SSAlign_workflow(1280, 2000, 0.2, 1000)

    SSAlign_prefilter_workflow(512, 1)
    SSAlign_workflow(512, 2000, 0.3, 1000)

    SSAlign_prefilter_workflow(256, 1)
    SSAlign_workflow(256, 2000, 0.45, 1000)

    SSAlign_prefilter_workflow(128, 1)
    SSAlign_workflow(128, 2000, 0.6, 1000)

    SSAlign_prefilter_workflow(64, 1)
    SSAlign_workflow(64, 2000, 0.7, 1000)









