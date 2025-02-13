from utils.execFoldseek import exec_foldseek_easy_search_para
from utils.execTMalign import tm_align_exec,extract_key_info,sort_Talign_chain,compare_with_target
from utils.SSAlign_utils import load_vectors_from_file_and_queryvector, build_faiss_index_IP_gpu, faiss_align_vector, \
    load_squeue,add_prefilter_tmscore,add_foldseek_tmscore
from SAligner.example import use_pairalign
import faiss
import os
import pandas as pd

"""

三种工具的运行

全队全搜索，

"""

def tmalign_workflow():

    file_dir = "../data/pdb/Scope40/"
    structure_dir = "../data/pdb/Scope40/"

    basenames = []

    for root, dirs, files in os.walk(file_dir):
        for file in files:
            basenames.append(file)

    for basename in basenames:
        query_file_path = f"../data/pdb/Scope40/{basename}"
        tmp_output_file = f"../data/pdb/Scope40/{basename}_tmp"
        output_file = f"../data/pdb/Scope40/tmalign/{basename}.result"

        compare_with_target(structure_dir, query_file_path, tmp_output_file)
        sort_Talign_chain(tmp_output_file, output_file)

        os.remove(tmp_output_file)


def foldseek_workflow():
    file_dir = "../data/pdb/Scope40/"

    foldseek = "../bin/foldseek"

    basenames = []

    for root, dirs, files in os.walk(file_dir):
        for file in files:
            basenames.append(file)

    output_directory = "../data/result/Scope40/foldseek"

    exec_foldseek_easy_search_para(foldseek, '../data/foldseekDB/Scope40DB/DB/scope40DB', basenames, output_directory,
                                   9.5, 10, 1000)

    # 添加tmalign计算过的指标
    for basename in basenames:
        foldseek_file = f"../data/pdb/Scope40/foldseek/{basename}_foldseek"

        tmalign_file = f"../data/pdb/Scope40/tmalign/{basename}.result"

        output_file = f"../data/pdb/Scope40/foldseek/{basename}.result"

        add_foldseek_tmscore(foldseek_file,tmalign_file,output_file,basename)







"""
dim  1280 512 256 128 64
"""
def SSAlign_prefilter_workflow(dim,gpu_id):

    whitened_vector_file_path = "../data/result/Scope40/scope40_vector_results_whitening"

    file_dir = "../data/pdb/Scope40"

    basenames = []

    for root, dirs, files in os.walk(file_dir):
        for file in files:
            basenames.append(file)

    random_filenames = basenames

    all_filenames, all_vectors , query_filenames,query_vectors = load_vectors_from_file_and_queryvector(whitened_vector_file_path,1,random_filenames,dim)

    faiss.normalize_L2(all_vectors)
    faiss.normalize_L2(query_vectors)

    # print(all_vectors.dtype)
    # print(query_vectors.dtype)

    index = build_faiss_index_IP_gpu(all_vectors, gpu_id,dim)



    for i, query_vector in enumerate(query_vectors):
        # basename = query_filenames[i].replace(".cif","")

        basename = os.path.basename(query_filenames[i])  # 提取文件名

        query_vector = query_vector.reshape(1, -1)  # 保持查询向量为二维

        faiss_align_vector(index, all_filenames, query_vector, 8000,f"../data/result/Scope40/SSAlign/SVD{dim}/{basename}.result")


    # 添加tmalign计算过的指标
    for basename in basenames:
        faiss_file = f"../data/result/Scope40/SSAlign/SVD{dim}/{basename}.result"

        tmalign_file = f"../data/pdb/Scope40/tmalign/{basename}.result"

        output_file = f"../data/result/Scope40/SSAlign/SVD{dim}/{basename}.result"
        add_prefilter_tmscore(faiss_file,tmalign_file,output_file,basename,dim)





"""
SSAlign_prefilter -----  SAligner
"""
def SSAlign_SAligner_workflow(dim,topk):
    file_dir = "../data/pdb/Scope40/"
    basenames = [file for root, dirs, files in os.walk(file_dir) for file in files]

    for basename in basenames:
        csv_file_path = f"../data/result/Scope40/SSAlign/SVD{dim}/{basename}.result"

        df = pd.read_csv(csv_file_path)

        df_sorted = df.sort_values(by='Cosine_Similarity', ascending=False).head(topk)

        df_sorted['Avg_TM_Score'] = df_sorted.apply(lambda row: (row['TM-Score1'] + row['TM-Score2']) / 2, axis=1)

        prefix_path = "../data/pdb/Scope40/"
        fullname_list = df_sorted['File1'].apply(lambda x: prefix_path + x).tolist()

        # 查询他们的序列
        target_squeue_data, sequence_data = load_squeue(fullname_list,f'../data/pdb/Scope4/{basename}', 'lower')

        target_sequence = next(iter(target_squeue_data.values()))

        comparison_results = []

        for filename, seq in sequence_data.items():
            filename = os.path.basename(filename)

            result = use_pairalign(target_sequence, seq)

            avg_tm_score = df_sorted[df_sorted['File1'] == filename]['Avg_TM_Score'].values[0]
            cosine_Similarity = df_sorted[df_sorted['File1'] == filename]['Cosine_Similarity'].values[0]
            Aligned_Length = df_sorted[df_sorted['File1'] == filename]['Aligned Length'].values[0]
            RMSD = df_sorted[df_sorted['File1'] == filename]['RMSD'].values[0]
            Seq_ID = df_sorted[df_sorted['File1'] == filename]['Seq_ID'].values[0]

            comparison_results.append({
                "File1": filename,
                "Aligned Length": Aligned_Length,
                "RMSD": RMSD,
                "Seq_ID": Seq_ID,
                "length_squeue": len(seq),
                "Avg_TM_Score": avg_tm_score,
                "Cosine_Similarity": cosine_Similarity,
                "Score": result["Score"]
            })
            # 将结果写入 CSV
        df_results = pd.DataFrame(comparison_results)

        csv_output_path = f"../data/result/Scope40/SSAlign/SVD{dim}bio_{basename}_lower_global.csv"


        # 保存到 CSV 文件
        df_results.to_csv(csv_output_path, index=False)









def ssalign(dim,basename,faiss_topk,cos_threshold,final_number):
    file_path = f"../data/result/Scope40/SSAlign/SVD{dim}/bio_{basename}_lower_global.csv"

    # 1. 先预过滤出来 faiss_topk个
    df = pd.read_csv(file_path).sort_values(by='Cosine_Similarity',ascending=False).head(faiss_topk)

    # 2. 然后选出大于阈值的
    df_first = df[df['Cosine_Similarity'] >= cos_threshold]

    # 如果第一步的个数已经足够了，那就直接返回
    if len(df_first) >= final_number:
        df_all = df_first.sort_values(by='Cosine_Similarity',ascending=False).head(final_number)
    else:
        # 反之，进行第二步
        # 3. 小于阈值的，进行第二步，使用bio得分来筛选
        df_less = df[df['Cosine_Similarity'] < cos_threshold]
        #
        df_second = df_less.sort_values(by='Score',ascending=False).head(final_number-len(df_first))

        df_all = pd.concat([df_first,df_second])



    df_right = df_all[df_all['Avg_TM_Score'] >= 0.5]

    return df_all








