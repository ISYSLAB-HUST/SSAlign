from utils.execFoldseek import exec_foldseek_easy_search_para
from utils.execTMalign import tm_align_exec,extract_key_info,sort_Talign_chain,compare_with_target
from utils.SSAlign_utils import load_vectors_from_file_and_queryvector, build_faiss_index_IP_gpu, faiss_align_vector, \
    load_squeue,add_prefilter_tmscore,add_foldseek_tmscore
from SAligner.example import use_pairalign
import faiss
import os
import pandas as pd



"""



random files

"""

def tmalign_workflow():

    file_dir = "../data/pdb/Swissport/"
    structure_dir = "../data/pdb/Swissport/"


    with open("170filenames.txt","r") as f:
        basenames = [line.strip() for line in f if line.strip()]  

    for basename in basenames:
        query_file_path = f"../data/pdb/Swissport/{basename}"
        tmp_output_file = f"../data/pdb/Swissport/{basename}_tmp"
        output_file = f"../data/pdb/Swissport/tmalign/{basename}.result"

        compare_with_target(structure_dir, query_file_path, tmp_output_file)
        sort_Talign_chain(tmp_output_file, output_file)

        os.remove(tmp_output_file)


def foldseek_workflow():
    file_dir = "../data/pdb/Swissport/"

    foldseek = "../bin/foldseek"



    with open("170filenames.txt", "r") as f:
        basenames = [line.strip() for line in f if line.strip()]  

    output_directory = "../data/result/Swissport/foldseek"

    exec_foldseek_easy_search_para(foldseek, '../data/foldseekDB/SpDB/spDB', basenames, output_directory,9.5, 10, 1000)

    for basename in basenames:
        foldseek_file = f"../data/pdb/Swissport/foldseek/{basename}_foldseek"

        tmalign_file = f"../data/pdb/Swissport/tmalign/{basename}.result"

        output_file = f"../data/pdb/Swissport/foldseek/{basename}.result"

        add_foldseek_tmscore(foldseek_file, tmalign_file, output_file, basename)


def SSAlign_prefilter_workflow(dim,gpu_id):
    whitened_vector_file_path = "../data/result/Scope40/swissprot_cif_v4_files_results_whitening"

    file_dir = "../data/pdb/Swissport"


    with open("170filenames.txt", "r") as f:
        basenames = [line.strip() for line in f if line.strip()]  # 去掉空行和多余空格

    random_filenames = basenames

    all_filenames, all_vectors, query_filenames, query_vectors = load_vectors_from_file_and_queryvector(whitened_vector_file_path, 2, random_filenames, dim)

    faiss.normalize_L2(all_vectors)
    faiss.normalize_L2(query_vectors)

    # print(all_vectors.dtype)
    # print(query_vectors.dtype)

    index = build_faiss_index_IP_gpu(all_vectors, gpu_id, dim)

    for i, query_vector in enumerate(query_vectors):
        # basename = query_filenames[i].replace(".cif","")

        basename = os.path.basename(query_filenames[i])  # 

        query_vector = query_vector.reshape(1, -1)  # 

        faiss_align_vector(index, all_filenames, query_vector, 8000,f"../data/result/Swissport/SSAlign/SVD{dim}/{basename}.result")

    # 
    for basename in basenames:
        faiss_file = f"../data/result/Scope40/SSAlign/SVD{dim}/{basename}.result"

        tmalign_file = f"../data/pdb/Scope40/tmalign/{basename}.result"

        output_file = f"../data/result/Scope40/SSAlign/SVD{dim}/{basename}.result"
        add_prefilter_tmscore(faiss_file, tmalign_file, output_file, basename, dim)


"""
1280
"""
def SSAlign_SAligner_workflow(topk):
    file_dir = "../data/pdb/Swissport"
    with open("170filenames.txt", "r") as f:
        basenames = [line.strip() for line in f if line.strip()]  # 

    for basename in basenames:
        csv_file_path = f"../data/result/Scope40/SSAlign/SVD1280/{basename}.result"

        df = pd.read_csv(csv_file_path)

        df_sorted = df.sort_values(by='Cosine_Similarity', ascending=False).head(topk)

        df_sorted = df[df['Cosine_Similarity'] < 0.2]

        df_sorted['Avg_TM_Score'] = df_sorted.apply(lambda row: (row['TM-Score1'] + row['TM-Score2']) / 2, axis=1)

        prefix_path = "../data/pdb/Swissport"
        fullname_list = df_sorted['File1'].apply(lambda x: prefix_path + x).tolist()

        # 
        target_squeue_data, sequence_data = load_squeue(fullname_list,f'../data/pdb/Swissport/{basename}', 'lower')

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
            #  
        df_results = pd.DataFrame(comparison_results)

        csv_output_path = f"../data/result/Swissport/SSAlign/SVD1280/bio_{basename}_lower_global.csv"

        # to csv
        df_results.to_csv(csv_output_path, index=False)


"""
other dim
"""
def SSAlign_SAligner_workflow_otherdim(dim,topk):
    file_dir = "../data/pdb/Scope40/"
    basenames = [file for root, dirs, files in os.walk(file_dir) for file in files]

    for basename in basenames:
        csv_file_path = f"../data/result/Swissport/SSAlign/SVD{dim}/{basename}.result"

        df = pd.read_csv(csv_file_path)

        df_sorted = df.sort_values(by='Cosine_Similarity', ascending=False).head(topk)

        df_sorted['Avg_TM_Score'] = df_sorted.apply(lambda row: (row['TM-Score1'] + row['TM-Score2']) / 2, axis=1)

        prefix_path = "../data/pdb/Scope40/"
        fullname_list = df_sorted['File1'].apply(lambda x: prefix_path + x).tolist()

        # 
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
            #  CSV
        df_results = pd.DataFrame(comparison_results)

        csv_output_path = f"../data/result/Swissport/SSAlign/SVD{dim}/bio_{basename}_lower_global.csv"


        # to csv
        df_results.to_csv(csv_output_path, index=False)










def SSAlign_workflow(basename,faiss_topk,final_num,cosine_threshold):
    faiss_file_path = f"../data/result/Swissport/SSAlign/SVD1280/{basename}.result"

    bio_file_path = f"../data/result/Swissport/SSAlign/SVD1280/bio_{basename}_lower_global.csv"

    df_faiss = pd.read_csv(faiss_file_path)

    df_faiss['Avg_TM_Score'] = (df_faiss['TM-Score1'] + df_faiss['TM-Score2']) / 2

    df_faiss_sorted = df_faiss.sort_values(by='Cosine_Similarity', ascending=False)

    df_filtered_threshold = df_faiss_sorted[df_faiss_sorted['Cosine_Similarity'] >= cosine_threshold]


    df_faiss_matched = pd.DataFrame(columns=df_filtered_threshold.columns)


    if(len(df_filtered_threshold) < final_num):
        try:
            df_bio = pd.read_csv(bio_file_path)

            if df_bio.empty:
                print(f"文件 {bio_file_path} 是空的。创建一个占位 DataFrame。")
                # df_bio = pd.DataFrame(columns=df_filtered_threshold.columns)  # 创建空表结构
                df_bio = pd.DataFrame(columns=['filename_1', 'Score', 'Cosine_Similarity'])

            df_bio_sorted_cos = df_bio.sort_values(by='Cosine_Similarity', ascending=False).head(faiss_topk - len(df_filtered_threshold))

            df_bio_target = df_bio_sorted_cos.sort_values(by='Score', ascending=False).head(final_num - len(df_filtered_threshold))

            file_names_from_bio = df_bio_target['filename_1'].tolist()

            df_faiss_matched = df_faiss[df_faiss['File1'].isin(file_names_from_bio)]
            df_faiss_matched = df_faiss_matched.copy()  # 

            df_faiss_matched = df_faiss_matched.merge(df_bio_target[['filename_1', 'Score']],
                                                  left_on='File1',
                                                  right_on='filename_1',
                                                  how='left')

        except pd.errors.EmptyDataError:
            print(f"文件 {bio_file_path} 是空的，无法读取。")
            df_bio_target = pd.DataFrame(columns=df_filtered_threshold.columns)

        df_final = pd.concat([df_filtered_threshold, df_faiss_matched]).drop_duplicates()

        df_final = df_final.drop(columns=['filename_1', 'File2','TM-Score1','TM-Score2'], errors='ignore')

        df_high_cosine = df_final[df_final['Cosine_Similarity'] >= 0.2].sort_values(by='Cosine_Similarity', ascending=False)
        df_low_cosine = df_final[df_final['Cosine_Similarity'] < 0.2].sort_values(by='Score', ascending=False)

        # 
        df_final_sorted = pd.concat([df_high_cosine, df_low_cosine])

    if len(df_filtered_threshold) > final_num:
        print(basename)
        print(len(df_filtered_threshold))
        df_final_sorted = df_filtered_threshold.sort_values(by='Cosine_Similarity', ascending=False).head(final_num)
        # 
        required_columns = ['File1', 'Aligned Length', 'RMSD', 'Seq_ID', 'Cosine_Similarity', 'Avg_TM_Score', 'Score']
        for col in required_columns:
            if col not in df_final_sorted.columns:
                df_final_sorted[col] = None  # 
        df_final_sorted = df_final_sorted[required_columns]  # 


    # 
    #print(df_final_sorted)

    save_path = f"../data/result/Swissport/SSAlign/SVD1280/{basename}.result"

    df_final_sorted.to_csv(save_path, index=False)
    print(f"结果已保存到 {save_path}")



def SSAlign_workflow_otherdim(basename,dim,faiss_topk,final_num,cosine_threshold):
    bio_file_path = f"../data/result/Swissport/SSAlign/SVD{dim}/bio_{basename}_lower_global.csv"

    df = pd.read_csv(bio_file_path)

    df_topk = df.sort_values(by='Cosine_Similarity', ascending=False).head(faiss_topk)

    df_faiss = df_topk[df_topk['Cosine_Similarity'] >= cosine_threshold].head(final_num)

    if len(df_faiss) < final_num:
        df_cos_lower = df_topk[df_topk['Cosine_Similarity'] < cosine_threshold]
        # 
        df_bio = df_cos_lower.sort_values(by='Score', ascending=False).head(final_num-len(df_faiss))
    else:
        df_bio = pd.DataFrame(columns=df_faiss.columns)


    df_final = pd.concat([df_faiss, df_bio]).drop_duplicates()

    print(len(df_faiss))
    # print(df_final)

    save_path =f"../data/result/Swissport/SSAlign/SVD{dim}/ssalign/{basename}.result"

    df_final.to_csv(save_path, index=False)
    print(f"结果已保存到 {save_path}")
