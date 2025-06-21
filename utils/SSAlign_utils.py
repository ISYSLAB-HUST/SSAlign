import os
import numpy as np
import pandas as pd
import csv
import faiss


def load_vectors_from_file_and_queryvector(vector_file_path,index, random_filenames,dim):
    
    all_filenames=[]
    all_vectors=[]
    query_vectors = []
    query_filenames = []


    with open(vector_file_path, 'r') as f:
        for line in f:
            parts = line.strip().split(",")
            filename = parts[0]  # \
            base_filename = os.path.basename(filename)  

            # print(base_filename)

            # sequence = parts[1]  #
            vector_str = ",".join(parts[index:])  # 
            vector = np.array(eval(vector_str))[:,:dim]  # 

            all_filenames.append(filename)
            all_vectors.append(vector)

           
            if base_filename in random_filenames:
                # print(filename)
                # print(vector)
                query_filenames.append(filename)
                query_vectors.append(vector)

    all_vectors = np.vstack(all_vectors).astype('float32')# FAISS要求float32类型
    query_vectors = np.vstack(query_vectors).astype('float32')


    return all_filenames, all_vectors, query_filenames,query_vectors



def build_faiss_index_IP_gpu(vectors,gpu_id,dim):
  

    # d = SVD1280  # 

    index = faiss.IndexFlatIP(dim)  # 

    res = faiss.StandardGpuResources()  #

    index_gpu = faiss.index_cpu_to_gpu(res, gpu_id, index)
 
    index_gpu.add(vectors)  

    return index_gpu

def search_similar_vectors(index, query_vector, top_k):

    D, I = index.search(query_vector, top_k)
    return D, I 

def save_to_file(output_file, result):
    
    with open(output_file, 'a') as f:

       
        f.write(f"{result[0]},{result[1]}\n")


def faiss_align_vector(index, filenames, query_vector, top_k, result_file_path):
    distances, indices = search_similar_vectors(index, query_vector, top_k)

    for i in range(top_k):
       
        file_name = os.path.basename(filenames[indices[0][i]])

        print(distances[0][i].dtype)
        print(distances[0][i])
        save_to_file(result_file_path,(file_name,distances[0][i]))


def load_squeue(file_path,fullname_list,target_fullname,charset):

    target_sequence_data = {}

    sequence_data = {}

    with open(file_path, 'r') as f:
        for line in f:
            
            parts = line.strip().split(",")
            filename = parts[0]  
            sequence = parts[1]  


            if charset == "upper":
              
                sequence_upper = ''.join([char for char in sequence if char.isupper()])
            if charset == "lower":
                sequence_upper = ''.join([char.upper() for char in sequence if char.islower()])
            if charset == "all":
                sequence_upper = ''.join([char.upper() for char in sequence])

          
            if filename == target_fullname:
                target_sequence_data[filename] = sequence_upper
                #(filename)
           
            if filename in fullname_list:
                sequence_data[filename] = sequence_upper
                #print(filename)

            # sequence = parts[1]
    return target_sequence_data,sequence_data


def add_prefilter_tmscore(faiss_file,tmalign_file,output_file,basename,dim):

    tmalign_dict = {}
    try:
        with open(tmalign_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader)  
            for row in reader:
                file1_name = row[0].split("/")[-1] 
                file2_name = row[1].split("/")[-1]  
                metrics = row[2:]  
                tmalign_dict[file1_name] = metrics
    except Exception as e:
        print(f"读取 TM-align 文件时出错: {e}")
        return

    merged_data = []

    try:
        with open(faiss_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) < 2:
                    continue
                file_name = parts[0]  # Faiss 文件名
                cosine_similarity = parts[1]  # Cosine Similarity

                tmalign_metrics = tmalign_dict.get(file_name, ["N/A"] * 5)

                merged_row = [file_name,basename ] + tmalign_metrics + [cosine_similarity]
                merged_data.append(merged_row)
    except Exception as e:
        print(f"读取 Faiss 文件时出错: {e}")
        return

    try:
        with open(output_file, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            header = ["File1", "File2", "TM-Score1", "TM-Score2", "Aligned Length", "RMSD", "Seq_ID", "Cosine_Similarity"]
            writer.writerow(header)
            writer.writerows(merged_data)
        print(f"成功生成文件: {output_file}")
    except Exception as e:
        print(f"写入输出文件时出错: {e}")



def add_foldseek_tmscore(foldseek_file,tmalign_file,output_file,basename):

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

    merged_data = []
    try:
        with open(foldseek_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                file1_name = parts[0]  # FoldSeek 第一列
                file2_name = parts[1]  # FoldSeek 第二列
                foldseek_metrics = " ".join(parts[2:])  # FoldSeek 指标

                tmalign_metrics = tmalign_dict.get(file2_name, ["N/A"] * 5)

                
                merged_row = [ file2_name,file1_name] + tmalign_metrics + [foldseek_metrics]
                merged_data.append(merged_row)
    except Exception as e:
        print(f"读取 FoldSeek 文件时出错: {e}")

    # 
    try:
        with open(output_file, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            header = ["File1", "File2", "TM-Score1", "TM-Score2", "Aligned Length", "RMSD", "Seq_ID", "FoldSeek_Metrics"]
            writer.writerow(header)
            writer.writerows(merged_data)
        print(f"成功生成文件: {output_file}")
    except Exception as e:
        print(f"写入输出文件时出错: {e}")
