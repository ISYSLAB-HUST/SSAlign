import os
import random
import logging

def exec_foldseek_easy_search_para(foldseek,foldeseekDB,file_basenames,output_directory,param_s,param_e,param_max_seqs):
    # 遍历每个文件basename并执行foldseek命令
    for basename in file_basenames:

        file_path = f"../data/Scope40/{basename}"
        output_path = f"{basename}_foldseek"
        #output_directory = "/data/foldseek_database/sp_cif_file/100files_foldseek_faiss"

        # 构造并执行命令
        # cmd = f"{foldseek} easy-search {file_path} ../data/foldseekDB/Scope40DB/DB/scope40DB  {output_path} {output_directory} -s {param_s} -e {param_e} --max-seqs {param_max_seqs} >> {log_file} 2>&1"
        cmd = f"{foldseek} easy-search {file_path} {foldeseekDB}  {output_path} {output_directory} -s {param_s} -e {param_e} --max-seqs {param_max_seqs} "

        result = os.system(cmd)

        # 检查命令执行情况
        if result == 0:
            print(f"{basename}: foldseek 执行成功")
        else:
            print(f"{basename}: foldseek 执行失败")



if __name__=="__main__":

    file_dir = "../data/pdb/Scope40/"

    foldseek = "../bin/foldseek"


    basenames = []

    for root, dirs, files in os.walk(file_dir):
        for file in files:
            basenames.append(file)


    output_directory = "../data/result/Scope40/foldseek"

    exec_foldseek_easy_search_para(foldseek,'../data/foldseekDB/Scope40DB/DB/scope40DB', basenames, output_directory, 9.5, 10, 1000)

