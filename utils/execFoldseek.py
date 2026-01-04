import os
from utils.SSAlign_utils import add_foldseek_tmscore

"""
    运行foldseek并且添加指标！
"""


"""
    All-against-all performance evaluation on the SCOPe40 dataset.
    foldseek
"""
def exec_foldseek_easy_search_para_SCOPe40():

    foldseek = "../bin/foldseek"

    file_dir = "../pdbData/pdb/SCOPe40/"
    basenames = []
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            basenames.append(file)

    foldeseekDB = '../models/foldseekDB/SCOPe40/SCOPe40DB/'
    output_directory = "../benchmarkData/SCOPe40/foldseek"
    param_s = 9.5
    param_e = 10
    param_max_seqs = 1000


    # 遍历每个文件basename并执行foldseek命令
    for basename in basenames:

        file_path = f"../pdbData/SwissProt/{basename}"

        output_path = f"{basename}_foldseek"

        # 构造并执行命令
        cmd = f"{foldseek} easy-search {file_path} {foldeseekDB}  {output_path} {output_directory} -s {param_s} -e {param_e} --max-seqs {param_max_seqs} "

        result = os.system(cmd)

        # 检查命令执行情况
        if result == 0:
            print(f"{basename}: foldseek 执行成功")
        else:
            print(f"{basename}: foldseek 执行失败")

    """
        添加 TM-Score指标
    """
    # 添加tmalign计算过的指标
    for basename in basenames:
        foldseek_file = f"../benchmarkData/SCOPe40/foldseek/{basename}_foldseek"

        tmalign_file = f"../benchmarkData/SCOPe40/tmalign/{basename}.result"

        output_file = f"../benchmarkData/SCOPe40/foldseek/{basename}.result"

        add_foldseek_tmscore(foldseek_file, tmalign_file, output_file, basename)



"""
    This benchmark utilizes the Swiss-Prot dataset comprising 100 query proteins against a database of 542,378 entries
    foldseek
"""
def exec_foldseek_easy_search_para_SwissProt():

    foldseek = "../bin/foldseek"
    basenames = []

    with open("../SwissProt/100filenames.txt") as f:
        for line in f:
            line = line.strip()
            basenames.append(line)

    foldeseekDB = '../pdbData/foldseekDB/SwissProtDB/spDB/'
    output_directory = "../benchmarkData/SwissProt/foldseek"
    param_s = 9.5
    param_e = 10
    param_max_seqs =1000


    for basename in basenames:

        file_path = f"../pdbData/SwissProt/{basename}"

        output_path = f"{basename}_foldseek"

        # 构造并执行命令
        cmd = f"{foldseek} easy-search {file_path} {foldeseekDB}  {output_path} {output_directory} -s {param_s} -e {param_e} --max-seqs {param_max_seqs} "

        result = os.system(cmd)

        # 检查命令执行情况
        if result == 0:
            print(f"{basename}: foldseek 执行成功")
        else:
            print(f"{basename}: foldseek 执行失败")

    """
        添加 TM-Score指标
    """
    # 添加tmalign计算过的指标
    for basename in basenames:
        foldseek_file = f"../benchmarkData/SwissProt/foldseek/{basename}_foldseek"

        tmalign_file = f"../benchmarkData/SwissProt/tmalign/{basename}.result"

        output_file = f"../benchmarkData/SwissProt/foldseek/{basename}.result"

        add_foldseek_tmscore(foldseek_file, tmalign_file, output_file, basename)






