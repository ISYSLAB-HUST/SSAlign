import numpy as np

import faiss
import time
import multiprocessing





"""


模型：
/home/xuchaozhang/ssalign/ssalign/SaprotProject/models/SaProt_650M_AF2.pt

首先，怎么利用foldseek的库？ /data2/zxc_data/afdb50/

accession_list  这里是完整的 2亿条蛋白质名



afdb50_subset 就是氨基酸序列
虽然afdb50只有 5KW条，但是下面这些官方给的，都是 2亿条，
afdb50_subset.lookup    afdb50_subset    afdb50_subset.fasta(这个是自己创建的)  afdb50_subset_taxonomy   afdb50_subset_mapping afdb50_subset.dbtype  afdb50_subset.index

(base) xuchaozhang@ubuntu:/data2/zxc_data/afdb50$ head afdb50_subset.lookup 
两亿条
序号  蛋白质名字               
0	AF-A0A3S5XGV8-F1-model_v4	0
1	AF-A0A451ES27-F1-model_v4	0
2	AF-A0A142IBY3-F1-model_v4	0
3	AF-B9X0D7-F1-model_v4	0
4	AF-V5Q9C9-F1-model_v4	0
5	AF-D8XMR9-F1-model_v4	0
6	AF-A0A141GTK8-F1-model_v4	0
7	AF-A0A141GT99-F1-model_v4	0
8	AF-A0A0A0RIM5-F1-model_v4	0
9	AF-A0A141GT39-F1-model_v4	0

(base) xuchaozhang@ubuntu:/data2/zxc_data/afdb50$ head afdb50_subset
对应的蛋白质序列   这里就只有5Kw条了
VGTSLSVLIRAELGHPGALIGDDQIYNVIVTAHAFVMIFFMVMPIMIGGFGNWLVPLMLGAPDMAFPRMNNMSFWLLPPSLTLLLVSSMVENGAGTGWTVYPPLSASIAHGGASVDLAIFSLHLAGMSSILGAVNFITTVINMRSHGISYDRMPLFVWSVVITALLLLLSLPVLAGAITMLLTD
MIQIIYSSIIIILILIIFTLMRKIRRIKKEHRLRLANLYKLLSKLTSDEKIYRDKIKLDNSLAKKISEAKAQLNTDIFDLQINIFKKIIEK

(base) xuchaozhang@ubuntu:/data2/zxc_data/afdb50$ head afdb50_subset.index 
索引文件这里就只有5Kw条了。可以看到序号不连续，因为2亿条里面有很多不存在的

蛋白质序号  起始位置  结束位置
14	0	186
41	186	93
48	279	81
53	360	62
55	422	73
69	495	158
73	653	85
78	738	169
80	907	540
86	1447	321


所以可以利用索引快速读取
dd if=afdb50_subset  bs=1 skip=186 count=93 
读取afdb50_subset文件，从第186字节开始，读取93个。这样就可以非常快速地读取

MIQIIYSSIIIILILIIFTLMRKIRRIKKEHRLRLANLYKLLSKLTSDEKIYRDKIKLDNSLAKKISEAKAQLNTDIFDLQINIFKKIIEK




这里也是类似的

afdb50_subset_ss 就是3Di序列
afdb50_subset_ss       afdb50_subset_ss.lookup   afdb50_subset_ss.dbtype   afdb50_subset_ss.index  afdb50_subset_ss.fasta

------------------------------------
下面是应该是一些我们用不上的文件，代表结构文件：ca原子......
afdb50_subset_ss_h  afdb50_subset_h  afdb50_subset_h.dbtype afdb50_subset_ss_h.index  afdb50_subset_ss_h.dbtype
afdb50_ss       afdb50_ss.lookup   afdb50_ss.index   afdb50_ss.dbtype
afdb50_h     afdb50_h.index      afdb50_h.dbtype afdb50_subset_h.index
afdb50  afdb50.index  afdb50.dbtype    afdb50_mapping  afdb50_taxonomy   afdb50.lookup
afdb50_ca   afdb50_ca.index   afdb50_ca.dbtype





1. 首先，使用faiss
    faiss索引文件
    /data2/zxc_data/afdb50_combined_fasta/faiss_index/other/IVFPQ/afdb50_{dim}_IVFPQ_faiss.faiss

    /data2/zxc_data/afdb50/afdb50_combined.fasta  文件。这里面是将 5K w条构建的
    索引文件中的加入索引的顺序也是根据这个文件加入的
    所以如何根据 faiss返回的索引，快速地找到对应的序列?

    >AF-A0A0S2QPT1-F1-model_v4 AF-A0A0S2QPT1-F1-model_v4 Cytochrome c oxidase subunit 1
    VdGlTvSvLlSvVvLvIlRcAvEqLvGvHdPpGdAgLpIpGvDdDpQlIlYnNlVqIsVqTqAlHsAvFcVcMcIpFlFvMnVvMcPcIcMlIlGvGnFcGvNqWpLvVlPcLvMlLqGvAfPpDgMwApFcPvRpMlNsNvMlSlFvWvLlLlPvPqSlLnTvLlLsLvVvSlSvMvVpEdNsGhAdGsTcGhWnTvVcYdPpPpLcSnAaScIvAnHdGnGdAcSsVsDvLsAnIlFvSsLlHvLsAsGlMsSsSlIlLsGsAlVvNrFlIlTcTcVvIvNpMgRgSdHpGpIdSdYpDvRnMrPdLvFsVsWvSvVnVnIvTvAsLvLvLsLnLvSvLsPvVvLvAnGvAvIsTvMvLrLvTvDd
    >AF-A0A1Q3Z5S1-F1-model_v4 AF-A0A1Q3Z5S1-F1-model_v4 Uncharacterized protein
    MdIvQvIvIvYvSvSvIvIvIvIvLvIvLvIvIvFvTvLvMvRvKvIvRvRvIvKvKvEvHvRvLvRvLvAvNvLvYvKvLvLvSvKvLvTvSvDvEvKvIvYvRvDvKvIvKvLvDvNvSvLvAvKvKvIvSvEvAvKvAvQvLvNvTvDvIvFvDvLvQvIvNvIvFvKvKvIvIvEvKd



"""


"""

"""


"""
参考
afdb50_subset.lookup   afdb50_subset  afdb50_subset.index

ssalign_afdb50_combined_seq.lookup  序号、蛋白质名
ssalign_afdb50_combined_seq         
ssalign_afdb50_combined_seq.index  构建索引




"""
def build_indexDB_1():
    with open('/data2/zxc_data/afdb50/afdb50_combined.fasta', 'r') as fasta_file, open('./ssalign_afdb50_combined_seq.lookup', 'w') as lookup_file:
        seq_num = 0  # 用于生成序号
        protein_id = ''
        for line in fasta_file:
            if line.startswith('>'):  # 遇到蛋白质ID行
                if protein_id:  # 如果已经读取了一个蛋白质ID
                    lookup_file.write(f"{seq_num}\t{protein_id}\n")
                protein_id = line.strip().split()[0].lstrip('>')  # 获取蛋白质ID，取第一个字段
                seq_num += 1
        # 最后一行也需要写入
        if protein_id:
            lookup_file.write(f"{seq_num}\t{protein_id}\n")

def build_indexDB_2():
    with open('/data2/zxc_data/afdb50/afdb50_combined.fasta', 'r') as fasta_file, open('./ssalign_afdb50_combined_seq', 'w') as combined_file:
        sequence = ''
        for line in fasta_file:
            if line.startswith('>'):  # 如果是ID行，跳过
                if sequence:  # 如果之前有序列，写入
                    combined_file.write(sequence + '\n')
                sequence = ''  # 重置序列
            else:
                sequence += line.strip()  # 去掉行末的换行符，拼接序列
        # 最后一个序列也要写入
        if sequence:
            combined_file.write(sequence + '\n')

def build_indexDB_3():
    with open('./ssalign_afdb50_combined_seq', 'r') as seq_file, open('./ssalign_afdb50_combined_seq.index', 'w') as index_file:
        seq_num = 1  # 序号从1开始
        start_pos = 0  # 当前文件的字节位置
        for line in seq_file:
            sequence = line.strip()  # 去掉行末的换行符，得到序列
            end_pos = start_pos + len(sequence)  # 计算结束位置
            # 写入序号、开始位置、结束位置
            index_file.write(f"{seq_num}\t{start_pos}\t{end_pos}\n")
            start_pos = end_pos+1   # 更新当前的起始位置为当前序列的结束位置
            seq_num += 1  # 序号递增




def get_protein_by_index(lookup_file_path, index_file_path, seq_file_path, target_index):
   

    """
    先加载到内存当中，后面都需要用到，占用3G内存

    """
    protein_name_map = {}
    with open(lookup_file_path, 'r') as lookup_file:
        for line in lookup_file:
            seq_num, protein_name = line.strip().split('\t')
            protein_name_map[int(seq_num)] = protein_name

    index_map = {}
    with open(index_file_path, 'r') as index_file:
        for line in index_file:
            seq_num, start_pos, end_pos = map(int, line.strip().split())
            index_map[int(seq_num)] = (start_pos, end_pos)


    start = time.time()



    # 2. 从 protein_name_map 获取蛋白质名
    protein_name = protein_name_map.get(target_index, None)
    if not protein_name:
        print(f"序号 {target_index} 不存在于 lookup 文件中")
        

    # 3. 从 index_map 获取序列的起始和结束位置
    start_pos, end_pos = index_map.get(target_index, (None, None))
    if start_pos is None or end_pos is None:
        print(f"序号 {target_index} 不存在于 index 文件中")
        return None, None

    # 4. 读取序列文件中的对应序列
    with open(seq_file_path, 'r') as seq_file:
        seq_file.seek(start_pos)
        sequence = seq_file.read(end_pos - start_pos).strip()

    end = time.time()

    print(end-start)

    return protein_name, sequence






"""
批次查询
"""

"""
批次查询
target_indices  是第一阶段返回的  [索引，分数]

根据索引找到蛋白质名字。
1. 达到阈值的不需要序列
2. 未达到阈值的需要序列
"""
#def get_protein_by_index_batch(lookup_file_path, index_file_path, seq_file_path, remaining_results,prefilter_threshold):
def get_protein_by_index_batch(lookup_dict, index_dict, seq_file_path, remaining_results,prefilter_threshold):


    time1 = time.time()
    # 从 lookup 文件获取蛋白质名
#    lookup_dict = {}
#    with open(lookup_file_path, 'r') as lookup_file:
#        for line in lookup_file:
#            seq_num, name = line.strip().split('\t')
#            lookup_dict[int(seq_num)] = name
#
    # 从 index 文件获取序列的起始和结束位置
#    index_dict = {}
#    with open(index_file_path, 'r') as index_file:
#        for line in index_file:
#            seq_num, start, end = map(int, line.strip().split())
#            index_dict[seq_num] = (start, end)
#
#
    prefilter_results_pdb = [] # 达到了阈值
    saligner_pdb = []   # 未达到阈值
#
#    time2 = time.time()
#
#    print(f"加载两个索引文件，cost{time2 - time1}")

    # 从 seq 文件读取蛋白质序列
    with open(seq_file_path, 'r') as seq_file:
        for i, (target_index, prefilter_score) in enumerate(remaining_results):
            # 查询蛋白质名
            protein_name = lookup_dict.get(target_index, None)
            # 没查询到
            if protein_name is None:
                print(f"序号 {target_index} 不存在于 lookup 文件中")
                continue

            # 下面就是都查询到了
            # 如果是前 prefilter_threshold 个索引，只获取 protein_name
            if i < prefilter_threshold:
                prefilter_results_pdb.append((protein_name, prefilter_score))
            else:
                # 查询序列起始和结束位置
                start_pos, end_pos = index_dict.get(target_index, (None, None))
                if start_pos is None or end_pos is None:
                    print(f"序号 {target_index} 不存在于 index 文件中")
                    continue

                # 获取对应序列
                seq_file.seek(start_pos)
                sequence = seq_file.read(end_pos - start_pos).strip()

                # 添加到结果列表
                saligner_pdb.append((protein_name, prefilter_score, sequence))

    # 返回两个部分：只有蛋白质名的列表和包含序列的列表
    return prefilter_results_pdb, saligner_pdb



def get_protein_by_index_batch_ok(lookup_dict, index_dict, seq_file_path, all_remaining_results,prefilter_threshold):

    all_prefilter_results_pdb = []
    all_aligner_pdb = []

    # 从 seq 文件读取蛋白质序列
    with open(seq_file_path, 'r') as seq_file:
        for remaining_results in all_remaining_results:
            prefilter_results_pdb = []  # 达到了阈值
            saligner_pdb = []  # 未达到阈值
            for i, (target_index, prefilter_score) in enumerate(remaining_results):
                # 查询蛋白质名
                protein_name = lookup_dict.get(target_index, None)
                # 没查询到
                if protein_name is None:
                    print(f"序号 {target_index} 不存在于 lookup 文件中")
                    continue

                # 下面就是都查询到了
                # 如果是前 prefilter_threshold 个索引，只获取 protein_name
                if i < prefilter_threshold:
                    prefilter_results_pdb.append((protein_name, prefilter_score))
                else:
                    # 查询序列起始和结束位置
                    start_pos, end_pos = index_dict.get(target_index, (None, None))
                    if start_pos is None or end_pos is None:
                        print(f"序号 {target_index} 不存在于 index 文件中")
                        continue

                    # 获取对应序列
                    seq_file.seek(start_pos)
                    sequence = seq_file.read(end_pos - start_pos).strip()

                    # 添加到结果列表
                    saligner_pdb.append((protein_name, prefilter_score, sequence))

            all_prefilter_results_pdb.append(prefilter_results_pdb)
            all_aligner_pdb.append(saligner_pdb)


    # 返回两个部分：只有蛋白质名的列表和包含序列的列表
    # return prefilter_results_pdb, saligner_pdb
    return all_prefilter_results_pdb,all_aligner_pdb




if __name__=="__main__":
#    with multiprocessing.Pool(3) as pool:
#        # 将任务分发到进程池中，异步执行
#        pool.apply_async(build_indexDB_1)
#        pool.apply_async(build_indexDB_2)
#        pool.apply_async(build_indexDB_3)
#
#        # 等待所有进程完成
#        pool.close()
#        pool.join()

#    build_indexDB_3()
#    print("All index DBs are built.")
    
    # 示例调用：
    # 示例调用：
    lookup_file = './ssalign_afdb50_combined_seq.lookup'
    index_file = './ssalign_afdb50_combined_seq.index'
    seq_file = './ssalign_afdb50_combined_seq'

    # 获取第 1 个序号对应的蛋白质名和序列
    protein_name, sequence = get_protein_by_index(lookup_file, index_file, seq_file, 3000000)
    if protein_name and sequence:
        print(f"蛋白质名: {protein_name}")
        print(f"序列: {sequence}")
   




