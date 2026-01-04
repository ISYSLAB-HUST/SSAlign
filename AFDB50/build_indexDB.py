
"""
参考
afdb50_subset.lookup   afdb50_subset  afdb50_subset.index

ssalign_afdb50_combined_seq.lookup  序号、蛋白质名
ssalign_afdb50_combined_seq         
ssalign_afdb50_combined_seq.index  构建索引

"""
def build_lookup_tsv_from_fasta():
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

def build_one_record_per_line_seqfile():
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

def build_byte_offset_index_for_seqfile():
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



"""

"""
def get_protein_by_index(lookup_file_path, index_file_path, seq_file_path, target_index):
    # 从 lookup 文件获取蛋白质名
    protein_name = None
    with open(lookup_file_path, 'r') as lookup_file:
        for line in lookup_file:
            seq_num, name = line.strip().split('\t')
            if int(seq_num) == target_index:
                protein_name = name
                break
        else:
            print(f"序号 {target_index} 不存在于 lookup 文件中")
            return None, None
    
    # 从 index 文件获取序列的起始和结束位置
    start_pos, end_pos = None, None
    with open(index_file_path, 'r') as index_file:
        for line in index_file:
            seq_num, start, end = map(int, line.strip().split())
            if seq_num == target_index:
                start_pos, end_pos = start, end
                break
        else:
            print(f"序号 {target_index} 不存在于 index 文件中")
            return None, None


    # 从 seq 文件读取对应的蛋白质序列
    with open(seq_file_path, 'r') as seq_file:
        seq_file.seek(start_pos)
        sequence = seq_file.read(end_pos - start_pos).strip()

    return protein_name, sequence




"""
批次查询
target_indices  是第一阶段返回的  [索引，分数]
    
根据索引找到蛋白质名字。
1. 达到阈值的不需要序列
2. 未达到阈值的需要序列
"""
def get_protein_by_index_batch(lookup_dict, index_dict, seq_file_path, all_remaining_results,prefilter_threshold):

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
    return all_prefilter_results_pdb,all_aligner_pdb





if __name__=="__main__":

    
    # 示例调用：
    # 示例调用：
    lookup_file = '../models/SSAlignDB/AFDB50/ssalign_afdb50_combined_seq.lookup'
    index_file = '../models/SSAlignDB/AFDB50/ssalign_afdb50_combined_seq.index'
    seq_file = '../models/SSAlignDB/AFDB50/ssalign_afdb50_combined_seq'

    # 获取第 1 个序号对应的蛋白质名和序列
    protein_name, sequence = get_protein_by_index(lookup_file, index_file, seq_file, 3)
    if protein_name and sequence:
        print(f"蛋白质名: {protein_name}")
        print(f"序列: {sequence}")
   




