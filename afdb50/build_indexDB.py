import numpy as np

import faiss
import time
import multiprocessing

"""
Model: SaProt_650M_AF2.pt

First, how to use the foldseek library? 

accession_list  This is the complete list of 200 million protein names.

afdb50_subset is the amino acid sequence.
Although afdb50 only has 5,000 entries, the official ones below all have 200 million entries.

afdb50_subset.lookup    afdb50_subset    afdb50_subset.fasta (this is self-created)  afdb50_subset_taxonomy   afdb50_subset_mapping afdb50_subset.dbtype  afdb50_subset.index

head afdb50_subset.lookup 

Serial number  Protein name
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

head afdb50_subset
Corresponding protein sequences   There are only 5Kw entries here.
VGTSLSVLIRAELGHPGALIGDDQIYNVIVTAHAFVMIFFMVMPIMIGGFGNWLVPLMLGAPDMAFPRMNNMSFWLLPPSLTLLLVSSMVENGAGTGWTVYPPLSASIAHGGASVDLAIFSLHLAGMSSILGAVNFITTVINMRSHGISYDRMPLFVWSVVITALLLLLSLPVLAGAITMLLTD
MIQIIYSSIIIILILIIFTLMRKIRRIKKEHRLRLANLYKLLSKLTSDEKIYRDKIKLDNSLAKKISEAKAQLNTDIFDLQINIFKKIIEK

head afdb50_subset.index 
There are only 5Kw entries in the index file here. You can see that the serial numbers are not consecutive because there are many non-existent entries among the 200 million entries.
Protein serial number  Start position  End position
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


So you can use the index to read quickly.
dd if=afdb50_subset  bs=1 skip=186 count=93 
Read the afdb50_subset file, starting from the 186th byte, and read 93 bytes. This allows you to read very quickly.

MIQIIYSSIIIILILIIFTLMRKIRRIKKEHRLRLANLYKLLSKLTSDEKIYRDKIKLDNSLAKKISEAKAQLNTDIFDLQINIFKKIIEK

1. First, use faiss
    faiss index file : afdb50_{dim}_IVFPQ_faiss.faiss
    afdb50_combined.fasta  file. This file contains the 5K w-based index file,
    and the order in which the index is added is also based on this file.
    So, how can we quickly find the corresponding sequence based on the index returned by faiss?

    >AF-A0A0S2QPT1-F1-model_v4 AF-A0A0S2QPT1-F1-model_v4 Cytochrome c oxidase subunit 1
    VdGlTvSvLlSvVvLvIlRcAvEqLvGvHdPpGdAgLpIpGvDdDpQlIlYnNlVqIsVqTqAlHsAvFcVcMcIpFlFvMnVvMcPcIcMlIlGvGnFcGvNqWpLvVlPcLvMlLqGvAfPpDgMwApFcPvRpMlNsNvMlSlFvWvLlLlPvPqSlLnTvLlLsLvVvSlSvMvVpEdNsGhAdGsTcGhWnTvVcYdPpPpLcSnAaScIvAnHdGnGdAcSsVsDvLsAnIlFvSsLlHvLsAsGlMsSsSlIlLsGsAlVvNrFlIlTcTcVvIvNpMgRgSdHpGpIdSdYpDvRnMrPdLvFsVsWvSvVnVnIvTvAsLvLvLsLnLvSvLsPvVvLvAnGvAvIsTvMvLrLvTvDd
    >AF-A0A1Q3Z5S1-F1-model_v4 AF-A0A1Q3Z5S1-F1-model_v4 Uncharacterized protein
    MdIvQvIvIvYvSvSvIvIvIvIvLvIvLvIvIvFvTvLvMvRvKvIvRvRvIvKvKvEvHvRvLvRvLvAvNvLvYvKvLvLvSvKvLvTvSvDvEvKvIvYvRvDvKvIvKvLvDvNvSvLvAvKvKvIvSvEvAvKvAvQvLvNvTvDvIvFvDvLvQvIvNvIvFvKvKvIvIvEvKd


"""


"""
Reference
afdb50_subset.lookup   afdb50_subset  afdb50_subset.index

ssalign_afdb50_combined_seq.lookup  # Serial number, protein name
ssalign_afdb50_combined_seq         
ssalign_afdb50_combined_seq.index  # Build index

"""
def build_indexDB_1():
    with open('/data2/zxc_data/afdb50/afdb50_combined.fasta', 'r') as fasta_file, open('./ssalign_afdb50_combined_seq.lookup', 'w') as lookup_file:
        seq_num = 0 
        protein_id = ''
        for line in fasta_file:
            if line.startswith('>'):  
                if protein_id: 
                    lookup_file.write(f"{seq_num}\t{protein_id}\n")
                protein_id = line.strip().split()[0].lstrip('>') 
                seq_num += 1
        if protein_id:
            lookup_file.write(f"{seq_num}\t{protein_id}\n")

def build_indexDB_2():
    with open('/data2/zxc_data/afdb50/afdb50_combined.fasta', 'r') as fasta_file, open('./ssalign_afdb50_combined_seq', 'w') as combined_file:
        sequence = ''
        for line in fasta_file:
            if line.startswith('>'):
                if sequence:  
                    combined_file.write(sequence + '\n')
                sequence = ''
            else:
                sequence += line.strip()  
        if sequence:
            combined_file.write(sequence + '\n')

def build_indexDB_3():
    with open('./ssalign_afdb50_combined_seq', 'r') as seq_file, open('./ssalign_afdb50_combined_seq.index', 'w') as index_file:
        seq_num = 1 
        start_pos = 0  
        for line in seq_file:
            sequence = line.strip() 
            end_pos = start_pos + len(sequence) 
            index_file.write(f"{seq_num}\t{start_pos}\t{end_pos}\n")
            start_pos = end_pos+1
            seq_num += 1 




def get_protein_by_index(lookup_file_path, index_file_path, seq_file_path, target_index):
    """
    First load it into memory, as it will be needed later and occupies 3G of memory.
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



    # 2. Get protein names from protein_name_map
    protein_name = protein_name_map.get(target_index, None)
    if not protein_name:
        print(f"序号 {target_index} 不存在于 lookup 文件中")
        

    # 3. Get the start and end positions of the sequence from index_map
    start_pos, end_pos = index_map.get(target_index, (None, None))
    if start_pos is None or end_pos is None:
        print(f"序号 {target_index} 不存在于 index 文件中")
        return None, None

    # 4. Read the corresponding sequence from the sequence file
    with open(seq_file_path, 'r') as seq_file:
        seq_file.seek(start_pos)
        sequence = seq_file.read(end_pos - start_pos).strip()

    end = time.time()

    print(end-start)

    return protein_name, sequence

"""
Batch query
target_indices  is the [index, score] returned in the first stage

Find the protein name based on the index.
1. Sequences that reach the threshold are not needed
2. Sequences that do not reach the threshold are needed
"""
#def get_protein_by_index_batch(lookup_file_path, index_file_path, seq_file_path, remaining_results,prefilter_threshold):
def get_protein_by_index_batch(lookup_dict, index_dict, seq_file_path, remaining_results,prefilter_threshold):
    with open(seq_file_path, 'r') as seq_file:
        for i, (target_index, prefilter_score) in enumerate(remaining_results):
            protein_name = lookup_dict.get(target_index, None)
            if protein_name is None:
                print(f"序号 {target_index} 不存在于 lookup 文件中")
                continue

         
            if i < prefilter_threshold:
                prefilter_results_pdb.append((protein_name, prefilter_score))
            else:
                start_pos, end_pos = index_dict.get(target_index, (None, None))
                if start_pos is None or end_pos is None:
                    print(f"序号 {target_index} 不存在于 index 文件中")
                    continue

                seq_file.seek(start_pos)
                sequence = seq_file.read(end_pos - start_pos).strip()

                saligner_pdb.append((protein_name, prefilter_score, sequence))

    # Return two parts: a list of protein names only and a list containing sequences.
    return prefilter_results_pdb, saligner_pdb



def get_protein_by_index_batch_ok(lookup_dict, index_dict, seq_file_path, all_remaining_results,prefilter_threshold):

    all_prefilter_results_pdb = []
    all_aligner_pdb = []

    with open(seq_file_path, 'r') as seq_file:
        for remaining_results in all_remaining_results:
            prefilter_results_pdb = []  
            saligner_pdb = []  
            for i, (target_index, prefilter_score) in enumerate(remaining_results):
                protein_name = lookup_dict.get(target_index, None)
                if protein_name is None:
                    print(f"序号 {target_index} 不存在于 lookup 文件中")
                    continue
                if i < prefilter_threshold:
                    prefilter_results_pdb.append((protein_name, prefilter_score))
                else:
                    start_pos, end_pos = index_dict.get(target_index, (None, None))
                    if start_pos is None or end_pos is None:
                        print(f"序号 {target_index} 不存在于 index 文件中")
                        continue

                    seq_file.seek(start_pos)
                    sequence = seq_file.read(end_pos - start_pos).strip()

                    saligner_pdb.append((protein_name, prefilter_score, sequence))

            all_prefilter_results_pdb.append(prefilter_results_pdb)
            all_aligner_pdb.append(saligner_pdb)

    return all_prefilter_results_pdb,all_aligner_pdb




if __name__=="__main__":

    lookup_file = './ssalign_afdb50_combined_seq.lookup'
    index_file = './ssalign_afdb50_combined_seq.index'
    seq_file = './ssalign_afdb50_combined_seq'

    protein_name, sequence = get_protein_by_index(lookup_file, index_file, seq_file, 3000000)
    if protein_name and sequence:
        print(f"蛋白质名: {protein_name}")
        print(f"序列: {sequence}")
   




