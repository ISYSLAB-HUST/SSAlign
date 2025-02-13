import os
import re
import time
import json
import numpy as np
import sys
sys.path.append(".")


# 使用foldseek工具将PDB文件转为3Di结构描述符
#
def get_struc_seq(foldseek,
                  path,
                  chains: list = None,
                  process_id: int = 0,
                  plddt_mask: bool = False,
                  plddt_threshold: float = 70.,
                  foldseek_verbose: bool = False) -> dict:
    """

    Args:
        foldseek: Binary executable file of foldseek

        path: Path to pdb file

        所以foldseek是按蛋白质的 链 来提取3Di特征的，这里指定链，不然会提取所有的链
        返回的结果就是
            {
                "A": (seq_A, foldseek_seq_A, combined_seq_A),
                "B": (seq_B, foldseek_seq_B, combined_seq_B),
                ...
            }
        chains: Chains to be extracted from pdb file. If None, all chains will be extracted.

        process_id: Process ID for temporary files. This is used for parallel processing.

        是否启用 plddt 掩码，如果启用，将根据 plddt_threshold 过滤序列。
        plddt_mask: If True, mask regions with plddt < plddt_threshold. plddt scores are from the pdb file.

        plddt_threshold: Threshold for plddt. If plddt is lower than this value, the structure will be masked.

        foldseek_verbose: If True, foldseek will print verbose messages.

    Returns:
        seq_dict: A dict of structural seqs. The keys are chain IDs. The values are tuples of
        (seq, struc_seq, combined_seq).

    返回值是一个字典，其中每个链的键是链 ID，值是一个包含三个元素的元组 (seq, struc_seq, combined_seq)，分别表示：
    seq: 原始蛋白质序列（氨基酸序列,大写字母）。
    struc_seq: 结构序列（即 3Di 编码，也是大写字母）。
    combined_seq: 将蛋白质序列和结构序列按位置组合而成的新序列，每个位置的氨基酸与对应的结构编码组合。但是将struc_seq中的大写转为了小写，所以这里面的小写代表3Di了

    """
    assert os.path.exists(foldseek), f"Foldseek not found: {foldseek}"
    assert os.path.exists(path), f"PDB file not found: {path}"
    
    tmp_save_path = f"get_struc_seq_{process_id}_{time.time()}.tsv"

    # 这里是通过执行命令的方式来调用foldseek
    # 调用Foldseek工具的structureto3didescriptor功能，将PDB文件转为3Di描述符，保存临时文件到tmp_save_path
    if foldseek_verbose:
        cmd = f"{foldseek} structureto3didescriptor --threads 1 --chain-name-mode 1 {path} {tmp_save_path}"
    else:
        cmd = f"{foldseek} structureto3didescriptor -v 0 --threads 1 --chain-name-mode 1 {path} {tmp_save_path}"
    os.system(cmd)

    # 解析生成的 .tsv 文件，提取序列信息。
    # 如果启用了 plddt_mask，还会进一步根据 plddt 分数过滤低置信度区域。
    seq_dict = {}
    name = os.path.basename(path)
    with open(tmp_save_path, "r") as r:
        for i, line in enumerate(r):
            desc, seq, struc_seq = line.split("\t")[:3]
            
            # Mask low plddt
            if plddt_mask:
                plddts = extract_plddt(path)
                assert len(plddts) == len(struc_seq), f"Length mismatch: {len(plddts)} != {len(struc_seq)}"
                
                # Mask regions with plddt < threshold
                indices = np.where(plddts < plddt_threshold)[0]
                np_seq = np.array(list(struc_seq))
                np_seq[indices] = "#"
                struc_seq = "".join(np_seq)
            
            name_chain = desc.split(" ")[0]
            chain = name_chain.replace(name, "").split("_")[-1]
            # 根据指定的链提取结构序列（如果没有指定，则提取所有链），并将蛋白质序列和结构序列组合。
            if chains is None or chain in chains:
                if chain not in seq_dict:
                    # 注意看，这里将struc_seq中的大写转为了小写，然后再拼接给combined_seq！
                    combined_seq = "".join([a + b.lower() for a, b in zip(seq, struc_seq)])
                    seq_dict[chain] = (seq, struc_seq, combined_seq)
    # 清理临时文件
    os.remove(tmp_save_path)
    os.remove(tmp_save_path + ".dbtype")
    return seq_dict


def extract_plddt(pdb_path: str) -> np.ndarray:
    """
    从 PDB 文件中提取每个氨基酸位置的 plddt 分数。
    Plddt 分数通常用于衡量预测的蛋白质结构的置信度，范围为 0 到 100。
    Extract plddt scores from pdb file.
    Args:
        pdb_path: Path to pdb file.

    Returns:
        plddts: plddt scores.
    """
    with open(pdb_path, "r") as r:
        plddt_dict = {}
        for line in r:
            line = re.sub(' +', ' ', line).strip()
            splits = line.split(" ")
            
            if splits[0] == "ATOM":
                # If position < 1000
                if len(splits[4]) == 1:
                    pos = int(splits[5])
                
                # If position >= 1000, the blank will be removed, e.g. "A 999" -> "A1000"
                # So the length of splits[4] is not 1
                else:
                    pos = int(splits[4][1:])
                
                plddt = float(splits[-2])
                
                if pos not in plddt_dict:
                    plddt_dict[pos] = [plddt]
                else:
                    plddt_dict[pos].append(plddt)
    
    plddts = np.array([np.mean(v) for v in plddt_dict.values()])
    return plddts
