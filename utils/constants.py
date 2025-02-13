import itertools

"""
定义了一些关于蛋白质序列和结构信息的常量和工具函数。它主要为 SaProt 模型提供了一些基础词汇表和相关函数，用于表示蛋白质的序列和结构信息。
"""
# 包含标准的20种蛋白质氨基酸的单字母代码
aa_set = {"A", "C", "D", "E", "F", "G", "H", "I", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "V", "W", "Y"}
aa_list = ["A", "C", "D", "E", "F", "G", "H", "I", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "V", "W", "Y"]

# 表示氨基酸序列的词汇表，包括 20 种氨基酸以及一个额外的字符 #用于表示特殊标记或缺失数据。
foldseek_seq_vocab = "ACDEFGHIKLMNPQRSTVWY#"

# 这个是蛋白质的结构信息，也就是经过foldseek编码之后的3Di结构字母表的20个状态！
foldseek_struc_vocab = "pynwrqhgdlvtmfsaeikc#"

# 包含26个小写字母？
struc_unit = "abcdefghijklmnopqrstuvwxyz"


def create_vocab(size: int) -> dict:
    """
    创建一个词汇表vocab，根据输入的大小决定此表的长度和字符组合
    Args:
        size:   Size of the vocabulary

    Returns:
        vocab:  Vocabulary
    """
    # token_len 最初设为 1，表示最初假设每个词汇由一个字符组成。
    token_len = 1
    # 如果要求的词汇量size > len(struc_unit) ** token_len,则增加token_len的长度
    while size > len(struc_unit) ** token_len:
        token_len += 1

    vocab = {}
    for i, token in enumerate(itertools.product(struc_unit, repeat=token_len)):
        vocab[i] = "".join(token)
        if len(vocab) == size:
            vocab[i+1] = "#"
            return vocab