import esm
import itertools
import torch

from esm.model.esm2 import ESM2
from utils.constants import foldseek_seq_vocab, foldseek_struc_vocab

# 加载一个基于 ESM（Evolutionary Scale Modeling）的蛋白质语言模型（SaProt）

def load_esm_saprot(path: str):
    """
    Load SaProt model of esm version.
    Args:
        path: path to SaProt model
    """
    
    # Initialize the alphabet
    # 定义了一组基础标记（tokens），例如 <cls>（分类标记）、<pad>（填充标记）、<eos>（结束标记）、<unk>（未知标记）、<mask>（掩码标记）。
    tokens = ["<cls>", "<pad>", "<eos>", "<unk>", "<mask>"]
    # 使用 itertools.product 生成 序列和结构的组合标记。
    # 将 foldseek_seq_vocab 的序列信息 与来自 foldseek_struc_vocab 的结构信息进行两两组合，构建出一个自定义的词汇表。
    for seq_token, struc_token in itertools.product(foldseek_seq_vocab, foldseek_struc_vocab):
        token = seq_token + struc_token
        tokens.append(token)

    # ESM 模型使用的字母表对象，将tokens  编码成模型可接受的格式。
    alphabet = esm.data.Alphabet(standard_toks=tokens,
                                 prepend_toks=[], # 在每个序列的开头和结尾分别加上 <cls> 和 <eos>。
                                 append_toks=[],
                                 prepend_bos=True,
                                 append_eos=True,
                                 use_msa=False)  # 不使用MSA比对，所以就是单序列输入
    
    alphabet.all_toks = alphabet.all_toks[:-2]  # 移除最后两个特殊标记
    alphabet.unique_no_split_tokens = alphabet.all_toks
    alphabet.tok_to_idx = {tok: i for i, tok in enumerate(alphabet.all_toks)}  # 创建了一个字典，将每个标记映射到一个唯一的索引，用于数值化序列输入。
    
    # Load weights
    data = torch.load(path)  # 加载模型权重文件
    weights = data["model"]  # 提取权重矩阵
    config = data["config"]  # iqu模型的配置信息，层数、嵌入维度、注意力头数等，用于构建模型架构
    
    # Initialize the model初始化模型
    model = ESM2(
        num_layers=config["num_layers"],
        embed_dim=config["embed_dim"],
        attention_heads=config["attention_heads"],
        alphabet=alphabet,
        token_dropout=config["token_dropout"],
    )
    # 将加载的权重应用到模型的参数中
    load_weights(model, weights)
    return model, alphabet



def load_weights(model, weights):
    """
    加载模型的权重
    Args:
        model:
        weights:

    Returns:

    """
    model_dict = model.state_dict()

    unused_params = []
    missed_params = list(model_dict.keys())

    for k, v in weights.items():
        if k in model_dict.keys():
            model_dict[k] = v
            missed_params.remove(k)

        else:
            unused_params.append(k)

    if len(missed_params) > 0:
        print(f"\033[31mSome weights of {type(model).__name__} were not "
              f"initialized from the model checkpoint: {missed_params}\033[0m")

    if len(unused_params) > 0:
        print(f"\033[31mSome weights of the model checkpoint were not used: {unused_params}\033[0m")

    model.load_state_dict(model_dict)