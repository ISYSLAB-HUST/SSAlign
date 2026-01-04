from utils.esm_loader import load_esm_saprot
import torch
import numpy as np


def combined_seq_to_vector_exist_model_1_cpu_batch(combined_seqs,model,alphabet):

    batch_converter = alphabet.get_batch_converter()  # 获取批量转换器，用于将数据转为模型输入的格式
    data = [(f"protein{i + 1}", seq) for i, seq in enumerate(combined_seqs)]

    batch_labels, batch_strs, batch_tokens = batch_converter(data)  # 使用批量转化器，将蛋白质序列转为模型的所需的输入格式


    batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)  # 计算每个序列的实际长度

    with torch.no_grad():  # torch.no_grad()用来禁止梯度计算。在推理阶段（即预测时），不需要计算梯度，节省内存和计算资源。
        results_1 = model(batch_tokens, repr_layers=[1], return_contacts=True)

    token_representations_1 = results_1["representations"][1]


    # 针对每个序列，排除填充位并计算平均表示
    sequence_representations = []
    for i, tokens_len in enumerate(batch_lens):
        sequence_rep = token_representations_1[i, 1:tokens_len - 1, :].mean(0)
        avg_representation = sequence_rep.unsqueeze(0)
        sequence_representations.append(avg_representation)



    # return sequence_representations.numpy()
    # 将嵌入结果转为 numpy 格式，返回包含多个蛋白质序列的嵌入
    return torch.stack(sequence_representations, dim=0).numpy()




def combined_seq_to_vector_exist_model_1_gpu_batch(combined_seqs,model,alphabet,cuda_device):

    batch_converter = alphabet.get_batch_converter()  # 获取批量转换器，用于将数据转为模型输入的格式
    data = [(f"protein{i + 1}", seq) for i, seq in enumerate(combined_seqs)]

    batch_labels, batch_strs, batch_tokens = batch_converter(data)  # 使用批量转化器，将蛋白质序列转为模型的所需的输入格式


    batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)  # 计算每个序列的实际长度

    with torch.no_grad():  # torch.no_grad()用来禁止梯度计算。在推理阶段（即预测时），不需要计算梯度，节省内存和计算资源。
        batch_tokens = batch_tokens.to(cuda_device)  # 将数据转移到指定 GPU
        results_1 = model(batch_tokens, repr_layers=[1], return_contacts=True)

    token_representations_1 = results_1["representations"][1]


    # 针对每个序列，排除填充位并计算平均表示
    sequence_representations = []
    for i, tokens_len in enumerate(batch_lens):
        sequence_rep = token_representations_1[i, 1:tokens_len - 1, :].mean(0)
        avg_representation = sequence_rep.unsqueeze(0)
        sequence_representations.append(avg_representation)

    # 删除临时结果，释放显存
    # 删除不再需要的变量并释放显存
    # 释放显存
    del token_representations_1, batch_tokens, results_1
    torch.cuda.empty_cache()

    # return sequence_representations.numpy()
    # 将嵌入结果转为 numpy 格式，返回包含多个蛋白质序列的嵌入
    return torch.stack(sequence_representations, dim=0).cpu().numpy()


"""
一次性处理多个序列
获得第1层和第33层的平均值
"""

def combined_seq_to_vector_exist_model_1_33_avg_cpu_batch(combined_seqs,model,alphabet):
    batch_converter = alphabet.get_batch_converter()  # 获取批量转换器，用于将数据转为模型输入的格式

    data = [(f"protein{i + 1}", seq) for i, seq in enumerate(combined_seqs)]

    batch_labels, batch_strs, batch_tokens = batch_converter(data)  # 使用批量转化器，将蛋白质序列转为模型的所需的输入格式

    batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)  # 计算每个序列的实际长度

    with torch.no_grad():  # torch.no_grad()用来禁止梯度计算。在推理阶段（即预测时），不需要计算梯度，节省内存和计算资源。
        results_1 = model(batch_tokens, repr_layers=[1], return_contacts=True)
        results_33 = model(batch_tokens, repr_layers=[33], return_contacts=True)

    token_representations_1 = results_1["representations"][1]
    token_representations_33 = results_33["representations"][33]

    # 针对每个序列，排除填充位并计算平均表示
    # # 针对每个序列计算其第1层和第33层的平均表示，并存储到列表中
    sequence_representations = []
    for i, tokens_len in enumerate(batch_lens):
        avg_representation = (token_representations_1[i, 1:tokens_len - 1, :].mean(0) + token_representations_33[i,1:tokens_len - 1, :].mean(0)) / 2
        avg_representation = avg_representation.unsqueeze(0)  # 将每个向量形状扩展为 (1, SVD1280)
        #print(avg_representation.shape)
        sequence_representations.append(avg_representation)

    torch.set_printoptions(sci_mode=False, threshold=5000)
    np.set_printoptions(suppress=True, threshold=5000)  # 禁用科学计数法

    # return sequence_representations.numpy()
    # 将嵌入结果转为 numpy 格式，返回包含多个蛋白质序列的嵌入
    return torch.stack(sequence_representations, dim=0).numpy()


def combined_seq_to_vector_exist_model_1_33_avg_cpu(combined_seq, model, alphabet):
    """
        获得第1层和第33层的平均值
    """
    batch_converter = alphabet.get_batch_converter()  # 获取批量转换器，用于将数据转为模型输入的格式

    name = "test"
    data = [(name, combined_seq)]

    batch_labels, batch_strs, batch_tokens = batch_converter(data)  # 使用批量转化器，将蛋白质序列转为模型的所需的输入格式

    with torch.no_grad():  # torch.no_grad()用来禁止梯度计算。在推理阶段（即预测时），不需要计算梯度，节省内存和计算资源。
        results_1 = model(batch_tokens, repr_layers=[1], return_contacts=True)
        results_33 = model(batch_tokens, repr_layers=[33], return_contacts=True)


    token_representations_1 = results_1["representations"][1][:, 1:-1, :].mean(1)  # 提取第 33 层的输出表示  句向量

    token_representations_33 = results_33["representations"][33][:, 1:-1, :].mean(1)  # 提取第 33 层的输出表示  句向量

    token_representations = (token_representations_33+token_representations_1) / 2

    torch.set_printoptions(sci_mode=False, threshold=5000)

    np.set_printoptions(suppress=True, threshold=5000)  # 禁用科学计数法

    return token_representations.numpy()


def combined_seq_to_vector_exist_model_1_33_avg_gpu(combined_seq, model, alphabet,cuda_device):
    """
        获得第1层和第33层的平均值
    """
    #print(f"Using GPU device: {cuda_device}")


    batch_converter = alphabet.get_batch_converter()  # 获取批量转换器，用于将数据转为模型输入的格式

    name = "test"
    data = [(name, combined_seq)]

    batch_labels, batch_strs, batch_tokens = batch_converter(data)  # 使用批量转化器，将蛋白质序列转为模型的所需的输入格式

    with torch.no_grad():  # torch.no_grad()用来禁止梯度计算。在推理阶段（即预测时），不需要计算梯度，节省内存和计算资源。
        batch_tokens = batch_tokens.to(cuda_device)
        # results_1 = model(batch_tokens, repr_layers=[1], return_contacts=True)
        # results_33 = model(batch_tokens, repr_layers=[33], return_contacts=True)

        # 分步提取第1层和第33层，避免显存过度占用
        results_1 = model(batch_tokens, repr_layers=[1], return_contacts=True)
        token_representations_1 = results_1["representations"][1][:, 1:-1, :].mean(1)

        #results_33 = model(batch_tokens, repr_layers=[33], return_contacts=True)
        #token_representations_33 = results_33["representations"][33][:, 1:-1, :].mean(1)

    # token_representations_1 = results_1["representations"][1][:, 1:-1, :].mean(1)  # 提取第 33 层的输出表示  句向量
    #
    # token_representations_33 = results_33["representations"][33][:, 1:-1, :].mean(1)  # 提取第 33 层的输出表示  句向量

    token_representations = (token_representations_1+token_representations_1) / 2

    torch.set_printoptions(sci_mode=False, threshold=5000)

    np.set_printoptions(suppress=True, threshold=5000)  # 禁用科学计数法

    return token_representations.cpu().numpy()


def combined_seq_to_vector_exist_model(combined_seq, model, alphabet):
    batch_converter = alphabet.get_batch_converter()  # 获取批量转换器，用于将数据转为模型输入的格式

    name = "test"
    data = [(name, combined_seq)]

    batch_labels, batch_strs, batch_tokens = batch_converter(data)  # 使用批量转化器，将蛋白质序列转为模型的所需的输入格式

    with torch.no_grad():  # torch.no_grad()用来禁止梯度计算。在推理阶段（即预测时），不需要计算梯度，节省内存和计算资源。
        results = model(batch_tokens, repr_layers=[33], return_contacts=True)


    token_representations = results["representations"][33][:, 1:-1, :].mean(1)  # 提取第 33 层的输出表示  句向量

    torch.set_printoptions(sci_mode=False, threshold=5000)

    np.set_printoptions(suppress=True, threshold=5000)  # 禁用科学计数法

    return token_representations.numpy()

"""
使用gpu资源
"""
def combined_seq_to_vector_exist_model_gpu(combined_seq, model, alphabet,cuda_device):


    batch_converter = alphabet.get_batch_converter()  # 获取批量转换器，用于将数据转为模型输入的格式

    name = "test"
    data = [(name, combined_seq)]

    batch_labels, batch_strs, batch_tokens = batch_converter(data)  # 使用批量转化器，将蛋白质序列转为模型的所需的输入格式

    with torch.no_grad():  # torch.no_grad()用来禁止梯度计算。在推理阶段（即预测时），不需要计算梯度，节省内存和计算资源。
        batch_tokens = batch_tokens.to(cuda_device)
        results = model(batch_tokens, repr_layers=[33], return_contacts=True)


    token_representations = results["representations"][33][:, 1:-1, :].mean(1)  # 提取第 33 层的输出表示  句向量

    torch.set_printoptions(sci_mode=False, threshold=5000)

    np.set_printoptions(suppress=True, threshold=5000)  # 禁用科学计数法

    return token_representations.cpu().numpy()




def combined_seq_to_vector(combined_seq):
    model_path = "/home/chen/wanglei_workspace/ssalign/SaprotProject/models/SaProt_650M_AF2.pt"  # SaProt模型的路径

    """
    从蛋白质序列和结构信息中提取深度学习模型的高维特征表示和注意力权重。
    通过仔细的顺序安排和正确的参数设置，可以有效地利用预训练的 SaProt 模型进行蛋白质研究和分析。
    """

    # alphabet：模型使用的字母表 对象，包含了序列和结构的编码信息。
    model, alphabet = load_esm_saprot(model_path)  # 加载模型
    batch_converter = alphabet.get_batch_converter()  # 获取批量转换器，用于将数据转为模型输入的格式


    name = "test"
    data = [(name, combined_seq)]
    """
    batch_labels：通常包含输入序列的名称或标签。
    batch_strs：原始输入序列，保留为字符串形式。
    batch_tokens：转换后的张量表示，包含序列的数字化表示（tokens），即模型可以直接处理的输入。
    """
    batch_labels, batch_strs, batch_tokens = batch_converter(data)  # 使用批量转化器，将蛋白质序列转为模型的所需的输入格式

    """
    return_contacts=True：告诉模型在返回结果中包含 接触矩阵（contact map），用于预测蛋白质残基之间的接触关系。
    repr_layers=[33]：指定要提取模型中第 33 层的表示。

    """
    with torch.no_grad():  # torch.no_grad()用来禁止梯度计算。在推理阶段（即预测时），不需要计算梯度，节省内存和计算资源。
        results = model(batch_tokens, repr_layers=[33], return_contacts=True)

    """
    1. 从 results["representations"] 中提取第 33 层的输出表示。
    每个输入序列的每个位置（即每个氨基酸和其对应的结构信息）都会有一个向量表示，这个向量捕捉了模型对该位置的理解，通常用于后续的分析任务，如结构预测或功能预测。
    形状：token_representations 的维度应该是 (batch_size, seq_len, hidden_dim)，
    batch_size 就是输入的序列的条数
    seq_len 表示输入的蛋白质序列的长度(即有多少个氨基酸)
    hidden_dim 指的是，每个氨基酸对应的表示向量的维度

    2. 提取模型中的注意力权重（attention weights）。注意力机制用来衡量序列中的每个位置（氨基酸残基）对其他位置的相关性。
    [:, -1, :, 1:-1, 1:-1]
        [:, -1, :, :]：选择最后一个注意力头（通常是模型的第 -1 个注意力头）或者是第 -1 层的注意力权重。
        1:-1：去掉序列开头的 <CLS> 标记和结尾的 <EOS> 标记，只保留实际的蛋白质序列部分。
    形状：attentions 的形状应该是 (batch_size, num_heads, seq_len, seq_len)，其中 num_heads 是多头注意力中的头数，seq_len 是序列的长度。
    """
    token_representations = results["representations"][33][:, 1:-1, :].mean(1)  # 提取第 33 层的输出表示  句向量
    # attentions = results["attentions"][:, -1, :, 1:-1, 1:-1] # 提取注意力权重

    # torch.Size([1, SVD1280])
    # torch.Size([1, 20, 217, 217])
    #print(token_representations.shape)
    # print(attentions.shape)

    # 维度太大，torch会使用...来省略输出，那么写入csv文件也会含有...,并且不可以使用科学计数法
    torch.set_printoptions(sci_mode=False, threshold=5000)
    # print(token_representations)

    np.set_printoptions(suppress=True, threshold=5000)  # 禁用科学计数法

    return token_representations.numpy()
    #print(token_representations.numpy())

if __name__=="__main__":
    Saport_model_path = "/home/chen/wanglei_workspace/ssalign/SaprotProject/models/SaProt_650M_AF2.pt"  # SaProt模型的路径


    # 加载模型和字母表
    model, alphabet = load_esm_saprot(Saport_model_path)


    cuda_device = "cuda:0"  # 根据实际 GPU 设备
    model = model.to(cuda_device)

    combined_seq1 = "MdAdVdAdSdTdSdLdAdSdQdMdSdGdPdHdFdSdGdLdRdKdSdIdSdKdLdDdNdTdSdVdSdFdSdTdSdQdAdFdFdHdNdVdDdAdHdLdRdLdSdSdAdGdKdGdCdRdSdVqVqTqMfAlGaSsGaKfFfFeVeGeGeNeWaKdCpNdGdTdKlDvSlIvSvKvLlVqSvDlLqNqSpAfQdLfEdSqDrVyDaVaVeVyAePdPqFpLvYcIlDlQvVcKlNvSrLhTdDpRsIyEaVySeAhQqNaCaWaTlGaKdGdGdApFaTpGlEdIdRdAdHpYvYvAvMvIvRvYvKvLvSvSvLvLcHvAvVvGvFnTpDpFpVpTvVvTvDvGvIvRvLvLvSvKvNpLqLdArQsTrDhSyShVlEvQvVcKvDvLsGpChKqWeVyIeLfGpHpSpEcRcRcHpIvIvGnEdNdDlElFrIrGlKsKrAlAvYrAnLvSvQvGrVrGaVyIaAqCeIfGeEdLdLpQvEcRvEvAvGvKnTgFlDvVrCrFlQsQsLvKvAsFnAdGdSdWcDqNrVyVaVyAeYyEePhVvWnAqIaGpTvGpKdVdAdTdPlEvQnAvQlEvVvHlVvAsVvRlDvWsLcTcKvNpVpSrAvEnVsAsSnKhThRqIyIaYyGeGgSaVdNwGlSvNcSlSlDvLnAlKlKrErDrIhDsGaFyLhVhGyGpAnSsLsKpGsPvEtFsAsTsIsVsNrSsVvTvSsKsKvVvAnAd"
    combined_seq2 = "MdAaRaRqFfFeVaGeGeNeWaKdMpNdGdNdKlEvSlLlElQvLlIqTvTlLvNqTpAdSdLfDdDpQsTyEaVaVeCyAeAdPeSpIvYcLqDlFvAnRvSvLrLhDdPpRsIyGaVySeAhQaNdCaYfKlVdAdKdGdAdFdTpGpEgIhShPlAvMvIsKvDvCsGvAhDqWeVyIeLfGpHpSpEcRcRcHpVvFvGvEdSdDlElLrIrGlQsKrVlAlHrAnLvEvSsDrLhGqVyIaAyCeIfGeEdKeLpEvEcRvEvAvGvSnTgElEvVrVrYlAvQsTvQvVsInAvEvNsVdTpDdWcElKrVyVaLyAeYyEePhVvWnAqIaGpTvGpKdThAdTdPlEvQsAvQlEvVvHlEvKsLvRlAvWsFcRcAvNpVpShDpDvVcAsDrShLhRaIyIaYyGeGgSdVdTfGlAvNcClRlEvLnAvSpQrGpDrVhDrGyFyLhVhGyGpAcSsLsKdPsErFvIsDsInIsNcAsRvQpKdQpDpFpNpHpEpGpQpIpIpRpFpTpQdVdTdEdPdIdWdLdTdLdSdSdRdQdLdQdAdRdSdSdAdAdFdSdIdLdFdGdLdVdTdVdKdSdSdPdTdTdWdIdPdTdWdAdVdRdRdDdQdAdSdQdSdSdWdSdNdGdSdSdMdEdTdTdGdRdKdRdGdEdMdId"
    combined_seq3 = "MdVdRdFdSdPdAdSdFdLdCdHdRdSdSdVdVdLdLdVdLdLdFdFdLdLdSdPdCdPdLdLdIpQdTdCdPdIpSpSpFpLpTpKpHpTpFpKlMpPaRaQqFfFeVaGeGeNeFaKdMpNdGdSdAlEvSlIlTlAvIlIqKvNlLqNvDvAdKdLfDdElSsAyEaVaVeVySePdPaTpLvYcLqLlLvAnNcQvIrAhDpQpKsKrVyRaVyAeSyQaNdVaFfDlKdPaNdGdAdFdTpGpEgIhShVlEvQnLcQvDvAsKvIhQqWeTyIeIfGpHpSpEcRcRcVpIpLvKvEdTdDlElFrIsAlRsKrVlKlArAnVlDvGsGrIrSaVyIaFyCeIfGfEdTeLpEvEcRvEvAvNvKnTgIlEvVrVrTvKvQsLvNvAnAnAlKvErLdTaKlEvQsWlTlKsVyVaIyAeYyEePhVvWnAqIaGvTvGvKdVdAdTdTlQvQsAlQlEvVvHlAvAsIvRlKvWcLcAcDvSrIrShApEvAsSsAsNhThRaVyIaYyGeGgSaVdSfElKvNcClRlEvLnAvKpErPsDsVhDsGaFyLnVhGyGpArSsLsKdPsArFsVsDsIrVsNrAsRvLd"

    combined_seqs = [combined_seq1, combined_seq2, combined_seq3]

    combined_seq_to_vector_exist_model_1_cpu_batch(combined_seqs,model,alphabet,cuda_device)
