from utils.esm_loader import load_esm_saprot
import torch
import numpy as np


def combined_seq_to_vector_exist_model_1_cpu_batch(combined_seqs,model,alphabet):

    batch_converter = alphabet.get_batch_converter()  #
    data = [(f"protein{i + 1}", seq) for i, seq in enumerate(combined_seqs)]

    batch_labels, batch_strs, batch_tokens = batch_converter(data)  #


    batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)  #

    with torch.no_grad():  
        results_1 = model(batch_tokens, repr_layers=[1], return_contacts=True)

    token_representations_1 = results_1["representations"][1]


    sequence_representations = []
    for i, tokens_len in enumerate(batch_lens):
        sequence_rep = token_representations_1[i, 1:tokens_len - 1, :].mean(0)
        avg_representation = sequence_rep.unsqueeze(0)
        sequence_representations.append(avg_representation)



    return torch.stack(sequence_representations, dim=0).numpy()




def combined_seq_to_vector_exist_model_1_gpu_batch(combined_seqs,model,alphabet,cuda_device):

    batch_converter = alphabet.get_batch_converter()  
    data = [(f"protein{i + 1}", seq) for i, seq in enumerate(combined_seqs)]

    batch_labels, batch_strs, batch_tokens = batch_converter(data)  


    batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)  

    with torch.no_grad():  
        batch_tokens = batch_tokens.to(cuda_device) 
        results_1 = model(batch_tokens, repr_layers=[1], return_contacts=True)

    token_representations_1 = results_1["representations"][1]


   
    sequence_representations = []
    for i, tokens_len in enumerate(batch_lens):
        sequence_rep = token_representations_1[i, 1:tokens_len - 1, :].mean(0)
        avg_representation = sequence_rep.unsqueeze(0)
        sequence_representations.append(avg_representation)

   
    del token_representations_1, batch_tokens, results_1
    torch.cuda.empty_cache()

    # return sequence_representations.numpy()
    return torch.stack(sequence_representations, dim=0).cpu().numpy()




def combined_seq_to_vector_exist_model_1_33_avg_cpu_batch(combined_seqs,model,alphabet):
    batch_converter = alphabet.get_batch_converter() 

    data = [(f"protein{i + 1}", seq) for i, seq in enumerate(combined_seqs)]

    batch_labels, batch_strs, batch_tokens = batch_converter(data)  

    batch_lens = (batch_tokens != alphabet.padding_idx).sum(1) 

    with torch.no_grad():  
        results_1 = model(batch_tokens, repr_layers=[1], return_contacts=True)
        results_33 = model(batch_tokens, repr_layers=[33], return_contacts=True)

    token_representations_1 = results_1["representations"][1]
    token_representations_33 = results_33["representations"][33]

   
    sequence_representations = []
    for i, tokens_len in enumerate(batch_lens):
        avg_representation = (token_representations_1[i, 1:tokens_len - 1, :].mean(0) + token_representations_33[i,1:tokens_len - 1, :].mean(0)) / 2
        avg_representation = avg_representation.unsqueeze(0)  
        #print(avg_representation.shape)
        sequence_representations.append(avg_representation)

    torch.set_printoptions(sci_mode=False, threshold=5000)
    np.set_printoptions(suppress=True, threshold=5000)  
    return torch.stack(sequence_representations, dim=0).numpy()


def combined_seq_to_vector_exist_model_1_33_avg_cpu(combined_seq, model, alphabet):
   
    batch_converter = alphabet.get_batch_converter() 

    name = "test"
    data = [(name, combined_seq)]

    batch_labels, batch_strs, batch_tokens = batch_converter(data)  
    with torch.no_grad(): 
        results_1 = model(batch_tokens, repr_layers=[1], return_contacts=True)
        results_33 = model(batch_tokens, repr_layers=[33], return_contacts=True)


    token_representations_1 = results_1["representations"][1][:, 1:-1, :].mean(1)  
    token_representations_33 = results_33["representations"][33][:, 1:-1, :].mean(1)  

    token_representations = (token_representations_33+token_representations_1) / 2

    torch.set_printoptions(sci_mode=False, threshold=5000)

    np.set_printoptions(suppress=True, threshold=5000)  

    return token_representations.numpy()


def combined_seq_to_vector_exist_model_1_33_avg_gpu(combined_seq, model, alphabet,cuda_device):
   
   

    batch_converter = alphabet.get_batch_converter()  
    name = "test"
    data = [(name, combined_seq)]

    batch_labels, batch_strs, batch_tokens = batch_converter(data)  
    with torch.no_grad(): 
        batch_tokens = batch_tokens.to(cuda_device)
        # results_1 = model(batch_tokens, repr_layers=[1], return_contacts=True)
        # results_33 = model(batch_tokens, repr_layers=[33], return_contacts=True)

        results_1 = model(batch_tokens, repr_layers=[1], return_contacts=True)
        token_representations_1 = results_1["representations"][1][:, 1:-1, :].mean(1)

        #results_33 = model(batch_tokens, repr_layers=[33], return_contacts=True)
        #token_representations_33 = results_33["representations"][33][:, 1:-1, :].mean(1)

    # token_representations_1 = results_1["representations"][1][:, 1:-1, :].mean(1)  
    #
    # token_representations_33 = results_33["representations"][33][:, 1:-1, :].mean(1)  
    token_representations = (token_representations_1+token_representations_1) / 2

    torch.set_printoptions(sci_mode=False, threshold=5000)

    np.set_printoptions(suppress=True, threshold=5000) 

    return token_representations.cpu().numpy()


def combined_seq_to_vector_exist_model(combined_seq, model, alphabet):
    batch_converter = alphabet.get_batch_converter()  
    name = "test"
    data = [(name, combined_seq)]

    batch_labels, batch_strs, batch_tokens = batch_converter(data)  
    with torch.no_grad(): 
        results = model(batch_tokens, repr_layers=[33], return_contacts=True)


    token_representations = results["representations"][33][:, 1:-1, :].mean(1) 
    torch.set_printoptions(sci_mode=False, threshold=5000)

    np.set_printoptions(suppress=True, threshold=5000) 

    return token_representations.numpy()


def combined_seq_to_vector_exist_model_gpu(combined_seq, model, alphabet,cuda_device):


    batch_converter = alphabet.get_batch_converter() 
    name = "test"
    data = [(name, combined_seq)]

    batch_labels, batch_strs, batch_tokens = batch_converter(data)  
    with torch.no_grad(): 
        batch_tokens = batch_tokens.to(cuda_device)
        results = model(batch_tokens, repr_layers=[33], return_contacts=True)


    token_representations = results["representations"][33][:, 1:-1, :].mean(1)  
    torch.set_printoptions(sci_mode=False, threshold=5000)

    np.set_printoptions(suppress=True, threshold=5000) 

    return token_representations.cpu().numpy()




def combined_seq_to_vector(combined_seq):
    model_path = "../models/SaProt_650M_AF2.pt"  

    

    
    model, alphabet = load_esm_saprot(model_path) 
    batch_converter = alphabet.get_batch_converter() 


    name = "test"
    data = [(name, combined_seq)]
    
    batch_labels, batch_strs, batch_tokens = batch_converter(data)  
    with torch.no_grad(): 。
        results = model(batch_tokens, repr_layers=[33], return_contacts=True)

    
    token_representations = results["representations"][33][:, 1:-1, :].mean(1)  # 提取第 33 层的输出表示  句向量
    # attentions = results["attentions"][:, -1, :, 1:-1, 1:-1] # 提取注意力权重

    # torch.Size([1, SVD1280])
    # torch.Size([1, 20, 217, 217])
    #print(token_representations.shape)
    # print(attentions.shape)

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
