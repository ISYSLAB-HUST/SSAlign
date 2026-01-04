import numpy as np
import torch
import faiss
import os
from utils.foldseek_util import get_struc_seq
from utils.esm_loader import load_esm_saprot

import pandas as pd
from concurrent.futures import ProcessPoolExecutor


FOLDSEEK_PATH = "../bin/foldseek"
SAPROT_MODEL_PATH = "../models/SaProt_650M_AF2.pt"
CUDA_DEVICE = "cuda:1"

SwissPort_MU_FILENAME = "../models/SSAlignDB/SwissProt/SwissPort_whitening_mu.npy"
SwissPort_W_FILENAME = "../models/SSAlignDB/SwissProt/SwissPort_whitening_W.npy"
SwissPort_id_seqs_file = "../models/SSAlignDB/SwissProt/SwissPort_id_Seq.npz"


def generate_3di_sequences(pdb_list):
    foldseek_seqs = {}
    for pdb in pdb_list:
        parsed_seqs = get_struc_seq(FOLDSEEK_PATH, pdb, plddt_mask=False)
        key, (seq, foldseek_seq, combined_seq) = next(iter(parsed_seqs.items()))
        foldseek_seqs[pdb] = (foldseek_seq, combined_seq)
    return foldseek_seqs


def generate_saprot_embeddings_gpu(pdb_list):
    model, alphabet = load_esm_saprot(SAPROT_MODEL_PATH)
    model = model.to(CUDA_DEVICE)
    batch_converter = alphabet.get_batch_converter()

    foldseek_seqs = generate_3di_sequences(pdb_list)

    data = [(pdb, foldseek_seqs[pdb][1]) for pdb in pdb_list]  # use combined_seq
    batch_labels, batch_strs, batch_tokens = batch_converter(data)
    batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)

    with torch.no_grad():
        batch_tokens = batch_tokens.to(CUDA_DEVICE)
        results = model(batch_tokens, repr_layers=[33], return_contacts=False)

    token_representations = results["representations"][33]
    for i, tokens_len in enumerate(batch_lens):
        sequence_rep = token_representations[i, 1:tokens_len - 1].mean(0)
        avg_representation = sequence_rep.unsqueeze(0)
        pdb = pdb_list[i]
        foldseek_seq, _ = foldseek_seqs[pdb]
        foldseek_seqs[pdb] = (foldseek_seq, avg_representation.cpu().numpy())

    return foldseek_seqs


def generate_saprot_embeddings_cpu(pdb_list):
    model, alphabet = load_esm_saprot(SAPROT_MODEL_PATH)
    batch_converter = alphabet.get_batch_converter()

    foldseek_seqs = generate_3di_sequences(pdb_list)

    data = [(pdb, foldseek_seqs[pdb][1]) for pdb in pdb_list]  # use combined_seq
    batch_labels, batch_strs, batch_tokens = batch_converter(data)
    batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)

    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[33], return_contacts=False)

    token_representations = results["representations"][33]
    for i, tokens_len in enumerate(batch_lens):
        sequence_rep = token_representations[i, 1:tokens_len - 1].mean(0)
        avg_representation = sequence_rep.unsqueeze(0)
        pdb = pdb_list[i]
        foldseek_seq, _ = foldseek_seqs[pdb]
        foldseek_seqs[pdb] = (foldseek_seq, avg_representation.numpy())

    return foldseek_seqs


def generate_saprot_embeddings(pdb_list):
    """
    First try running SaProt on GPU, fall back to CPU if needed.
    """
    if not torch.cuda.is_available():
        print("[SaProt] No CUDA device available, using CPU.")
        return generate_saprot_embeddings_cpu(pdb_list)

    try:
        print(f"[SaProt] Using GPU device: {CUDA_DEVICE}")
        return generate_saprot_embeddings_gpu(pdb_list)
    except RuntimeError as e:
        msg = str(e).lower()
        if "out of memory" in msg or "cuda" in msg:
            print(f"[SaProt] GPU failed ({e}), falling back to CPU.")
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            return generate_saprot_embeddings_cpu(pdb_list)
        raise


def apply_whitening(foldseek_seqs):
    """
    Apply whitening transform and L2-normalization to SaProt embeddings.
    """
    mu = np.load(SwissPort_MU_FILENAME)
    W = np.load(SwissPort_W_FILENAME)

    for pdb, (foldseek_seq, saprot_embedding) in foldseek_seqs.items():
        X_centered = saprot_embedding - mu
        whitened_embedding = np.dot(X_centered, W)
        whitened_embedding = np.array(whitened_embedding, dtype=np.float32)
        faiss.normalize_L2(whitened_embedding)
        foldseek_seqs[pdb] = (foldseek_seq, whitened_embedding)

    return foldseek_seqs


def search_faiss(index, dim, query_vectors, prefilter_target):
    res = faiss.StandardGpuResources()
    gpu_id = 0
    index = faiss.index_cpu_to_gpu(res, gpu_id, index)

    q = np.asarray(query_vectors, dtype=np.float32)

    if q.ndim == 3:
        n, _, d = q.shape
        q = q.reshape(n, d)
    elif q.ndim == 1:
        q = q.reshape(1, -1)

    q = np.ascontiguousarray(q[:, :dim], dtype=np.float32)
    distances, indices = index.search(q, prefilter_target)
    return distances, indices


def _load_id_seq_mapping(npz_path: str):
    """
    Load mapping arrays: ids and seqs.
    """
    data = np.load(npz_path, allow_pickle=True)
    ids = data["ids"]
    seqs = data["seqs"]
    return ids, seqs


def _saligner_worker(args):
    """
    Worker function for multiprocessing.

    args = (query_seq, raw_target_seq)

    Note:
    - The target sequence keeps only lowercase characters and converts them to uppercase,
      following the original logic in your code.
    - Import saligner inside the worker to avoid pickling/global-state issues.
    """
    query_seq, raw_target_seq = args
    processed_target_seq = "".join(ch.upper() for ch in raw_target_seq if ch.islower())

    from SAligner.saligner import saligner as _saligner
    return float(_saligner(query_seq, processed_target_seq))


def runSAligner(threshold: float, prefilter_rows: list[dict], n_proc: int = 64) -> pd.DataFrame:
    """
    Run SAligner in two segments:

    1) Sort all candidates by prefilter_score descending.
    2) For rows with prefilter_score >= threshold:
       - Keep ordering by prefilter_score
       - Set saligner_score to NaN
    3) For rows with prefilter_score < threshold:
       - Compute saligner_score in parallel (multiprocessing)
       - Sort these rows by saligner_score descending
    4) Concatenate (high segment first, low segment second) and return a single DataFrame.
    """
    df = pd.DataFrame(prefilter_rows)
    df = df.sort_values("prefilter_score", ascending=False).reset_index(drop=True)

    mask_low = df["prefilter_score"] < threshold

    if not mask_low.any():
        df["saligner_score"] = np.nan
        return df

    df_high = df[~mask_low].copy()
    df_high["saligner_score"] = np.nan

    df_low = df[mask_low].copy()

    tasks = list(zip(df_low["query_Seq"].tolist(), df_low["target_Seq"].tolist()))

    with ProcessPoolExecutor(max_workers=n_proc) as ex:
        scores = list(ex.map(_saligner_worker, tasks))

    df_low["saligner_score"] = scores
    df_low = df_low.sort_values("saligner_score", ascending=False)

    df_final = pd.concat([df_high, df_low], ignore_index=True)
    return df_final


def runSSAlign_cmd(pdb_list,dim, cos_threshold, prefilter_topk, final_number, n_proc: int = 64) -> pd.DataFrame:

    SwissPort_faiss_index_file = f"../models/SSAlignDB/SwissProt/SwissPort_IndexFlatIP_{dim}_faiss.index"


    """
    For each query PDB:
    - Run prefilter (FAISS) + SAligner re-ranking
    - Write the final top-N result DataFrame to CSV with name: {query_pdb}.ssalign
      in the current working directory

    Returns:
    - A single DataFrame that concatenates all queries' top-N results.
    """
    foldseek_seqs = generate_saprot_embeddings(pdb_list)
    foldseek_seqs = apply_whitening(foldseek_seqs)

    query_embs = []
    for pdb in pdb_list:
        _, emb = foldseek_seqs[pdb]
        query_embs.append(emb)
    query_vectors = np.stack(query_embs, axis=0)

    if not os.path.exists(SwissPort_faiss_index_file):
        raise FileNotFoundError(f" FAISS index file not found: {SwissPort_faiss_index_file}")

    index = faiss.read_index(SwissPort_faiss_index_file)
    dim = index.d

    distances, indices = search_faiss(
        index=index,
        dim=dim,
        query_vectors=query_vectors,
        prefilter_target=prefilter_topk,
    )

    target_ids, target_seqs = _load_id_seq_mapping(SwissPort_id_seqs_file)

    all_top_dfs = []

    for qi, query_pdb in enumerate(pdb_list):
        query_seq = foldseek_seqs[query_pdb][0]

        prefilter_rows = []
        for rank, (idx, dist) in enumerate(
            zip(indices[qi].tolist(), distances[qi].tolist()),
            start=1,
        ):
            target_id = str(target_ids[idx])
            target_seq = str(target_seqs[idx])

            prefilter_rows.append(
                {
                    "query_pdb": query_pdb,
                    "query_Seq": query_seq,
                    "rank": rank,
                    "target_id": target_id,
                    "target_Seq": target_seq,
                    "prefilter_score": float(dist),
                }
            )

        result_df = runSAligner(cos_threshold, prefilter_rows, n_proc=n_proc)

        # Keep your original SS_score logic as-is
        result_df["SS_score"] = 0.45 * result_df["prefilter_score"] + 0.56

        top_hits_df = result_df.head(final_number).reset_index(drop=True)

        out_name = f"{query_pdb}.result"
        top_hits_df.to_csv(out_name, index=False)
        print(f"[Saved] {out_name}  rows={len(top_hits_df)}")

        all_top_dfs.append(top_hits_df)

    if all_top_dfs:
        return pd.concat(all_top_dfs, ignore_index=True)
    return pd.DataFrame()


if __name__ == "__main__":
    # Simple test (fill in your PDB paths or identifiers)
    test_pdbs = [
        "../pdbData/pdb/SwissProt/AF-A0BLX0-F1-model_v4.cif",
        "../pdbData/pdb/SwissProt/AF-Q6ME97-F1-model_v4.cif"
    ]

    final_df = runSSAlign_cmd(
        test_pdbs,
        dim=1280,
        cos_threshold=0.2,
        prefilter_topk=2000,
        final_number=1000,
        n_proc=64,
    )

