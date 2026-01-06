
import os
import sys
import time
import argparse
import numpy as np
import pandas as pd
import multiprocessing as mp

import torch
import faiss

from utils.esm_loader import load_esm_saprot
from utils.foldseek_util import get_struc_seq


DEFAULT_FOLDSEEK_PATH = "../bin/foldseek"
DEFAULT_SAPROT_MODEL_PATH = "../../models/SaProt_650M_AF2.pt"

DEFAULT_W_FILENAME = "../models/SSAlignDB/AFDB50/AFDB50_whitening_W.npy"
DEFAULT_MU_FILENAME = "../models/SSAlignDB/AFDB50/AFDB50_whitening_mu.npy"

DEFAULT_LOOKUP_FILE = '../models/SSAlignDB/AFDB50/ssalign_afdb50_combined_seq.lookup'
DEFAULT_INDEX_FILE = '../models/SSAlignDB/AFDB50/ssalign_afdb50_combined_seq.index'
DEFAULT_SEQ_FILE = '../models/SSAlignDB/AFDB50/ssalign_afdb50_combined_seq'




# =========================
# worker 全局（fork 后共享）
# =========================
G_LOOKUP = None
G_INDEX = None
G_SEQ_FILE = None


# =========================
# 基础工具
# =========================
def parse_list_from_file(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return [x.strip() for x in f if x.strip()]





def load_lookup_dict(lookup_file):
    d = {}
    with open(lookup_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            seq_num, name = line.split("\t")
            d[int(seq_num)] = name
    return d


def load_index_dict(index_file):
    d = {}
    with open(index_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            seq_num, start, end = map(int, line.split())
            d[int(seq_num)] = (start, end)
    return d


def load_mu_and_W(mu_path, w_path):
    mu = np.load(mu_path)
    W = np.load(w_path)
    return mu, W


def apply_whitening_and_l2(mu, W, emb_mat):
    X = emb_mat - mu.reshape(1, -1)
    Xw = X @ W
    Xw = np.asarray(Xw, dtype=np.float32)
    faiss.normalize_L2(Xw)
    return Xw


# =========================
# Query：生成 foldseek_seq + combined_seq
# =========================
def generate_query_seqs(foldseek_exec, query_paths):
    """
    返回：
      valid_paths, foldseek_seq_list, combined_seq_list
    三者顺序严格一致
    """
    valid_paths = []
    foldseek_seqs = []
    combined_seqs = []

    for p in query_paths:
        if not os.path.exists(p):
            print(f"[WARN] missing query file: {p}", file=sys.stderr)
            continue

        parsed = get_struc_seq(foldseek_exec, p, plddt_mask=False)
        _, (seq, foldseek_seq, combined_seq) = next(iter(parsed.items()))

        valid_paths.append(p)
        foldseek_seqs.append(foldseek_seq)
        combined_seqs.append(combined_seq)

    if not valid_paths:
        raise RuntimeError("No valid query structures found.")

    return valid_paths, foldseek_seqs, combined_seqs


# =========================
# SaProt embedding（只做推理；模型在外面预先加载好）
# =========================
def saprot_embed_infer(model, alphabet, batch_converter, cuda_device, labels, seqs, batch_size=20):
    """
    返回 emb_mat: (N, D)
    """
    data = list(zip(labels, seqs))
    embs = []

    if cuda_device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()

    with torch.no_grad():
        for bs in range(0, len(data), batch_size):
            batch = data[bs:bs + batch_size]
            _, _, batch_tokens = batch_converter(batch)
            batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)

            batch_tokens = batch_tokens.to(cuda_device)
            results = model(batch_tokens, repr_layers=[33], return_contacts=False)
            reps = results["representations"][33]  # (B, L, D)

            for i, L in enumerate(batch_lens.tolist()):
                vec = reps[i, 1:L - 1].mean(0).unsqueeze(0)  # (1, D)
                embs.append(vec.detach().cpu().numpy())

    if cuda_device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()

    return np.vstack(embs).astype(np.float32)


# =========================
# Stage2：saligner worker
# =========================
def _init_worker(lookup_dict, index_dict, seq_file_path):
    global G_LOOKUP, G_INDEX, G_SEQ_FILE
    G_LOOKUP = lookup_dict
    G_INDEX = index_dict
    G_SEQ_FILE = seq_file_path


def _fetch_combined_by_index(seq_num: int, fh) -> str:
    """
    按 byte offset 读取 combined_seq（rb 模式最稳）
    """
    se = G_INDEX.get(seq_num, None)
    if se is None:
        return ""
    start, end = se
    fh.seek(start)
    data = fh.read(end - start)
    return data.decode("utf-8", errors="ignore").strip()


def _extract_3di_from_combined(combined_seq: str) -> str:
    # 旧逻辑：candidate 的 3Di 是 combined_seq 里的小写部分，转大写
    return "".join([c for c in combined_seq if c.islower()]).upper()


def _process_one_query(args):
    """
    args:
      (q_idx, q_path, q_foldseek_seq, high_pairs, low_pairs, max_target, out_dir, overwrite)
    """
    q_idx, q_path, q_foldseek_seq, high_pairs, low_pairs, max_target, out_dir, overwrite = args

    qname = os.path.splitext(os.path.basename(q_path))[0]
    out_path = os.path.join(out_dir, f"{qname}.ssalign")

    if (not overwrite) and os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        return q_idx, qname, "SKIP"

    # 1) 高分部分：直接输出（Score=NA）
    high_rows = []
    for tid, score in high_pairs:
        pname = G_LOOKUP.get(tid, "NA")
        high_rows.append((pname, float(score), np.nan))

    high_df = pd.DataFrame(high_rows, columns=["File1", "prefilter_score", "Score"])

    # FAISS(IndexFlatIP) 本身已降序，这里再保险
    if len(high_df) > 0:
        high_df = high_df.sort_values("Cosine_Similarity", ascending=False)

    # 如果高分已经够 max_target：直接截断写出，不跑第二阶段
    if len(high_df) >= max_target:
        out_df = high_df.head(max_target)
        os.makedirs(out_dir, exist_ok=True)
        out_df.to_csv(out_path, index=False)
        return q_idx, qname, "OK"

    need = max_target - len(high_df)
    if need <= 0 or not low_pairs:
        os.makedirs(out_dir, exist_ok=True)
        high_df.to_csv(out_path, index=False)
        return q_idx, qname, "OK"

    # 2) 低分部分：跑 saligner，按 Score 取 top need
    from saligner import saligner  # worker 内导入更稳

    sal_rows = []
    with open(G_SEQ_FILE, "rb") as fh:
        for tid, score in low_pairs:
            combined = _fetch_combined_by_index(tid, fh)
            if not combined:
                continue
            d3i = _extract_3di_from_combined(combined)
            if not d3i:
                continue
            s_score = saligner(q_foldseek_seq, d3i)
            sal_rows.append((G_LOOKUP.get(tid, "NA"), float(score), float(s_score)))

    if sal_rows:
        sal_df = pd.DataFrame(sal_rows, columns=["File1", "Cosine_Similarity", "Score"])
        # ✅ 只取需要的 need 条，避免超过 max_target
        sal_df = sal_df.sort_values("Score", ascending=False).head(need)
        final_df = pd.concat([high_df, sal_df], ignore_index=True)
    else:
        final_df = high_df

    os.makedirs(out_dir, exist_ok=True)
    final_df.to_csv(out_path, index=False)
    return q_idx, qname, "OK"


# =========================
# 主流程
# =========================
def main():
    ap = argparse.ArgumentParser(description="SSAlign two-stage: score<threshold -> saligner refine.")

    ap.add_argument("--querypdbs", required=True, help="Comma-separated list of query PDB files")
    ap.add_argument("--dim", type=int, default=512)  # 必须等于 index.d
    ap.add_argument("--prefilter_target", type=int, default=2000)  # FAISS topK
    ap.add_argument("--prefilter_threshold", type=float,default=0.3)  # 分数阈值（例如 0.3）
    ap.add_argument("--max_target", type=int, default=1000)  # 最终输出行数（<= prefilter_target）
    ap.add_argument("--mode", type=int, required=True)  # 0：预过滤/ 1：完整阶段
    ap.add_argument("--prefilter_mode", type=str, default="cpu", choices=["cpu", "gpu"],
                    help="FAISS prefilter on CPU or GPU (multi-gpu sharded)")
    ap.add_argument("--out_dir", required=True)

    ap.add_argument("--nproc", type=int, default=64)
    ap.add_argument("--cuda_device", default="cuda:0")
    ap.add_argument("--batch_size", type=int, default=20)



    args = ap.parse_args()

    if args.mode not in (0, 1):
        raise ValueError("--mode must be 0 or 1")

    if args.max_target > args.prefilter_target:
        raise ValueError("max_target must be <= prefilter_target")

    os.makedirs(args.out_dir, exist_ok=True)

    # ==========================================================
    # 这些加载不计入耗时：SaProt、faiss、lookup_dict、index_dict
    # ==========================================================
    t_load0 = time.time()

    # 1) load SaProt model
    t = time.time()
    if args.cuda_device.startswith("cuda") and (not torch.cuda.is_available()):
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False.")
    model, alphabet = load_esm_saprot(args.saprot_model)
    model = model.to(args.cuda_device)
    model.eval()
    batch_converter = alphabet.get_batch_converter()
    t_load_saprot = time.time() - t

    # 2) load FAISS index
    t = time.time()
    faiss_index = f"../model/SSAlignDB/AFDB50/afdb50_{args.dim}_IndexFlatIP_faiss.faiss"
    index = faiss.read_index(faiss_index)
    # ===== prefilter on GPU (multi GPU sharded) =====
    if args.prefilter_mode == "gpu":
        # faiss-gpu 才会有这些接口
        if not hasattr(faiss, "get_num_gpus") or faiss.get_num_gpus() <= 0:
            raise RuntimeError("prefilter_mode=gpu but FAISS GPU is not available (faiss-gpu not installed or no GPU).")

        ngpu = faiss.get_num_gpus()
        gpu_resources = [faiss.StandardGpuResources() for _ in range(ngpu)]

        co = faiss.GpuMultipleClonerOptions()
        co.shard = True  # 多 GPU 分片
        # move CPU index -> multi GPU
        index = faiss.index_cpu_to_gpu_multiple_py(gpu_resources, index, co)
    t_load_faiss = time.time() - t
    if index.d != args.dim:
        raise ValueError(f"--dim={args.dim} but index.d={index.d}. Must match for IndexFlatIP.")

    # 3) load lookup_dict
    t = time.time()
    lookup_dict = load_lookup_dict(args.lookup_file)
    t_load_lookup = time.time() - t

    # 4) load index_dict
    t = time.time()
    index_dict = load_index_dict(args.index_file)
    t_load_index = time.time() - t

    # mu/W（小，提前加载更干净）
    t = time.time()
    mu, W = load_mu_and_W(args.mu_file, args.w_file)
    t_load_muW = time.time() - t

    print("[PRELOAD DONE] (not counted)")
    print(f"  load_saprot   = {t_load_saprot:.2f}s")
    print(f"  load_faiss    = {t_load_faiss:.2f}s")
    print(f"  load_lookup   = {t_load_lookup:.2f}s")
    print(f"  load_index    = {t_load_index:.2f}s")
    print(f"  load_muW      = {t_load_muW:.2f}s")
    print(f"  preload_total = {time.time() - t_load0:.2f}s\n")

    # =========================
    # 从这里开始计入有效耗时
    # =========================
    time_start = time.time()

    # A) query list -> paths
    if args.querypdbs:
        query_pdbs = [p.strip() for p in args.querypdbs.split(',')]
    else:
        print("Error: --querypdbs cannot be empty")
        sys.exit(1)

    # B) query foldseek_seq + combined_seq
    t = time.time()
    valid_paths, foldseek_seq_list, combined_seq_list = generate_query_seqs(args.foldseek_exec, query_pdbs)
    print(f"[QSEQ] n={len(valid_paths)} cost={time.time() - t:.2f}s")

    # C) SaProt infer
    t = time.time()
    emb_mat = saprot_embed_infer(
        model=model,
        alphabet=alphabet,
        batch_converter=batch_converter,
        cuda_device=args.cuda_device,
        labels=valid_paths,
        seqs=combined_seq_list,
        batch_size=args.batch_size,
    )
    print(f"[EMB] emb_mat shape={emb_mat.shape} cost={time.time() - t:.2f}s")

    # D) whitening + L2
    t = time.time()
    if emb_mat.shape[1] != mu.shape[0]:
        raise ValueError(f"Embedding dim={emb_mat.shape[1]} but mu dim={mu.shape[0]} mismatch.")
    whitened = apply_whitening_and_l2(mu, W, emb_mat)
    print(f"[WHITEN] whitened shape={whitened.shape} cost={time.time() - t:.2f}s")

    # E) FAISS search (IndexFlatIP)
    t = time.time()
    q = np.ascontiguousarray(whitened[:, :index.d], dtype=np.float32)
    distances, indices0 = index.search(q, args.prefilter_target)
    indices1 = indices0 + 1
    print(f"[SSAlign-Prefilter] topK={args.prefilter_target} cost={time.time() - t:.2f}s")

    # F) 组织任务：按 mode 决定是否需要 low_pairs
    score_th = float(args.prefilter_threshold)
    t = time.time()
    tasks = []
    for i in range(len(valid_paths)):
        idxs = indices1[i].tolist()
        scs = distances[i].tolist()

        high_pairs = []
        low_pairs = []

        if args.mode == 0:
            # ✅ 预过滤模式：全部作为 high_pairs；low_pairs 为空 -> worker 不会触发 saligner
            for tid, score in zip(idxs, scs):
                high_pairs.append((tid, score))
        else:
            # ✅ 完整模式：按阈值分 high/low
            for tid, score in zip(idxs, scs):
                if score >= score_th:
                    high_pairs.append((tid, score))
                else:
                    low_pairs.append((tid, score))

        tasks.append((
            i + 1,
            valid_paths[i],
            foldseek_seq_list[i],
            high_pairs,
            low_pairs,
            args.max_target,
            args.out_dir,
            args.overwrite,
        ))
    print(f"[TASKS] n={len(tasks)} threshold={score_th} cost={time.time() - t:.2f}s")

    # G) Stage2: saligner（mode=0 时不会真的触发，因为 low_pairs 为空）
    print(f"[STAGE2] saligner nproc={args.nproc} (only score<{score_th}) ...")
    t = time.time()
    try:
        ctx = mp.get_context("fork")
    except ValueError:
        ctx = mp.get_context()

    with ctx.Pool(
        processes=args.nproc,
        initializer=_init_worker,
        initargs=(lookup_dict, index_dict, args.seq_file),
    ) as pool:
        done = 0
        for (q_idx, qname, status) in pool.imap_unordered(_process_one_query, tasks, chunksize=1):
            done += 1
            if done % 10 == 0 or done == len(tasks):
                print(f"[PROGRESS] {done}/{len(tasks)} last={qname} status={status}")



if __name__ == "__main__":
    main()

    """
    python AFDB50_SSAlign.py \
    --querypdbs "../pdbData/pdb/SwissProt/AF-A0BLX0-F1-model_v4.cif,../pdbData/pdb/SwissProt/AF-Q6ME97-F1-model_v4.cif" \
    --dim 512 \
    --prefilter_target 2000 \
    --prefilter_threshold 0.3 \
    --max_target 1000 \
    --mode 1 \
    --prefilter_mode cpu \
    --out_dir "./results" \
    --n_proc 64
    """
