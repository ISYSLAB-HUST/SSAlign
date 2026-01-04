import numpy as np
import os

def build_id_Seq_npz(intput_file_path,output_npz):

    ids = []
    seqs = []

    with open(intput_file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # 按逗号切三段：path, seq, embedding
            parts = line.split(",", 2)
            if len(parts) < 2:
                continue  # 格式异常，跳过

            path = parts[0]
            seq = parts[1]

            # 取 basename 作为结构名，比如 d2fnba_
            name = os.path.basename(path)

            ids.append(name)
            seqs.append(seq)

    # 转成 numpy 数组；用 dtype=object 可以容纳变长字符串
    ids = np.array(ids, dtype=object)
    seqs = np.array(seqs, dtype=object)

    # 保存到 npz
    np.savez(output_npz, ids=ids, seqs=seqs)

    print(f"Saved {len(ids)} entries to {output_npz}")

def build_id_embedding_npz(whitening_file_path,output_npz):
    paths = []
    vecs = []

    with open(whitening_file_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            p, s = line.split(",", 1)  # 只切第一个逗号
            s = s.strip()

            # 去掉外层 [[...]] 或 [...]
            if s.startswith("[[") and s.endswith("]]"):
                s = s[2:-2]
            elif s.startswith("[") and s.endswith("]"):
                s = s[1:-1]

            v = np.array([float(x) for x in s.split(",")], dtype=np.float32)

            # 你的文件是 1280 维，这里强制取前 1280（安全起见）
            v = v[:1280]

            paths.append(os.path.basename(p))
            vecs.append(v)

    embeddings = np.vstack(vecs).astype(np.float32)

    # 保存 npz（后续你原逻辑就能继续用）
    np.savez(output_npz,
             embeddings=embeddings,
             ids=np.array(paths, dtype=object))


if __name__ == "__main__":


    orgin_embedding_file = '/data2/zxc_data/foldseek_database/foldseek_database/Scope40/scope40_vector_results'
    """
        组织为npz文件，方便后续使用
    """
    output_npz_1 = "SCOPe40_id_Seq.npz"
    build_id_Seq_npz(orgin_embedding_file, output_npz_1)

    embedding_file = '/data2/zxc_data/foldseek_database/foldseek_database/Scope40/scope40_vector_results_whitening'
    output_npz_2 = "SCOPe40_id_embedding.npz"
    build_id_Seq_npz(orgin_embedding_file, output_npz_2)



