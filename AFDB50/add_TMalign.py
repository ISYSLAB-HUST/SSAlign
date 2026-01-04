#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import re
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# -----------------------------
# TMalign 输出解析
# -----------------------------
RE_ALIGNED = re.compile(
    r"Aligned length=\s*(\d+),\s*RMSD=\s*([\d.]+),\s*Seq_ID=n_identical/n_aligned=\s*([\d.]+)"
)
RE_TM1 = re.compile(r"TM-score=\s*([\d.]+)\s*\(if normalized by length of Chain_1", re.I)
RE_TM2 = re.compile(r"TM-score=\s*([\d.]+)\s*\(if normalized by length of Chain_2", re.I)


def parse_tmalign(stdout: str):
    """Return: (tm1, tm2, aln_len, rmsd, seq_id) as strings (or '')"""
    tm1 = tm2 = aln_len = rmsd = seq_id = ""

    m = RE_ALIGNED.search(stdout)
    if m:
        aln_len, rmsd, seq_id = m.group(1), m.group(2), m.group(3)

    m = RE_TM1.search(stdout)
    if m:
        tm1 = m.group(1)

    m = RE_TM2.search(stdout)
    if m:
        tm2 = m.group(1)

    return tm1, tm2, aln_len, rmsd, seq_id


def to_v6_id(v4_id: str) -> str:
    v4_id = v4_id.strip()
    if v4_id.endswith("-model_v4"):
        return v4_id[:-len("-model_v4")] + "-model_v6"
    return v4_id


def run_tmalign(tm_exec: str, q_path: Path, t_path: Path, timeout_s: int):
    """Return: (ok: bool, tm1, tm2, aln_len, rmsd, seq_id)"""
    if (not q_path.exists()) or q_path.stat().st_size == 0:
        return False, "", "", "", "", ""
    if (not t_path.exists()) or t_path.stat().st_size == 0:
        return False, "", "", "", "", ""
    try:
        res = subprocess.run(
            [tm_exec, str(q_path), str(t_path)],
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout_s,
        )
        if res.returncode != 0:
            return False, "", "", "", "", ""
        tm1, tm2, aln_len, rmsd, seq_id = parse_tmalign(res.stdout or "")
        return True, tm1, tm2, aln_len, rmsd, seq_id
    except Exception:
        return False, "", "", "", "", ""


def avg_tm(tm1: str, tm2: str) -> str:
    """Avg_TM_Score: (tm1+tm2)/2 -> 保留 5 位小数，匹配你示例 0.96995"""
    try:
        a = float(tm1)
        b = float(tm2)
        return f"{(a + b) / 2.0:.5f}"
    except Exception:
        # 只要有一个不可用就空着（你后续统计时也更干净）
        return ""


def read_id_list(txt_path: Path):
    ids = []
    with txt_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip().strip("\ufeff").strip()
            if not s:
                continue
            ids.append(s)
    return ids


def read_ssalign_csv(path: Path):
    """
    读取你的 .ssalign:
    File1,Cosine_Similarity,Score
    AF-xxx,0.606...,   (Score 可能为空)
    """
    rows = []
    with path.open("r", encoding="utf-8", errors="ignore", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header is None:
            return [], []
        header = [h.strip() for h in header]

        # 兼容某些行列数不齐（Score 空）
        for parts in reader:
            if not parts:
                continue
            parts = [p.strip() for p in parts]
            # pad
            while len(parts) < len(header):
                parts.append("")
            row = {header[i]: parts[i] for i in range(len(header))}
            rows.append(row)
    return header, rows


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


# -----------------------------
# foldseek: timebenchmark -> add TM -> foldseek/*.result
# -----------------------------
def process_foldseek_one(query_id: str, foldseek_in_dir: Path, foldseek_out_dir: Path,
                         query_dir: Path, v6_dir: Path, tm_exec: str,
                         timeout_s: int, nworker: int, overwrite: bool):
    in_fp = foldseek_in_dir / f"{query_id}.foldseek"
    out_fp = foldseek_out_dir / f"{query_id}.result"
    if (not in_fp.exists()) or in_fp.stat().st_size == 0:
        print(f"[WARN] foldseek missing: {in_fp}")
        return
    if out_fp.exists() and (not overwrite):
        print(f"[SKIP] foldseek exists: {out_fp}")
        return

    # 读取 foldseek 原始行：空白分隔
    raw_lines = []
    targets = []
    with in_fp.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            cols = line.split()
            if len(cols) < 2:
                continue
            q_cif = cols[0].strip()
            t_v4 = cols[1].strip()
            metrics = " ".join(cols[2:]).strip()
            raw_lines.append((q_cif, t_v4, metrics))
            targets.append(t_v4)

    # 并行跑 TM-align（按 target 去重）
    q_path = query_dir / f"{query_id}.cif"
    uniq_targets = sorted(set(targets))

    def one_target(t_v4: str):
        t_path = v6_dir / (to_v6_id(t_v4) + ".pdb")
        ok, tm1, tm2, aln_len, rmsd, seq_id = run_tmalign(tm_exec, q_path, t_path, timeout_s)
        return t_v4, (tm1, tm2, aln_len, rmsd, seq_id)

    tmalign_map = {}
    with ThreadPoolExecutor(max_workers=max(1, nworker)) as ex:
        futs = [ex.submit(one_target, t) for t in uniq_targets]
        for fu in as_completed(futs):
            t_v4, vals = fu.result()
            tmalign_map[t_v4] = vals

    # 输出
    ensure_dir(foldseek_out_dir)
    with out_fp.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["File1", "File2", "TM-Score1", "TM-Score2", "Aligned Length", "RMSD", "Seq_ID", "FoldSeek_Metrics"])
        for q_cif, t_v4, metrics in raw_lines:
            tm1, tm2, aln_len, rmsd, seq_id = tmalign_map.get(t_v4, ("", "", "", "", ""))
            file2 = to_v6_id(t_v4) + ".pdb"
            w.writerow([q_cif, file2, tm1, tm2, aln_len, rmsd, seq_id, metrics])

    print(f"[OK] foldseek -> {out_fp}")


# -----------------------------
# ssalign_prefilter: add TM -> ssalign_prefilter/*.result
# 并返回映射 dict: target_v4 -> (Aligned, RMSD, Seq_ID, Avg_TM_Score)
# -----------------------------
def process_ssalign_prefilter_one(query_id: str, ssalign_prefilter_in_dir: Path, ssalign_prefilter_out_dir: Path,
                                  query_dir: Path, v6_dir: Path, tm_exec: str,
                                  timeout_s: int, nworker: int, overwrite: bool):
    in_fp = ssalign_prefilter_in_dir / f"{query_id}.ssalign"
    out_fp = ssalign_prefilter_out_dir / f"{query_id}.result"

    if (not in_fp.exists()) or in_fp.stat().st_size == 0:
        print(f"[WARN] ssalign_prefilter missing: {in_fp}")
        return {}, None
    if out_fp.exists() and (not overwrite):
        # 仍然要提供映射：从已有 result 读一遍（避免你后面 ssalign 映射断掉）
        mapping = {}
        try:
            with out_fp.open("r", encoding="utf-8", errors="ignore", newline="") as f:
                dr = csv.DictReader(f)
                for r in dr:
                    t = (r.get("File1") or "").strip()
                    aln = (r.get("Aligned Length") or "").strip()
                    rmsd = (r.get("RMSD") or "").strip()
                    seqid = (r.get("Seq_ID") or "").strip()
                    avg = avg_tm((r.get("TM-Score1") or "").strip(), (r.get("TM-Score2") or "").strip())
                    if t:
                        mapping[t] = (aln, rmsd, seqid, avg)
            print(f"[SKIP] ssalign_prefilter exists: {out_fp} (reuse mapping)")
            return mapping, out_fp
        except Exception:
            print(f"[WARN] ssalign_prefilter exists but failed to reuse: {out_fp} -> will recompute")

    header, rows = read_ssalign_csv(in_fp)
    # 期望列名：File1,Cosine_Similarity,Score
    # 但这里不强依赖，只取 File1/Cosine_Similarity
    q_path = query_dir / f"{query_id}.cif"

    targets = []
    for r in rows:
        t = (r.get("File1") or "").strip()
        if t:
            targets.append(t)

    uniq_targets = sorted(set(targets))

    def one_target(t_v4: str):
        t_path = v6_dir / (to_v6_id(t_v4) + ".pdb")
        ok, tm1, tm2, aln_len, rmsd, seq_id = run_tmalign(tm_exec, q_path, t_path, timeout_s)
        return t_v4, (tm1, tm2, aln_len, rmsd, seq_id)

    tmalign_map = {}
    with ThreadPoolExecutor(max_workers=max(1, nworker)) as ex:
        futs = [ex.submit(one_target, t) for t in uniq_targets]
        for fu in as_completed(futs):
            t_v4, vals = fu.result()
            tmalign_map[t_v4] = vals

    ensure_dir(ssalign_prefilter_out_dir)
    mapping_for_full = {}

    with out_fp.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["File1", "File2", "TM-Score1", "TM-Score2", "Aligned Length", "RMSD", "Seq_ID", "Cosine_Similarity"])
        for r in rows:
            t_v4 = (r.get("File1") or "").strip()
            cos = (r.get("Cosine_Similarity") or "").strip()
            tm1, tm2, aln_len, rmsd, seq_id = tmalign_map.get(t_v4, ("", "", "", "", ""))
            w.writerow([t_v4, query_id, tm1, tm2, aln_len, rmsd, seq_id, cos])

            a = avg_tm(tm1, tm2)
            if t_v4:
                mapping_for_full[t_v4] = (aln_len, rmsd, seq_id, a)

    print(f"[OK] ssalign_prefilter -> {out_fp}")
    return mapping_for_full, out_fp


# -----------------------------
# ssalign(full): 从 prefilter 的 mapping 映射 TM 指标 -> ssalign/*.result
# -----------------------------
def process_ssalign_full_one(query_id: str, ssalign_in_dir: Path, ssalign_out_dir: Path,
                             prefilter_map: dict, overwrite: bool):
    in_fp = ssalign_in_dir / f"{query_id}.ssalign"
    out_fp = ssalign_out_dir / f"{query_id}.result"

    if (not in_fp.exists()) or in_fp.stat().st_size == 0:
        print(f"[WARN] ssalign missing: {in_fp}")
        return
    if out_fp.exists() and (not overwrite):
        print(f"[SKIP] ssalign exists: {out_fp}")
        return

    _, rows = read_ssalign_csv(in_fp)
    ensure_dir(ssalign_out_dir)

    with out_fp.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["File1", "Aligned Length", "RMSD", "Seq_ID", "Cosine_Similarity", "Avg_TM_Score", "Score"])

        for r in rows:
            t_v4 = (r.get("File1") or "").strip()
            cos = (r.get("Cosine_Similarity") or "").strip()
            score = (r.get("Score") or "").strip()

            aln_len, rmsd, seq_id, a = prefilter_map.get(t_v4, ("", "", "", ""))
            w.writerow([t_v4, aln_len, rmsd, seq_id, cos, a, score])

    print(f"[OK] ssalign(full mapped) -> {out_fp}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selected", required=True, help="random_100_queries.txt")
    ap.add_argument("--query_dir", required=True, help="query cif dir (AF-...-model_v4.cif)")
    ap.add_argument("--v6_dir", required=True, help="target v6 pdb dir (AF-...-model_v6.pdb)")
    ap.add_argument("--tmalign_exec", required=True, help="TMalign_cpp path")

    ap.add_argument("--foldseek_in_dir", required=True, help=".../foldseek/timebenchmark")
    ap.add_argument("--foldseek_out_dir", required=True, help=".../foldseek (write *.result)")

    ap.add_argument("--ssalign_in_dir", required=True, help=".../SSAlign/SVD512/timechmark")
    ap.add_argument("--ssalign_out_dir", required=True, help=".../SSAlign/SVD512/ssalign (write *.result)")

    ap.add_argument("--ssalign_prefilter_in_dir", required=True, help=".../SSAlign/SVD512/ssalign_prefilter")
    ap.add_argument("--ssalign_prefilter_out_dir", required=True, help=".../SSAlign/SVD512/ssalign_prefilter (write *.result)")

    ap.add_argument("--nworker", type=int, default=16, help="并发跑 TMalign 的 worker 数（线程数）")
    ap.add_argument("--timeout", type=int, default=300, help="单次 TMalign 超时秒数")
    ap.add_argument("--overwrite", action="store_true", help="覆盖已存在的 *.result")

    args = ap.parse_args()

    selected = Path(args.selected)
    query_ids = read_id_list(selected)

    foldseek_in_dir = Path(args.foldseek_in_dir)
    foldseek_out_dir = Path(args.foldseek_out_dir)

    ssalign_in_dir = Path(args.ssalign_in_dir)
    ssalign_out_dir = Path(args.ssalign_out_dir)

    ssalign_prefilter_in_dir = Path(args.ssalign_prefilter_in_dir)
    ssalign_prefilter_out_dir = Path(args.ssalign_prefilter_out_dir)

    query_dir = Path(args.query_dir)
    v6_dir = Path(args.v6_dir)

    print(f"[START] queries={len(query_ids)} nworker={args.nworker} timeout={args.timeout}")

    for qid in query_ids:
        # 1) foldseek 加 TM
        process_foldseek_one(
            qid, foldseek_in_dir, foldseek_out_dir,
            query_dir, v6_dir, args.tmalign_exec,
            args.timeout, args.nworker, args.overwrite
        )

        # 2) ssalign_prefilter 跑 TM，并返回 mapping
        pre_map, _ = process_ssalign_prefilter_one(
            qid, ssalign_prefilter_in_dir, ssalign_prefilter_out_dir,
            query_dir, v6_dir, args.tmalign_exec,
            args.timeout, args.nworker, args.overwrite
        )

        # 3) ssalign(full) 用 mapping 填 TM 指标（不再跑 TM）
        process_ssalign_full_one(
            qid, ssalign_in_dir, ssalign_out_dir,
            pre_map, args.overwrite
        )

    print("[DONE]")


if __name__ == "__main__":
    main()

"""
python add_TMalign.py \
  --selected random_100_queries.txt \
  --query_dir ../pdbData/pdb/AFDB50 \
  --v6_dir ../pdbData/pdb/AFDB50 \
  --tmalign_exec ../bin/TMalign_cpp \
  --foldseek_in_dir ../benchmarkData/AFDB50/foldseek/timebenchmark \
  --foldseek_out_dir ../benchmarkData/AFDB50/foldseek \
  --ssalign_in_dir ../benchmarkData/AFDB50/SSAlign/SVD512/timechmark \
  --ssalign_out_dir ../benchmarkData/AFDB50/SSAlign/SVD512/ssalign \
  --ssalign_prefilter_in_dir ../benchmarkData/AFDB50/SSAlign/SVD512/ssalign_prefilter \
  --ssalign_prefilter_out_dir ../benchmarkData/AFDB50/SSAlign/SVD512/ssalign_prefilter \
  --nworker 24 \
  --timeout 300
"""