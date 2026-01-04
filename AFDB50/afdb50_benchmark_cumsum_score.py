"""
random_100_queries.txt

../benchmarkData/AFDB50/SSAlign/SVD512/ssalign/
File1,Aligned Length,RMSD,Seq_ID,Cosine_Similarity,Avg_TM_Score,Score


../benchmarkData/AFDB50/SSAlign/SVD512/ssalign_prefilter/
File1,File2,TM-Score1,TM-Score2,Aligned Length,RMSD,Seq_ID,Cosine_Similarity


../benchmarkData/AFDB50/foldseek/
File1,File2,TM-Score1,TM-Score2,Aligned Length,RMSD,Seq_ID,FoldSeek_Metrics

"""

from __future__ import annotations

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


# -----------------------------
# Utils
# -----------------------------
def read_query_list(txt_path: str) -> list[str]:
    basenames = []
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            # 允许输入 AF-xxx... 或 AF-xxx...cif/pdb/result/ssalign
            s = re.sub(r"\.(cif|pdb|result|ssalign|csv)$", "", s, flags=re.IGNORECASE)
            basenames.append(s)
    # 去重但保序
    seen = set()
    out = []
    for b in basenames:
        if b not in seen:
            out.append(b)
            seen.add(b)
    return out


def find_result_file(root: Path, basename: str) -> Path | None:
    """
    兼容 basename.result / basename.ssalign
    """
    p1 = root / f"{basename}.result"
    if p1.exists():
        return p1
    p2 = root / f"{basename}.ssalign"
    if p2.exists():
        return p2
    return None


def normalize_af_id(x: str) -> str:
    """
    用于做差集/交集时统一 target id（foldseek 的 File2 是 *_v6.pdb，ssalign 是 *_v4）
    目标：统一到 AF-XXXXXX-F1 这种粒度（不含 model_v? 也不含扩展名）
    """
    if x is None:
        return ""
    s = str(x).strip()

    # 去扩展名
    s = re.sub(r"\.(pdb|cif)$", "", s, flags=re.IGNORECASE)

    # 尝试直接抓 AF-...-F\d
    m = re.search(r"(AF-[A-Za-z0-9]+-F\d+)", s)
    if m:
        return m.group(1)

    # 否则退化：去掉 _model_v\d 以及后缀
    s = re.sub(r"-model_v\d+$", "", s)
    s = re.sub(r"_model_v\d+$", "", s)
    return s


def to_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def ensure_avg_tm(df: pd.DataFrame) -> pd.DataFrame:
    """
    让 df 一定有 Avg_TM_Score（若已有则跳过；若有 TM-Score1/2 则计算）
    """
    if "Avg_TM_Score" in df.columns:
        df["Avg_TM_Score"] = pd.to_numeric(df["Avg_TM_Score"], errors="coerce")
        return df
    if "TM-Score1" in df.columns and "TM-Score2" in df.columns:
        df = to_numeric(df, ["TM-Score1", "TM-Score2"])
        df["Avg_TM_Score"] = (df["TM-Score1"] + df["TM-Score2"]) / 2.0
    else:
        df["Avg_TM_Score"] = np.nan
    return df


def parse_foldseek_evalue(series: pd.Series) -> pd.Series:
    """
    复刻你 SwissProt 版本：E-value = FoldSeek_Metrics.split()[8] :contentReference[oaicite:2]{index=2}
    """
    def _one(x):
        if pd.isna(x):
            return np.nan
        s = str(x).strip()
        if not s:
            return np.nan
        parts = s.split()
        if len(parts) <= 8:
            return np.nan
        try:
            return float(parts[8])
        except Exception:
            return np.nan

    return series.apply(_one)


# -----------------------------
# Load per-query data
# -----------------------------
def load_foldseek_one(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, na_values=["NA", "NaN", ""], keep_default_na=True)
    # 必要列：File2 / TM / RMSD / FoldSeek_Metrics
    df = to_numeric(df, ["TM-Score1", "TM-Score2", "Aligned Length", "RMSD", "Seq_ID"])
    df = ensure_avg_tm(df)
    if "FoldSeek_Metrics" in df.columns:
        df["E-value"] = parse_foldseek_evalue(df["FoldSeek_Metrics"])
    else:
        df["E-value"] = np.nan

    # 统一 target key（foldseek 里 target 在 File2）
    if "File2" in df.columns:
        df["target_norm"] = df["File2"].apply(normalize_af_id)
    else:
        df["target_norm"] = ""
    return df


def load_ssalign_prefilter_one(path: Path) -> pd.DataFrame:
    """
    你新统一的 prefilter 格式：
    File1,File2,TM-Score1,TM-Score2,Aligned Length,RMSD,Seq_ID,Cosine_Similarity
    """
    df = pd.read_csv(path, na_values=["NA", "NaN", ""], keep_default_na=True)
    df = to_numeric(df, ["TM-Score1", "TM-Score2", "Aligned Length", "RMSD", "Seq_ID", "Cosine_Similarity"])
    df = ensure_avg_tm(df)

    # target 在 File1
    if "File1" in df.columns:
        df["target_norm"] = df["File1"].apply(normalize_af_id)
    else:
        df["target_norm"] = ""
    return df


def load_ssalign_two_stage_one(path: Path) -> pd.DataFrame:
    """
    你新统一的 two-stage SSAlign 格式：
    File1,Aligned Length,RMSD,Seq_ID,Cosine_Similarity,Avg_TM_Score,Score
    """
    df = pd.read_csv(path, na_values=["NA", "NaN", ""], keep_default_na=True)
    df = to_numeric(df, ["Aligned Length", "RMSD", "Seq_ID", "Cosine_Similarity", "Avg_TM_Score", "Score"])
    df = ensure_avg_tm(df)

    if "File1" in df.columns:
        df["target_norm"] = df["File1"].apply(normalize_af_id)
    else:
        df["target_norm"] = ""
    return df


# -----------------------------
# Sorting logic (mirror SwissProt)
# -----------------------------
def sort_df_for_tool(df: pd.DataFrame, tool: str, sort_mode: str, cos_threshold: float) -> pd.DataFrame:
    """
    tool:
      - foldseek
      - ssalign
      - ssalign_top1000
      - ssalign_top2000
    """
    if sort_mode == "avg_TM-score":
        return df.sort_values(by="Avg_TM_Score", ascending=False)

    # method_measure
    if tool == "foldseek":
        # foldseek: E-value asc :contentReference[oaicite:3]{index=3}
        return df.sort_values(by="E-value", ascending=True)

    if tool == "ssalign":
        # ssalign: cos>=thr 按cos desc；cos<thr 按 Score desc；拼接 :contentReference[oaicite:4]{index=4}
        part1 = df[df["Cosine_Similarity"] >= cos_threshold].sort_values(by="Cosine_Similarity", ascending=False)
        part2 = df[df["Cosine_Similarity"] < cos_threshold].sort_values(by="Score", ascending=False)
        return pd.concat([part1, part2], ignore_index=True)

    # prefilter top1000/top2000：直接 cosine desc :contentReference[oaicite:5]{index=5}
    return df.sort_values(by="Cosine_Similarity", ascending=False)


def extract_scores(df_sorted: pd.DataFrame, score_measure: str, max_points: int) -> np.ndarray:
    if score_measure == "avg_TM-score":
        s = pd.to_numeric(df_sorted["Avg_TM_Score"], errors="coerce")
    else:
        s = pd.to_numeric(df_sorted["RMSD"], errors="coerce")

    s = s.dropna().values
    if max_points is not None and max_points > 0:
        s = s[:max_points]
    return s


# -----------------------------
# Benchmark: cumsum (overall)
# -----------------------------
def benchmark_cumsum_afdb50(
    selected_txt: str,
    foldseek_dir: str,
    ssalign_prefilter_dir: str,
    ssalign_dir: str,
    out_dir: str,
    dim: int,
    cos_threshold: float,
    sort_mode: str,
    score_measure: str,
    max_points: int,
    topk1000: int = 1000,
    topk2000: int = 2000,
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    basenames = read_query_list(selected_txt)

    merged_foldseek = []
    merged_ssalign = []
    merged_ssa1000 = []
    merged_ssa2000 = []

    missing_foldseek = 0
    missing_pref = 0
    missing_ssalign = 0

    foldseek_dir = Path(foldseek_dir)
    ssalign_prefilter_dir = Path(ssalign_prefilter_dir)
    ssalign_dir = Path(ssalign_dir)

    for b in basenames:
        p_fold = find_result_file(foldseek_dir, b)
        p_pref = find_result_file(ssalign_prefilter_dir, b)
        p_ssa = find_result_file(ssalign_dir, b)

        if p_fold is None:
            missing_foldseek += 1
        else:
            df_fold = load_foldseek_one(p_fold)
            merged_foldseek.append(df_fold)

        if p_pref is None:
            missing_pref += 1
        else:
            df_pref = load_ssalign_prefilter_one(p_pref)
            df_pref_sorted = df_pref.sort_values(by="Cosine_Similarity", ascending=False)
            merged_ssa1000.append(df_pref_sorted.head(topk1000))
            merged_ssa2000.append(df_pref_sorted.head(topk2000))

        if p_ssa is None:
            missing_ssalign += 1
        else:
            df_ssa = load_ssalign_two_stage_one(p_ssa)
            merged_ssalign.append(df_ssa)

    merged_foldseek = pd.concat(merged_foldseek, ignore_index=True) if merged_foldseek else pd.DataFrame()
    merged_ssalign = pd.concat(merged_ssalign, ignore_index=True) if merged_ssalign else pd.DataFrame()
    merged_ssa1000 = pd.concat(merged_ssa1000, ignore_index=True) if merged_ssa1000 else pd.DataFrame()
    merged_ssa2000 = pd.concat(merged_ssa2000, ignore_index=True) if merged_ssa2000 else pd.DataFrame()

    print(f"[INFO] selected={len(basenames)} "
          f"missing: foldseek={missing_foldseek} prefilter={missing_pref} ssalign={missing_ssalign}")
    print(f"[INFO] merged rows: foldseek={len(merged_foldseek)} ssalign={len(merged_ssalign)} "
          f"ssa1000={len(merged_ssa1000)} ssa2000={len(merged_ssa2000)}")

    # 排序
    fold_sorted = sort_df_for_tool(merged_foldseek, "foldseek", sort_mode, cos_threshold) if len(merged_foldseek) else merged_foldseek
    ssa_sorted = sort_df_for_tool(merged_ssalign, "ssalign", sort_mode, cos_threshold) if len(merged_ssalign) else merged_ssalign
    ssa1000_sorted = sort_df_for_tool(merged_ssa1000, "ssalign_top1000", sort_mode, cos_threshold) if len(merged_ssa1000) else merged_ssa1000
    ssa2000_sorted = sort_df_for_tool(merged_ssa2000, "ssalign_top2000", sort_mode, cos_threshold) if len(merged_ssa2000) else merged_ssa2000

    # 分数数组（只用非空）
    fold_scores = extract_scores(fold_sorted, score_measure, max_points)
    ssa_scores = extract_scores(ssa_sorted, score_measure, max_points)
    ssa1000_scores = extract_scores(ssa1000_sorted, score_measure, max_points)
    ssa2000_scores = extract_scores(ssa2000_sorted, score_measure, max_points)

    # cumsum
    fold_cumsum = np.cumsum(fold_scores)
    ssa_cumsum = np.cumsum(ssa_scores)
    ssa1000_cumsum = np.cumsum(ssa1000_scores)
    ssa2000_cumsum = np.cumsum(ssa2000_scores)

    out_path = out_dir / f"AFDB50_dim_{dim}_cumsum_{score_measure}_sorted_by_{sort_mode}.npz"
    np.savez(
        out_path,
        foldseek=fold_cumsum,
        ssalign=ssa_cumsum,
        ssalign_prefilter_1000=ssa1000_cumsum,
        ssalign_prefilter_2000=ssa2000_cumsum,
    )
    print(f"[DONE] saved: {out_path}")


# -----------------------------
# Benchmark: except/diff cumsum
# -----------------------------
def benchmark_except_cumsum_afdb50(
    selected_txt: str,
    foldseek_dir: str,
    ssalign_prefilter_dir: str,
    ssalign_dir: str,
    out_dir: str,
    dim: int,
    cos_threshold: float,
    sort_mode: str,
    score_measure: str,
    max_points: int,
    topk1000: int = 1000,
    topk2000: int = 2000,
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    basenames = read_query_list(selected_txt)

    merged_ssalign_except_foldseek = []
    merged_foldseek_except_ssalign = []

    merged_ssa1000_except_foldseek = []
    merged_ssa2000_except_foldseek = []

    merged_foldseek_except_ssa1000 = []
    merged_foldseek_except_ssa2000 = []

    foldseek_dir = Path(foldseek_dir)
    ssalign_prefilter_dir = Path(ssalign_prefilter_dir)
    ssalign_dir = Path(ssalign_dir)

    miss_any = 0

    for b in basenames:
        p_fold = find_result_file(foldseek_dir, b)
        p_pref = find_result_file(ssalign_prefilter_dir, b)
        p_ssa = find_result_file(ssalign_dir, b)

        if p_fold is None or p_pref is None or p_ssa is None:
            miss_any += 1
            continue

        df_fold = load_foldseek_one(p_fold)
        df_pref = load_ssalign_prefilter_one(p_pref)
        df_ssa = load_ssalign_two_stage_one(p_ssa)

        df_pref_sorted = df_pref.sort_values(by="Cosine_Similarity", ascending=False)
        df_ssa1000 = df_pref_sorted.head(topk1000)
        df_ssa2000 = df_pref_sorted.head(topk2000)

        # sets（按 target_norm）
        fold_set = set(df_fold["target_norm"].dropna().astype(str))
        ssa_set = set(df_ssa["target_norm"].dropna().astype(str))
        ssa1000_set = set(df_ssa1000["target_norm"].dropna().astype(str))
        ssa2000_set = set(df_ssa2000["target_norm"].dropna().astype(str))

        # 差集过滤
        ssalign_except_foldseek = df_ssa[~df_ssa["target_norm"].isin(fold_set)]
        foldseek_except_ssalign = df_fold[~df_fold["target_norm"].isin(ssa_set)]

        ssa1000_except_foldseek = df_ssa1000[~df_ssa1000["target_norm"].isin(fold_set)]
        ssa2000_except_foldseek = df_ssa2000[~df_ssa2000["target_norm"].isin(fold_set)]

        fold_except_ssa1000 = df_fold[~df_fold["target_norm"].isin(ssa1000_set)]
        fold_except_ssa2000 = df_fold[~df_fold["target_norm"].isin(ssa2000_set)]

        merged_ssalign_except_foldseek.append(ssalign_except_foldseek)
        merged_foldseek_except_ssalign.append(foldseek_except_ssalign)

        merged_ssa1000_except_foldseek.append(ssa1000_except_foldseek)
        merged_ssa2000_except_foldseek.append(ssa2000_except_foldseek)

        merged_foldseek_except_ssa1000.append(fold_except_ssa1000)
        merged_foldseek_except_ssa2000.append(fold_except_ssa2000)

    if miss_any:
        print(f"[WARN] skipped queries (missing any of foldseek/prefilter/ssalign) = {miss_any}")

    merged_ssalign_except_foldseek = pd.concat(merged_ssalign_except_foldseek, ignore_index=True) if merged_ssalign_except_foldseek else pd.DataFrame()
    merged_foldseek_except_ssalign = pd.concat(merged_foldseek_except_ssalign, ignore_index=True) if merged_foldseek_except_ssalign else pd.DataFrame()

    merged_ssa1000_except_foldseek = pd.concat(merged_ssa1000_except_foldseek, ignore_index=True) if merged_ssa1000_except_foldseek else pd.DataFrame()
    merged_ssa2000_except_foldseek = pd.concat(merged_ssa2000_except_foldseek, ignore_index=True) if merged_ssa2000_except_foldseek else pd.DataFrame()

    merged_foldseek_except_ssa1000 = pd.concat(merged_foldseek_except_ssa1000, ignore_index=True) if merged_foldseek_except_ssa1000 else pd.DataFrame()
    merged_foldseek_except_ssa2000 = pd.concat(merged_foldseek_except_ssa2000, ignore_index=True) if merged_foldseek_except_ssa2000 else pd.DataFrame()

    # 排序（差集也沿用同样排序逻辑）
    ssa_ex_fold_sorted = sort_df_for_tool(merged_ssalign_except_foldseek, "ssalign", sort_mode, cos_threshold) if len(merged_ssalign_except_foldseek) else merged_ssalign_except_foldseek
    fold_ex_ssa_sorted = sort_df_for_tool(merged_foldseek_except_ssalign, "foldseek", sort_mode, cos_threshold) if len(merged_foldseek_except_ssalign) else merged_foldseek_except_ssalign

    ssa1000_ex_fold_sorted = sort_df_for_tool(merged_ssa1000_except_foldseek, "ssalign_top1000", sort_mode, cos_threshold) if len(merged_ssa1000_except_foldseek) else merged_ssa1000_except_foldseek
    ssa2000_ex_fold_sorted = sort_df_for_tool(merged_ssa2000_except_foldseek, "ssalign_top2000", sort_mode, cos_threshold) if len(merged_ssa2000_except_foldseek) else merged_ssa2000_except_foldseek

    fold_ex_ssa1000_sorted = sort_df_for_tool(merged_foldseek_except_ssa1000, "foldseek", sort_mode, cos_threshold) if len(merged_foldseek_except_ssa1000) else merged_foldseek_except_ssa1000
    fold_ex_ssa2000_sorted = sort_df_for_tool(merged_foldseek_except_ssa2000, "foldseek", sort_mode, cos_threshold) if len(merged_foldseek_except_ssa2000) else merged_foldseek_except_ssa2000

    # 分数数组 + cumsum
    ssa_ex_fold_scores = extract_scores(ssa_ex_fold_sorted, score_measure, max_points)
    fold_ex_ssa_scores = extract_scores(fold_ex_ssa_sorted, score_measure, max_points)

    ssa1000_ex_fold_scores = extract_scores(ssa1000_ex_fold_sorted, score_measure, max_points)
    ssa2000_ex_fold_scores = extract_scores(ssa2000_ex_fold_sorted, score_measure, max_points)

    fold_ex_ssa1000_scores = extract_scores(fold_ex_ssa1000_sorted, score_measure, max_points)
    fold_ex_ssa2000_scores = extract_scores(fold_ex_ssa2000_sorted, score_measure, max_points)

    out_path = out_dir / f"AFDB50_dim_{dim}_except_cumsum_{score_measure}_sorted_by_{sort_mode}.npz"
    np.savez(
        out_path,
        ssalign_except_foldseek_cumsum=np.cumsum(ssa_ex_fold_scores),
        foldseek_except_ssalign_cumsum=np.cumsum(fold_ex_ssa_scores),
        ssalign_prefilter_1000_except_foldseek_cumsum=np.cumsum(ssa1000_ex_fold_scores),
        ssalign_prefilter_2000_foldseek_cumsum=np.cumsum(ssa2000_ex_fold_scores),
        foldseek_except_ssalign_prefilter_1000_cumsum=np.cumsum(fold_ex_ssa1000_scores),
        foldseek_except_ssalign_prefilter_2000_cumsum=np.cumsum(fold_ex_ssa2000_scores),
    )
    print(f"[DONE] saved: {out_path}")


# -----------------------------
# CLI
# -----------------------------
def parse_args():
    p = argparse.ArgumentParser("AFDB50 benchmark cumsum npz generator")

    p.add_argument("--except_mode", type=int, required=True, help="0=overall cumsum, 1=except/diff cumsum")
    p.add_argument("--dim", type=int, default=512, help="only for naming (e.g., 512)")
    p.add_argument("--cosine", type=float, default=0.3, help="ssalign cosine threshold")
    p.add_argument("--sort_mode", type=str, required=True, choices=["avg_TM-score", "method_measure"])
    p.add_argument("--score_measure", type=str, required=True, choices=["avg_TM-score", "RMSD"])
    p.add_argument("--max_points", type=int, default=200000)

    p.add_argument("--selected", type=str, required=True, help="random_100_queries.txt")
    p.add_argument("--foldseek_dir", type=str, required=True)
    p.add_argument("--ssalign_prefilter_dir", type=str, required=True)
    p.add_argument("--ssalign_dir", type=str, required=True)
    p.add_argument("--out_dir", type=str, required=True)

    p.add_argument("--topk1000", type=int, default=1000)
    p.add_argument("--topk2000", type=int, default=2000)

    return p.parse_args()


def main():
    args = parse_args()

    if args.except_mode == 0:
        benchmark_cumsum_afdb50(
            selected_txt=args.selected,
            foldseek_dir=args.foldseek_dir,
            ssalign_prefilter_dir=args.ssalign_prefilter_dir,
            ssalign_dir=args.ssalign_dir,
            out_dir=args.out_dir,
            dim=args.dim,
            cos_threshold=args.cosine,
            sort_mode=args.sort_mode,
            score_measure=args.score_measure,
            max_points=args.max_points,
            topk1000=args.topk1000,
            topk2000=args.topk2000,
        )
    else:
        benchmark_except_cumsum_afdb50(
            selected_txt=args.selected,
            foldseek_dir=args.foldseek_dir,
            ssalign_prefilter_dir=args.ssalign_prefilter_dir,
            ssalign_dir=args.ssalign_dir,
            out_dir=args.out_dir,
            dim=args.dim,
            cos_threshold=args.cosine,
            sort_mode=args.sort_mode,
            score_measure=args.score_measure,
            max_points=args.max_points,
            topk1000=args.topk1000,
            topk2000=args.topk2000,
        )


if __name__ == "__main__":
    main()

"""
python afdb50_benchmark_cumsum_score.py \
  --except_mode 0 \
  --dim 512 \
  --cosine 0.3 \
  --sort_mode avg_TM-score \
  --score_measure avg_TM-score \
  --max_points 200000 \
  --selected random_100_queries.txt \
  --foldseek_dir ../benchmarkData/AFDB50/foldseek/ \
  --ssalign_prefilter_dir ../benchmarkData/AFDB50/SSAlign/SVD512/ssalign_prefilter/ \
  --ssalign_dir ../benchmarkData/AFDB50/SSAlign/SVD512/ssalign/ \
  --out_dir ../benchmarkData/AFDB50/cumsumNpz

python afdb50_benchmark_cumsum_score.py \
  --except_mode 0 \
  --dim 512 \
  --cosine 0.3 \
  --sort_mode avg_TM-score \
  --score_measure RMSD \
  --max_points 200000 \
  --selected random_100_queries.txt \
  --foldseek_dir ../benchmarkData/AFDB50/foldseek/ \
  --ssalign_prefilter_dir ../benchmarkData/AFDB50/SSAlign/SVD512/ssalign_prefilter/ \
  --ssalign_dir ../benchmarkData/AFDB50/SSAlign/SVD512/ssalign/ \
  --out_dir ../benchmarkData/AFDB50/cumsumNpz
  
  
python afdb50_benchmark_cumsum_score.py \
  --except_mode 0 \
  --dim 512 \
  --cosine 0.3 \
  --sort_mode method_measure \
  --score_measure avg_TM-score \
  --max_points 200000 \
  --selected random_100_queries.txt \
  --foldseek_dir ../benchmarkData/AFDB50/foldseek/ \
  --ssalign_prefilter_dir ../benchmarkData/AFDB50/SSAlign/SVD512/ssalign_prefilter/ \
  --ssalign_dir ../benchmarkData/AFDB50/SSAlign/SVD512/ssalign/ \
  --out_dir ../benchmarkData/AFDB50/cumsumNpz
  
  python afdb50_benchmark_cumsum_score.py \
  --except_mode 0 \
  --dim 512 \
  --cosine 0.3 \
  --sort_mode method_measure \
  --score_measure RMSD \
  --max_points 200000 \
  --selected random_100_queries.txt \
  --foldseek_dir ../benchmarkData/AFDB50/foldseek/ \
  --ssalign_prefilter_dir ../benchmarkData/AFDB50/SSAlign/SVD512/ssalign_prefilter/ \
  --ssalign_dir ../benchmarkData/AFDB50/SSAlign/SVD512/ssalign/ \
  --out_dir ../benchmarkData/AFDB50/cumsumNpz
  
  
  
  差集
  
  python afdb50_benchmark_cumsum_score.py \
  --except_mode 1 \
  --dim 512 \
  --cosine 0.3 \
  --sort_mode avg_TM-score \
  --score_measure avg_TM-score \
  --max_points 200000 \
  --selected random_100_queries.txt \
  --foldseek_dir ../benchmarkData/AFDB50/foldseek/ \
  --ssalign_prefilter_dir ../benchmarkData/AFDB50/SSAlign/SVD512/ssalign_prefilter/ \
  --ssalign_dir ../benchmarkData/AFDB50/SSAlign/SVD512/ssalign/ \
  --out_dir ../benchmarkData/AFDB50/cumsumNpz
  
  
  python afdb50_benchmark_cumsum_score.py \
  --except_mode 1 \
  --dim 512 \
  --cosine 0.3 \
  --sort_mode avg_TM-score \
  --score_measure RMSD \
  --max_points 200000 \
  --selected random_100_queries.txt \
  --foldseek_dir ../benchmarkData/AFDB50/foldseek/ \
  --ssalign_prefilter_dir ../benchmarkData/AFDB50/SSAlign/SVD512/ssalign_prefilter/ \
  --ssalign_dir ../benchmarkData/AFDB50/SSAlign/SVD512/ssalign/ \
  --out_dir ../benchmarkData/AFDB50/cumsumNpz
  
  
  python afdb50_benchmark_cumsum_score.py \
  --except_mode 1 \
  --dim 512 \
  --cosine 0.3 \
  --sort_mode method_measure \
  --score_measure avg_TM-score \
  --max_points 200000 \
  --selected random_100_queries.txt \
  --foldseek_dir ../benchmarkData/AFDB50/foldseek/ \
  --ssalign_prefilter_dir ../benchmarkData/AFDB50/SSAlign/SVD512/ssalign_prefilter/ \
  --ssalign_dir ../benchmarkData/AFDB50/SSAlign/SVD512/ssalign/ \
  --out_dir ../benchmarkData/AFDB50/cumsumNpz
  
  
python afdb50_benchmark_cumsum_score.py \
  --except_mode 1 \
  --dim 512 \
  --cosine 0.3 \
  --sort_mode method_measure \
  --score_measure RMSD \
  --max_points 200000 \
  --selected random_100_queries.txt \
  --foldseek_dir ../benchmarkData/AFDB50/foldseek/ \
  --ssalign_prefilter_dir ../benchmarkData/AFDB50/SSAlign/SVD512/ssalign_prefilter/ \
  --ssalign_dir ../benchmarkData/AFDB50/SSAlign/SVD512/ssalign/ \
  --out_dir ../benchmarkData/AFDB50/cumsumNpz
"""

