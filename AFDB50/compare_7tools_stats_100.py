#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
compare_7tools_stats_100.py

仅对 random_100_queries.txt 中的 query 统计 7 种 tool 的指标：

tool:
  1) foldseek
  2) ssalign
  3) ssalign_prefilter_2000
  4) foldseek_except_ssalign
  5) ssalign_except_foldseek
  6) foldseek_except_ssalign_prefilter_2000
  7) ssalign_prefilter_2000_except_foldseek

输出字段保持不变：
query, tool,
total_rows, na_rows,
tm_non_na_rows, avg_tmscore, sum_tmscore,
rmsd_non_na_rows, avg_RMSD, sum_RMSD

差集按“行”统计（不做 unique target 去重）。
foldseek 的 target 用 File2；ssalign/ssalign_prefilter 用 File1。
normalize 后比较（去 .pdb/.cif + 去 _v4/_v6/... 版本后缀）。
"""

import argparse
import csv
import re
from pathlib import Path

_re_ext = re.compile(r"\.(pdb|cif)$", re.IGNORECASE)
_re_ver = re.compile(r"_v\d+$", re.IGNORECASE)


def normalize_id(s: str) -> str:
    """统一 target ID：去扩展名、去版本号 _v4/_v6/..."""
    if s is None:
        return ""
    x = s.strip()
    if not x:
        return ""
    x = _re_ext.sub("", x)
    x = _re_ver.sub("", x)
    return x


def is_empty_or_na(val: str) -> bool:
    if val is None:
        return True
    v = str(val).strip()
    return (v == "") or (v.upper() == "NA")


def safe_float(val: str):
    try:
        return float(val)
    except Exception:
        return None


def read_csv_rows(fp: Path):
    """读取 CSV 文件，返回 (header, rows)；文件不存在则返回空。"""
    if not fp.exists():
        return [], []
    with fp.open("r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header is None:
            return [], []
        rows = [r for r in reader if r]
    return header, rows


def build_colmap(header):
    return {name: i for i, name in enumerate(header)} if header else {}


def compute_stats(rows, tm_idx: int, rmsd_idx: int):
    """
    na_rows：tm 列为空/NA/无法解析 float 的行数
    """
    total_rows = 0
    na_rows = 0

    tm_sum = 0.0
    tm_cnt = 0

    rmsd_sum = 0.0
    rmsd_cnt = 0

    for row in rows:
        if not row:
            continue
        total_rows += 1

        # TM
        if tm_idx is None or tm_idx >= len(row):
            na_rows += 1
        else:
            tm_val = row[tm_idx]
            if is_empty_or_na(tm_val):
                na_rows += 1
            else:
                x = safe_float(tm_val)
                if x is None:
                    na_rows += 1
                else:
                    tm_sum += x
                    tm_cnt += 1

        # RMSD
        if rmsd_idx is not None and rmsd_idx < len(row):
            rmsd_val = row[rmsd_idx]
            if not is_empty_or_na(rmsd_val):
                y = safe_float(rmsd_val)
                if y is not None:
                    rmsd_sum += y
                    rmsd_cnt += 1

    avg_tm = (tm_sum / tm_cnt) if tm_cnt > 0 else 0.0
    avg_rmsd = (rmsd_sum / rmsd_cnt) if rmsd_cnt > 0 else 0.0

    return {
        "total_rows": total_rows,
        "na_rows": na_rows,
        "tm_non_na_rows": tm_cnt,
        "avg_tmscore": avg_tm,
        "sum_tmscore": tm_sum,
        "rmsd_non_na_rows": rmsd_cnt,
        "avg_RMSD": avg_rmsd,
        "sum_RMSD": rmsd_sum,
    }


def build_norm_targets(rows, col_idx: int):
    """
    与 rows 等长的 norm_list + 去重 norm_set
    """
    norm_list = []
    norm_set = set()
    for row in rows:
        if col_idx is None or col_idx >= len(row):
            tid = ""
        else:
            tid = normalize_id(row[col_idx])
        norm_list.append(tid)
        if tid:
            norm_set.add(tid)
    return norm_list, norm_set


def filter_rows_by_set(rows, norm_list, keep_if_not_in_set=None):
    """保留 tid 不在集合中的行（按行统计差集）。"""
    if keep_if_not_in_set is None:
        return list(rows)
    S = keep_if_not_in_set
    out = []
    for row, tid in zip(rows, norm_list):
        if tid and (tid not in S):
            out.append(row)
    return out


def load_query_list(fp: Path):
    qs = []
    with fp.open("r", encoding="utf-8") as f:
        for line in f:
            q = line.strip()  # 自动吃掉 \r\n
            if q:
                qs.append(q)
    return qs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--query_list", required=True, help="random_100_queries.txt")
    ap.add_argument("--foldseek_dir", required=True, help="foldseek *.result 目录")
    ap.add_argument("--ssalign_dir", required=True, help="SSAlign/SVD512/ssalign/*.result 目录")
    ap.add_argument("--prefilter_dir", required=True, help="SSAlign/SVD512/ssalign_prefilter/*.result 目录")
    ap.add_argument("--prefilter_top", type=int, default=2000, help="ssalign_prefilter_2000 取前N行（默认2000）")
    ap.add_argument("--out_csv", default="compare_7tools_stats_100.csv")
    args = ap.parse_args()

    q_list = load_query_list(Path(args.query_list))
    print(f"[INFO] queries in list = {len(q_list)} (from {args.query_list})")

    fold_dir = Path(args.foldseek_dir)
    ssa_dir = Path(args.ssalign_dir)
    pre_dir = Path(args.prefilter_dir)

    if not fold_dir.exists():
        raise SystemExit(f"[ERROR] foldseek_dir 不存在: {fold_dir}")
    if not ssa_dir.exists():
        raise SystemExit(f"[ERROR] ssalign_dir 不存在: {ssa_dir}")
    if not pre_dir.exists():
        raise SystemExit(f"[ERROR] prefilter_dir 不存在: {pre_dir}")

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "query", "tool",
        "total_rows", "na_rows",
        "tm_non_na_rows", "avg_tmscore", "sum_tmscore",
        "rmsd_non_na_rows", "avg_RMSD", "sum_RMSD"
    ]

    with out_path.open("w", encoding="utf-8", newline="") as fw:
        writer = csv.DictWriter(fw, fieldnames=fieldnames)
        writer.writeheader()

        miss_fold = 0
        miss_ssa = 0
        miss_pre = 0

        for q in q_list:
            fold_fp = fold_dir / f"{q}.result"
            ssa_fp = ssa_dir / f"{q}.result"
            pre_fp = pre_dir / f"{q}.result"

            # ---------- foldseek ----------
            fold_header, fold_rows = read_csv_rows(fold_fp)
            if not fold_header and not fold_rows:
                miss_fold += 1
            fold_col = build_colmap(fold_header)
            fold_file2_idx = fold_col.get("File2", 1)
            fold_tm_idx = fold_col.get("TM-Score1", None)
            fold_rmsd_idx = fold_col.get("RMSD", None)
            fold_norm_list, fold_norm_set = build_norm_targets(fold_rows, fold_file2_idx)

            # ---------- ssalign (最终阶段，Avg_TM_Score) ----------
            ssa_header, ssa_rows = read_csv_rows(ssa_fp)
            if not ssa_header and not ssa_rows:
                miss_ssa += 1
            ssa_col = build_colmap(ssa_header)
            ssa_file1_idx = ssa_col.get("File1", 0)
            ssa_tm_idx = ssa_col.get("Avg_TM_Score", None)
            ssa_rmsd_idx = ssa_col.get("RMSD", None)
            ssa_norm_list, ssa_norm_set = build_norm_targets(ssa_rows, ssa_file1_idx)

            # ---------- ssalign_prefilter_2000 (取前 prefilter_top 行) ----------
            pre_header, pre_rows_all = read_csv_rows(pre_fp)
            if not pre_header and not pre_rows_all:
                miss_pre += 1
            pre_rows = pre_rows_all[:args.prefilter_top]
            pre_col = build_colmap(pre_header)
            pre_file1_idx = pre_col.get("File1", 0)
            pre_tm_idx = pre_col.get("TM-Score1", None)
            pre_rmsd_idx = pre_col.get("RMSD", None)
            pre_norm_list, pre_norm_set = build_norm_targets(pre_rows, pre_file1_idx)

            # ===== 3 个“全集” =====
            writer.writerow({"query": q, "tool": "foldseek", **compute_stats(fold_rows, fold_tm_idx, fold_rmsd_idx)})
            writer.writerow({"query": q, "tool": "ssalign", **compute_stats(ssa_rows, ssa_tm_idx, ssa_rmsd_idx)})
            writer.writerow({"query": q, "tool": "ssalign_prefilter_2000", **compute_stats(pre_rows, pre_tm_idx, pre_rmsd_idx)})

            # ===== 4 个“差集” =====
            # foldseek_except_ssalign
            fold_ex_ssa_rows = filter_rows_by_set(fold_rows, fold_norm_list, keep_if_not_in_set=ssa_norm_set)
            writer.writerow({"query": q, "tool": "foldseek_except_ssalign",
                             **compute_stats(fold_ex_ssa_rows, fold_tm_idx, fold_rmsd_idx)})

            # ssalign_except_foldseek
            ssa_ex_fold_rows = filter_rows_by_set(ssa_rows, ssa_norm_list, keep_if_not_in_set=fold_norm_set)
            writer.writerow({"query": q, "tool": "ssalign_except_foldseek",
                             **compute_stats(ssa_ex_fold_rows, ssa_tm_idx, ssa_rmsd_idx)})

            # foldseek_except_ssalign_prefilter_2000
            fold_ex_pre_rows = filter_rows_by_set(fold_rows, fold_norm_list, keep_if_not_in_set=pre_norm_set)
            writer.writerow({"query": q, "tool": "foldseek_except_ssalign_prefilter_2000",
                             **compute_stats(fold_ex_pre_rows, fold_tm_idx, fold_rmsd_idx)})

            # ssalign_prefilter_2000_except_foldseek
            pre_ex_fold_rows = filter_rows_by_set(pre_rows, pre_norm_list, keep_if_not_in_set=fold_norm_set)
            writer.writerow({"query": q, "tool": "ssalign_prefilter_2000_except_foldseek",
                             **compute_stats(pre_ex_fold_rows, pre_tm_idx, pre_rmsd_idx)})

    print("[INFO] missing files:",
          f"foldseek={miss_fold}, ssalign={miss_ssa}, ssalign_prefilter={miss_pre}")
    print(f"[DONE] wrote csv: {out_path.resolve()}")


if __name__ == "__main__":
    main()

"""
python compare_7tools_stats_100.py \
  --query_listrandom_100_queries.txt \
  --foldseek_dir ../benchmarkData/AFDB50/foldseek \
  --ssalign_dir ../benchmarkData/AFDB50/SSAlign/SVD512/ssalign \
  --prefilter_dir ../benchmarkData/AFDB50/SSAlign/SVD512/ssalign_prefilter \
  --prefilter_top 2000 \
  --out_csv ../benchmarkData/AFDB50/benchmark/compare_7tools_stats_100.csv

"""