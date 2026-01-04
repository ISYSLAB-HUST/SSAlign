#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv

NUM_KEYS = [
    "total_rows",
    "na_rows",
    "tm_non_na_rows",
    "avg_tmscore",
    "sum_tmscore",
    "rmsd_non_na_rows",
    "avg_RMSD",
    "sum_RMSD",
]

def load_selected(path: str) -> list[str]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                out.append(s)
    return out

def to_float(x: str) -> float:
    try:
        return float(x)
    except Exception:
        return 0.0

def mean(xs):
    return 0.0 if not xs else sum(xs) / len(xs)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selected", required=True, help="random_100_queries.txt")
    ap.add_argument("--csv", required=True, help="compare_7tools_stats.csv")
    ap.add_argument("--tool", default="foldseek", help="默认 foldseek，可改成 ssalign等")
    args = ap.parse_args()

    selected_list = load_selected(args.selected)
    selected = set(selected_list)
    if not selected:
        raise SystemExit("[ERROR] selected list is empty")

    # 先把 csv 中 tool 对应的 query->row 映射出来（每个 query/tool 理论只有一行）
    tool_map = {}
    with open(args.csv, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            q = (row.get("query") or "").strip()
            tool = (row.get("tool") or "").strip()
            if tool == args.tool and q:
                tool_map[q] = row

    # 收集数值
    vals = {k: [] for k in NUM_KEYS}
    missing = []

    for q in selected_list:
        row = tool_map.get(q)
        if row is None:
            missing.append(q)
            continue
        for k in NUM_KEYS:
            vals[k].append(to_float((row.get(k) or "").strip()))

    found = len(selected_list) - len(missing)

    print("[INFO]")
    print(f"  tool               = {args.tool}")
    print(f"  selected_queries   = {len(selected_list)}")
    print(f"  found_in_csv       = {found}")
    print(f"  missing_in_csv     = {len(missing)}")

    print("\n[MEAN PER-QUERY (over found queries)]")
    for k in NUM_KEYS:
        print(f"  mean_{k:14s} = {mean(vals[k]):.6f}")

    # 你要的“100的和”（严格说是 found 的和，因为可能 foldseek 少3个）
    sum_tmscore_total = sum(vals["sum_tmscore"])
    sum_rmsd_total = sum(vals["sum_RMSD"])

    print("\n[SUM OVER QUERIES]")
    print(f"  SUM(sum_tmscore) = {sum_tmscore_total:.6f}")
    print(f"  SUM(sum_RMSD)    = {sum_rmsd_total:.6f}")

    # 额外：也给你一个更“数学正确”的总体平均（用总和/总有效行数）
    # 这比 mean(avg_tmscore) 更不受每个query有效行数不同的影响
    tm_total = sum(vals["sum_tmscore"])
    tm_n = sum(vals["tm_non_na_rows"])
    rmsd_total = sum(vals["sum_RMSD"])
    rmsd_n = sum(vals["rmsd_non_na_rows"])

    overall_avg_tm = 0.0 if tm_n == 0 else (tm_total / tm_n)
    overall_avg_rmsd = 0.0 if rmsd_n == 0 else (rmsd_total / rmsd_n)

    print("\n[OVERALL AVERAGE (sum / non_na_rows)]")
    print(f"  overall_avg_tmscore = {overall_avg_tm:.6f}  (SUM(sum_tmscore)/SUM(tm_non_na_rows))")
    print(f"  overall_avg_RMSD    = {overall_avg_rmsd:.6f}  (SUM(sum_RMSD)/SUM(rmsd_non_na_rows))")

    if missing:
        print("\n[MISSING QUERIES (first 30)]")
        for q in missing[:30]:
            print(" ", q)
        if len(missing) > 30:
            print(f"  ... ({len(missing)-30} more)")

if __name__ == "__main__":
    main()

"""
python test_100_queries.py  --selected random_100_queries.txt --csv ../benchmarkData/AFDB50/benchmark/compare_7tools_stats_100.csv --tool foldseek  
[INFO]
  tool               = foldseek
  selected_queries   = 100
  found_in_csv       = 100
  missing_in_csv     = 0

[MEAN PER-QUERY (over found queries)]
  mean_total_rows     = 1160.770000
  mean_na_rows        = 88.070000
  mean_tm_non_na_rows = 1072.700000
  mean_avg_tmscore    = 0.697784
  mean_sum_tmscore    = 751.126022
  mean_rmsd_non_na_rows = 1072.700000
  mean_avg_RMSD       = 2.580847
  mean_sum_RMSD       = 2750.976300

[SUM OVER QUERIES]
  SUM(sum_tmscore) = 75112.602180
  SUM(sum_RMSD)    = 275097.630000

[OVERALL AVERAGE (sum / non_na_rows)]
  overall_avg_tmscore = 0.700220  (SUM(sum_tmscore)/SUM(tm_non_na_rows))
  overall_avg_RMSD    = 2.564535  (SUM(sum_RMSD)/SUM(rmsd_non_na_rows))

---------------------------------------------------------------------------------------------------------------------------------------------------------------------

python test_100_queries.py  --selected random_100_queries.txt --csv ../benchmarkData/AFDB50/benchmark/compare_7tools_stats_100.csv  --tool ssalign  
[INFO]
  tool               = ssalign
  selected_queries   = 100
  found_in_csv       = 100
  missing_in_csv     = 0

[MEAN PER-QUERY (over found queries)]
  mean_total_rows     = 1000.000000
  mean_na_rows        = 70.550000
  mean_tm_non_na_rows = 929.450000
  mean_avg_tmscore    = 0.774128
  mean_sum_tmscore    = 719.118324
  mean_rmsd_non_na_rows = 929.450000
  mean_avg_RMSD       = 2.457357
  mean_sum_RMSD       = 2283.556700

[SUM OVER QUERIES]
  SUM(sum_tmscore) = 71911.832360
  SUM(sum_RMSD)    = 228355.670000

[OVERALL AVERAGE (sum / non_na_rows)]
  overall_avg_tmscore = 0.773703  (SUM(sum_tmscore)/SUM(tm_non_na_rows))
  overall_avg_RMSD    = 2.456890  (SUM(sum_RMSD)/SUM(rmsd_non_na_rows))

---------------------------------------------------------------------------------------------------------------------------------------------------------------------

python test_100_queries.py  --selected random_100_queries.txt --csv ../benchmarkData/AFDB50/benchmark/compare_7tools_stats_100.csv  --tool ssalign_prefilter_2000
[INFO]
  tool               = ssalign_prefilter_2000
  selected_queries   = 100
  found_in_csv       = 100
  missing_in_csv     = 0

[MEAN PER-QUERY (over found queries)]
  mean_total_rows     = 2000.000000
  mean_na_rows        = 147.030000
  mean_tm_non_na_rows = 1852.970000
  mean_avg_tmscore    = 0.705252
  mean_sum_tmscore    = 1305.911834
  mean_rmsd_non_na_rows = 1852.970000
  mean_avg_RMSD       = 2.591949
  mean_sum_RMSD       = 4802.821100

[SUM OVER QUERIES]
  SUM(sum_tmscore) = 130591.183370
  SUM(sum_RMSD)    = 480282.110000

[OVERALL AVERAGE (sum / non_na_rows)]
  overall_avg_tmscore = 0.704767  (SUM(sum_tmscore)/SUM(tm_non_na_rows))
  overall_avg_RMSD    = 2.591958  (SUM(sum_RMSD)/SUM(rmsd_non_na_rows))


---------------------------------------------------------------------------------------------------------------------------------------------------------------------



差集

python test_100_queries.py  --selected random_100_queries.txt --csv ../benchmarkData/AFDB50/benchmark/compare_7tools_stats_100.csv  --tool foldseek_except_ssalign
[INFO]
  tool               = foldseek_except_ssalign
  selected_queries   = 100
  found_in_csv       = 100
  missing_in_csv     = 0

[MEAN PER-QUERY (over found queries)]
  mean_total_rows     = 753.250000
  mean_na_rows        = 59.080000
  mean_tm_non_na_rows = 694.170000
  mean_avg_tmscore    = 0.649752
  mean_sum_tmscore    = 446.531299
  mean_rmsd_non_na_rows = 694.170000
  mean_avg_RMSD       = 2.685277
  mean_sum_RMSD       = 1922.343900

[SUM OVER QUERIES]
  SUM(sum_tmscore) = 44653.129870
  SUM(sum_RMSD)    = 192234.390000

[OVERALL AVERAGE (sum / non_na_rows)]
  overall_avg_tmscore = 0.643259  (SUM(sum_tmscore)/SUM(tm_non_na_rows))
  overall_avg_RMSD    = 2.769270  (SUM(sum_RMSD)/SUM(rmsd_non_na_rows))

 
---------------------------------------------------------------------------------------------------------------------------------------------------------------------

 
  
 python test_100_queries.py  --selected random_100_queries.txt --csv ../benchmarkData/AFDB50/benchmark/compare_7tools_stats_100.csv  --tool ssalign_except_foldseek
[INFO]
  tool               = ssalign_except_foldseek
  selected_queries   = 100
  found_in_csv       = 100
  missing_in_csv     = 0

[MEAN PER-QUERY (over found queries)]
  mean_total_rows     = 592.480000
  mean_na_rows        = 41.560000
  mean_tm_non_na_rows = 550.920000
  mean_avg_tmscore    = 0.759319
  mean_sum_tmscore    = 410.554350
  mean_rmsd_non_na_rows = 550.920000
  mean_avg_RMSD       = 2.522376
  mean_sum_RMSD       = 1454.924300

[SUM OVER QUERIES]
  SUM(sum_tmscore) = 41055.434990
  SUM(sum_RMSD)    = 145492.430000

[OVERALL AVERAGE (sum / non_na_rows)]
  overall_avg_tmscore = 0.745216  (SUM(sum_tmscore)/SUM(tm_non_na_rows))
  overall_avg_RMSD    = 2.640899  (SUM(sum_RMSD)/SUM(rmsd_non_na_rows))
  
  
---------------------------------------------------------------------------------------------------------------------------------------------------------------------


  
python test_100_queries.py  --selected random_100_queries.txt --csv ../benchmarkData/AFDB50/benchmark/compare_7tools_stats_100.csv  --tool foldseek_except_ssalign_prefilter_2000
[INFO]
  tool               = foldseek_except_ssalign_prefilter_2000
  selected_queries   = 100
  found_in_csv       = 100
  missing_in_csv     = 0

[MEAN PER-QUERY (over found queries)]
  mean_total_rows     = 527.380000
  mean_na_rows        = 41.700000
  mean_tm_non_na_rows = 485.680000
  mean_avg_tmscore    = 0.629552
  mean_sum_tmscore    = 295.552779
  mean_rmsd_non_na_rows = 485.680000
  mean_avg_RMSD       = 2.760439
  mean_sum_RMSD       = 1431.702300

[SUM OVER QUERIES]
  SUM(sum_tmscore) = 29555.277950
  SUM(sum_RMSD)    = 143170.230000

[OVERALL AVERAGE (sum / non_na_rows)]
  overall_avg_tmscore = 0.608534  (SUM(sum_tmscore)/SUM(tm_non_na_rows))
  overall_avg_RMSD    = 2.947830  (SUM(sum_RMSD)/SUM(rmsd_non_na_rows))


---------------------------------------------------------------------------------------------------------------------------------------------------------------------


python test_100_queries.py  --selected random_100_queries.txt --csv ../benchmarkData/AFDB50/benchmark/compare_7tools_stats_100.csv  --tool ssalign_prefilter_2000_except_foldseek
[INFO]
  tool               = ssalign_prefilter_2000_except_foldseek
  selected_queries   = 100
  found_in_csv       = 100
  missing_in_csv     = 0

[MEAN PER-QUERY (over found queries)]
  mean_total_rows     = 1366.610000
  mean_na_rows        = 100.660000
  mean_tm_non_na_rows = 1265.950000
  mean_avg_tmscore    = 0.678052
  mean_sum_tmscore    = 850.338591
  mean_rmsd_non_na_rows = 1265.950000
  mean_avg_RMSD       = 2.679169
  mean_sum_RMSD       = 3483.547100

[SUM OVER QUERIES]
  SUM(sum_tmscore) = 85033.859140
  SUM(sum_RMSD)    = 348354.710000

[OVERALL AVERAGE (sum / non_na_rows)]
  overall_avg_tmscore = 0.671700  (SUM(sum_tmscore)/SUM(tm_non_na_rows))
  overall_avg_RMSD    = 2.751726  (SUM(sum_RMSD)/SUM(rmsd_non_na_rows))
"""