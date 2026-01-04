#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import csv
import time
import random
import argparse
from pathlib import Path
from multiprocessing import Pool

try:
    import requests
except ImportError:
    print("[ERROR] missing requests. Install: pip install requests", file=sys.stderr)
    sys.exit(1)

AFDB_BASE = "https://alphafold.ebi.ac.uk/files"


# -------------------------
# utils
# -------------------------
def norm_line(s: str) -> str:
    """strip + remove CR + ignore empty/comment"""
    if s is None:
        return ""
    s = str(s).strip().replace("\r", "")
    if not s or s.startswith("#"):
        return ""
    return s


def strip_ext(x: str) -> str:
    """remove common extensions"""
    x = norm_line(x)
    for suf in (".pdb", ".cif", ".mmcif", ".pdb.gz", ".cif.gz"):
        if x.endswith(suf):
            x = x[: -len(suf)]
            break
    return x


def to_v6_id(af_id: str) -> str:
    """AF-...-model_v4 -> AF-...-model_v6"""
    af_id = strip_ext(af_id)
    if af_id.endswith("-model_v4"):
        return af_id[:-len("-model_v4")] + "-model_v6"
    return af_id


def unique_keep_order(items):
    seen = set()
    out = []
    for x in items:
        if x and x not in seen:
            seen.add(x)
            out.append(x)
    return out


def file_nonempty(p: Path) -> bool:
    return p.exists() and p.is_file() and p.stat().st_size > 0


# -------------------------
# parse foldseek / ssalign
# -------------------------
def parse_foldseek_targets(foldseek_file: Path):
    """
    foldseek raw output example:
    AF-XXX-model_v4.cif  AF-YYY-model_v4  0.676 ...
    take col2 as target
    """
    targets = []
    if not foldseek_file.exists():
        return targets

    with foldseek_file.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = norm_line(line)
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            t = strip_ext(parts[1])
            if t and t != "NA":
                targets.append(t)
    return targets


def parse_ssalign_targets(ssalign_file: Path):
    """
    ssalign timebenchmark example:
    "File1", "Cosine_Similarity", "Score"
    AF-AAA-model_v4,0.60,
    """
    targets = []
    if not ssalign_file.exists():
        return targets

    with ssalign_file.open("r", encoding="utf-8", errors="ignore", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return targets

        # 兼容：优先 File1
        col = "File1" if "File1" in reader.fieldnames else reader.fieldnames[0]
        for row in reader:
            t = strip_ext(row.get(col, ""))
            if t and t != "NA":
                targets.append(t)
    return targets


# -------------------------
# downloader
# -------------------------
def download_with_retries(url: str, out_path: Path, retries: int = 8, timeout: int = 120):
    tmp_path = out_path.with_suffix(out_path.suffix + ".part")
    sess = requests.Session()
    sess.headers.update({"User-Agent": "afdb-v6-downloader/1.0"})

    for attempt in range(1, retries + 1):
        try:
            with sess.get(url, stream=True, timeout=(10, timeout)) as r:
                if r.status_code == 404:
                    return False, "HTTP 404"
                if r.status_code != 200:
                    raise RuntimeError(f"HTTP {r.status_code}")

                out_path.parent.mkdir(parents=True, exist_ok=True)
                with open(tmp_path, "wb") as w:
                    for chunk in r.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            w.write(chunk)

            if not tmp_path.exists() or tmp_path.stat().st_size == 0:
                raise RuntimeError("downloaded empty file")

            os.replace(tmp_path, out_path)
            return True, f"OK ({out_path.stat().st_size} bytes)"

        except Exception as e:
            try:
                if tmp_path.exists() and tmp_path.stat().st_size == 0:
                    tmp_path.unlink()
            except Exception:
                pass

            if attempt == retries:
                return False, f"FAIL after {retries} tries: {e}"

            sleep_s = (0.8 * (2 ** (attempt - 1))) + random.uniform(0.0, 0.3)
            time.sleep(min(sleep_s, 20.0))

    return False, "FAIL (unexpected)"


def _worker_download(args):
    idx, total, raw_id, out_dir_str = args
    raw_id = strip_ext(raw_id)
    if not raw_id:
        return idx, total, "", "", "SKIP", "empty", ""

    v6_id = to_v6_id(raw_id)
    out_dir = Path(out_dir_str)
    out_path = out_dir / f"{v6_id}.pdb"

    if file_nonempty(out_path):
        return idx, total, raw_id, v6_id, "SKIP", "already exists", str(out_path)

    url = f"{AFDB_BASE}/{v6_id}.pdb"
    ok, msg = download_with_retries(url, out_path, retries=8, timeout=120)
    return idx, total, raw_id, v6_id, ("OK" if ok else "FAIL"), msg, str(out_path)


# -------------------------
# main
# -------------------------
def main():
    ap = argparse.ArgumentParser(
        description="For random_100_queries: merge Foldseek+SSAlign targets and download AFDB v6 pdbs."
    )
    ap.add_argument("--query_list", required=True, help="random_100_queries.txt (one query per line)")
    ap.add_argument("--foldseek_dir", required=True, help="foldseek/timebenchmark dir (contains *.foldseek)")
    ap.add_argument("--ssalign_dir", required=True, help="SSAlign/SVD512/timechmark dir (contains *.ssalign)")
    ap.add_argument("--out_dir", required=True, help="output dir for v6 pdbs")
    ap.add_argument("--list_out", default="benchmark_download_name_100.txt", help="output id list (raw v4 ids)")
    ap.add_argument("--nproc", type=int, default=20, help="download processes")
    ap.add_argument("--log_file", default="download_benchmark_v6_pdb.log", help="download log file")
    args = ap.parse_args()

    query_list = Path(args.query_list)
    foldseek_dir = Path(args.foldseek_dir)
    ssalign_dir = Path(args.ssalign_dir)
    out_dir = Path(args.out_dir)
    list_out = Path(args.list_out)
    log_file = Path(args.log_file)

    if not query_list.exists():
        print(f"[ERROR] query_list not found: {query_list}", file=sys.stderr)
        sys.exit(2)

    queries = []
    for line in query_list.read_text(encoding="utf-8", errors="ignore").splitlines():
        q = strip_ext(line)
        if q:
            queries.append(q)

    queries = unique_keep_order(queries)
    print(f"[INFO] queries={len(queries)}")

    all_targets = []
    missing_fold = 0
    missing_ssa = 0

    for i, q in enumerate(queries, 1):
        f1 = foldseek_dir / f"{q}.foldseek"
        f2 = ssalign_dir / f"{q}.ssalign"

        ft = parse_foldseek_targets(f1)
        st = parse_ssalign_targets(f2)

        if not f1.exists():
            missing_fold += 1
        if not f2.exists():
            missing_ssa += 1

        all_targets.extend(ft)
        all_targets.extend(st)

        if i % 10 == 0:
            print(f"[PARSE] {i}/{len(queries)} total_targets_acc={len(all_targets)}")

    all_targets = unique_keep_order(all_targets)
    print(f"[INFO] union_unique_targets={len(all_targets)} "
          f"(missing_foldseek_files={missing_fold}, missing_ssalign_files={missing_ssa})")

    # filter existed v6
    need = []
    existed = 0
    for raw in all_targets:
        v6 = to_v6_id(raw)
        fp = out_dir / f"{v6}.pdb"
        if file_nonempty(fp):
            existed += 1
        else:
            need.append(raw)

    list_out.parent.mkdir(parents=True, exist_ok=True)
    list_out.write_text("\n".join(need) + ("\n" if need else ""), encoding="utf-8")
    print(f"[DONE] list_out={list_out}")
    print(f"[DONE] existed_v6={existed} need_download={len(need)}")

    if not need:
        print("[DONE] nothing to download.")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("w", encoding="utf-8") as w:
        w.write(f"[START] query_list={query_list} out_dir={out_dir} nproc={args.nproc} need={len(need)}\n")

    def log(msg: str):
        print(msg, flush=True)
        with log_file.open("a", encoding="utf-8") as w:
            w.write(msg + "\n")

    tasks = [(i + 1, len(need), need[i], str(out_dir)) for i in range(len(need))]
    ok = skip = fail = 0

    with Pool(processes=args.nproc) as pool:
        for (idx, total, raw_id, v6_id, status, info, out_path) in pool.imap_unordered(_worker_download, tasks, chunksize=20):
            msg = f"[{idx}/{total}] {status} raw={raw_id} v6={v6_id} :: {info}"
            log(msg)
            if status == "OK":
                ok += 1
            elif status == "SKIP":
                skip += 1
            else:
                fail += 1

    log(f"[DONE] OK={ok} SKIP={skip} FAIL={fail} log={log_file}")
    print(f"[DONE] OK={ok} SKIP={skip} FAIL={fail} log={log_file}")


if __name__ == "__main__":
    main()

"""
python download_target_structfiles.py \
  --query_list  random_100_queries.txt \
  --foldseek_dir ../benchmarkData/AFDB50/foldseek/timebenchmark \
  --ssalign_dir  ../AFDB50/SSAlign/SVD512/timechmark \
  --out_dir      ../pdbData/pdb/AFDB50/ \
  --list_out     benchmark_download_name_100.txt \
  --nproc 20 \
  --log_file     ./logs/download_benchmark_v6_pdb.log


"""