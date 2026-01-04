#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Extract unique 2nd-column entries from all *.foldseek files in a directory.
Output: download_name.txt (one entry per line).
"""

import argparse
from pathlib import Path


def iter_second_column(fpath: Path):
    with fpath.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()  # split by any whitespace
            if len(parts) >= 2:
                yield parts[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--in_dir",
        default="/SSAlign/afdb50/test_pdb/foldseek_result",
        help="Directory containing *.foldseek files",
    )
    ap.add_argument(
        "--out",
        default=None,
        help="Output file path. Default: <in_dir>/download_name.txt",
    )
    ap.add_argument(
        "--sort",
        action="store_true",
        help="Sort output names (default keeps arbitrary set order after de-dup).",
    )
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    if not in_dir.is_dir():
        raise SystemExit(f"[ERROR] Not a directory: {in_dir}")

    foldseek_files = sorted(in_dir.glob("*.foldseek"))
    if not foldseek_files:
        raise SystemExit(f"[ERROR] No *.foldseek files found in: {in_dir}")

    names = set()
    for fp in foldseek_files:
        for name in iter_second_column(fp):
            names.add(name)

    out_path = Path(args.out) if args.out else (in_dir / "download_name.txt")

    out_list = sorted(names) if args.sort else list(names)
    with out_path.open("w", encoding="utf-8") as w:
        for n in out_list:
            w.write(n + "\n")

    print(f"[OK] Parsed files: {len(foldseek_files)}")
    print(f"[OK] Unique names: {len(names)}")
    print(f"[OK] Written: {out_path}")


if __name__ == "__main__":
    main()
