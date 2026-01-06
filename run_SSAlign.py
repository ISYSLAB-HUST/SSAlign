#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import subprocess
from pathlib import Path
import argparse


DB_TO_SCRIPT = {
    "afdb50": "./AFDB50/AFDB50_SSAlign.py",
    "scope40": "./SCOPe40/SCOPe40_SSAlign.py",
    "swissprot": "./SwissProt/SwissProt_SSAlign.py",
}


def main():
    parser = argparse.ArgumentParser(
        description="Unified SSAlign runner via --db (pass-through all other args)."
    )
    parser.add_argument(
        "--db",
        required=True,
        choices=sorted(DB_TO_SCRIPT.keys()),
        help="Which database runner to use: afdb50 | scope40 | swissprot",
    )

    # 只解析 --db，其它参数全部透传给子脚本
    args, rest = parser.parse_known_args()

    here = Path(__file__).resolve().parent
    rel = DB_TO_SCRIPT[args.db]
    script_path = (here / rel).resolve()

    if not script_path.exists():
        print(f"[ERROR] script not found: {script_path}", file=sys.stderr)
        sys.exit(2)

    cmd = [sys.executable, str(script_path)] + rest

    print(f"[RUN] db={args.db} -> {script_path}")
    print("[CMD] " + " ".join(cmd))

    r = subprocess.run(cmd)
    sys.exit(r.returncode)


if __name__ == "__main__":
    main()

"""
python run_SSAlign.py \
  --db afdb50 \
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