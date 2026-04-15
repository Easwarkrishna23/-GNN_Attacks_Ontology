"""
Download Planetoid raw dataset files (Cora/Citeseer/PubMed) into data/<Name>/raw.

Why this exists in this repo:
- torch_geometric may fail to import on some environments due to binary extension mismatches.
- Our loader has a raw Planetoid fallback that works without torch_geometric,
  but it requires the raw files to exist on disk.

This script downloads the classic Planetoid raw files from the original public source:
  https://github.com/kimiyoung/planetoid/tree/master/data

Usage:
  python3 datasets/download_planetoid_raw.py --dataset Cora
  python3 datasets/download_planetoid_raw.py --dataset all
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from urllib.request import urlopen, Request


PLANETOID_BASE = "https://raw.githubusercontent.com/kimiyoung/planetoid/master/data"


def _download(url: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(req, timeout=60) as r:
        data = r.read()
    out_path.write_bytes(data)


def download_dataset(name: str, root: str = "data") -> None:
    name = str(name)
    name_l = name.lower()
    files = ["x", "y", "tx", "ty", "allx", "ally", "graph", "test.index"]
    out_dir = Path(root) / name / "raw"
    for f in files:
        url = f"{PLANETOID_BASE}/ind.{name_l}.{f}"
        out_path = out_dir / f"ind.{name_l}.{f}"
        if out_path.exists():
            continue
        print(f"Downloading {url} -> {out_path}")
        _download(url, out_path)
    print(f"Done: {name} raw files in {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="all", choices=["Cora", "Citeseer", "PubMed", "all"])
    parser.add_argument("--root", default="data")
    args = parser.parse_args()

    if args.dataset == "all":
        for d in ["Cora", "Citeseer", "PubMed"]:
            download_dataset(d, root=args.root)
    else:
        download_dataset(args.dataset, root=args.root)


if __name__ == "__main__":
    main()

