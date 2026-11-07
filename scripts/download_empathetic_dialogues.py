#!/usr/bin/env python3
"""Download and prepare the EmpatheticDialogues dataset.

Usage:
    python scripts/download_empathetic_dialogues.py
    python scripts/download_empathetic_dialogues.py --output-dir slm-ft/data/ed_docraw

Fetches the public tarball with the standard library only (no `datasets`
dependency) and extracts train.csv, valid.csv, and test.csv into the output
directory, which is the layout `slm-ft/prepare_data.py --ed-csv-dir` expects.
Existing files are kept, so re-running is cheap.
"""

from __future__ import annotations

import argparse
import sys
import tarfile
import urllib.request
from pathlib import Path

DEFAULT_URL = (
    "https://dl.fbaipublicfiles.com/parlai/empatheticdialogues/"
    "empatheticdialogues.tar.gz"
)
CSV_NAMES = ("train.csv", "valid.csv", "test.csv")


def main() -> int:
    ap = argparse.ArgumentParser(description="Download EmpatheticDialogues dataset")
    ap.add_argument(
        "--output-dir",
        default="slm-ft/data/ed_docraw",
        help="Directory to extract train/valid/test.csv into",
    )
    ap.add_argument("--url", default=DEFAULT_URL, help="Tarball URL to fetch")
    ap.add_argument(
        "--force", action="store_true", help="Re-download even if the CSVs exist"
    )
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    missing = [name for name in CSV_NAMES if not (out / name).exists()]
    if not missing:
        print(f"EmpatheticDialogues CSVs already present in {out}; nothing to do.")
        return 0
    if len(missing) != len(CSV_NAMES):
        print(f"Partial dataset in {out}; missing {', '.join(missing)}.")

    archive = out / "empatheticdialogues.tar.gz"
    print(f"Downloading {args.url} -> {archive} ...")
    try:
        urllib.request.urlretrieve(args.url, archive)
    except Exception as exc:
        print(f"Download failed: {exc}")
        return 1

    print(f"Extracting {', '.join(missing)} ...")
    try:
        with tarfile.open(archive, "r:gz") as tar:
            members = {
                Path(m.name).name: m
                for m in tar.getmembers()
                if Path(m.name).name in CSV_NAMES and m.isfile()
            }
            for name in missing:
                member = members.get(name)
                if member is None:
                    print(f"{name} not found in archive; skipping")
                    continue
                source = tar.extractfile(member)
                if source is None:
                    print(f"Could not read {name} from archive; skipping")
                    continue
                (out / name).write_bytes(source.read())
                print(f"  {name}: {(out / name).stat().st_size} bytes")
    except Exception as exc:
        print(f"Extraction failed: {exc}")
        return 1

    still_missing = [name for name in CSV_NAMES if not (out / name).exists()]
    if still_missing:
        print(f"Incomplete dataset, missing: {', '.join(still_missing)}")
        return 1

    archive.unlink(missing_ok=True)
    print(f"Done. Next: python slm-ft/prepare_data.py --ed-csv-dir {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
