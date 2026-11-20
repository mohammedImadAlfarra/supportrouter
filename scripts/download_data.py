"""Download the pinned Bitext dataset to data/raw/ and print a first look (Phase 1)."""

import argparse
import json

from supportrouter.config import load_config
from supportrouter.data.load import describe_raw, download_raw, file_sha256, read_raw


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="re-download even if the file exists")
    args = parser.parse_args()

    cfg = load_config()
    path = download_raw(cfg.data, force=args.force)
    summary = {
        "path": str(path),
        "revision": cfg.data.hf_revision,
        "sha256": file_sha256(path),
        **describe_raw(read_raw(path)),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
