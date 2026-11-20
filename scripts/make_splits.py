"""Build the frozen train/val/test split and its manifest (Phase 3)."""

import argparse
import json

from supportrouter.config import load_config
from supportrouter.data.split import make_splits


def main() -> None:
    cfg = load_config()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sample",
        type=int,
        nargs="?",
        const=cfg.sample_size,
        default=None,
        help=f"stratified N-row subset to data/processed/sample/ (default N={cfg.sample_size})",
    )
    args = parser.parse_args()

    manifest, written = make_splits(cfg, sample=args.sample)
    summary = {
        "written": written,
        "mode": manifest["mode"],
        "rows": {name: part["rows"] for name, part in manifest["splits"].items()},
        "sha256": {name: part["sha256"] for name, part in manifest["splits"].items()},
        "near_duplicates": manifest["leakage"]["near_duplicates"],
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
