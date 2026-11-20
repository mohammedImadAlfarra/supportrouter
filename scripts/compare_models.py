"""Build `reports/model_comparison.md`, or `--errors` / `--robustness` reports; thin wrapper."""

import argparse

from supportrouter.config import load_config
from supportrouter.evaluate import write_comparison, write_error_analysis, write_robustness


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--errors", action="store_true", help="error analysis instead")
    parser.add_argument("--robustness", action="store_true", help="robustness report instead")
    args = parser.parse_args()
    cfg = load_config()
    if args.robustness:
        print(write_robustness(cfg))
    else:
        print(write_error_analysis(cfg) if args.errors else write_comparison(cfg))


if __name__ == "__main__":
    main()
