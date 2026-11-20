"""Send the reference / normal / shifted batches to a running API; thin wrapper.

Start the API first (`make serve`), then `make simulate-traffic`. Writes the request_ids of each
batch (no text) to `monitoring.batches_path` so the drift report can find their log lines.
"""

import argparse
import json
import urllib.request
from collections.abc import Sequence
from datetime import UTC, datetime
from urllib.parse import urlparse

from supportrouter.config import load_config
from supportrouter.data.load import load_split
from supportrouter.monitoring.traffic import build_batches, send_batches


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000", help="API base URL")
    args = parser.parse_args()
    if urlparse(args.url).scheme not in ("http", "https"):
        raise SystemExit("--url must be http(s)")
    cfg = load_config()

    def post(texts: Sequence[str]) -> str:
        req = urllib.request.Request(
            f"{args.url}/classify",
            data=json.dumps({"texts": list(texts)}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=60) as resp:  # noqa: S310 (scheme checked)
            return str(json.load(resp)["request_id"])

    with urllib.request.urlopen(f"{args.url}/health", timeout=10) as resp:  # noqa: S310
        model_version = json.load(resp)["model_version"]
    batches = build_batches(load_split(cfg, "val"), cfg)
    ids = send_batches(batches, post, cfg.api.max_batch)
    out = {
        "created": datetime.now(UTC).isoformat(),
        "model_version": model_version,
        "batches": {
            name: {"messages": len(batches[name]), "request_ids": ids[name]} for name in batches
        },
    }
    cfg.monitoring.batches_path.parent.mkdir(parents=True, exist_ok=True)
    cfg.monitoring.batches_path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    for name in batches:
        print(f"{name}: {len(batches[name])} messages in {len(ids[name])} requests")
    print(cfg.monitoring.batches_path)


if __name__ == "__main__":
    main()
