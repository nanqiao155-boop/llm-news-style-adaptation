from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset.sft_v2_complete_build import build, validate


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/dataset/task011d_complete_sft_v2_build.json"),
    )
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    config = args.config if args.config.is_absolute() else args.root / args.config
    result = validate(args.root, config) if args.validate_only else build(args.root, config)
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
