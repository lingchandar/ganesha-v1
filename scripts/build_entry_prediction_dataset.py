"""Build the point-in-time historical dataset used by entry prediction."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow this script to be executed directly from the repository root.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.prediction.dataset import build_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Ganesha v1 entry prediction dataset")
    parser.add_argument("--output", default="data/prediction/entry_dataset.csv")
    parser.add_argument("--start-date", default=None, help="Inclusive YYYY-MM-DD")
    parser.add_argument("--end-date", default=None, help="Inclusive YYYY-MM-DD")
    args = parser.parse_args()

    path = build_dataset(
        output_path=Path(args.output),
        start_date=args.start_date,
        end_date=args.end_date,
    )
    print(f"saved_dataset={path}")


if __name__ == "__main__":
    main()
