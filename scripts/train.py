"""Train, compare and evaluate salary models; save artifacts.

Usage (from the repository root):
    python scripts/train.py                       # synthetic demo data
    python scripts/train.py --csv path/to/jobs.csv
    python -m scripts.train --output artifacts
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:  # allow `python scripts/train.py`
    sys.path.insert(0, str(ROOT))

from jobscope.data import load_csv, load_demo  # noqa: E402
from jobscope.modeling import save_artifacts, train_and_compare  # noqa: E402
from jobscope.validation import is_synthetic, validate  # noqa: E402


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", type=Path, help="CSV using the documented schema (default: bundled synthetic demo)")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts", help="Output directory for model + metrics")
    args = parser.parse_args(argv)

    raw = load_csv(args.csv) if args.csv else load_demo()
    df, quality = validate(raw, return_report=True)
    data_type = "synthetic_demo" if is_synthetic(df) else "user_supplied"
    print(f"Validated {quality.output_rows} of {quality.input_rows} rows ({data_type}).")

    result = train_and_compare(df)
    save_artifacts(result, args.output, data_type)

    print("\nCross-validation on training split (5-fold):")
    print(result.cv_results.round(3).to_string(index=False))
    print("\nHoldout (20%) comparison:")
    print(result.holdout_results.round(3).to_string(index=False))
    summary = {**result.summary(), "data_type": data_type}
    print("\nSelected model summary:")
    print(json.dumps(summary, indent=2))
    print(f"\nSaved artifacts to {args.output}")
    return summary


if __name__ == "__main__":
    main()
