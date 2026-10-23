"""Validate CCO V2 training annotations without making model/API calls."""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from medical_ai_selective_context.training_dataset import (
    DatasetValidationError,
    TrainingExample,
    validate_training_dataset,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=PROJECT_ROOT,
        help=(
            "Root whose data/ directory contains ccoFile paths. "
            "Defaults to the repository root."
        ),
    )
    parser.add_argument(
        "--print-schema",
        action="store_true",
        help="Print the annotation JSON Schema before validating the dataset.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.print_schema:
        print(json.dumps(TrainingExample.model_json_schema(by_alias=True), ensure_ascii=False, indent=2))
    try:
        report = validate_training_dataset(
            args.dataset.resolve(), args.manifest.resolve(), args.project_root.resolve()
        )
    except DatasetValidationError as exc:
        print(json.dumps({"valid": False, "error": str(exc)}, ensure_ascii=False, indent=2))
        return 1
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

