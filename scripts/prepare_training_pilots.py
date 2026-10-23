"""Prepare a local 60-item gold queue and controlled synthetic pilot."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from medical_ai_selective_context.synthetic_training import (
    generate_synthetic_pilot,
    prepare_gold_review_queue,
)
from medical_ai_selective_context.training_dataset import validate_training_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workspace",
        required=True,
        type=Path,
        help="Standalone local workspace for synthetic CCOs, graphs and annotations.",
    )
    parser.add_argument("--synthetic-count", type=int, default=480)
    parser.add_argument("--seed", type=int, default=20261021)
    parser.add_argument("--cco-dir", type=Path)
    parser.add_argument("--gold-output", type=Path)
    parser.add_argument("--gold-count", type=int, default=60)
    parser.add_argument("--questions", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    package = generate_synthetic_pilot(
        args.workspace,
        count=args.synthetic_count,
        seed=args.seed,
    )
    validation = validate_training_dataset(
        package.annotations,
        package.manifest,
        package.project_root,
    )
    result = {
        "synthetic": {
            "requested": args.synthetic_count,
            "valid": validation["valid"],
            "examples": validation["examples"],
            "eventGraphs": args.synthetic_count,
            "workspace": str(package.package_root),
            "externalModelCalls": 0,
            "realHealthDataUsed": False,
        },
        "goldReview": None,
        "loraStarted": False,
    }
    if args.cco_dir is not None or args.gold_output is not None:
        if args.cco_dir is None or args.gold_output is None:
            raise SystemExit("--cco-dir and --gold-output must be provided together")
        gold = prepare_gold_review_queue(
            args.cco_dir,
            args.gold_output,
            count=args.gold_count,
            questions_path=args.questions,
        )
        result["goldReview"] = {
            "requested": args.gold_count,
            "tasks": args.gold_count,
            "reviewStatus": "pending",
            "goldLabelsCreated": False,
            "containsSourceCcoText": False,
            "containsReviewQuestions": args.questions is not None,
            "workspace": str(gold.package_root),
        }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
