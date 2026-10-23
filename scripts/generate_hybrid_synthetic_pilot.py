"""Generate a 20-root/80-variant hybrid synthetic pilot without starting training."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from medical_ai_selective_context.hybrid_synthetic import (
    build_fixture_drafts,
    generate_hybrid_pilot,
    request_openai_drafts,
    request_openrouter_drafts,
)


def _load_local_env(path: Path) -> None:
    """Load simple KEY=VALUE entries without printing or persisting secrets."""
    if not path.is_file():
        return
    for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key or key in os.environ:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        os.environ[key] = value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", required=True, type=Path)
    parser.add_argument(
        "--provider", choices=("openai", "openrouter", "fixture"), default="openai"
    )
    parser.add_argument("--model", default="")
    parser.add_argument("--root-count", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20261021)
    parser.add_argument("--package-name", default="hybrid-pilot")
    parser.add_argument(
        "--base-url",
        default="",
    )
    return parser.parse_args()


def main() -> int:
    _load_local_env(PROJECT_ROOT / ".env")
    args = parse_args()
    if args.root_count != 20:
        raise SystemExit("The first hybrid pilot must use exactly 20 roots")
    api_calls = 0
    request_hash = None
    if args.provider in {"openai", "openrouter"}:
        if args.provider == "openai":
            model = args.model or os.getenv("OPENAI_SYNTHETIC_MODEL", "")
            base_url = args.base_url or os.getenv(
                "OPENAI_BASE_URL", "https://api.openai.com/v1"
            )
            api_key = os.getenv("OPENAI_API_KEY", "")
            requester = request_openai_drafts
        else:
            model = args.model or os.getenv("OPENROUTER_SYNTHETIC_MODEL", "")
            base_url = args.base_url or os.getenv(
                "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
            )
            api_key = os.getenv("OPENROUTER_API_KEY", "")
            requester = request_openrouter_drafts
        if not model.strip():
            raise SystemExit(f"--model or {args.provider.upper()}_SYNTHETIC_MODEL is required")
        batch, request_hash = requester(
            api_key=api_key,
            model=model,
            root_count=args.root_count,
            seed=args.seed,
            base_url=base_url,
        )
        api_calls = 1
    else:
        batch = build_fixture_drafts(args.root_count)

    package = generate_hybrid_pilot(
        args.workspace,
        batch,
        seed=args.seed,
        package_name=args.package_name,
        draft_provider=args.provider,
        external_api_calls=api_calls,
        request_sha256=request_hash,
    )
    print(package.report.read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
