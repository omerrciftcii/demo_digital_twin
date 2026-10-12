"""Run the model gateway. Default bind is 127.0.0.1:8095.

Public addresses are refused unless GATEWAY_ALLOW_PUBLIC_BIND is true.
This process does not start the QA API on 8091.
"""

from __future__ import annotations

import os

import uvicorn

from medical_ai_selective_context.model_gateway import app, resolve_bind_host

_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})


def main() -> None:
    allowed = os.environ.get("GATEWAY_ALLOW_PUBLIC_BIND", "").strip().lower() in _TRUE_VALUES
    host = resolve_bind_host(os.environ.get("GATEWAY_HOST", "127.0.0.1"), allowed)
    port = int(os.environ.get("GATEWAY_PORT", "8095"))
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
