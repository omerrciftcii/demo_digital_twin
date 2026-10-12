#!/bin/sh
# Operator template. Replace with the real vLLM launch for a downloaded model.
# Arguments come from the model config (served name, quantization, port).
# This file does not download weights and does not start a server.
echo "start-model template (no weights launched): $*" >&2
exit 0
