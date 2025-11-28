#!/usr/bin/env bash
set -euo pipefail

cd /workspace
exec python3 scripts/runpod_worker.py
