#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="$PWD/src"

streamlit run src/perp_intel/dashboard/app.py