#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

python -c 'import sys; assert sys.version_info >= (3, 11), "V-JEPA 2 requires Python >= 3.11"'

python -m pip install -e "${PROJECT_ROOT}/third_party/vjepa2"
python -m pip install -e "${PROJECT_ROOT}"
