"""Import helpers for the pinned, unmodified V-JEPA 2.1 checkout."""

from __future__ import annotations

import sys
from pathlib import Path


def upstream_root() -> Path:
    return Path(__file__).resolve().parents[2] / "third_party" / "vjepa2"


def activate_upstream() -> Path:
    root = upstream_root()
    if not (root / "app" / "vjepa_2_1" / "models" / "vision_transformer.py").is_file():
        raise FileNotFoundError(
            f"Missing official V-JEPA 2.1 checkout at {root}. "
            "Clone https://github.com/facebookresearch/vjepa2.git there first."
        )
    root_text = str(root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    return root

