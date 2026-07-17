"""
sd_webui_apg — Adaptive Projected Guidance (APG) package.

Re-exports the public surface of the self-contained algorithm module.
"""

from .core import (
    MARKER,
    MomentumBuffer,
    apply_apg,
    remove_apg_patches,
)

__all__ = [
    "MARKER",
    "MomentumBuffer",
    "apply_apg",
    "remove_apg_patches",
]
