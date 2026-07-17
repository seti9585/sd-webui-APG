"""
sd-webui-APG — Adaptive Projected Guidance for Forge-derived WebUIs
====================================================================
Location: extensions/sd-webui-APG/scripts/sd_webui_apg.py

Hook:  Pre-CFG (reForge / Forge Classic) / Post-CFG (Forge Neo)
Paper: "Eliminating Oversaturation and Artifacts of High Guidance Scales in
        Diffusion Models" (arXiv:2410.02416, ICLR 2025)

sorting_priority: 14.5
    TCFG (13.0) -> SkimmedCFG (14.0) -> DifferenceCFG (14.2) -> APG (14.5)
    -> CFG -> CFGZeroStar (15.0) -> MaHiRo (15.5)

APG decomposes the guidance vector (cond - uncond) into components parallel
and orthogonal to the conditional prediction. The parallel component is the
main source of oversaturation at high CFG; eta scales how much of it is kept.
An optional per-sample L2 clamp (norm_threshold) and an optional running
average (momentum) complete the paper's Algorithm 1. See
sd_webui_apg/core.py for the formulation details and how this port differs
from the ComfyUI built-in node.

Defaults follow the paper: eta = 0.0 (recommended default), norm_threshold =
15.0 (Table 10, SDXL row), momentum = 0.0 (Algorithm 1 default; the momentum
buffer carries state across model evaluations and therefore interacts with
multi-stage / adaptive ODE samplers -- see core.py).

Neutral settings that reproduce standard CFG exactly (fixed-seed A/B
baseline): eta = 1.0, norm_threshold = 0.0, momentum = 0.0.

Inspiration:
    The author first learned of APG through note.com articles by
    Shiba-2-shiba, whose APGForge implementation for Forge Classic was also
    consulted. This port is written from the paper above; the pointer that
    made it knowable is gratefully acknowledged.
"""

import logging
import os
import sys
import traceback
from functools import partial
from typing import Any

import gradio as gr
from modules import scripts, script_callbacks

# ---------------------------------------------------------------------------
# sys.path — ensure the extension root is importable
# ---------------------------------------------------------------------------
_EXT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _EXT_ROOT not in sys.path:
    sys.path.insert(0, _EXT_ROOT)
# ---------------------------------------------------------------------------

from sd_webui_apg import apply_apg, remove_apg_patches

logger = logging.getLogger(__name__)


def _has_forge_backend(p) -> bool:
    return hasattr(p, "sd_model") and hasattr(p.sd_model, "forge_objects")


def _build_infotext_params(cfg: dict) -> dict:
    """Build the infotext key/value dict.

    Keys use the "APG" prefix. "APG Eta" is written only when the extension is
    active, so its presence doubles as the enable marker on the read side.
    """
    return {
        "APG Eta":            cfg["eta"],
        "APG Norm Threshold": cfg["norm_threshold"],
        "APG Momentum":       cfg["momentum"],
    }


# ---------------------------------------------------------------------------
# Script
# ---------------------------------------------------------------------------

class APGScript(scripts.Script):

    sorting_priority = 14.5

    def __init__(self):
        self.enabled = False

    def title(self) -> str:
        return "APG (Adaptive Projected Guidance)"

    def show(self, is_img2img: bool):
        return scripts.AlwaysVisible

    def ui(self, is_img2img: bool):
        with gr.Accordion(open=False, label=self.title()):
            gr.HTML(
                "<p><i>"
                "<b>Pre-CFG</b>: Splits the guidance vector into components "
                "parallel and orthogonal to the conditional prediction and "
                "down-weights the parallel one (the main source of "
                "oversaturation at high CFG), with optional norm rescale and "
                "momentum. Paper formulation (arXiv:2410.02416)."
                "</i></p>"
            )
            enabled = gr.Checkbox(label="Enable APG", value=False)
            eta = gr.Slider(
                0.0, 2.0, value=0.0, step=0.05,
                label="Eta (parallel component; 0 = paper default, 1 = keep fully)",
            )
            norm_threshold = gr.Slider(
                0.0, 50.0, value=15.0, step=0.5,
                label="Norm Threshold (guidance L2 clamp; 0 = off)",
            )
            momentum = gr.Slider(
                -1.5, 1.0, value=0.0, step=0.05,
                label="Momentum (beta; 0 = off, paper uses negative values)",
            )
            gr.HTML(
                "<p style='color:gray;font-size:0.9em;'>"
                "Neutral settings (identical to standard CFG): Eta 1.0, "
                "Norm Threshold 0, Momentum 0. Momentum accumulates across "
                "model evaluations, so its effect depends on the sampler "
                "(multi-stage / adaptive solvers update it more often per "
                "step); it is OFF by default."
                "</p>"
            )

        for slider in (eta, norm_threshold, momentum):
            slider.do_not_save_to_config = True

        # Infotext round-trip (PNG Info -> Send to txt2img / img2img).
        # Metadata is written in process(). "APG Eta" is written only when
        # active, so its presence means ON, absence OFF; Enable therefore
        # binds to a callable that forces OFF when the key is absent. The
        # other keys use plain strings (absent keys leave the component
        # untouched).
        self.infotext_fields = [
            (enabled,        lambda d: "APG Eta" in d),
            (eta,            "APG Eta"),
            (norm_threshold, "APG Norm Threshold"),
            (momentum,       "APG Momentum"),
        ]

        return [enabled, eta, norm_threshold, momentum]

    # ------------------------------------------------------------------
    # Effective configuration (UI args + XYZ Grid override)
    # ------------------------------------------------------------------

    def _resolve(self, p, args):
        if len(args) < 4:
            return None
        (enabled, eta, norm_threshold, momentum) = args[:4]

        xyz = getattr(p, "_apg_xyz", {})
        if "enabled" in xyz:
            enabled = (xyz["enabled"] == "True")
        if "eta" in xyz:
            eta = xyz["eta"]
        if "norm_threshold" in xyz:
            norm_threshold = xyz["norm_threshold"]
        if "momentum" in xyz:
            momentum = xyz["momentum"]

        return {
            "enabled":        bool(enabled),
            "eta":            float(eta),
            "norm_threshold": float(norm_threshold),
            "momentum":       float(momentum),
        }

    # ------------------------------------------------------------------
    # Metadata write (runs once before sampling so create_infotext captures it)
    # ------------------------------------------------------------------

    def process(self, p, *args):
        cfg = self._resolve(p, args)
        if cfg is None or not cfg["enabled"]:
            return
        p.extra_generation_params.update(_build_infotext_params(cfg))

    # ------------------------------------------------------------------
    # Hook application (correct timing for forge_objects.unet)
    # ------------------------------------------------------------------

    def process_before_every_sampling(self, p, *args, **kwargs):
        cfg = self._resolve(p, args)
        if cfg is None:
            logger.warning("[APG] process_before_every_sampling: missing args")
            return

        self.enabled = cfg["enabled"]

        if not cfg["enabled"]:
            return

        if not _has_forge_backend(p):
            logger.warning("[APG] Requires Forge backend.")
            return

        unet = p.sd_model.forge_objects.unet.clone()

        apply_apg(
            unet,
            eta=cfg["eta"],
            norm_threshold=cfg["norm_threshold"],
            momentum=cfg["momentum"],
        )

        p.sd_model.forge_objects.unet = unet
        logger.debug(
            "[APG] applied: eta=%s norm_threshold=%s momentum=%s",
            cfg["eta"], cfg["norm_threshold"], cfg["momentum"],
        )


# ---------------------------------------------------------------------------
# XYZ Grid
# ---------------------------------------------------------------------------

def _set_xyz(p, x: Any, xs: Any, *, field: str) -> None:
    if not hasattr(p, "_apg_xyz"):
        p._apg_xyz = {}
    p._apg_xyz[field] = x


def _register_xyz() -> None:
    xyz_grid = None
    for script in scripts.scripts_data:
        if script.script_class.__module__ == "xyz_grid.py":
            xyz_grid = script.module
            break
    if xyz_grid is None:
        return

    new_axes = [
        xyz_grid.AxisOption(
            "(APG) Enabled",
            str,
            partial(_set_xyz, field="enabled"),
            choices=lambda: ["True", "False"],
        ),
        xyz_grid.AxisOption(
            "(APG) Eta",
            float,
            partial(_set_xyz, field="eta"),
        ),
        xyz_grid.AxisOption(
            "(APG) Norm Threshold",
            float,
            partial(_set_xyz, field="norm_threshold"),
        ),
        xyz_grid.AxisOption(
            "(APG) Momentum",
            float,
            partial(_set_xyz, field="momentum"),
        ),
    ]

    if not any(x.label.startswith("(APG)") for x in xyz_grid.axis_options):
        xyz_grid.axis_options.extend(new_axes)


def _on_before_ui() -> None:
    try:
        _register_xyz()
    except Exception:
        print(f"[sd-webui-APG] XYZ Grid error:\n{traceback.format_exc()}")


script_callbacks.on_before_ui(_on_before_ui)
