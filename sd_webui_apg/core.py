"""
core.py — Adaptive Projected Guidance (APG) algorithm
======================================================
Location: extensions/sd-webui-APG/sd_webui_apg/core.py

Based on:
    "Eliminating Oversaturation and Artifacts of High Guidance Scales in
     Diffusion Models" (arXiv:2410.02416, ICLR 2025), Algorithm 1.

This port follows the PAPER formulation, not the ComfyUI built-in node:

    diff   = cond - uncond                       (denoised / x0 space)
    diff   = momentum_buffer.update(diff)        (optional, beta != 0)
    diff   = clamp L2 norm to norm_threshold     (optional, threshold > 0)
    par, orth = project diff onto cond
    update = orth + eta * par
    final  = cond + (cond_scale - 1) * update

Differences from the ComfyUI built-in node (comfy_extras/nodes_apg.py),
kept deliberately:
  * Final combination uses (cond_scale - 1), matching paper Algorithm 1.
    The ComfyUI node effectively yields cond + cond_scale * update, which
    is one guidance unit stronger and never reduces to standard CFG.
    With the paper form, the NEUTRAL settings eta = 1.0, norm_threshold = 0,
    momentum = 0 reduce to standard CFG algebraically. This is an algebraic
    identity, not a bitwise one: _project() decomposes the guidance vector
    in double precision and casts the parts back, so the round trip
    perturbs the low-order bits even though parallel + orthogonal sums to
    the original diff. The CPU numerical suite confirms the identity to a
    tolerance of 0.000001, but on a real sampling run the perturbation is
    amplified by the solver. Measured on SDXL with kutta4, Align Your
    Steps, 35 steps, CFG 7 and a fixed seed, the neutral setting differs
    from a bare run by a mean absolute RGB difference of about 0.74.
    Do NOT use the neutral setting as a bitwise A/B baseline; disable the
    extension instead. The ComfyUI node does not reduce to standard CFG at
    all, algebraically or otherwise.
  * Projection / norm reduction runs over ALL non-batch dimensions
    (range(1, ndim)) instead of the fixed dim=[-1, -2, -3]. For 4-D SDXL
    latents (B, C, H, W) this is identical to the paper's Algorithm 1.
    For 5-D Anima / NextDiT latents (B, C, T, H, W) it keeps the paper's
    per-sample semantics instead of silently becoming per-channel.
    HuggingFace diffusers' AdaptiveProjectedGuidance makes the same choice
    (norm_dim=None -> all non-batch dims).

Defaults (paper-derived):
    eta            = 0.0   Paper Appendix C.8: "We recommend setting eta=0 by
                           default and only increasing it if more saturation
                           is desired."
    norm_threshold = 15.0  Paper Table 10, the Stable Diffusion XL row (r=15).
    momentum       = 0.0   Paper Algorithm 1 signature default (buffer=None).
                           Table 10 uses beta=-0.5 for SDXL. See the momentum
                           section below for why this port defaults it OFF.

Momentum and ODE samplers:
    The momentum buffer carries state across model EVALUATIONS, which breaks
    the stateless right-hand-side assumption of ODE samplers. Multi-stage
    solvers (fe_kutta4 = 4 evaluations per step) and adaptive step control
    (a rejected step is re-evaluated) change how fast the buffer accumulates,
    so the same beta behaves differently per sampler and per rtol/atol
    setting.

    Fixed-seed A/B measurements (SDXL, 35 steps, Align Your Steps, CFG 7,
    beta = -0.15) showed that this is not merely a scaling difference:

      * With a single-stage solver (euler), varying an early-decay parameter
        produced changes of roughly half the magnitude seen with a 4-stage
        solver (kutta4) under identical settings.
      * With kutta4, a parameter change of 0.01 moved the result about as far
        as the result was from the baseline in the first place -- i.e. the
        output jumped to an unrelated solution rather than changing by a
        proportional amount.

    Interpretation: once momentum is enabled the integrand is no longer a
    function of (x, sigma) alone, so a high-order solver's intermediate
    stages no longer improve the estimate; they amplify small early
    perturbations instead. The higher the order, the stronger the
    amplification.

    Practical consequence: momentum is not recommended together with
    high-order solvers (TDE Sampler / RK Sampler multi-stage methods,
    Heun, DPM++ 2M/3M and similar). Momentum defaults to OFF, which keeps
    APG a pure stateless per-evaluation transform; the slider remains
    available for single-stage samplers.

Momentum reset:
    A fresh closure (and thus a fresh MomentumBuffer) is created per sampling
    pass by the script layer, so the running average never carries across
    passes. No sigma-based reset guard is used: it would misfire on
    adaptive-step solvers, which legitimately re-try a rejected step at a
    larger sigma, and the per-pass closure already covers the case it was
    meant to catch.

Backend-adaptive hooking (same pattern as sd-webui-DifferenceCFG / TCFG):
    * reForge / Forge Classic -> Pre-CFG (dict args, "conds_out" style).
      Write-back trick: overwriting the uncond slot with (cond - update)
      makes the subsequent standard CFG step
          out = uncond' + cond_scale * (cond - uncond')
              = cond + (cond_scale - 1) * update
      reproduce the paper formula for ANY cond_scale (no scale dependence in
      the write-back itself).
    * Forge Neo               -> Post-CFG (dict args, "denoised" style);
      the final prediction is recomputed directly as
          cond + (cond_scale - 1) * update.

Composition with the SETI suite:
    sorting_priority 14.5 places APG last in the pre-CFG chain --
    TCFG (13.0) -> SkimmedCFG (14.0) -> DifferenceCFG (14.2) -> APG (14.5)
    -> CFG -> CFGZeroStar (15.0) -> MaHiRo (15.5) -- matching the
    "final polish before CFG" role recommended for APG.
    On Forge Neo, TCFG's damped uncond is read from
    model_options["_tcfg_damped_uncond"] when present (same known limitation
    as DifferenceCFG: Forge Neo hands every post-CFG hook the raw cond/uncond, so
    modifications by other post-CFG extensions that do not stash their state
    are not visible here).
"""

import logging
import os
import sys

import torch

logger = logging.getLogger(__name__)

MARKER = "sd_webui_apg_v1"

# Mirrors APGScript.sorting_priority in scripts/sd_webui_apg.py. Kept in sync
# manually; used only to order this extension's hook within Forge Neo's
# sampler_post_cfg_function list relative to other SETI extensions.
_PRIORITY = 14.5

# Suite-wide debug convention: 0 = off, 1 = apply-time settings + chain dump.
DEBUG_ENV_VAR = "SD_WEBUI_SETI_DEBUG"

# One chain dump per sampling pass. Reset by apply_apg().
_CHAIN_DUMPED = False


def _debug_level():
    try:
        return int(os.environ.get(DEBUG_ENV_VAR, "0"))
    except Exception:
        return 0


def _emit(level, fmt, *args):
    """Emit to both logging and stderr; some forks suppress module loggers."""
    if _debug_level() < level:
        return
    try:
        msg = (fmt % args) if args else fmt
    except Exception:
        msg = str(fmt)
    text = "[APG] " + msg
    logger.warning(text)
    try:
        print(text, file=sys.stderr, flush=True)
    except Exception:
        pass


def _describe_chain(fns):
    """Render a hook list as 'name(priority)' in actual execution order."""
    parts = []
    for fn in fns or []:
        name = getattr(fn, "__name__", None) or type(fn).__name__
        prio = getattr(fn, "_sd_webui_priority", None)
        parts.append("%s(%s)" % (name, "-" if prio is None else prio))
    return " -> ".join(parts) if parts else "(empty)"


# ---------------------------------------------------------------------------
# Backend detection (duplicated; identical logic to sd-webui-DifferenceCFG)
# ---------------------------------------------------------------------------

_BACKEND_IS_NEO = None  # cached


def _is_forge_neo_backend() -> bool:
    """Return True if the active backend is Forge Neo.

    Forge Neo's sampler_pre_cfg_function is called BEFORE model evaluation,
    so denoised predictions are not available there. On reForge / Forge
    Classic the pre-CFG hook receives a single dict whose "conds_out" already
    holds the predictions.
    """
    global _BACKEND_IS_NEO
    if _BACKEND_IS_NEO is not None:
        return _BACKEND_IS_NEO

    is_neo = False
    try:
        from backend.sampling import sampling_function as _sf
        is_neo = (
            hasattr(_sf, "sampling_function_inner")
            and hasattr(_sf, "calc_cond_uncond_batch")
        )
    except Exception:
        is_neo = False

    _BACKEND_IS_NEO = is_neo
    logger.debug(
        "[APG] backend detected: %s",
        "Forge Neo" if is_neo else "reForge / Forge Classic",
    )
    return is_neo


# ---------------------------------------------------------------------------
# Priority-ordered insertion for Forge Neo's post-cfg list (duplicated)
# ---------------------------------------------------------------------------

def _priority_insert_post_cfg(unet, fn) -> None:
    """Insert fn into unet.model_options["sampler_post_cfg_function"] at the
    position that keeps SETI-suite hooks (those carrying a _sd_webui_priority
    attribute) in ascending priority order. Third-party hooks without that
    attribute are left exactly where they already are; only the new fn's
    position relative to them is decided (inserted before the first tracked
    hook with a strictly greater priority, otherwise appended at the end).
    """
    key = "sampler_post_cfg_function"
    existing = unet.model_options.get(key, [])
    priority = fn._sd_webui_priority

    insert_at = len(existing)
    for i, other in enumerate(existing):
        other_priority = getattr(other, "_sd_webui_priority", None)
        if other_priority is not None and other_priority > priority:
            insert_at = i
            break

    unet.model_options[key] = existing[:insert_at] + [fn] + existing[insert_at:]


# ---------------------------------------------------------------------------
# Priority-ordered insertion for the reForge / Forge Classic pre-cfg list
# ---------------------------------------------------------------------------

def _priority_insert_pre_cfg(unet, fn, disable_cfg1_optimization: bool = False) -> None:
    """
    Twin of _priority_insert_post_cfg for the pre-CFG list. Identical
    semantics, different key.

    Replaces the plain append that set_model_sampler_pre_cfg_function
    performs. That append made execution order depend on extension load
    order rather than on _sd_webui_priority, so the documented chain
    TCFG (13.0) -> SkimmedCFG (14.0) -> DifferenceCFG (14.2) -> APG (14.5)
    was not actually enforced on reForge. Forge Neo was already correct
    because that path used _priority_insert_post_cfg.

    disable_cfg1_optimization mirrors the flag that
    set_model_sampler_pre_cfg_function sets, so callers relying on it keep
    working.

    A new list is built rather than mutating in place, matching the backend
    helper's semantics, so a cloned unet never leaks the change into its
    source. Duplicated deliberately: each extension carries its own copy so
    no cross-extension import dependency exists.
    """
    key = "sampler_pre_cfg_function"
    existing = unet.model_options.get(key, [])
    priority = fn._sd_webui_priority

    insert_at = len(existing)
    for i, other in enumerate(existing):
        other_priority = getattr(other, "_sd_webui_priority", None)
        if other_priority is not None and other_priority > priority:
            insert_at = i
            break

    unet.model_options[key] = existing[:insert_at] + [fn] + existing[insert_at:]

    if disable_cfg1_optimization:
        unet.model_options["disable_cfg1_optimization"] = True


def _stashed_tcfg_uncond(args: dict):
    """Return TCFG's damped uncond from model_options if TCFG ran earlier in
    this same post-cfg call, else None."""
    model_options = args.get("model_options")
    if not isinstance(model_options, dict):
        return None
    return model_options.get("_tcfg_damped_uncond")


# ---------------------------------------------------------------------------
# APG core math (paper Algorithm 1)
# ---------------------------------------------------------------------------

class MomentumBuffer:
    """Running average of the guidance vector across model evaluations.

        running_average <- update + momentum * running_average

    Verbatim port of the paper's MomentumBuffer. A negative coefficient
    subtracts a fraction of the previously accumulated guidance, damping
    abrupt evaluation-to-evaluation changes.
    """

    __slots__ = ("momentum", "running_average")

    def __init__(self, momentum: float):
        self.momentum = momentum
        self.running_average = 0.0        # scalar 0 broadcasts on first update

    def update(self, value: torch.Tensor) -> torch.Tensor:
        self.running_average = value + self.momentum * self.running_average
        return self.running_average


def _reduce_dims(t: torch.Tensor):
    """All non-batch dimensions of t.

    Identical to the paper's dim=[-1, -2, -3] for 4-D (B, C, H, W) latents;
    extends the same per-sample semantics to 5-D (B, C, T, H, W) Anima /
    NextDiT latents.
    """
    return list(range(1, t.ndim))


def _project(v0: torch.Tensor, v1: torch.Tensor):
    """Decompose v0 into components parallel and orthogonal to v1.

    Verbatim port of the paper's project() except for the rank-agnostic
    reduction dims. Computation is done in double precision as in the paper,
    then cast back to v0's original dtype.
    """
    dims = _reduce_dims(v0)
    dtype = v0.dtype
    v0d = v0.double()
    v1d = v1.double()
    v1n = torch.nn.functional.normalize(v1d, dim=dims)
    parallel = (v0d * v1n).sum(dim=dims, keepdim=True) * v1n
    orthogonal = v0d - parallel
    return parallel.to(dtype), orthogonal.to(dtype)


def _apg_update(
    cond: torch.Tensor,
    uncond: torch.Tensor,
    buffer: MomentumBuffer,
    eta: float,
    norm_threshold: float,
) -> torch.Tensor:
    """Compute the APG-reshaped guidance update vector.

    Order matches paper Algorithm 1: momentum -> rescale -> projection.
    The momentum stage is skipped entirely when the coefficient is 0 so that
    the default configuration stays a stateless per-evaluation transform.

    The final combination with cond_scale is done by the caller (it differs
    between the pre-CFG write-back and the post-CFG direct recomputation).
    """
    diff = cond - uncond

    # 1. Momentum (running average across model evaluations). Skipped when
    #    the coefficient is 0 so the default configuration stays stateless.
    if buffer.momentum != 0.0:
        diff = buffer.update(diff)

    # 2. Rescale: clamp the per-sample L2 norm to the threshold (0 disables).
    #    No epsilon is added, matching the paper: a zero norm makes the ratio
    #    infinite and torch.minimum then selects 1.0, leaving diff unchanged.
    if norm_threshold > 0.0:
        diff_norm = diff.norm(p=2, dim=_reduce_dims(diff), keepdim=True)
        ones = torch.ones_like(diff_norm)
        scale = torch.minimum(ones, norm_threshold / diff_norm)
        diff = diff * scale

    # 3. Projection onto the cond prediction.
    parallel, orthogonal = _project(diff, cond)
    return orthogonal + eta * parallel


# ---------------------------------------------------------------------------
# Pre-CFG chain observability
# ---------------------------------------------------------------------------

def _maybe_dump_chain(args) -> None:
    """Emit the pre-CFG chain once per pass, from inside the hook, so what is
    printed is the list as the sampler actually holds it at call time. The
    suite's post-CFG dump (sd-webui-FreSca) reads sampler_post_cfg_function
    and cannot see this list."""
    global _CHAIN_DUMPED
    if _CHAIN_DUMPED or _debug_level() < 1:
        return
    _CHAIN_DUMPED = True
    try:
        opts = args.get("model_options") or {}
        _emit(1, "pre-CFG chain: %s",
              _describe_chain(opts.get("sampler_pre_cfg_function")))
    except Exception as exc:
        _emit(1, "pre-CFG chain dump failed: %r", exc)


# ---------------------------------------------------------------------------
# Pre-CFG factory (reForge / Forge Classic)
# ---------------------------------------------------------------------------
# reForge pre-CFG args dict keys used here:
#   "conds_out"  — [cond, uncond] denoised predictions (uncond may be absent
#                  or all-zero when CFG == 1)
#   "cond_scale" — CFG scale (not needed by the write-back; it is
#                  scale-independent)
# ---------------------------------------------------------------------------

def _make_apg_pre_fn(
    eta: float,
    norm_threshold: float,
    momentum: float,
):
    """APG — Pre-CFG (reForge / Forge Classic).

    Write-back trick: overwriting conds_out[1] with (cond - update) makes the
    backend's standard CFG combination produce the paper output
    cond + (cond_scale - 1) * update for any cond_scale.
    """
    buffer = MomentumBuffer(momentum)

    @torch.no_grad()
    def _fn(args):
        conds_out = args["conds_out"]
        try:
            if conds_out is None or len(conds_out) < 2:
                return conds_out
            if conds_out[1] is None or not torch.any(conds_out[1]):
                # CFG == 1 optimization: no usable uncond; nothing to do.
                return conds_out

            cond = conds_out[0]
            uncond = conds_out[1]

            _maybe_dump_chain(args)

            update = _apg_update(cond, uncond, buffer, eta, norm_threshold)
            conds_out[1] = cond - update
            return conds_out
        except Exception:
            logger.exception("[APG] pre-CFG function failed; passing through")
            return conds_out

    # Factory-made closures are all called "_fn"; give them a stable name so
    # the chain dump is readable.
    _fn.__name__ = "_apg_pre_cfg_fn"
    _fn._sd_webui_apg_marker = MARKER
    # Ordering tag read by _priority_insert_pre_cfg. Previously only the
    # Forge Neo post-CFG factory carried this, so the reForge pre-CFG hook
    # was invisible to priority-based insertion.
    _fn._sd_webui_priority = _PRIORITY
    return _fn


# ---------------------------------------------------------------------------
# Post-CFG factory (Forge Neo)
# ---------------------------------------------------------------------------
# Forge Neo post-CFG args dict keys used here:
#   "denoised"        — current CFG result (returned unchanged on early exit)
#   "cond_denoised"   — positive prediction
#   "uncond_denoised" — negative prediction (None when CFG == 1 / uncond off)
#   "cond_scale"      — CFG scale
#   "model_options"   — shared dict; read for TCFG's stashed damped uncond
# ---------------------------------------------------------------------------

def _make_apg_post_fn(
    eta: float,
    norm_threshold: float,
    momentum: float,
):
    """APG — Post-CFG (Forge Neo).

    Recomputes the final prediction directly with the paper formula
    cond + (cond_scale - 1) * update. Early returns hand back
    args["denoised"] unchanged.
    """
    buffer = MomentumBuffer(momentum)

    @torch.no_grad()
    def _fn(args):
        try:
            uncond_denoised = args.get("uncond_denoised")
            if uncond_denoised is None or not torch.any(uncond_denoised):
                return args["denoised"]

            cond_scale = args["cond_scale"]

            cond = args["cond_denoised"]
            tcfg_uncond = _stashed_tcfg_uncond(args)
            uncond = tcfg_uncond if tcfg_uncond is not None else uncond_denoised

            update = _apg_update(cond, uncond, buffer, eta, norm_threshold)
            return cond + (cond_scale - 1.0) * update
        except Exception:
            logger.exception("[APG] post-CFG function failed; passing through")
            return args["denoised"]

    _fn.__name__ = "_apg_post_cfg_fn"
    _fn._sd_webui_apg_marker = MARKER
    _fn._sd_webui_priority = _PRIORITY
    return _fn


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def _is_apg_fn(fn) -> bool:
    return getattr(fn, "_sd_webui_apg_marker", None) == MARKER


def remove_apg_patches(unet) -> None:
    """Remove all APG patches from both pre- and post-CFG lists.

    Only this extension's own hooks (identified by MARKER) are removed, so
    other extensions' pre/post-CFG functions are left untouched.
    """
    for key in ("sampler_pre_cfg_function", "sampler_post_cfg_function"):
        existing = unet.model_options.get(key)
        if isinstance(existing, list):
            unet.model_options[key] = [fn for fn in existing if not _is_apg_fn(fn)]


def apply_apg(
    unet,
    eta: float,
    norm_threshold: float,
    momentum: float,
):
    """Register APG on unet, choosing the correct hook for the backend.

      * Forge Neo               -> Post-CFG, priority-ordered so it runs after
                                    TCFG / SkimmedCFG / DifferenceCFG and
                                    before CFGZeroStar / MaHiRo.
      * reForge / Forge Classic -> Pre-CFG.

    A fresh closure (and momentum buffer) is created on every call, so
    invoking this from process_before_every_sampling() resets momentum for
    each sampling pass (txt2img and hires.fix get independent state).

    Parameters:
      eta            : parallel-component scale (paper recommends 0.0;
                       1.0 keeps the parallel component fully = projection
                       off)
      norm_threshold : per-sample L2 clamp on the guidance vector
                       (0 disables; paper Table 10 uses 15.0 for SDXL)
      momentum       : running-average coefficient (0 disables; the paper's
                       experiments use negative values such as -0.5). Not
                       recommended with high-order solvers -- see the module
                       docstring.
    """
    global _CHAIN_DUMPED
    _CHAIN_DUMPED = False   # one chain dump per sampling pass

    remove_apg_patches(unet)

    logger.info(
        "[APG] eta: %s / norm threshold: %s / momentum: %s",
        eta, norm_threshold, momentum,
    )

    if _is_forge_neo_backend():
        _priority_insert_post_cfg(
            unet,
            _make_apg_post_fn(eta, norm_threshold, momentum),
        )
        logger.debug("[APG] registered post-CFG hook (Forge Neo backend)")
    else:
        # v1.1: priority-ordered insertion replaces the plain append that
        # set_model_sampler_pre_cfg_function performs. See
        # _priority_insert_pre_cfg for why.
        _priority_insert_pre_cfg(
            unet,
            _make_apg_pre_fn(eta, norm_threshold, momentum),
        )
        _emit(1, "registered pre-CFG hook (reForge / Forge Classic), "
                 "priority=%s", _PRIORITY)

    return unet
