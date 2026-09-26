"""
core.py - Adaptive Projected Guidance (APG) algorithm
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
                           Table 10 uses beta=-0.5 for SDXL. Kept OFF by
                           default so the transform stays stateless.

Momentum and ODE samplers:
    The momentum buffer carries state across model EVALUATIONS, not solver
    steps, so the guidance is no longer a function of (x, sigma) alone.
    Fixed-seed measurements on reForge (SDXL, 35 steps, Align Your Steps,
    CFG 7, Eta 1.0, Norm Threshold 15) showed:

      * With kutta4, beta = -0.1 ... -0.5 moved the result by a steadily
        growing amount (mean absolute RGB difference 2.3 -> 5.6 against
        beta = 0). The value maps to the result monotonically.
      * The same beta acts more strongly on a single-stage solver: at
        beta = -0.15, euler changed by about 1.8 times as much as kutta4.
        Re-tune beta when the sampler changes.
      * beta = -1.0 does not converge (past values never decay) and the
        image collapses into a flat, grey result. Keep |beta| below 1.

    An earlier release documented momentum as unpredictable with
    multi-stage solvers. That measurement was taken with the since-removed
    Adaptive Momentum option and does not apply to plain momentum.
    Adaptive-step solvers re-evaluate rejected steps, so the effective
    strength also depends on their tolerance settings.

Momentum reset:
    A fresh closure (and thus a fresh MomentumBuffer) is created per sampling
    pass by the script layer, so the running average never carries across
    passes. No sigma-based reset guard is used: it would misfire on
    adaptive-step solvers, which legitimately re-try a rejected step at a
    larger sigma, and the per-pass closure already covers the case it was
    meant to catch.

Hooking (v3.0: post-CFG on every backend):
    APG is registered in sampler_post_cfg_function on reForge / Forge
    Classic AND on Forge Neo, and works on the chained prediction
    args["denoised"] instead of the raw cond/uncond pair:

        G        = denoised - cond          (guidance the chain produced;
                                             already carries (s - 1))
        diff_eff = G / (s - 1)              (s = cond_scale, s > 1 only)
        update   = APG(diff_eff)            (momentum -> clamp -> projection)
        out      = cond + (s - 1) * update

    With no earlier post-CFG hook, denoised = uncond + s * (cond - uncond),
    so diff_eff = cond - uncond and the result is algebraically identical
    to the paper formula (and to the pre-CFG write-back used up to v2.x). On
    reForge, pre-CFG extensions (TCFG / SkimmedCFG / DifferenceCFG) are
    already folded into denoised and into cond_denoised, so they are
    preserved as before. The same formula now runs on both backends.

Composition with the SETI suite (v3.0):
    _PRIORITY = 15.4 places APG after FreSca and before MaHiRo:
    CFGZeroStar (15.0) -> FreSca (15.2) -> APG (15.4) -> MaHiRo (15.5)
    -> CFGNorm (16.0) -> CFGRegulator (16.5)
    Rationale: FreSca's frequency rescaling can create components parallel
    to cond, which APG should see; MaHiRo is the final guidance decision
    layer and must run after APG (APG after MaHiRo would erase MaHiRo's
    positive leap entirely).

    Behaviour change versus v2.x on reForge: MaHiRo, FreSca and
    CFGZeroStar read args["uncond_denoised"]. v2.x rewrote that slot to the
    synthetic uncond (cond - update) in pre-CFG; v3.0 leaves it as the
    TCFG / SkimmedCFG / DifferenceCFG adjusted uncond. Images generated
    with the same seed and settings therefore differ from v2.x. Fixed-seed
    comparison (SDXL, CFG 15, TCFG + APG Eta 0.5 + MaHiRo) showed the same
    saturation level in both versions, while FreSca's settings now carry
    through to the result as intended (with v2.x, lowering FreSca's
    high-frequency scale had almost no effect once MaHiRo was on).

    On Forge Neo, APG now also receives the SkimmedCFG / DifferenceCFG
    result through args["denoised"] instead of overwriting it.

    Zero-init: when an earlier hook (CFGZeroStar zero-init) returns an
    all-zero prediction, APG passes it through unchanged.
"""

import logging
import os
import sys

import torch

logger = logging.getLogger(__name__)

MARKER = "sd_webui_apg_v1"

# Mirrors APGScript.sorting_priority in scripts/sd_webui_apg.py. Kept in sync
# manually; used to order this extension's hook within the
# sampler_post_cfg_function list relative to other SETI extensions.
_PRIORITY = 15.4

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

    The final combination with cond_scale is done by the caller.
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
# Post-CFG chain observability
# ---------------------------------------------------------------------------

def _maybe_dump_chain(args) -> None:
    """Emit the post-CFG chain once per pass, from inside the hook, so what
    is printed is the list as the sampler actually holds it at call time."""
    global _CHAIN_DUMPED
    if _CHAIN_DUMPED or _debug_level() < 1:
        return
    _CHAIN_DUMPED = True
    try:
        opts = args.get("model_options") or {}
        _emit(1, "post-CFG chain: %s",
              _describe_chain(opts.get("sampler_post_cfg_function")))
    except Exception as exc:
        _emit(1, "post-CFG chain dump failed: %r", exc)


# ---------------------------------------------------------------------------
# Post-CFG factory (all backends)
# ---------------------------------------------------------------------------
# Post-CFG args dict keys used here (same on reForge and Forge Neo):
#   "denoised"        - prediction produced by the chain so far (anchor)
#   "cond_denoised"   - positive prediction (on reForge: after pre-CFG hooks)
#   "uncond_denoised" - only checked for presence (None / zero = no CFG)
#   "cond_scale"      - CFG scale
# ---------------------------------------------------------------------------

def _make_apg_post_fn(
    eta: float,
    norm_threshold: float,
    momentum: float,
):
    """APG - Post-CFG, chain-respecting (v3.0).

    Recovers the guidance vector from the chained prediction, applies the
    paper's APG update to it and rebuilds the output around cond. Every
    early exit returns args["denoised"] unchanged so earlier hooks survive.
    """
    buffer = MomentumBuffer(momentum)

    @torch.no_grad()
    def _fn(args):
        denoised = args["denoised"]
        try:
            _maybe_dump_chain(args)

            uncond_denoised = args.get("uncond_denoised")
            if uncond_denoised is None or not torch.any(uncond_denoised):
                # CFG == 1 optimization / uncond skipped: nothing to reshape.
                return denoised

            cond = args.get("cond_denoised")
            if cond is None or denoised is None:
                return denoised

            s = float(args["cond_scale"])
            if s <= 1.0:
                # (s - 1) is the divisor below; no guidance to reshape.
                return denoised

            if not torch.any(denoised):
                # An earlier hook zeroed the prediction on purpose
                # (CFGZeroStar zero-init). Keep it zero.
                return denoised

            # Guidance actually produced by the chain, divided back to the
            # paper's (cond - uncond) scale so norm_threshold keeps its
            # meaning. Done in fp32: the operands are close in magnitude.
            orig_dtype = denoised.dtype
            cond_f = cond.float()
            diff_eff = (denoised.float() - cond_f) / (s - 1.0)

            # _apg_update computes cond - uncond internally, so pass an
            # uncond that reproduces diff_eff exactly.
            update = _apg_update(cond_f, cond_f - diff_eff, buffer,
                                 eta, norm_threshold)
            return (cond_f + (s - 1.0) * update).to(orig_dtype)
        except Exception:
            logger.exception("[APG] post-CFG function failed; passing through")
            return denoised

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
    """Register APG on unet as a post-CFG hook (all backends, v3.0).

    Inserted by priority (15.4): after CFGZeroStar / FreSca, before MaHiRo /
    CFGNorm / CFGRegulator. Any APG hook left in the pre-CFG list by an
    older version is removed first.

    A fresh closure (and momentum buffer) is created on every call, so
    invoking this from process_before_every_sampling() resets momentum for
    each sampling pass (txt2img and hires.fix get independent state).

    Parameters:
      eta            : parallel-component scale (paper recommends 0.0;
                       1.0 keeps the parallel component fully = projection
                       off)
      norm_threshold : per-sample L2 clamp on the guidance vector
                       (0 disables; paper Table 10 uses 15.0 for SDXL)
      momentum       : running-average coefficient (0 disables). Not
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

    _priority_insert_post_cfg(
        unet,
        _make_apg_post_fn(eta, norm_threshold, momentum),
    )
    _emit(1, "registered post-CFG hook (%s backend), priority=%s",
          "Forge Neo" if _is_forge_neo_backend() else "reForge / Forge Classic",
          _PRIORITY)

    return unet
