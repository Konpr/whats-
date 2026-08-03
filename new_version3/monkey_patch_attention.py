"""Monkey-patch CrossAttention.forward for per-region attention control.

Used on vanilla A1111 (no Forge API). Replaces attn2 forward on all
CrossAttention modules in the UNet with a wrapper that injects
per-region conditioning into the COND k/v and blends outputs by masks.

Unlike regional-prompter (4000 lines of xformers/sdp/doggettx rewrites),
we keep the ORIGINAL attention forward and just expand k/v along the
token dimension — the kernel doesn't care about token count.
"""
import logging
import weakref

import torch

from modules.region_utils import _repeat_div, lcm_for_list, downsample_mask, clear_mask_cache, region_curve_factor

logger = logging.getLogger(__name__)

_warned_single_pass_messages = set()


class MonkeyPatchRegionPatcher:
    """Monkey-patches CrossAttention.forward for per-region attention.

    Architecture: region tokens are concatenated to the COND k/v along
    the token dimension (dim=1). UNCOND k/v is padded with zeros to
    match. One forward pass per region, outputs blended by spatial masks.

    Usage:
        patcher = MonkeyPatchRegionPatcher(region_conds, masks,
                                           latent_h, latent_w,
                                           batch_size=1, stop_ratio=1.0)
        patcher.hook(sd_model.model.diffusion_model)
        # ... sampling ...
        patcher.unhook()
    """

    def __init__(
        self,
        region_conds: list[torch.Tensor],
        masks: torch.Tensor,
        latent_h: int,
        latent_w: int,
        batch_size: int = 1,
        stop_ratio: float = 1.0,
        start_ratio: float = 0.0,
        curves: list[str] | None = None,
    ):
        device = masks.device
        dtype = masks.dtype
        self.region_conds = [c.to(device=device, dtype=dtype) for c in region_conds]
        self.masks = masks  # [num_regions, H, W] or [num_regions, 1, H, W], normalized per pixel
        clear_mask_cache()
        self.latent_h = latent_h
        self.latent_w = latent_w
        self.batch_size = batch_size
        self.stop_ratio = stop_ratio
        self.start_ratio = start_ratio
        self._num_conds = len(region_conds)
        self.curves = curves if curves is not None else ["linear"] * self._num_conds
        self._saved: list[tuple[torch.nn.Module, callable]] = []
        self._hooked = False
        self._step = 0
        self._total_steps = 1
        self._callback_registered = False
        self._cfg_callback = None

    def _curve_factors(self) -> torch.Tensor | None:
        """Per-step [num_conds] weight multipliers from per-region curves.

        Returns None when no region uses a curve (all 'linear') — callers
        skip the modulation entirely (zero overhead on the common path).
        """
        if not any(c != "linear" for c in self.curves):
            return None
        factors = torch.tensor(
            [
                region_curve_factor(self._step, self._total_steps, self.start_ratio, self.stop_ratio, c)
                for c in self.curves
            ],
            device=self.masks.device,
            dtype=self.masks.dtype,
        )
        return factors

    def set_hires_dims(self, latent_h: int, latent_w: int):
        self.latent_h = latent_h
        self.latent_w = latent_w
        clear_mask_cache()

    def _is_active(self) -> bool:
        if self.start_ratio <= 0.0 and self.stop_ratio >= 1.0:
            return True
        progress = self._step / max(1, self._total_steps)
        return progress >= self.start_ratio and progress <= self.stop_ratio

    def _denoiser_callback(self, params):
        self._step = params.sampling_step
        self._total_steps = params.total_sampling_steps

    def hook(self, model: torch.nn.Module):
        """Find all CrossAttention attn2 modules, save + replace forward."""
        if self._hooked:
            return
        count = 0
        for _name, _mod in model.named_modules():
            if "attn2" not in _name:
                continue
            if _mod.__class__.__name__ != "CrossAttention":
                continue
            if not hasattr(_mod, "to_q"):
                continue
            orig = _mod.forward
            self._saved.append((_mod, orig))
            _mod.forward = self._build_forward(_mod, orig)
            count += 1
        if count == 0:
            logger.warning("MonkeyPatchRegionPatcher: no CrossAttention attn2 modules found — region control disabled")
        else:
            logger.info("MonkeyPatchRegionPatcher hooked %d CrossAttention modules", count)
        self._hooked = count > 0

    def _build_forward(self, module: torch.nn.Module, original: callable) -> callable:
        """Create wrapper closure for one CrossAttention module.

        Detection of cond vs uncond:
          - Batched mode (batch_cond_uncond=True):
            batch == 2*batch_size → first half=cond, second half=uncond
          - Single-pass mode (batch_cond_uncond=False):
            batch == batch_size — no reliable heuristic, region control
            disabled with logger.warning.
        """

        def _forward(x, context=None, mask=None, **kwargs):
            if not self._is_active():
                return original(x, context=context, mask=mask, **kwargs)

            batch = x.shape[0]

            if batch == 2 * self.batch_size:
                half = self.batch_size
                cond_x = x[:half]
                uncond_x = x[half:]
                cond_ctx = context[:half] if context is not None else None
                uncond_ctx = context[half:] if context is not None else None
            else:
                # Single-pass mode (batch_cond_uncond=False):
                # cond/uncond processed separately, can't distinguish.
                # Injecting regions into uncond corrupts CFG, so skip.
                _msg = "MonkeyPatchRegionPatcher: single-pass mode (batch_cond_uncond=False) — region control disabled, standard forward used"
                if _msg not in _warned_single_pass_messages:
                    _warned_single_pass_messages.add(_msg)
                    logger.warning(_msg)
                return original(x, context=context, mask=mask, **kwargs)

            # --- Cond path: separate forwards, no padding, no mask ---
            # Method C: each branch (base + per-region) runs its own forward
            # at natural token length. No padding, no LCM expansion, no mask.
            # Works identically on ALL 8 A1111 attention backends because
            # there's nothing to mask. N+1 kernel launches vs 1 batched, but
            # no mask-dependent silent degradation on xformers/doggettx/etc.
            if cond_x is not None:
                orig_tokens = cond_ctx.shape[1] if cond_ctx is not None else 0
                if orig_tokens == 0:
                    return original(x, context=context, mask=mask, **kwargs)
                bs = cond_x.shape[0]

                # Base forward: natural length, no padding, no mask
                base_out = original(cond_x, context=cond_ctx, mask=None, **kwargs)

                # Per-region forwards: LC-align base+part WITHIN each branch
                # to give equal softmax weight regardless of token counts.
                # Without this, a 77-token base + 11-token region gives
                # 7:1 attention bias towards base, weakening region control.
                region_outs = []
                for i in range(self._num_conds):
                    part = self.region_conds[i].expand(bs, -1, -1).to(device=x.device, dtype=x.dtype)
                    if cond_ctx is not None:
                        L = lcm_for_list([cond_ctx.shape[1], part.shape[1]])
                        ctx_lcm = cond_ctx.repeat(1, L // cond_ctx.shape[1], 1)
                        part_lcm = part.repeat(1, L // part.shape[1], 1)
                        region_ctx = torch.cat([ctx_lcm, part_lcm], dim=1)
                    else:
                        region_ctx = part
                    r_out = original(cond_x, context=region_ctx, mask=None, **kwargs)
                    region_outs.append(r_out)

                # Blend: base*(1-Σm) + Σ(m_i * region_i)
                num_tokens = cond_x.shape[1]
                mask_down = downsample_mask(
                    self.masks, batch_size=bs, num_tokens=num_tokens,
                    latent_h=self.latent_h, latent_w=self.latent_w,
                )
                if region_outs:
                    stacked = torch.stack(region_outs, dim=0)  # [N, bs, tokens, d]
                    mask_r = mask_down.view(self._num_conds, bs, num_tokens, 1)
                    curve_factors = self._curve_factors()
                    if curve_factors is not None:
                        mask_r = mask_r * curve_factors.view(self._num_conds, 1, 1, 1)
                        stacked = stacked * curve_factors.view(self._num_conds, 1, 1, 1)
                    mask_sum = mask_r.sum(dim=0)
                    cond_out = base_out * (1.0 - mask_sum) + (stacked * mask_r).sum(dim=0)
                else:
                    cond_out = base_out
            else:
                cond_out = None

            # --- Uncond path: standard forward ---
            if uncond_x is not None:
                uncond_out = original(uncond_x, context=uncond_ctx, mask=mask, **kwargs)
            else:
                uncond_out = None

            if cond_out is not None and uncond_out is not None:
                return torch.cat([cond_out, uncond_out], dim=0)
            elif cond_out is not None:
                return cond_out
            else:
                return uncond_out

        return _forward

    def unhook(self):
        """Restore all original forwards and remove CFG denoiser callback."""
        if self._cfg_callback is not None:
            try:
                from modules.script_callbacks import remove_callbacks_for_function
                remove_callbacks_for_function(self._cfg_callback)
            except Exception:
                pass
            self._cfg_callback = None
        for mod, orig in self._saved:
            mod.forward = orig
        self._saved.clear()
        self._hooked = False

    def register_callbacks(self):
        """Register on_cfg_denoiser for step tracking.

        Uses weakref to prevent GC-leak: after unhook()+ dereference, the
        patcher object (with its tensors) becomes collectable immediately.
        The stale callback entry in script_callbacks persists as a tiny
        closure that checks `inst._hooked` and does nothing when unhooked.
        """
        if self._callback_registered:
            return
        try:
            from modules.script_callbacks import on_cfg_denoiser
            weak_self = weakref.ref(self)
            def _callback(params):
                inst = weak_self()
                if inst is not None and inst._hooked:
                    inst._step = params.sampling_step
                    inst._total_steps = params.total_sampling_steps
            on_cfg_denoiser(_callback)
            self._cfg_callback = _callback
            self._callback_registered = True
        except ImportError:
            logger.warning("Cannot register CFG denoiser callback")
