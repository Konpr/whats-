"""
Shim that adapts a Forge-style attn2 patcher (methods _attn2_patch(x,context,value,extra_options)
and _attn2_output_patch(out,extra_options), like ForgeRegionPatcher) to run via direct
monkeypatch of CrossAttention.forward on vanilla A1111.

Key insight verified against real Forge source (backend/nn/unet.py):
  attn2_patch operates on (x, context, value) BEFORE the to_q/to_k/to_v linear
  projections, not after. The ordinary (non-"replace") path then calls
  self.attn2(patched_x, context=patched_context, value=patched_value) as a
  black box -- exactly equivalent to calling A1111's original(x, context, mask)
  as a black box. This means NO mask is needed and NO backend-specific code
  path is needed: whatever attention optimization A1111 has selected
  (xformers/sdp/doggettx/etc.) is irrelevant, since we never look inside it.

Constraint: A1111's CrossAttention.forward has no separate `value` parameter
(only `context`, shared for both K and V). This shim therefore requires the
patcher's _attn2_patch to always return k==v (true for ForgeRegionPatcher's
current design: `return qs, ks, ks`). If that ever changes, this shim needs
a rewrite to compute attention manually.
"""
import logging
import torch

logger = logging.getLogger(__name__)

_warned_single_pass = set()


class ForgeAttn2ShimForA1111:
    COND = 0
    UNCOND = 1

    def __init__(self, forge_patcher, latent_h: int, latent_w: int, batch_size: int = 1):
        self.forge_patcher = forge_patcher
        self.latent_h = latent_h
        self.latent_w = latent_w
        self.batch_size = batch_size
        self._saved: list[tuple[torch.nn.Module, callable]] = []
        self._hooked = False

    def set_hires_dims(self, latent_h: int, latent_w: int):
        self.latent_h = latent_h
        self.latent_w = latent_w

    def hook(self, model: torch.nn.Module):
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
            logger.warning("ForgeAttn2ShimForA1111: no CrossAttention attn2 modules found — region control disabled")
        else:
            logger.info("ForgeAttn2ShimForA1111 hooked %d CrossAttention modules", count)
        self._hooked = count > 0

    def _build_forward(self, module, original):
        def _forward(x, context=None, mask=None, **kwargs):
            if not self.forge_patcher._is_active():
                return original(x, context=context, mask=mask, **kwargs)

            batch = x.shape[0]
            if batch != 2 * self.batch_size:
                # Single-pass mode (batch_cond_uncond=False): same bail-out
                # rationale as monkey_patch_attention.py -- cannot safely
                # distinguish cond/uncond from shape alone.
                _msg = "ForgeAttn2ShimForA1111: single-pass mode (batch_cond_uncond=False) — region control disabled, standard forward used"
                if _msg not in _warned_single_pass:
                    _warned_single_pass.add(_msg)
                    logger.warning(_msg)
                return original(x, context=context, mask=mask, **kwargs)

            # ONE entry per logical chunk (cond-group, uncond-group), NOT one
            # per image -- confirmed against real Forge sampling_function.py.
            # Order matches A1111's own batch layout: cond half first.
            cond_or_uncond = [self.COND, self.UNCOND]
            extra_options = {
                "cond_or_uncond": cond_or_uncond,
                "original_shape": [batch, 0, self.latent_h, self.latent_w],
            }

            n2, ctx2, val2 = self.forge_patcher._attn2_patch(x, context, context, extra_options)
            # ForgeRegionPatcher always returns k==v (return qs, ks, ks) --
            # A1111 has no separate value param, so ctx2 alone is sufficient.
            raw_out = original(n2, context=ctx2, mask=None, **kwargs)
            out = self.forge_patcher._attn2_output_patch(raw_out, extra_options)
            return out
        return _forward

    def unhook(self):
        for mod, orig in self._saved:
            mod.forward = orig
        self._saved.clear()
        self._hooked = False
