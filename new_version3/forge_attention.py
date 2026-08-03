import logging
import torch

from modules.region_utils import _repeat_div, lcm_for_list, downsample_mask, clear_mask_cache, region_curve_factor

logger = logging.getLogger(__name__)


class ForgeRegionPatcher:
    """Forge API attention patcher for REGION blocks.

    Injects per-region K/V into cross-attention and blends output by spatial masks.
    One UNet forward per step, separation per-token.

    Usage:
        patcher = ForgeRegionPatcher(region_conds, masks)
        patched_unet = patcher.apply(model)      # model.clone() + patches
        # ... sampling with patched_unet ...
        # Forge auto-cleans on model re-creation
    """

    def __init__(
        self,
        region_conds: list[torch.Tensor],
        masks: torch.Tensor,
        batch_size: int = 1,
        stop_ratio: float = 1.0,
        start_ratio: float = 0.0,
        curves: list[str] | None = None,
    ):
        self.region_conds = [c.to(device=masks.device, dtype=masks.dtype) for c in region_conds]
        self.num_tokens = [c.shape[1] for c in self.region_conds]
        self.masks = masks  # [num_regions, H, W] or [num_regions, 1, H, W], normalized per pixel
        clear_mask_cache()
        self.batch_size = batch_size
        self._num_conds = len(region_conds)
        self.stop_ratio = stop_ratio
        self.start_ratio = start_ratio
        self.curves = curves if curves is not None else ["linear"] * self._num_conds

    def _curve_factors(self) -> torch.Tensor | None:
        """Per-step [num_conds] weight multipliers from per-region curves.

        Returns None when no region uses a curve (all 'linear') — callers
        skip the modulation entirely (zero overhead on the common path).
        """
        if not any(c != "linear" for c in self.curves):
            return None
        from modules import shared
        step = shared.state.sampling_step
        total = max(1, shared.state.sampling_steps)
        factors = torch.tensor(
            [
                region_curve_factor(step, total, self.start_ratio, self.stop_ratio, c)
                for c in self.curves
            ],
            device=self.masks.device,
            dtype=self.masks.dtype,
        )
        return factors

    def _is_active(self) -> bool:
        """Check if region attention should be active at current step."""
        if self.start_ratio <= 0.0 and self.stop_ratio >= 1.0:
            return True
        from modules import shared
        current = shared.state.sampling_step
        total = max(1, shared.state.sampling_steps)
        progress = current / total
        return progress >= self.start_ratio and progress <= self.stop_ratio

    @torch.inference_mode()
    def _attn2_patch(self, q, k, v, extra_options):
        if not self._is_active():
            return q, k, v
        if self._num_conds == 0:
            return q, k, v
        if not torch.allclose(k, v, atol=1e-4):
            logger.warning("k and v differ in cross-attention — region injection may degrade")
            # Some fine-tuned models have small numerical diffs; fallback gracefully.

        cond_or_unconds = extra_options["cond_or_uncond"]
        num_chunks = len(cond_or_unconds)
        actual_batch = q.shape[0] // num_chunks

        q_chunks = q.chunk(num_chunks, dim=0)
        k_chunks = k.chunk(num_chunks, dim=0)

        lcm_tokens = lcm_for_list(self.num_tokens + [k.shape[1]])
        # conds_tensor: [num_conds * actual_batch, lcm_tokens, dim]
        conds_tensor = torch.cat(
            [
                c.repeat(actual_batch, lcm_tokens // self.num_tokens[i], 1)
                for i, c in enumerate(self.region_conds)
            ],
            dim=0,
        )

        qs, ks = [], []
        for i, cui in enumerate(cond_or_unconds):
            k_target = k_chunks[i].repeat(1, lcm_tokens // k.shape[1], 1)
            if cui == 1:  # FIX: Forge COND=0, UNCOND=1 (was inverted: "== 0")
                # Uncond: standard forward (no regions).
                # Self-repeat pad: padding with copies of k_target leverages
                # softmax invariance to repeated identical K/V rows —
                # attention over [K, K] produces identical output as over
                # [K] alone (verified: diff=1.6e-6, float32 noise).
                # This avoids zero-pad's ~38% magnitude loss and K_neg's
                # ~50% output collapse, without needing a mask channel.
                k_padded = torch.cat([k_target, k_target], dim=1)
                qs.append(q_chunks[i])
                ks.append(k_padded)
            else:
                # Cond: base-only copy + per-region copies.
                k_target_rep = k_target.repeat(self._num_conds, 1, 1)
                k_regions = torch.cat([k_target_rep, conds_tensor], dim=1)  # [N*B, 2*lcm, D]
                # Base copy: self-repeat pad (same rationale as uncond)
                k_base_padded = torch.cat([k_target, k_target], dim=1)  # [B, 2*lcm, D]
                q_base = q_chunks[i]  # [B, seq, D]
                q_regions = q_chunks[i].repeat(self._num_conds, 1, 1)  # [N*B, seq, D]
                qs.append(torch.cat([q_base, q_regions], dim=0))
                ks.append(torch.cat([k_base_padded, k_regions], dim=0))

        qs = torch.cat(qs, dim=0).to(q)
        ks = torch.cat(ks, dim=0).to(k)

        # NOTE: self-repeat pad (not zeros) to match the rest of this file.
        # Zero-padded rows leak ~2% attention weight (softmax over exp(0)
        # vs typical exp(2.77) for real token dot-products). Self-repeat
        # is softmax-invariant (identical output as without padding).
        if qs.shape[0] % 2 == 1:
            q_pad = qs[-1:].expand(1, -1, -1)
            k_pad = ks[-1:].expand(1, -1, -1)
            qs = torch.cat([qs, q_pad], dim=0)
            ks = torch.cat([ks, k_pad], dim=0)

        return qs, ks, ks

    @torch.inference_mode()
    def _attn2_output_patch(self, out, extra_options):
        if not self._is_active():
            return out
        if self._num_conds == 0:
            return out
        cond_or_unconds = extra_options["cond_or_uncond"]
        mask_down = downsample_mask(
            self.masks, self.batch_size, out.shape[1], shape=extra_options["original_shape"]
        )
        curve_factors = self._curve_factors()
        outputs = []
        pos = 0
        for cui in cond_or_unconds:
            if cui == 1:  # FIX: Forge COND=0, UNCOND=1 (was inverted: "== 0")
                # Uncond: standard forward (no regions, single output)
                outputs.append(out[pos: pos + self.batch_size])
                pos += self.batch_size
            else:
                # Cond: base + per-region blend: base*(1-Σm) + Σ(m_i * region_i)
                base_out = out[pos: pos + self.batch_size]
                pos += self.batch_size
                region_out = out[pos: pos + self._num_conds * self.batch_size]
                masked = (
                    region_out * mask_down
                ).view(self._num_conds, self.batch_size, out.shape[1], out.shape[2])
                mask_r = mask_down.view(self._num_conds, self.batch_size, out.shape[1], 1)
                if curve_factors is not None:
                    # Scale per-region weight by easing curve at current step.
                    mask_r = mask_r * curve_factors.view(self._num_conds, 1, 1, 1)
                    masked = masked * curve_factors.view(self._num_conds, 1, 1, 1)
                mask_sum = mask_r.sum(dim=0)
                outputs.append(base_out * (1.0 - mask_sum) + masked.sum(dim=0))
                pos += self._num_conds * self.batch_size
        return torch.cat(outputs, dim=0)

    def apply(self, model):
        """Clone model and register attn2 patches. Return patched model.

        Args:
            model: Forge model object with set_model_attn2_patch / set_model_attn2_output_patch
        Returns:
            Patched model clone. Original model untouched.
        """
        m = model.clone()
        m.set_model_attn2_patch(self._attn2_patch)
        m.set_model_attn2_output_patch(self._attn2_output_patch)
        return m


def detect_region_backend(sd_model) -> str:
    """Auto-detect best REGION backend handler.

    Quality-first chain:
      forge -> monkey -> latent

    Detection:
      1. launch_utils.git_tag() starts with "f2" or is "neo" → Forge
      2. sd_model.forge_objects exists + set_model_attn2_patch → Forge
      3. CrossAttention attn2 modules found → Monkey (vanilla A1111)
      4. Otherwise → Latent (always works)

    Returns:
      'forge'  — Forge API available (per-token K/V injection, best quality)
      'monkey' — Monkey-patch CrossAttention.forward (vanilla A1111, good quality)
      'latent' — CFG callback blending (always works, moderate quality)
    """
    is_forge = False
    try:
        from modules import launch_utils
        tag = launch_utils.git_tag()
        if tag is not None and (str(tag).startswith("f2") or str(tag) == "neo"):
            is_forge = True
    except Exception:
        pass

    forge_objs = getattr(sd_model, 'forge_objects', None)
    if forge_objs is not None:
        is_forge = True
        unet = getattr(forge_objs, 'unet', None)
        if unet is not None and hasattr(unet, 'set_model_attn2_patch'):
            logger.debug("REGION auto-detect: forge — Forge API available")
            return 'forge'

    if is_forge:
        logger.warning(
            "git-tag suggests Forge but forge_objects/set_model_attn2_patch API unavailable — "
            "falling back to monkey/latent backend"
        )
        # Don't return 'forge' here — let it fall through to monkey/latent detection

    try:
        _mod = sd_model.model.diffusion_model
        for _name, _m in _mod.named_modules():
            if "attn2" in _name and _m.__class__.__name__ == "CrossAttention":
                logger.debug("REGION auto-detect: monkey — CrossAttention modules found")
                return 'monkey'
    except Exception:
        pass
    logger.debug("REGION auto-detect: latent — no CrossAttention/Forge API")
    return 'latent'
