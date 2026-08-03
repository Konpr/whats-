import math
import logging
import torch
from torch.nn.functional import interpolate

logger = logging.getLogger(__name__)

_mask_cache: dict[tuple, torch.Tensor] = {}
_mask_cache_keys: set[int] = set()


def clear_mask_cache():
    _mask_cache.clear()
    _mask_cache_keys.clear()


def lcm_for_list(numbers: list[int]) -> int:
    current = numbers[0]
    for n in numbers[1:]:
        current = current * n // math.gcd(current, n)
    return current


def region_curve_factor(step: int, total_steps: int, start_ratio: float, stop_ratio: float, curve: str) -> float:
    """Per-step multiplier for a region's weight curve over its active window.

    Progress = (step/total - start_ratio) / (stop_ratio - start_ratio),
    clamped to [0,1], then mapped through the easing curve.
    curve == "linear" (or unknown) → 1.0 (constant weight).
    """
    if not curve or curve == "linear":
        return 1.0
    total = max(1, int(total_steps or 1))
    window = stop_ratio - start_ratio
    if window <= 0:
        return 1.0
    t = (step / total - start_ratio) / window
    t = max(0.0, min(1.0, t))
    try:
        from modules.prompt_parser import _apply_easing
        return max(0.0, _apply_easing(t, curve))
    except Exception:
        return 1.0


def _repeat_div(value: int, iterations: int) -> int:
    for _ in range(iterations):
        value = math.ceil(value / 2)
    return value


def downsample_mask(
    mask: torch.Tensor,
    batch_size: int,
    num_tokens: int,
    shape: tuple | None = None,
    latent_h: int | None = None,
    latent_w: int | None = None,
) -> torch.Tensor:
    """Downsample spatial mask to match attention layer token count.

    Supports two calling conventions:
      forge: downsample_mask(mask, bs, tokens, shape=(B,C,H,W))
      monkey: downsample_mask(mask, bs, tokens, latent_h=h, latent_w=w)

    Results are cached by (height, width, num_tokens, num_conds, batch_size)
    since these don't change between denoising steps — only per UNet layer.
    Clear cache via clear_mask_cache() when region masks are updated.
    """
    if shape is not None:
        height, width = shape[2], shape[3]
    else:
        height = latent_h
        width = latent_w
    num_conds = mask.shape[0]
    cache_key = (height, width, num_tokens, num_conds, batch_size)
    if cache_key in _mask_cache:
        return _mask_cache[cache_key]

    scale = max(0, math.ceil(math.log2(math.sqrt(height * width / num_tokens))))
    size = (_repeat_div(height, scale), _repeat_div(width, scale))
    if mask.dim() == 3:
        m = mask.unsqueeze(1)
    else:
        m = mask
    down = interpolate(m, size=size, mode="bilinear")
    flat = down.reshape(num_conds, -1)
    tok = flat.shape[1]
    if tok > num_tokens:
        flat = flat[:, :num_tokens]
    elif tok < num_tokens:
        pad = torch.zeros(num_conds, num_tokens - tok, device=flat.device, dtype=flat.dtype)
        flat = torch.cat([flat, pad], dim=1)
    result = flat.view(num_conds, num_tokens, 1).repeat_interleave(batch_size, dim=0)
    _mask_cache[cache_key] = result
    return result