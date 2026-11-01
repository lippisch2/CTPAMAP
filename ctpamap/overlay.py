import numpy as np


def compute_iodine_overlay(
    ctpa: np.ndarray,
    iodine_map: np.ndarray,
    alpha: float = 0.6,
    lung_mask: np.ndarray | None = None,
    window_center: float = -600,
    window_width: float = 1500,
    no_seg: bool = False,
    iodine_map_neg: np.ndarray | None = None,
) -> np.ndarray:
    import matplotlib as mpl

    low = window_center - window_width / 2
    high = window_center + window_width / 2
    ctpa_norm = np.clip((ctpa - low) / (high - low), 0, 1)

    if no_seg and iodine_map_neg is not None:
        cmap_pos = mpl.colormaps["hot"]
        cmap_neg = mpl.colormaps["Blues_r"]

        color_pos = cmap_pos(iodine_map)[..., :3]
        color_neg = cmap_neg(iodine_map_neg)[..., :3]
    else:
        cmap = mpl.colormaps["hot"]
        color_pos = cmap(iodine_map)[..., :3]
        color_neg = None

    overlay = np.broadcast_to(ctpa_norm[..., None], (*ctpa.shape, 3)).copy()

    if lung_mask is None:
        lung_mask = np.ones_like(ctpa, dtype=bool)
    else:
        lung_mask = lung_mask.astype(bool)

    # Proper alpha compositing. The previous code did
    #   blended = (1-α)*bg + α*pos   then   blended += α*neg
    # so with α=0.6 the effective weight on the background could be as low as
    # 1-α-α = -0.2 and the two colour contributions summed to 1.2α, which
    # oversaturated/shifted colours. Here we accumulate a total weight and
    # renormalise so the result is always a convex combination.
    if color_neg is not None:
        total_color = color_pos + color_neg
        weight = np.clip(total_color.sum(axis=-1, keepdims=True), 0.0, 1.0)
        # Blend background with the (normalised) combined colour.
        blended = (1 - alpha * weight) * ctpa_norm[..., None] + alpha * total_color * weight
        # Guard against the degenerate case where both pos and neg are 0:
        blended = np.where(weight > 0, blended, ctpa_norm[..., None])
    else:
        blended = (1 - alpha) * ctpa_norm[..., None] + alpha * color_pos

    overlay = np.where(lung_mask[..., None], blended, overlay)

    return np.clip(overlay, 0, 1).astype(np.float32)
