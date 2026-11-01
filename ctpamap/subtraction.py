import numpy as np


def compute_iodine_map(
    ctpa: np.ndarray,
    pre_registered: np.ndarray,
    lung_mask: np.ndarray,
    vessel_mask: np.ndarray | None = None,
    airway_mask: np.ndarray | None = None,
    robust_percentile: float = 99.5,
    no_seg: bool = False,
) -> dict:
    raw = ctpa - pre_registered

    clean = raw.copy()
    clean[lung_mask == 0] = 0

    if no_seg:
        clean[np.abs(clean) > 150] = 0

        # Positive component
        pos = clean.copy()
        pos[pos < 0] = 0
        lung_pos = pos[lung_mask.astype(bool)]
        vmin_pos = 0.0
        vmax_pos = min(float(np.percentile(lung_pos, 99.5)) if len(lung_pos) > 0 else 1.0, 200.0)
        if vmax_pos < 1.0:
            vmax_pos = 1.0
        normalized_pos = np.clip((pos - vmin_pos) / (vmax_pos - vmin_pos), 0, 1)

        # Negative component (absolute value, 0 = no negative, max_neg = strongest)
        neg = clean.copy()
        neg[neg > 0] = 0
        neg = -neg
        lung_neg = neg[lung_mask.astype(bool)]
        vmax_neg = max(float(np.percentile(lung_neg, 99.5)) if len(lung_neg) > 0 else 1.0, 1.0)
        normalized_neg = 1.0 - np.clip(neg / vmax_neg, 0, 1)

        return {
            "raw": raw.astype(np.float32),
            "clean": clean.astype(np.float32),
            "normalized": normalized_pos.astype(np.float32),
            "normalized_neg": normalized_neg.astype(np.float32),
            "norm_min": float(vmin_pos),
            "norm_max": float(vmax_pos),
            "norm_max_neg": float(vmax_neg),
        }

    if vessel_mask is not None:
        clean[vessel_mask.astype(bool)] = 0
    if airway_mask is not None:
        clean[airway_mask.astype(bool)] = 0
    clean = np.clip(clean, -5, None)

    lung_voxels = clean[lung_mask.astype(bool)]
    if len(lung_voxels) > 0:
        upper = np.percentile(lung_voxels, robust_percentile)
        clean = np.clip(clean, -5, upper)

    lung_v = clean[lung_mask.astype(bool)]
    vmin = 0.0
    vmax = min(float(np.percentile(lung_v, 99.5)) if len(lung_v) > 0 else 1.0, 200.0)
    if vmax < 1.0:
        vmax = 1.0
    normalized = np.clip((clean - vmin) / (vmax - vmin), 0, 1)

    return {
        "raw": raw.astype(np.float32),
        "clean": clean.astype(np.float32),
        "normalized": normalized.astype(np.float32),
        "normalized_neg": None,
        "norm_min": float(vmin),
        "norm_max": float(vmax),
        "norm_max_neg": 0.0,
    }
