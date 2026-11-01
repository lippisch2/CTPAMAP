import numpy as np
import json


def _infer_z_axis(affine: np.ndarray | None) -> int:
    """Infer the axial (superior/inferior) array axis from a volume affine.

    Volumes are reoriented to canonical RAS and stored in nibabel (X, Y, Z)
    memory order, so the axial axis is the one whose affine column points most
    along the physical +Z (superior) direction. Falls back to axis 2 (the
    nibabel convention) when no affine is available.
    """
    if affine is None:
        return 2
    aff = np.asarray(affine, dtype=float)
    if aff.shape[0] < 3 or aff.shape[1] < 3:
        return 2
    cols = aff[:3, :3]
    z = np.array([0.0, 0.0, 1.0])
    dot = np.abs(cols.T @ z)
    return int(np.argmax(dot))


def _compute_diaphragm_displacement(
    ctpa: np.ndarray, pre_reg: np.ndarray, lung_mask: np.ndarray,
    z_spacing_mm: float = 1.0, z_axis: int = 2,
) -> float:
    """Estimate diaphragm mis-registration in millimetres.

    The previous implementation scanned axis 0 and labelled the result "mm",
    so slices were compared against the mm threshold. The useful axis is the
    axial (I/S) one (nibabel axis 2 in RAS); the best slice shift is multiplied
    by the axial slice spacing to yield mm. The search range (-20..20 slices)
    is reported in slices.
    """
    z_slices = ctpa.shape[z_axis]
    other = tuple(a for a in range(3) if a != z_axis)

    lung_axial = lung_mask.sum(axis=other) > 0
    lung_present = np.where(lung_axial)[0]

    if len(lung_present) < 10:
        return 0.0

    bottom_slice_ctpa = lung_present[-1]

    corr = []
    for d in range(-20, 21):
        shift_slice = bottom_slice_ctpa + d
        if 0 <= shift_slice < z_slices:
            slice_ctpa = np.take(ctpa, bottom_slice_ctpa, axis=z_axis)
            slice_pre = np.take(pre_reg, min(shift_slice, pre_reg.shape[z_axis] - 1), axis=z_axis)
            if slice_ctpa.std() > 0 and slice_pre.std() > 0:
                c = np.corrcoef(slice_ctpa.ravel(), slice_pre.ravel())[0, 1]
                corr.append((d, c))

    if not corr:
        return 0.0
    best_d = max(corr, key=lambda x: x[1])[0]
    return float(best_d * z_spacing_mm)


def _compute_registration_mismatch(
    ctpa: np.ndarray, pre_reg: np.ndarray, lung_mask: np.ndarray
) -> float:
    """Normalised residual intensity mismatch after registration.

    The previous formula `std / max(1, |mean|)` was backwards: a well-registered
    pair has a near-zero mean residual, which made the metric *larger*, so good
    registration was penalised. The natural, well-behaved measure of residual
    mismatch (used widely for CT subtraction QC) is the coefficient of variation
    of the signed difference within the lung: std(diff) / mean(|ctpa|). Smaller
    is better, and it is scale-stable (it doesn't blow up when mean→0).
    """
    mask_bool = lung_mask.astype(bool)
    lung_ctpa = ctpa[mask_bool]
    lung_pre = pre_reg[mask_bool]

    if len(lung_ctpa) < 100:
        return 1.0

    diff = lung_ctpa - lung_pre
    # Restrict to voxels where signal actually exists (avoids pure-air floor).
    diff_lung = diff[lung_ctpa > -500]
    if len(diff_lung) < 100:
        return 1.0

    denom = max(1.0, float(np.mean(np.abs(lung_ctpa))))
    return float(np.std(diff_lung) / denom)


def _compute_invalid_lung_voxels(
    iodine_map: np.ndarray, lung_mask: np.ndarray,
    lo: float = -5.0, hi: float = 400.0,
) -> float:
    """Fraction of lung voxels with implausible iodine values.

    The previous check used `< 0 | > 200`, but the iodine map was already
    clipped to [-5, upper] and zeroed outside the lung, so it was always ~0 and
    thus a vacuous gate. A more meaningful out-of-range band for a clipped map
    is a physiologically implausible negative or very-high positive residual.
    """
    mask_bool = lung_mask.astype(bool)
    total_lung = int(mask_bool.sum())
    if total_lung == 0:
        return 0.0
    invalid = int(((iodine_map < lo) | (iodine_map > hi))[mask_bool].sum())
    return float(invalid / total_lung) * 100.0


def _detect_artifact_flags(
    iodine_map: np.ndarray, lung_mask: np.ndarray, ctpa: np.ndarray,
    z_axis: int = 2,
) -> list[dict]:
    flags = []

    z_slices = iodine_map.shape[z_axis]
    other = tuple(a for a in range(3) if a != z_axis)
    lung_axial = lung_mask.sum(axis=other) > 0
    lung_present = np.where(lung_axial)[0]

    if len(lung_present) > 5:
        for region_name, frac in [("heart_base", 0.15), ("fissure_region", 0.1)]:
            n_slices = max(1, int(len(lung_present) * frac))
            if region_name == "heart_base":
                check_slices = lung_present[-n_slices:]
            else:
                check_slices = lung_present[len(lung_present) // 3: 2 * len(lung_present) // 3]
            for z in check_slices:
                slice_iodine = np.take(iodine_map, z, axis=z_axis)
                slice_lung = np.take(lung_mask, z, axis=z_axis).astype(bool)
                if slice_lung.sum() > 50:
                    mean_val = float(slice_iodine[slice_lung].mean())
                    max_val = float(slice_iodine[slice_lung].max())
                    if max_val > 100 and mean_val < 10:
                        flags.append({
                            "slice": int(z),
                            "region": region_name,
                            "type": "possible_high_frequency_noise",
                            "mean_iodine": mean_val,
                            "max_iodine": max_val,
                        })
                        break

    return flags


def compute_qc_metrics(
    ctpa: np.ndarray,
    pre_registered: np.ndarray,
    lung_mask: np.ndarray,
    iodine_map: np.ndarray,
    z_spacing_mm: float = 1.0,
    affine: np.ndarray | None = None,
) -> dict:
    z_axis = _infer_z_axis(affine)
    displacement = _compute_diaphragm_displacement(
        ctpa, pre_registered, lung_mask, z_spacing_mm, z_axis)
    mismatch = _compute_registration_mismatch(ctpa, pre_registered, lung_mask)
    pct_invalid = _compute_invalid_lung_voxels(iodine_map, lung_mask)
    artifact_flags = _detect_artifact_flags(iodine_map, lung_mask, ctpa, z_axis)

    return {
        "diaphragm_displacement_mm": displacement,
        "registration_mismatch_cov": mismatch,
        "percent_invalid_lung_voxels": pct_invalid,
        "artifact_flags": artifact_flags,
        # Thresholds: small displacement (mm), small residual CV (<0.5 ≈ good
        # registration), few invalid voxels.
        "qc_pass": pct_invalid < 10 and mismatch < 0.5 and displacement < 15,
    }


def save_qc_report(qc_metrics: dict, path: str) -> None:
    with open(path, "w") as f:
        json.dump(qc_metrics, f, indent=2)