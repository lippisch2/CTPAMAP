import numpy as np
from scipy.ndimage import gaussian_filter


def displacement_to_jacobian(disp: np.ndarray, spacing: tuple) -> np.ndarray:
    """Compute Jacobian determinant from displacement field.

    disp:    (3, Z, Y, X) displacement in physical units (mm), in sitk voxel
             order — components indexed x, y, z and array axes z, y, x in mm.
    spacing: (sz, sy, sx) voxel spacing in mm (z, y, x).

    Returns: (X, Y, Z) Jacobian determinant J in nibabel RAS order, matching
    the pipeline volumes (compute_iodine_map, overlay, qc).
    """
    sz, sy, sx = spacing

    dux_dz, dux_dy, dux_dx = np.gradient(disp[0], sz, sy, sx)
    duy_dz, duy_dy, duy_dx = np.gradient(disp[1], sz, sy, sx)
    duz_dz, duz_dy, duz_dx = np.gradient(disp[2], sz, sy, sx)

    f00 = 1.0 + dux_dx
    f01 = dux_dy
    f02 = dux_dz
    f10 = duy_dx
    f11 = 1.0 + duy_dy
    f12 = duy_dz
    f20 = duz_dx
    f21 = duz_dy
    f22 = 1.0 + duz_dz

    det = (f00 * (f11 * f22 - f12 * f21)
         - f01 * (f10 * f22 - f12 * f20)
         + f02 * (f10 * f21 - f11 * f20))

    # (Z, Y, X) sitk order -> (X, Y, Z) nibabel order.
    det = np.transpose(det, (2, 1, 0))
    det = gaussian_filter(det, sigma=2.0, mode='nearest')

    return det.astype(np.float32)


def correct_aeration(ctpa: np.ndarray, jac_det: np.ndarray) -> np.ndarray:
    """Correct CTPA for expiratory/inspiratory aeration differences.

    ctpa:    (X, Y, Z) post-contrast CTPA in HU (nibabel RAS order)
    jac_det: (X, Y, Z) Jacobian determinant from registration

    Returns: aeration-corrected CTPA.
    """
    j = np.clip(jac_det, 0.1, 3.0)
    return (-1000.0 + (ctpa + 1000.0) / j).astype(np.float32)
